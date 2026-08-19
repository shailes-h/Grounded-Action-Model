"""Masked RGB-D patch tokens from the WildDet3D backbone.

One forward of WildDet3D (SAM3 detector + LingBot RGB-D depth backend, metric depth
injected) per object click gives
    * the backbone's 49x49x1024 RGB-D feature map  -> 3x3 average pooled to a 16x16 grid,
    * a 2D box and a 3D oriented box for the clicked object.
Cells outside (object boxes U robot-arm mask) are zeroed ("masked patch tokens", 256x1024).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
import torch

INPUT = 1008                    # backbone input resolution (square)
FEAT = 49                       # backbone feature map is 49x49; pooled to grid x grid tokens
IMEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
ISTD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _quat_to_R(q):
    w, x, y, z = q / max(np.linalg.norm(q), 1e-9)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


@dataclass
class Detection:
    box2d: np.ndarray                    # (4,) pixel xyxy in the input image
    center: Optional[np.ndarray] = None  # (3,) world
    extents: Optional[np.ndarray] = None  # (3,) full size
    R: Optional[np.ndarray] = None       # (3,3) columns = axes (world)
    score: float = 0.0
    hidden: Optional[np.ndarray] = None  # (256,) detector query hidden state


@dataclass
class PatchResult:
    patches: torch.Tensor                # (grid*grid, 1024) float32, masked
    mask16: np.ndarray                   # (grid, grid) bool kept cells
    detections: List[Detection] = field(default_factory=list)


class PatchTokenExtractor:
    """Wraps a WildDet3D checkpoint and produces masked patch tokens + per-click 3D boxes.

    Args:
        checkpoint: WildDet3D .pt (see scripts/download_weights.sh).
        channel_swap: feed the image with swapped colour channels. Set this to whatever
            the policy checkpoint was trained with (stored in its config as
            `image_channel_swap`).
        mask_level: "feature" zeroes grid cells outside the mask after pooling;
            "pixel" blackens image pixels outside the mask before the backbone.
        grid: patch-token grid size (grid x grid tokens; 16 or 24 are sensible; must match
            the policy checkpoint's `encoder.n_patch = grid**2`).
    """

    def __init__(self, checkpoint: str, device: str = "cuda", channel_swap: bool = False,
                 mask_level: str = "feature", grid: int = 16):
        from wilddet3d.inference import build_model
        assert mask_level in ("feature", "pixel")
        assert 1 <= int(grid) <= FEAT
        self.grid = int(grid)
        self.dev = device
        self.channel_swap = bool(channel_swap)
        self.mask_level = mask_level
        self.pred = build_model(checkpoint=checkpoint, score_threshold=0.0, nms=False,
                                iou_threshold=0.6, skip_pretrained=True).to(device).eval()
        # build_model() only builds the structure: load the weights explicitly and verify.
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        sd = ck.get("model") or ck
        self.pred.wilddet3d.load_state_dict(sd, strict=False)
        k = [k for k in sd if "bbox3d_head" in k and k.endswith("weight")][0]
        a = float(sd[k].float().abs().sum()); b = float(self.pred.wilddet3d.state_dict()[k].float().abs().sum())
        assert abs(a - b) < 1e-2, f"weight load failed for {k}"
        del ck, sd
        gb = [m for n, m in self.pred.named_modules() if n.endswith("geometry_backend")][0]
        head = [m for n, m in self.pred.named_modules() if type(m).__name__ == "Det3DHead"][0]
        self._g = {}

        def hk_enc(m, i, o):
            for x in (o if isinstance(o, (list, tuple)) else [o]):
                if torch.is_tensor(x) and x.dim() == 4 and x.shape[-1] == 49 and x.shape[1] >= 512:
                    self._g["enc"] = x.detach()

        def hk_hid(m, inp):
            x = inp[0]
            if torch.is_tensor(x) and x.dim() == 3 and x.shape[-1] == 256:
                self._g["hid"] = x.detach()

        # metric depth injection: the predictor's depth_gt argument does not reach the
        # geometry backend at inference; route it through forward_test explicitly.
        gb.forward_test = type(gb).forward_test.__get__(gb)
        orig = gb.forward_test

        def inj(*a, **k):
            if self._dgt is not None:
                k["depth_gt"] = self._dgt
            return orig(*a, **k)
        gb.forward_test = inj
        self._dgt = None
        gb.encoder.register_forward_hook(hk_enc)
        for rb in head.reg_branches:
            rb.register_forward_pre_hook(hk_hid)

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray, depth_m: np.ndarray, K: np.ndarray, E: np.ndarray,
                 clicks_px: List, arm_mask: Optional[np.ndarray] = None,
                 extra_boxes_px: Optional[List] = None) -> PatchResult:
        """rgb (H,W,3) uint8 RGB; depth_m (H,W) metres; K (3,3); E (3,4) world->cam;
        clicks_px [(u,v), ...] one per object; arm_mask optional (H,W) bool;
        extra_boxes_px optional extra pixel boxes to keep in the mask."""
        H0, W0 = rgb.shape[:2]
        E = np.asarray(E, np.float64).reshape(3, 4); K = np.asarray(K, np.float64)
        sx, sy = INPUT / W0, INPUT / H0
        img = rgb
        if self.mask_level == "pixel":
            m = np.zeros((H0, W0), bool)
            if arm_mask is not None:
                m |= arm_mask
            for b in (extra_boxes_px or []):
                x0, y0, x1, y1 = [int(v) for v in b]
                m[max(y0, 0):min(y1 + 1, H0), max(x0, 0):min(x1 + 1, W0)] = True
            img = np.where(m[..., None], rgb, 0).astype(np.uint8)
        im = cv2.resize(img, (INPUT, INPUT))
        if self.channel_swap:
            im = im[:, :, ::-1]
        x = torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float()[None] / 255.0
        x = ((x - IMEAN) / ISTD).to(self.dev)
        Ks = torch.from_numpy((np.diag([sx, sy, 1.0]) @ K).astype(np.float32)).to(self.dev)
        dgt = torch.from_numpy(cv2.resize(np.asarray(depth_m, np.float32), (INPUT, INPUT),
                                          interpolation=cv2.INTER_NEAREST)).to(self.dev)[None, None]
        dets: List[Detection] = []
        for (u, v) in clicks_px:
            self._g.pop("hid", None)
            self._dgt = dgt
            out = self.pred(images=x, intrinsics=Ks[None], input_hw=[(INPUT, INPUT)],
                            original_hw=[(INPUT, INPUT)], padding=[(0, 0, 0, 0)],
                            input_points=[[(float(u) * sx, float(v) * sy, 1)]],
                            prompt_text="visual", depth_gt=dgt)
            self._dgt = None
            b2, b3, sc = out[0][0], out[1][0], out[2][0]
            if not (torch.is_tensor(sc) and sc.numel()):
                dets.append(Detection(box2d=np.zeros(4, np.float32)))
                continue
            j = int(sc.argmax())
            box2d = b2[j].float().cpu().numpy() / INPUT * [W0, H0, W0, H0]
            d = Detection(box2d=box2d.astype(np.float32), score=float(sc[j]))
            hid = self._g.get("hid")
            if hid is not None and hid.shape[1] > j:
                d.hidden = hid[0, j].float().cpu().numpy()
            if torch.is_tensor(b3) and b3.numel():
                bv = b3[j].float().cpu().numpy()
                c_cam, whl, quat = bv[:3], bv[3:6], bv[6:10]
                Rc = _quat_to_R(quat)
                d.center = (E[:, :3].T @ (c_cam - E[:, 3])).astype(np.float32)
                d.extents = whl.astype(np.float32)
                d.R = (E[:, :3].T @ Rc).astype(np.float32)
            dets.append(d)

        # 49x49 -> grid x grid pooled tokens, then the object/arm mask
        G = self.grid
        F = self._g["enc"][0].float().cpu().numpy().transpose(1, 2, 0)   # (49,49,C)
        T16 = np.zeros((G, G, F.shape[-1]), np.float32)
        for a in range(G):
            for b in range(G):
                a0, a1 = a * FEAT // G, FEAT if a == G - 1 else (a + 1) * FEAT // G
                b0, b1 = b * FEAT // G, FEAT if b == G - 1 else (b + 1) * FEAT // G
                T16[a, b] = F[a0:a1, b0:b1].mean((0, 1))
        m16 = np.zeros((G, G), bool)
        boxes = [d.box2d for d in dets if d.box2d[2] > d.box2d[0]] + list(extra_boxes_px or [])
        for (x0, y0, x1, y1) in boxes:
            c0, r0 = int(x0) * G // W0, int(y0) * G // H0
            c1, r1 = -(-int(x1) * G // W0), -(-int(y1) * G // H0)
            m16[max(r0 - 1, 0):min(r1 + 2, G), max(c0 - 1, 0):min(c1 + 2, G)] = True
        if arm_mask is not None:
            m16 |= cv2.resize(arm_mask.astype(np.float32), (G, G),
                              interpolation=cv2.INTER_AREA) > 0.10
        if self.mask_level == "feature":
            T16[~m16] = 0.0
        return PatchResult(patches=torch.from_numpy(T16.reshape(G * G, -1)),
                           mask16=m16, detections=dets)
