"""WildDet3D inference wrapper.

Provides a simple forward() interface for WildDet3D inference:

Supports three prompt types with 5-mode text labels:
- Text prompt: input_texts=["chair", "table"]
- Box prompt: input_boxes=[[x1, y1, x2, y2]] (pixel xyxy)
- Point prompt: input_points=[[(x, y, label), ...]] (pixel coords,
  label: 1=pos, 0=neg)

5-mode support via prompt_text parameter (for box/point prompts):
- "visual"          -> VISUAL mode (one-to-many, no category label)
- "visual: car"     -> VISUAL+LABEL mode (one-to-many, with category)
- "geometric"       -> GEOMETRY mode (one-to-one, no category label)
- "geometric: car"  -> GEOMETRY+LABEL mode (one-to-one, with category)
- "object"          -> default (backward compatible)

Example usage:
    from wilddet3d.inference import build_model
    from wilddet3d.preprocessing import preprocess

    # Build model
    model = build_model(
        checkpoint="path/to/checkpoint.ckpt"
    )

    # Preprocess data
    data = preprocess(image, intrinsics)

    # TEXT mode
    boxes, boxes3d, scores, class_ids, depth_maps = model(
        images=data["images"],
        intrinsics=data["intrinsics"],
        input_hw=[data["input_hw"]],
        original_hw=[data["original_hw"]],
        padding=[data["padding"]],
        input_texts=["chair", "table"],
    )

    # VISUAL mode (box prompt, one-to-many)
    boxes, boxes3d, scores, class_ids, depth_maps = model(
        ...,
        input_boxes=[[100, 200, 300, 400]],
        prompt_text="visual",
    )

    # GEOMETRY mode (box prompt, one-to-one)
    boxes, boxes3d, scores, class_ids, depth_maps = model(
        ...,
        input_boxes=[[100, 200, 300, 400]],
        prompt_text="geometric",
    )

    # Point prompt (works with any prompt_text)
    boxes, boxes3d, scores, class_ids, depth_maps = model(
        ...,
        input_points=[[(150, 250, 1), (200, 300, 0)]],
        prompt_text="geometric",
    )
"""

from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

from wilddet3d.data_types import WildDet3DInput
from wilddet3d.depth import LingbotDepthBackend
from wilddet3d.depth.depth_fusion import EarlyDepthFusionLingbot
from wilddet3d.head import Det3DCoder, RoI2Det3D
from wilddet3d.model import WildDet3D


class WildDet3DPredictor(nn.Module):
    """WildDet3D wrapper with a simple forward() interface.

    Provides a simple forward() interface:
        boxes, boxes3d, scores, class_ids, depth_maps = model(
            images=...,
            intrinsics=...,
            input_texts=["chair", "table"],
        )
    """

    def __init__(
        self,
        wilddet3d: WildDet3D,
        score_threshold: float = 0.3,
    ):
        super().__init__()
        self.wilddet3d = wilddet3d
        self.score_threshold = score_threshold

    def forward(
        self,
        images: Tensor,
        intrinsics: Optional[Tensor],
        input_hw: List[Tuple[int, int]],
        original_hw: List[Tuple[int, int]],
        padding: List[Tuple[int, int, int, int]],
        # Prompt types (mutually exclusive)
        input_texts: Optional[List[str]] = None,
        input_boxes: Optional[List[List[float]]] = None,
        input_points: Optional[
            List[List[Tuple[float, float, int]]]
        ] = None,
        # Text label for box/point prompts (5-mode support)
        # e.g. "visual", "visual: car", "geometric", "geometric: car"
        prompt_text: str = "object",
        return_predicted_intrinsics: bool = False,
        # Optional depth input (e.g., from LiDAR)
        depth_gt: Optional[Tensor] = None,  # (B, 1, H, W) meters
    ) -> Tuple[
        List[Tensor],
        List[Tensor],
        List[Tensor],
        List[Tensor],
        Optional[List[Tensor]],
        Optional[Tensor],
    ]:
        """Forward with simple interface.

        Args:
            images: (B, 3, H, W) preprocessed images
            intrinsics: (B, 3, 3) camera intrinsics, or None to use
                predicted
            input_hw: List of (H, W) tuples for each image
            original_hw: List of original (H, W) tuples
            padding: List of (left, right, top, bottom) padding tuples
            input_texts: Text prompts (e.g., ["chair", "table"])
            input_boxes: Box prompts per image, pixel xyxy
                [[x1,y1,x2,y2], ...]
            input_points: Point prompts per image
                [[(x,y,label), ...], ...]
            prompt_text: Text label for box/point prompts. Controls
                5-mode: "object" (default), "visual", "visual: car",
                "geometric", "geometric: car"
            return_predicted_intrinsics: Whether to return predicted
                intrinsics
            depth_gt: Optional depth input (B, 1, H, W) in meters

        Returns:
            boxes: List of 2D boxes per image (pixel xyxy)
            boxes3d: List of 3D boxes per image
            scores: List of confidence scores per image
            class_ids: List of class IDs per image
            depth_maps: List of depth maps per image (or None)
            predicted_intrinsics: (B, 3, 3) predicted intrinsics
                (if requested)
        """
        device = images.device
        B = images.shape[0]
        H, W = input_hw[0]

        # Determine prompt type and create batch
        if input_texts is not None:
            batch = self._create_text_batch(
                images,
                intrinsics,
                input_texts,
                device,
                padding=padding,
            )
            class_names = input_texts
        elif input_boxes is not None:
            batch = self._create_box_batch(
                images,
                intrinsics,
                input_boxes,
                (H, W),
                device,
                text=prompt_text,
                padding=padding,
            )
            class_names = [prompt_text]
        elif input_points is not None:
            batch = self._create_point_batch(
                images,
                intrinsics,
                input_points,
                (H, W),
                device,
                text=prompt_text,
                padding=padding,
            )
            class_names = [prompt_text]
        else:
            raise ValueError(
                "Must provide one of: input_texts, input_boxes, "
                "input_points"
            )

        # Attach depth input if provided
        if depth_gt is not None:
            batch.depth_gt = depth_gt

        # Run inference
        with torch.no_grad():
            output = self.wilddet3d(batch)

        # Output is Det3DOut with per-image lists
        boxes = output.boxes
        boxes3d = output.boxes3d
        scores = output.scores
        scores_2d = output.scores_2d
        scores_3d = output.scores_3d
        class_ids = output.class_ids
        depth_maps = output.depth_maps

        # Apply score threshold and rescale boxes to original size
        boxes_out = []
        boxes3d_out = []
        scores_out = []
        scores_2d_out = []
        scores_3d_out = []
        class_ids_out = []

        for i in range(B):
            # Filter by 2D score
            mask = scores[i] >= self.score_threshold
            img_scores = scores[i][mask]
            img_scores_2d = (
                scores_2d[i][mask]
                if scores_2d is not None
                else torch.zeros_like(img_scores)
            )
            img_scores_3d = (
                scores_3d[i][mask]
                if scores_3d is not None
                else torch.zeros_like(img_scores)
            )
            img_boxes = boxes[i][mask]
            img_boxes3d = boxes3d[i][mask]
            img_class_ids = class_ids[i][mask]

            # Rescale 2D boxes from input_hw to original_hw
            # Account for padding
            pad_left, pad_right, pad_top, pad_bottom = padding[i]
            orig_h, orig_w = original_hw[i]

            # Remove padding offset and rescale
            img_boxes = img_boxes.clone()
            img_boxes[:, 0] -= pad_left  # x1
            img_boxes[:, 2] -= pad_left  # x2
            img_boxes[:, 1] -= pad_top  # y1
            img_boxes[:, 3] -= pad_top  # y2

            # Scale from padded size to original
            padded_h = H - pad_top - pad_bottom
            padded_w = W - pad_left - pad_right
            scale_x = orig_w / padded_w
            scale_y = orig_h / padded_h

            img_boxes[:, 0::2] *= scale_x
            img_boxes[:, 1::2] *= scale_y

            # Clamp to image bounds
            img_boxes[:, 0::2] = img_boxes[:, 0::2].clamp(0, orig_w)
            img_boxes[:, 1::2] = img_boxes[:, 1::2].clamp(0, orig_h)

            boxes_out.append(img_boxes)
            boxes3d_out.append(img_boxes3d)
            scores_out.append(img_scores)
            scores_2d_out.append(img_scores_2d)
            scores_3d_out.append(img_scores_3d)
            class_ids_out.append(img_class_ids)

        # Get predicted intrinsics and confidence maps if available
        predicted_K = output.predicted_intrinsics
        confidence_maps = output.confidence_maps

        if return_predicted_intrinsics:
            return (
                boxes_out,
                boxes3d_out,
                scores_out,
                scores_2d_out,
                scores_3d_out,
                class_ids_out,
                depth_maps,
                predicted_K,
                confidence_maps,
            )
        else:
            return (
                boxes_out,
                boxes3d_out,
                scores_out,
                scores_2d_out,
                scores_3d_out,
                class_ids_out,
                depth_maps,
            )

    def _create_text_batch(
        self,
        images: Tensor,
        intrinsics: Tensor,
        texts: List[str],
        device: torch.device,
        padding: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> WildDet3DInput:
        """Create batch for text prompts."""
        n_prompts = len(texts)

        return WildDet3DInput(
            images=images,
            intrinsics=intrinsics,
            img_ids=torch.zeros(
                n_prompts, dtype=torch.long, device=device
            ),
            text_ids=torch.arange(
                n_prompts, dtype=torch.long, device=device
            ),
            unique_texts=texts,
            padding=padding,
        )

    def _create_box_batch(
        self,
        images: Tensor,
        intrinsics: Tensor,
        boxes_xyxy: List[List[float]],
        input_hw: Tuple[int, int],
        device: torch.device,
        text: str = "object",
        padding: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> WildDet3DInput:
        """Create batch for box prompts.

        Args:
            text: Text label for the prompt. Controls 5-mode behavior:
                "visual" / "visual: car" for one-to-many matching,
                "geometric" / "geometric: car" for one-to-one matching.
        """
        H, W = input_hw
        n_prompts = len(boxes_xyxy)

        # Convert pixel xyxy to normalized cxcywh
        boxes_cxcywh = []
        for box in boxes_xyxy:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2 / W
            cy = (y1 + y2) / 2 / H
            w = (x2 - x1) / W
            h = (y2 - y1) / H
            boxes_cxcywh.append([cx, cy, w, h])

        geo_boxes = torch.tensor(
            boxes_cxcywh, dtype=torch.float32, device=device
        )
        geo_boxes = geo_boxes.unsqueeze(1)  # (n_prompts, 1, 4)

        return WildDet3DInput(
            images=images,
            intrinsics=intrinsics,
            img_ids=torch.zeros(
                n_prompts, dtype=torch.long, device=device
            ),
            text_ids=torch.zeros(
                n_prompts, dtype=torch.long, device=device
            ),
            unique_texts=[text],
            geo_boxes=geo_boxes,
            geo_boxes_mask=torch.zeros(
                n_prompts, 1, dtype=torch.bool, device=device
            ),
            geo_box_labels=torch.ones(
                n_prompts, 1, dtype=torch.long, device=device
            ),
            padding=padding,
        )

    def _create_point_batch(
        self,
        images: Tensor,
        intrinsics: Tensor,
        points_list: List[List[Tuple[float, float, int]]],
        input_hw: Tuple[int, int],
        device: torch.device,
        text: str = "object",
        padding: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> WildDet3DInput:
        """Create batch for point prompts.

        Args:
            text: Text label for the prompt. Controls 5-mode behavior:
                "visual" / "visual: car" for one-to-many matching,
                "geometric" / "geometric: car" for one-to-one matching.
        """
        H, W = input_hw
        n_prompts = len(points_list)

        # Find max points per prompt for padding
        max_points = max(len(pts) for pts in points_list)

        # Normalize and pad points
        geo_points = torch.zeros(
            n_prompts, max_points, 2, device=device
        )
        geo_point_labels = torch.zeros(
            n_prompts, max_points, dtype=torch.long, device=device
        )
        geo_points_mask = torch.ones(
            n_prompts, max_points, dtype=torch.bool, device=device
        )

        for i, pts in enumerate(points_list):
            for j, (x, y, label) in enumerate(pts):
                geo_points[i, j] = torch.tensor([x / W, y / H])
                geo_point_labels[i, j] = label
                geo_points_mask[i, j] = False  # False = valid

        return WildDet3DInput(
            images=images,
            intrinsics=intrinsics,
            img_ids=torch.zeros(
                n_prompts, dtype=torch.long, device=device
            ),
            text_ids=torch.zeros(
                n_prompts, dtype=torch.long, device=device
            ),
            unique_texts=[text],
            geo_points=geo_points,
            geo_points_mask=geo_points_mask,
            geo_point_labels=geo_point_labels,
            padding=padding,
        )


def build_model(
    checkpoint: str,
    sam3_checkpoint: str = "pretrained/sam3/sam3_detector.pt",
    score_threshold: float = 0.3,
    nms: bool = True,
    iou_threshold: float = 0.6,
    device: str = "cuda",
    backbone_freeze_blocks: int = 28,
    lingbot_encoder_freeze_blocks: int = 21,
    ambiguous_rotation: bool = False,
    canonical_rotation: bool = False,
    use_depth_input_test: bool = False,
    use_predicted_intrinsics: bool = False,
    skip_pretrained: bool = False,
) -> WildDet3DPredictor:
    """Build WildDet3D model with LingBot-Depth backend.

    Args:
        checkpoint: Path to trained WildDet3D checkpoint (.ckpt file)
        sam3_checkpoint: Path to SAM3 pretrained weights
        score_threshold: Confidence threshold for filtering
        nms: Whether to apply NMS
        iou_threshold: IoU threshold for NMS
        device: Device to load model on
        backbone_freeze_blocks: Number of SAM3 ViT blocks to freeze.
        lingbot_encoder_freeze_blocks: Number of LingBot encoder blocks
            to freeze.
        use_predicted_intrinsics: If True, use geometry backend's
            predicted intrinsics (K_pred) for 3D box decoding instead of
            the input intrinsics. Useful for in-the-wild images without
            GT intrinsics.
        skip_pretrained: If True, skip loading pretrained weights for
            SAM3 and LingBot-Depth. Use this for inference when the
            training checkpoint already contains all weights (avoids
            loading ~4GB of pretrained weights that get immediately
            overwritten).

    Returns:
        WildDet3DPredictor model ready for inference
    """
    print("Building WildDet3D model with LingBot-Depth backend...")

    # When skip_pretrained=True, patch MDMModel.from_pretrained to build
    # model structure from config without loading weights (~1GB saved).
    _mdm_patch_cleanup = None
    if skip_pretrained:
        from mdm.model.v2 import MDMModel

        _orig_from_pretrained = MDMModel.from_pretrained

        @classmethod
        def _from_pretrained_config_only(cls, path, **kwargs):
            from pathlib import Path as P

            from huggingface_hub import hf_hub_download

            if P(path).exists():
                cp = path
            else:
                cp = hf_hub_download(
                    repo_id=path,
                    repo_type="model",
                    filename="model.pt",
                    **kwargs,
                )
            ckpt = torch.load(
                cp, map_location="cpu", weights_only=True
            )
            model = cls(**ckpt["model_config"])
            print(
                f"[LingbotDepth] Built model structure from config "
                f"(skipped pretrained weights)"
            )
            return model

        MDMModel.from_pretrained = _from_pretrained_config_only
        _mdm_patch_cleanup = lambda: setattr(
            MDMModel, "from_pretrained", _orig_from_pretrained
        )

    # Build geometry backend (LingBot-Depth)
    geometry_backend = LingbotDepthBackend(
        pretrained_model="robbyant/lingbot-depth-postrain-dc-vitl14",
        num_tokens=2400,
        target_latent_dim=256,
        depth_loss_weight=1.0,
        silog_loss_weight=0.5,
        monocular_prob=0.7,
        masked_prob=0.2,
        mask_ratio_range=(0.6, 0.9),
        mask_patch_size=14,
        camera_loss_weight=1.0,
        detach_depth_latents=True,
        encoder_freeze_blocks=lingbot_encoder_freeze_blocks,
    )

    # Restore original from_pretrained
    if _mdm_patch_cleanup is not None:
        _mdm_patch_cleanup()

    # Build components
    box_coder = Det3DCoder(
        ambiguous_rotation=ambiguous_rotation,
        canonical_rotation=canonical_rotation,
    )
    roi2det3d = RoI2Det3D(
        box_coder=box_coder,
        score_threshold=0.0,  # Threshold in wrapper
        nms=nms,
        iou_threshold=iou_threshold,
    )

    # ControlNet-style fusion for LingBot-Depth
    early_depth_fusion = EarlyDepthFusionLingbot(
        visual_dim=256,
        depth_dim=256,
        zero_init=True,
    )

    # Build WildDet3D
    # When skip_pretrained=True, build SAM3 model structure without
    # loading pretrained weights (~3.2GB) since the training checkpoint
    # already contains all weights.
    if skip_pretrained:
        from sam3.model_builder import build_sam3_image_model

        print(
            "[skip_pretrained] Building SAM3 structure without "
            "pretrained weights..."
        )
        sam3_model = build_sam3_image_model(
            checkpoint_path=None,
            load_from_HF=False,
            device="cpu",
            eval_mode=False,
            enable_segmentation=False,
        )
    else:
        sam3_model = None

    wilddet3d = WildDet3D(
        sam3_model=sam3_model if skip_pretrained else None,
        sam3_checkpoint=None if skip_pretrained else sam3_checkpoint,
        box_coder=box_coder,
        geometry_backend=geometry_backend,
        roi2det3d=roi2det3d,
        early_depth_fusion=early_depth_fusion,
        backbone_freeze_blocks=backbone_freeze_blocks,
        use_depth_input_test=use_depth_input_test,
        use_predicted_intrinsics=use_predicted_intrinsics,
    )

    # Load trained checkpoint
    import os as _os
    if _os.environ.get("EVAL_SKIP_TRUNK_SD", "0") == "1":
        # eval-server fast path (2026-07-09): the policy checkpoint's state_dict already
        # contains every frozen-trunk weight and is loaded right after instantiate --
        # reading the 4.7GB base ckpt here is duplicate I/O. NEVER set for from-scratch
        # training (no policy ckpt to overwrite these weights there).
        print(f"EVAL_SKIP_TRUNK_SD=1: skipping base ckpt weights ({checkpoint})")
    else:
        print(f"Loading checkpoint: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        # Remove "model." prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = (
                k.replace("model.", "") if k.startswith("model.") else k
            )
            new_state_dict[new_key] = v

        wilddet3d.load_state_dict(new_state_dict, strict=False)
    wilddet3d = wilddet3d.to(device)
    wilddet3d.eval()

    # Wrap with predictor interface
    model = WildDet3DPredictor(
        wilddet3d, score_threshold=score_threshold
    )
    model = model.to(device)
    model.eval()

    print("Model ready!")
    return model
