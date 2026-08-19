"""WildDet3D data types."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import List, NamedTuple

import torch
from torch import Tensor


class Det3DOut(NamedTuple):
    """Output of the detection model.

    boxes (list[Tensor]): 2D bounding boxes of shape [N, 4] in xyxy format.
    boxes3d (list[Tensor]): 3D bounding boxes of shape [N, 10].
    scores (list[Tensor]): 2D confidence scores of shape [N,].
    class_ids (list[Tensor]): class ids of shape [N,].
    depth_maps (list[Tensor] | None): depth maps for each image.
    categories (list[list[str]] | None): category names for each detection.
    predicted_intrinsics (Tensor | None): predicted camera intrinsics (B, 3, 3).
    scores_3d (list[Tensor] | None): 3D confidence scores of shape [N,].
    scores_2d (list[Tensor] | None): pure 2D confidence scores of shape [N,].
    """

    boxes: list[Tensor]
    boxes3d: list[Tensor]
    scores: list[Tensor]
    class_ids: list[Tensor]
    depth_maps: list[Tensor] | None
    categories: list[list[str]] | None = None
    predicted_intrinsics: Tensor | None = None
    scores_3d: list[Tensor] | None = None
    scores_2d: list[Tensor] | None = None
    confidence_maps: list[Tensor] | None = None


class WildDet3DOut(NamedTuple):
    """Output of WildDet3D model.

    All tensors use batch-first format: (B, num_queries, dim)
    where B = N_prompts (per-prompt batch).

    Coordinate formats:
    - pred_boxes_2d: normalized xyxy [0, 1]
    - pred_boxes_3d: encoded 3D params (delta_center, log_depth, log_dims, rot_6d)
    """
    # 2D Detection (from SAM3 decoder) - O2O outputs
    pred_logits: Tensor  # (N_prompts, num_queries, 1) - objectness
    pred_boxes_2d: Tensor  # (N_prompts, num_queries, 4) - normalized xyxy

    # 3D Detection (from 3D head) - O2O outputs
    pred_boxes_3d: Tensor | None  # (N_prompts, num_queries, 12) - encoded 3D params

    # Auxiliary outputs for each decoder layer (for deep supervision)
    aux_outputs: list[dict] | None

    # Geometry backend losses (SILog depth, phi, theta)
    geom_losses: dict[str, Tensor] | None

    # SAM3 specific outputs
    presence_logits: Tensor | None  # (N_prompts, num_queries, 1)
    queries: Tensor | None  # (N_prompts, num_queries, d_model) - for segmentation

    # Encoder hidden states (for depth head if needed)
    encoder_hidden_states: Tensor | None  # (H*W, N_prompts, d_model)

    # Matching indices from SAM3 (for loss computation)
    # Format: (batch_idx, src_idx, tgt_idx) from Hungarian matching
    indices: tuple | None = None

    # Normalized cxcywh boxes (needed by SAM3's Boxes loss for L1)
    pred_boxes_2d_cxcywh: Tensor | None = None  # (N_prompts, num_queries, 4) - normalized cxcywh

    # O2M (One-to-Many) outputs from SAM3 DAC mechanism
    # These are separate outputs from the second half of queries in DAC mode
    pred_logits_o2m: Tensor | None = None  # (N_prompts, num_queries, 1)
    pred_boxes_2d_o2m: Tensor | None = None  # (N_prompts, num_queries, 4) - normalized xyxy
    pred_boxes_2d_cxcywh_o2m: Tensor | None = None  # (N_prompts, num_queries, 4) - normalized cxcywh
    pred_boxes_3d_o2m: Tensor | None = None  # (N_prompts, num_queries, 12) - encoded 3D params

    # 3D confidence head outputs (camera+depth conditioned)
    pred_conf_3d: Tensor | None = None  # (N_prompts, num_queries, 1)
    pred_conf_3d_o2m: Tensor | None = None  # (N_prompts, num_queries, 1)

    def __getitem__(self, key: str):
        """Support dict-like access for vis4d data connector compatibility."""
        return getattr(self, key)

    def keys(self):
        """Return field names for dict-like iteration."""
        return [f.name for f in fields(self)]

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for dict-like access."""
        return hasattr(self, key)


@dataclass
class WildDet3DInput:
    """WildDet3D batched input format (per-prompt batch).

    Design Principles:
    1. Aligned with SAM3's BatchedDatapoint
    2. Added 3D detection required fields (intrinsics, gt_boxes3d)
    3. Supports three modes: TEXT / GEOMETRIC / TEXT_GEOMETRIC

    Coordinate Format Convention:
    - geo_boxes: normalized [0,1] cxcywh (SAM3 Geometry Encoder input)
    - gt_boxes2d: normalized [0,1] xyxy (for loss computation)
    - Model output pred_boxes_2d: normalized xyxy [0,1]
    """

    # ========== Image-level (Backbone processing) ==========
    images: Tensor                    # (B_images, 3, H, W)
    intrinsics: Tensor                # (B_images, 3, 3)

    # ========== Prompt-level (expanded) ==========
    img_ids: Tensor                   # (N_prompts,) - which image each prompt belongs to
    text_ids: Tensor                  # (N_prompts,) - text index for each prompt
    unique_texts: List[str]           # deduplicated texts (including "visual" placeholder)

    # Geometry input - batch-first: (N_prompts, max_K, 4) - normalized cxcywh
    # Converted to sequence-first when passed to SAM3 Prompt class
    geo_boxes: Tensor | None = None          # (N_prompts, max_K, 4)
    geo_boxes_mask: Tensor | None = None     # (N_prompts, max_K) - True=padding
    geo_box_labels: Tensor | None = None     # (N_prompts, max_K) - 0/1 for neg/pos

    # Point prompts (optional)
    geo_points: Tensor | None = None         # (N_prompts, max_P, 2) - (x, y)
    geo_points_mask: Tensor | None = None    # (N_prompts, max_P) - True=padding
    geo_point_labels: Tensor | None = None   # (N_prompts, max_P) - 0/1 for neg/pos

    # Ground Truth - normalized xyxy (training)
    gt_boxes2d: Tensor | None = None         # (N_prompts, max_gt, 4) - xyxy
    gt_boxes3d: Tensor | None = None         # (N_prompts, max_gt, 12) - 3D params
    num_gts: Tensor | None = None            # (N_prompts,) - number of GTs per prompt
    gt_category_ids: Tensor | None = None    # (N_prompts, max_gt)

    # Ignore boxes for negative loss suppression (per-prompt, same category)
    # Objects marked ignore in Omni3D (truncated, occluded, behind camera, etc.)
    # are not used as GT but should not cause FP penalty either.
    ignore_boxes2d: Tensor | None = None     # (N_prompts, max_ignore, 4) normalized xyxy
    num_ignores: Tensor | None = None        # (N_prompts,) number of ignore boxes per prompt

    # Query type tracking (collator-level label, does NOT control SAM3 internal matching).
    # 0=TEXT, 1=VISUAL, 2=GEOMETRY, 3=VISUAL+LABEL, 4=GEOMETRY+LABEL
    # "multi-target" (0,1,3): num_gts can be > 1 (all instances of a category)
    # "single-target" (2,4): num_gts = 1 (one selected instance)
    # NOTE: SAM3's DAC mechanism (internal o2o/o2m matcher) always runs
    # both branches regardless of this field.
    query_types: Tensor | None = None        # (N_prompts,) int

    # Metadata for evaluation/visualization
    sample_names: List[str] | None = None    # (B_images,) - image identifiers
    dataset_name: List[str] | None = None    # (B_images,) - dataset names for evaluator
    original_hw: List[tuple] | None = None   # (B_images,) - original (H, W) per image
    original_images: Tensor | None = None    # (B_images, 3, H_orig, W_orig) - unresized
    original_intrinsics: Tensor | None = None  # (B_images, 3, 3) - intrinsics before resize

    # CenterPad offsets [pad_left, pad_right, pad_top, pad_bottom]
    padding: List | None = None               # (B_images,) - padding offsets per image

    # Depth Ground Truth (for geometry backend supervision)
    depth_gt: Tensor | None = None            # (B_images, 1, H, W) depth map
    depth_mask: Tensor | None = None          # (B_images, H, W) valid depth mask

    # Key aliases for vis4d DataConnector compatibility
    # Maps expected DataLoader keys to actual dataclass field names
    _KEY_ALIASES = {
        # Target boxes (for loss computation)
        "boxes2d": "gt_boxes2d",
        "boxes3d": "gt_boxes3d",
        "boxes2d_classes": "gt_category_ids",
        # Geometric prompts (for SAM3 input)
        "prompt_boxes": "geo_boxes",
        "prompt_box_labels": "geo_box_labels",
        # Not available in per-prompt batch
        "depth_maps": None,
        "original_hw": None,
        "original_images": None,
        "padding": None,
    }

    def __getitem__(self, key: str):
        """Support dict-like access for vis4d data connector compatibility.

        Supports both actual field names and aliased keys from raw DataLoader.
        """
        # Check alias first
        if key in self._KEY_ALIASES:
            aliased_key = self._KEY_ALIASES[key]
            if aliased_key is None:
                return None  # Field not available
            return getattr(self, aliased_key)

        # Handle special computed fields
        if key == "input_hw":
            # Return (H, W) from images shape
            return (self.images.shape[2], self.images.shape[3])

        # Direct field access
        if hasattr(self, key):
            return getattr(self, key)

        # Return None for unknown keys instead of raising error
        return None

    def keys(self):
        """Return field names for dict-like iteration."""
        return [f.name for f in fields(self)]

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for dict-like access."""
        return hasattr(self, key)

    @property
    def num_images(self) -> int:
        """Number of unique images."""
        return self.images.shape[0]

    @property
    def num_prompts(self) -> int:
        """Number of prompts (batch size for decoder)."""
        return self.img_ids.shape[0]

    @property
    def device(self) -> torch.device:
        """Device of the batch."""
        return self.images.device
