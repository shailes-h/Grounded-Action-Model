"""WildDet3D: Open-Vocabulary Monocular 3D Object Detection in the Wild."""

import sys
from pathlib import Path

# Add third_party submodules to Python path
_third_party = Path(__file__).parent.parent / "third_party"
_sam3_path = str(_third_party / "sam3")
_lingbot_path = str(_third_party / "lingbot_depth")

if _sam3_path not in sys.path:
    sys.path.insert(0, _sam3_path)
if _lingbot_path not in sys.path:
    sys.path.insert(0, _lingbot_path)

from .data_types import Det3DOut, WildDet3DInput, WildDet3DOut
from .inference import WildDet3DPredictor, build_model
from .model import WildDet3D


def __getattr__(name):
    # `preprocess` lives in `.preprocessing`, which depends on `wilddet3d.data.transforms.*`
    # (a training-only submodule not shipped in this release). Defer the import so the
    # core inference path works without it. Only diagnostic scripts that explicitly call
    # `wilddet3d.preprocess` will trigger the import (and fail with a clear error).
    if name == "preprocess":
        from .preprocessing import preprocess
        return preprocess
    raise AttributeError(f"module 'wilddet3d' has no attribute {name!r}")


__all__ = [
    "WildDet3D",
    "WildDet3DPredictor",
    "WildDet3DInput",
    "WildDet3DOut",
    "Det3DOut",
    "build_model",
    "preprocess",
]
