"""GAM: grounded action model — language planner (Molmo2) -> object grounding -> flow-matching action policy."""
import os as _os
import sys as _sys

__version__ = "0.1.0"

# vendored perception backbone packages (third_party/) importable when running from source
_tp = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "third_party")
if _os.path.isdir(_tp) and _tp not in _sys.path:
    _sys.path.append(_tp)
