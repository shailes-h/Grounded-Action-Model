"""YAML config loading with a few sanity checks."""
from __future__ import annotations

import yaml


def load_config(path: str) -> dict:
    cfg = yaml.safe_load(open(path))
    for sec in ("data", "perception", "train"):
        assert sec in cfg, f"config missing section '{sec}'"
    assert cfg["data"].get("slots"), "data.slots must list at least one object slot"
    return cfg
