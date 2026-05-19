"""
YAML config loader.  Returns a nested SimpleNamespace so fields are
accessible as  cfg.training.lr  rather than  cfg["training"]["lr"].

Usage:
    cfg = load_config("configs/unet_r18_v1.yaml")
    print(cfg.training.lr)
    print(cfg.model.encoder)
"""

from pathlib import Path
from types import SimpleNamespace

import yaml


def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(i) for i in obj]
    return obj


def load_config(path: str | Path) -> SimpleNamespace:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return _to_ns(data)
