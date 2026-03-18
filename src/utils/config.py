from __future__ import annotations
import yaml
from pathlib import Path
from types import SimpleNamespace


def _dict_to_ns(d: dict) -> SimpleNamespace:
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _dict_to_ns(v) if isinstance(v, dict) else v)
    return ns


def load_config(path=None) -> SimpleNamespace:
    if path is None:
        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / "configs" / "config.yaml"
            if candidate.exists():
                path = candidate
                break
        else:
            raise FileNotFoundError("configs/config.yaml not found")
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _dict_to_ns(raw)


CFG = load_config()
