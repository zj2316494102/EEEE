from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class Q1Config:
    """Small path-aware wrapper around the YAML configuration."""

    def __init__(self, values: dict[str, Any], path: Path):
        self.values = values
        self.path = path.resolve()
        self.project_root = self.path.parent.parent.resolve()

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self.values:
            raise KeyError(f"Missing configuration key: {key}")
        return self.values[key]

    def resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.project_root / path).resolve()

    def path_for(self, key: str) -> Path:
        return self.resolve(self.require(key))

    @property
    def aligned_length(self) -> int:
        return int(self.get("aligned_length", 50))

    @property
    def main_feature_dims(self) -> tuple[int, int, int]:
        dims = self.get("main_feature_dims", [768, 768, 768])
        return tuple(int(x) for x in dims)

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(path: str | Path) -> Q1Config:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return Q1Config(values, config_path)


def dump_yaml(values: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(values, handle, allow_unicode=True, sort_keys=False)

