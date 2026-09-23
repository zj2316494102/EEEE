from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass
class Q2Config:
    """Path-aware configuration wrapper for the Question 2 pipeline."""

    values: dict[str, Any]
    path: Path

    def __post_init__(self) -> None:
        self.path = self.path.resolve()
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
    def feature_version(self) -> str:
        return str(self.get("feature_version", "aligned_50"))

    @property
    def max_length(self) -> int:
        return int(self.get("max_length", 50))

    @property
    def input_dims(self) -> tuple[int, int, int]:
        values = self.get("input_dims", [768, 74, 35])
        return tuple(int(value) for value in values)

    @property
    def projection_dim(self) -> int:
        return int(self.get("projection_dim", 128))

    @property
    def model_kwargs(self) -> dict[str, Any]:
        keys = (
            "projection_dim",
            "transformer_layers",
            "attention_heads",
            "feedforward_dim",
            "dropout",
            "use_mask_input",
            "fusion",
            "use_transformer",
            "use_coverage_features",
        )
        return {key: self.get(key) for key in keys if key in self.values}

    @property
    def config_hash(self) -> str:
        payload = json.dumps(
            self.values, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(path: str | Path) -> Q2Config:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, Mapping):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return Q2Config(dict(values), config_path)


def dump_yaml(values: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(values), handle, allow_unicode=True, sort_keys=False)

