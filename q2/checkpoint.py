from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def _to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def capture_rng_state(generator: np.random.Generator | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "torch": torch.get_rng_state().cpu(),
        "numpy_generator": None if generator is None else generator.bit_generator.state,
    }
    if torch.cuda.is_available():
        state["cuda"] = [item.cpu() for item in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state: Mapping[str, Any], generator: np.random.Generator | None = None) -> None:
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy_global") is not None:
        np.random.set_state(state["numpy_global"])
    if state.get("torch") is not None:
        torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8).cpu())
    if generator is not None and state.get("numpy_generator") is not None:
        generator.bit_generator.state = state["numpy_generator"]
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([torch.as_tensor(item, dtype=torch.uint8) for item in state["cuda"]])


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> None:
    """Write a checkpoint through a sibling temporary file then atomically replace it."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
        ) as handle:
            temporary_name = handle.name
        torch.save(dict(payload), temporary_name)
        os.replace(temporary_name, output)
        temporary_name = None
    finally:
        if temporary_name:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def load_torch_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(Path(path), map_location=device, weights_only=False)
    except TypeError:  # compatibility with older torch releases
        return torch.load(Path(path), map_location=device)


def make_training_checkpoint(
    *,
    stage: str,
    seed: int,
    epoch: int,
    completed: bool,
    model_config: Mapping[str, Any],
    model_state_dict: Mapping[str, Any],
    optimizer_state_dict: Mapping[str, Any],
    best_state_dict: Mapping[str, Any] | None,
    best_epoch: int,
    best_score: float,
    stale_epochs: int,
    history: list[dict[str, Any]],
    rng_state: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    scheduler_state_dict: Mapping[str, Any] | None = None,
    scaler_state_dict: Mapping[str, Any] | None = None,
    global_step: int = 0,
    augmentation_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "checkpoint_version": 2,
        "stage": stage,
        "seed": int(seed),
        "epoch": int(epoch),
        "completed": bool(completed),
        "model_config": dict(model_config),
        "model_state_dict": _to_cpu(dict(model_state_dict)),
        "optimizer_state_dict": _to_cpu(dict(optimizer_state_dict)),
        "best_state_dict": None if best_state_dict is None else _to_cpu(dict(best_state_dict)),
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "stale_epochs": int(stale_epochs),
        "history": list(history),
        "rng_state": dict(rng_state),
        "scheduler_state_dict": None if scheduler_state_dict is None else _to_cpu(dict(scheduler_state_dict)),
        "scaler_state_dict": None if scaler_state_dict is None else _to_cpu(dict(scaler_state_dict)),
        "global_step": int(global_step),
        "augmentation_state": None if augmentation_state is None else dict(augmentation_state),
        "metadata": dict(metadata or {}),
    }
