from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def render_sample_figure(
    sample_id: str,
    aligned: dict[str, Any],
    output_path: Path,
) -> None:
    """Render a compact static audit figure for one aligned sample."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.asarray(aligned["time_grid"], dtype=np.float64)
    centers = grid.mean(axis=1)
    masks = aligned["masks"]
    text = np.asarray(aligned["text"], dtype=np.float32)
    audio = np.asarray(aligned["audio"], dtype=np.float32)
    vision = np.asarray(aligned["vision"], dtype=np.float32)
    emotion = np.asarray(aligned["vision_emotion_probs"], dtype=np.float32)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    safe_title = str(sample_id).replace("$", r"\$")
    fig.suptitle(f"Question 1 temporal alignment audit: {safe_title}")
    axes[0].step(centers, masks["text"], where="mid", label="text mask")
    axes[0].step(centers, masks["audio"] + 1.1, where="mid", label="audio mask")
    axes[0].step(centers, masks["vision"] + 2.2, where="mid", label="vision mask")
    axes[0].set_yticks([0.5, 1.6, 2.7], ["text", "audio", "vision"])
    axes[0].set_ylabel("valid")
    axes[0].set_title("Three modalities on the same 50-bin time grid")
    axes[0].legend(loc="upper right", ncol=3)
    axes[1].plot(centers, np.linalg.norm(text, axis=1), label="text norm")
    axes[1].plot(centers, np.linalg.norm(audio, axis=1), label="audio norm")
    axes[1].plot(centers, np.linalg.norm(vision, axis=1), label="vision norm")
    axes[1].set_ylabel("feature L2 norm")
    axes[1].set_title("Standardized feature magnitude by time")
    axes[1].legend(loc="upper right", ncol=3)
    axes[2].plot(centers, emotion.max(axis=1), label="max face emotion probability")
    axes[2].plot(centers, masks["vision"], label="face-valid mask", alpha=0.75)
    axes[2].set_xlabel("clip time (s)")
    axes[2].set_ylabel("value")
    axes[2].set_title("Vision quality and auxiliary emotion signal")
    axes[2].legend(loc="upper right")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
