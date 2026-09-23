from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def configure_logging(log_path: Path, verbose: bool = True) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("q1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if verbose:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=json_default)


def write_csv(rows: Iterable[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def atomic_pickle_dump(value: Any, path: Path, protocol: int = 5) -> None:
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=protocol)
    temporary.replace(path)


def run_command(args: list[str], *, input_bytes: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        args,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def command_version(command: list[str]) -> str:
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        return result.stdout.decode("utf-8", errors="replace").splitlines()[0].strip()
    except (OSError, IndexError):
        return "unavailable"


def probe_media(path: Path) -> dict[str, Any]:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(result.stdout.decode("utf-8"))
    streams = payload.get("streams", [])
    video = next((item for item in streams if item.get("codec_type") == "video"), {})
    audio = next((item for item in streams if item.get("codec_type") == "audio"), {})
    duration_values = [
        video.get("duration"),
        audio.get("duration"),
        payload.get("format", {}).get("duration"),
    ]
    duration = next((float(x) for x in duration_values if x not in (None, "N/A")), 0.0)
    fps_text = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
    try:
        numerator, denominator = fps_text.split("/")
        fps = float(numerator) / float(denominator) if float(denominator) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    return {
        "duration": duration,
        "video_fps": fps,
        "video_width": int(video.get("width") or 0),
        "video_height": int(video.get("height") or 0),
        "video_codec": video.get("codec_name", ""),
        "audio_codec": audio.get("codec_name", ""),
        "audio_sample_rate_original": int(audio.get("sample_rate") or 0),
        "has_audio": bool(audio),
        "format_name": payload.get("format", {}).get("format_name", ""),
    }


def extract_audio(path: Path, sample_rate: int = 16000) -> np.ndarray:
    result = run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "wav",
            "pipe:1",
        ]
    )
    import io
    import soundfile as sf

    audio, actual_rate = sf.read(io.BytesIO(result.stdout), dtype="float32")
    if actual_rate != sample_rate:
        raise RuntimeError(f"Unexpected extracted sample rate {actual_rate}, expected {sample_rate}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32)


def environment_snapshot(model_manifest: Path | None = None) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "ffmpeg": command_version(["ffmpeg", "-version"]),
        "ffprobe": command_version(["ffprobe", "-version"]),
    }
    for package in ["numpy", "yaml", "transformers", "torch", "librosa", "soundfile", "cv2", "mediapipe", "openpyxl"]:
        try:
            module = __import__(package)
            snapshot[package] = str(getattr(module, "__version__", "installed"))
        except Exception as exc:  # pragma: no cover - diagnostic path
            snapshot[package] = f"unavailable: {exc}"
    try:
        import torch

        snapshot["cuda_available"] = bool(torch.cuda.is_available())
        snapshot["cuda_version"] = str(torch.version.cuda)
        snapshot["gpu_name"] = str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else ""
    except Exception as exc:  # pragma: no cover - diagnostic path
        snapshot["cuda_error"] = str(exc)
    if model_manifest and model_manifest.exists():
        snapshot["model_manifest"] = str(model_manifest)
    return snapshot


def verify_model_manifest(config: Any, logger: logging.Logger, verify_hashes: bool = True) -> dict[str, Any]:
    manifest_path = config.path_for("model_manifest")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Model manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    model_specs = [
        ("text_model", "text_model_path"),
        ("asr_timestamp_model", "asr_timestamp_model_path"),
        ("audio_model", "audio_model_path"),
        ("vision_model", "vision_model_path"),
    ]
    checked = []
    for repo_key, path_key in model_specs:
        repo_id = config.require(repo_key)
        model_dir = config.path_for(path_key)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found for {repo_id}: {model_dir}")
        entry = next((item for item in manifest.get("models", []) if item.get("repo_id") == repo_id), None)
        if entry is None:
            raise RuntimeError(f"No manifest entry for {repo_id}")
        if entry.get("revision") != config.get(f"{repo_key}_revision"):
            raise RuntimeError(f"Revision mismatch for {repo_id}: manifest={entry.get('revision')}")
        for record in entry.get("files", []):
            file_path = model_dir / str(record["path"])
            if not file_path.exists():
                raise FileNotFoundError(f"Missing model file: {file_path}")
            if verify_hashes:
                actual = sha256_file(file_path)
                if actual != record.get("sha256"):
                    raise RuntimeError(f"SHA-256 mismatch for {file_path}: {actual} != {record.get('sha256')}")
        checked.append({"repo_id": repo_id, "revision": entry.get("revision"), "path": str(model_dir)})
        logger.info("Model manifest verified: %s", repo_id)
    return {"manifest_path": str(manifest_path), "models": checked, "hashes_verified": verify_hashes}
