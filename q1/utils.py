from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import platform
import random
import struct
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


def _optional_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and number > 0 else 0.0


def _decoded_stream_duration(path: Path, selector: str) -> float:
    """Return the last decoded frame/packet end on FFmpeg's presentation timeline."""
    try:
        result = run_command(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                selector,
                "-show_frames",
                "-show_entries",
                "frame=best_effort_timestamp_time,pkt_duration_time",
                "-of",
                "csv=p=0",
                str(path),
            ]
        )
    except subprocess.CalledProcessError:
        return 0.0
    last_end = 0.0
    for line in result.stdout.decode("utf-8", errors="replace").splitlines():
        values = [item.strip() for item in line.split(",")]
        if not values:
            continue
        timestamp = _optional_float(values[0])
        packet_duration = _optional_float(values[1]) if len(values) > 1 else 0.0
        if timestamp or packet_duration:
            last_end = max(last_end, timestamp + packet_duration, timestamp)
    return last_end


def _iter_mp4_boxes(blob: bytes, start: int, end: int) -> Iterable[tuple[str, int, int]]:
    offset = int(start)
    end = min(int(end), len(blob))
    while offset + 8 <= end:
        size = struct.unpack_from(">I", blob, offset)[0]
        box_type = blob[offset + 4 : offset + 8].decode("latin-1")
        header_size = 8
        if size == 1:
            if offset + 16 > end:
                break
            size = struct.unpack_from(">Q", blob, offset + 8)[0]
            header_size = 16
        elif size == 0:
            size = end - offset
        if box_type == "uuid":
            header_size += 16
        if size < header_size or offset + size > end:
            break
        yield box_type, offset + header_size, offset + size
        offset += int(size)


def _first_mp4_box(blob: bytes, start: int, end: int, box_type: str) -> tuple[int, int] | None:
    for current_type, payload_start, payload_end in _iter_mp4_boxes(blob, start, end):
        if current_type == box_type:
            return payload_start, payload_end
    return None


def _parse_mp4_timing(path: Path) -> dict[str, Any]:
    """Read mvhd/mdhd/elst timing without confusing it with FFprobe format duration."""
    try:
        blob = path.read_bytes()
    except OSError:
        return {}
    moov = _first_mp4_box(blob, 0, len(blob), "moov")
    if not moov:
        return {}

    def parse_header(box: tuple[int, int] | None) -> tuple[float, float]:
        if not box:
            return 0.0, 0.0
        start, end = box
        if end - start < 20:
            return 0.0, 0.0
        version = blob[start]
        if version == 0 and end - start >= 20:
            timescale = struct.unpack_from(">I", blob, start + 12)[0]
            duration = struct.unpack_from(">I", blob, start + 16)[0]
        elif version == 1 and end - start >= 32:
            timescale = struct.unpack_from(">I", blob, start + 20)[0]
            duration = struct.unpack_from(">Q", blob, start + 24)[0]
        else:
            return 0.0, 0.0
        return float(timescale), float(duration) / float(timescale) if timescale else 0.0

    moov_start, moov_end = moov
    movie_timescale, duration_mvhd = parse_header(_first_mp4_box(blob, moov_start, moov_end, "mvhd"))
    tracks: list[dict[str, Any]] = []
    for box_type, trak_start, trak_end in _iter_mp4_boxes(blob, moov_start, moov_end):
        if box_type != "trak":
            continue
        mdia = _first_mp4_box(blob, trak_start, trak_end, "mdia")
        if not mdia:
            continue
        mdia_start, mdia_end = mdia
        handler_box = _first_mp4_box(blob, mdia_start, mdia_end, "hdlr")
        handler = ""
        if handler_box and handler_box[1] - handler_box[0] >= 12:
            handler = blob[handler_box[0] + 8 : handler_box[0] + 12].decode("latin-1")
        media_timescale, media_duration = parse_header(_first_mp4_box(blob, mdia_start, mdia_end, "mdhd"))
        edts = _first_mp4_box(blob, trak_start, trak_end, "edts")
        elst = _first_mp4_box(blob, edts[0], edts[1], "elst") if edts else None
        edit_entries = 0
        edit_duration = 0.0
        if elst and elst[1] - elst[0] >= 8:
            start, end = elst
            version = blob[start]
            edit_entries = struct.unpack_from(">I", blob, start + 4)[0]
            cursor = start + 8
            for _ in range(edit_entries):
                if version == 1:
                    if cursor + 20 > end:
                        break
                    segment_duration = struct.unpack_from(">Q", blob, cursor)[0]
                    cursor += 20
                else:
                    if cursor + 12 > end:
                        break
                    segment_duration = struct.unpack_from(">I", blob, cursor)[0]
                    cursor += 12
                if movie_timescale:
                    edit_duration += float(segment_duration) / movie_timescale
        tracks.append(
            {
                "handler": handler,
                "media_timescale": media_timescale,
                "media_duration": media_duration,
                "edit_duration": edit_duration,
                "edit_entries": int(edit_entries),
            }
        )
    video_track = next((track for track in tracks if track["handler"] == "vide"), {})
    audio_track = next((track for track in tracks if track["handler"] == "soun"), {})
    return {
        "duration_mvhd": duration_mvhd,
        "mvhd_timescale": movie_timescale,
        "duration_video_edit": float(video_track.get("edit_duration", 0.0)),
        "duration_audio_edit": float(audio_track.get("edit_duration", 0.0)),
        "video_edit_list_entries": int(video_track.get("edit_entries", 0)),
        "audio_edit_list_entries": int(audio_track.get("edit_entries", 0)),
        "video_edit_list_present": bool(video_track.get("edit_entries", 0)),
        "audio_edit_list_present": bool(audio_track.get("edit_entries", 0)),
    }


def probe_media(path: Path) -> dict[str, Any]:
    """Inspect all relevant MP4 duration conventions.

    ``duration_alignment`` intentionally follows the video stream's FFmpeg
    presentation duration (the edit-list-aware stream duration used by the
    actual decoder).  The container/movie-header duration is retained for
    source-range auditing and is never silently substituted into the feature
    time axis.
    """
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
    format_info = payload.get("format", {})
    duration_format = _optional_float(format_info.get("duration"))
    duration_video_stream = _optional_float(video.get("duration"))
    duration_audio_stream = _optional_float(audio.get("duration"))
    mp4_timing = _parse_mp4_timing(path)
    duration_mvhd = _optional_float(mp4_timing.get("duration_mvhd")) or duration_format
    duration_container = duration_mvhd
    duration_video_edit = _optional_float(mp4_timing.get("duration_video_edit")) or duration_video_stream
    duration_audio_edit = _optional_float(mp4_timing.get("duration_audio_edit")) or duration_audio_stream
    duration_decoded_video = _decoded_stream_duration(path, "v:0")
    duration_decoded_audio = _decoded_stream_duration(path, "a:0")
    duration_decoded = max(duration_decoded_video, duration_decoded_audio)
    duration_alignment = duration_video_edit or duration_audio_edit or duration_decoded or duration_container
    if mp4_timing.get("duration_video_edit"):
        duration_source = "video_edit_list_presentation"
    elif duration_video_stream:
        duration_source = "video_stream_presentation"
    elif duration_audio_edit:
        duration_source = "audio_edit_list_presentation"
    elif duration_audio_stream:
        duration_source = "audio_stream_presentation_edit_list"
    elif duration_decoded:
        duration_source = "decoded_presentation_pts"
    else:
        duration_source = "container_duration_fallback"
    fps_text = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
    try:
        numerator, denominator = fps_text.split("/")
        fps = float(numerator) / float(denominator) if float(denominator) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    return {
        "duration": duration_alignment,
        "duration_container": duration_container,
        "duration_format": duration_format,
        "duration_mvhd": duration_mvhd,
        "duration_video_stream": duration_video_stream,
        "duration_audio_stream": duration_audio_stream,
        "duration_video_edit": duration_video_edit,
        "duration_audio_edit": duration_audio_edit,
        "duration_decoded_video": duration_decoded_video,
        "duration_decoded_audio": duration_decoded_audio,
        "duration_decoded": duration_decoded,
        "duration_alignment": duration_alignment,
        "duration_source": duration_source,
        "duration_start_time": _optional_float(video.get("start_time")),
        "video_edit_list_entries": int(mp4_timing.get("video_edit_list_entries", 0)),
        "audio_edit_list_entries": int(mp4_timing.get("audio_edit_list_entries", 0)),
        "video_edit_list_present": bool(mp4_timing.get("video_edit_list_present", False)),
        "audio_edit_list_present": bool(mp4_timing.get("audio_edit_list_present", False)),
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
