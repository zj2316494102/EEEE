from __future__ import annotations

import io
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .alignment import convolution_spans, frame_spans_from_centers, tokenize_text
from .utils import run_command


@dataclass
class TextFeatures:
    features: np.ndarray
    word_indices: list[int]
    word_tokens: list[dict[str, Any]]
    overflow_chunks: int


@dataclass
class AudioFeatures:
    features: np.ndarray
    spans: np.ndarray
    mask: np.ndarray
    prosody: np.ndarray
    prosody_mask: np.ndarray
    timing: dict[str, int]


@dataclass
class VisionFeatures:
    features: np.ndarray
    emotion_probs: np.ndarray
    spans: np.ndarray
    mask: np.ndarray
    quality: np.ndarray
    face_bbox: np.ndarray
    face_confidence: np.ndarray
    track_ids: np.ndarray
    frame_indices: np.ndarray
    frame_count: int
    detector_stats: dict[str, Any]


def _model_dtype(torch: Any, device: Any) -> Any:
    return torch.float16 if getattr(device, "type", str(device)) == "cuda" else torch.float32


class TextExtractor:
    def __init__(self, model_path: Path, device: Any, max_length: int = 8192):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, use_fast=True)
        self.model = AutoModelForMaskedLM.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=_model_dtype(torch, device)
        )
        self.model.to(device).eval()
        hidden_size = int(getattr(self.model.config, "hidden_size", 0))
        if hidden_size != 768:
            raise RuntimeError(f"ModernBERT hidden size must be 768, got {hidden_size}")

    def extract(self, text: str) -> TextFeatures:
        tokens = tokenize_text(text)
        if not tokens:
            return TextFeatures(np.zeros((0, 768), dtype=np.float32), [], [], 0)
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            truncation=True,
            max_length=self.max_length,
            stride=0,
            padding=True,
        )
        offsets = encoded.pop("offset_mapping").cpu().numpy()
        model_inputs = {key: value.to(self.device) for key, value in encoded.items() if key != "overflow_to_sample_mapping"}
        with self.torch.inference_mode():
            outputs = self.model(**model_inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1] if outputs.hidden_states else outputs.last_hidden_state
        hidden = hidden_states.detach().float().cpu().numpy()
        accum = np.zeros((len(tokens), 768), dtype=np.float32)
        counts = np.zeros(len(tokens), dtype=np.int32)
        for chunk_index in range(hidden.shape[0]):
            for token_index, (start, end) in enumerate(offsets[chunk_index]):
                start, end = int(start), int(end)
                if end <= start:
                    continue
                centers = np.array([(start + end) / 2.0])
                word_index = int(np.searchsorted([token.end_char for token in tokens], centers[0], side="left"))
                if word_index >= len(tokens):
                    continue
                token_word = tokens[word_index]
                if end <= token_word.start_char or start >= token_word.end_char:
                    continue
                accum[word_index] += hidden[chunk_index, token_index]
                counts[word_index] += 1
        valid_indices = [index for index, count in enumerate(counts) if count > 0]
        features = np.zeros((len(valid_indices), 768), dtype=np.float32)
        for output_index, source_index in enumerate(valid_indices):
            features[output_index] = accum[source_index] / float(counts[source_index])
        word_tokens = [
            {
                "word": token.raw,
                "normalized": token.normalized,
                "start_char": token.start_char,
                "end_char": token.end_char,
            }
            for token in tokens
        ]
        return TextFeatures(features, valid_indices, word_tokens, int(hidden.shape[0]))


class ASRAligner:
    def __init__(
        self,
        model_path: Path,
        device: Any,
        chunk_length_s: float = 30.0,
        stride_s: float = 5.0,
    ):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=_model_dtype(torch, device)
        )
        self.model.to(device).eval()
        device_index = 0 if getattr(device, "type", str(device)) == "cuda" else -1
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=float(chunk_length_s),
            stride_length_s=(float(stride_s), float(stride_s)),
            torch_dtype=_model_dtype(torch, device),
            device=device_index,
        )

    def extract(
        self,
        audio: np.ndarray,
        duration: float,
        *,
        num_beams: int = 1,
        temperature: float = 0.0,
        context_s: float = 0.0,
        sample_rate: int = 16000,
    ) -> list[dict[str, Any]]:
        if audio.size == 0:
            return []
        context_samples = max(int(round(float(context_s) * int(sample_rate))), 0)
        if context_samples:
            model_audio = np.pad(audio, (context_samples, context_samples), mode="constant")
            timestamp_shift = float(context_samples) / max(int(sample_rate), 1)
        else:
            model_audio = audio
            timestamp_shift = 0.0
        result = self.pipeline(
            model_audio,
            return_timestamps="word",
            generate_kwargs={
                "language": "english",
                "task": "transcribe",
                "temperature": float(temperature),
                "num_beams": int(num_beams),
                "do_sample": bool(float(temperature) > 0.0),
            },
        )
        words: list[dict[str, Any]] = []
        for chunk in result.get("chunks", []) if isinstance(result, dict) else []:
            timestamp = chunk.get("timestamp")
            if not timestamp or timestamp[0] is None:
                continue
            start = float(timestamp[0]) - timestamp_shift
            end = (
                float(timestamp[1]) - timestamp_shift
                if timestamp[1] is not None
                else min(float(duration), start + 0.05)
            )
            if end <= start:
                end = min(duration, start + 0.05)
            if end <= 0 or start >= duration:
                continue
            words.append(
                {
                    "text": str(chunk.get("text", "")).strip(),
                    "start": max(0.0, start),
                    "end": min(float(duration), end),
                    "confidence": 1.0,
                }
            )
        return words


class AudioExtractor:
    def __init__(self, model_path: Path, device: Any, sample_rate: int = 16000):
        import torch
        from transformers import AutoFeatureExtractor, AutoModel

        self.torch = torch
        self.device = device
        self.sample_rate = int(sample_rate)
        # WavLM is an encoder-only audio model. The pinned local asset contains
        # a feature-extractor config but intentionally does not contain a CTC
        # tokenizer, so AutoProcessor would try to load a nonexistent tokenizer.
        self.processor = AutoFeatureExtractor.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModel.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=_model_dtype(torch, device)
        )
        self.model.to(device).eval()
        if int(getattr(self.model.config, "hidden_size", 0)) != 768:
            raise RuntimeError("WavLM output hidden size must be 768")

    def extract(self, audio: np.ndarray, duration: float) -> AudioFeatures:
        if audio.size == 0:
            empty = np.zeros((0, 768), dtype=np.float32)
            return AudioFeatures(empty, np.zeros((0, 2)), np.zeros(0, dtype=np.uint8), np.zeros((0, 5)), np.zeros(0, dtype=np.uint8), {})
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        parameter_dtype = next(self.model.parameters()).dtype
        model_inputs = {
            key: value.to(self.device, dtype=parameter_dtype if value.is_floating_point() else None)
            for key, value in inputs.items()
        }
        with self.torch.inference_mode():
            outputs = self.model(**model_inputs)
        hidden = outputs.last_hidden_state[0].detach().float().cpu().numpy()
        attention_mask = inputs.get("attention_mask")
        feature_mask = None
        if attention_mask is not None and hasattr(self.model, "_get_feature_vector_attention_mask"):
            feature_mask = self.model._get_feature_vector_attention_mask(hidden.shape[0], attention_mask)[0].cpu().numpy()
        spans, valid, timing = convolution_spans(
            hidden.shape[0],
            self.sample_rate,
            getattr(self.model.config, "conv_kernel"),
            getattr(self.model.config, "conv_stride"),
            duration,
            feature_mask,
        )
        prosody, prosody_mask = self._prosody(audio, spans, valid)
        return AudioFeatures(hidden, spans, valid.astype(np.uint8), prosody, prosody_mask, timing)

    def _prosody(self, audio: np.ndarray, spans: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.zeros((len(spans), 5), dtype=np.float32)
        if not len(spans):
            return values, np.zeros(0, dtype=np.uint8)
        try:
            import librosa

            frame_length = 400
            hop_length = 320
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length, center=True)[0]
            centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate, n_fft=frame_length, hop_length=hop_length, center=True)[0]
            zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=frame_length, hop_length=hop_length, center=True)[0]
            try:
                f0, voiced_flag, _ = librosa.pyin(
                    audio,
                    fmin=50.0,
                    fmax=500.0,
                    sr=self.sample_rate,
                    frame_length=frame_length,
                    hop_length=hop_length,
                    center=True,
                )
                voiced_flag = np.asarray(voiced_flag, dtype=bool)
            except Exception:
                try:
                    f0 = librosa.yin(
                        audio,
                        fmin=50.0,
                        fmax=500.0,
                        sr=self.sample_rate,
                        frame_length=frame_length,
                        hop_length=hop_length,
                        center=True,
                    )
                    voiced_flag = np.isfinite(f0)
                except Exception:
                    f0 = np.full_like(rms, np.nan, dtype=np.float32)
                    voiced_flag = np.zeros_like(rms, dtype=bool)
            frame_times = librosa.times_like(rms, sr=self.sample_rate, hop_length=hop_length)
            centers = spans.mean(axis=1)
            indices = np.clip(np.searchsorted(frame_times, centers), 0, len(frame_times) - 1)
            values[:, 0] = np.nan_to_num(f0[indices], nan=0.0, posinf=0.0, neginf=0.0)
            values[:, 1] = np.nan_to_num(rms[indices], nan=0.0, posinf=0.0, neginf=0.0)
            values[:, 2] = np.asarray(voiced_flag[indices], dtype=np.float32)
            values[:, 3] = np.nan_to_num(centroid[indices], nan=0.0, posinf=0.0, neginf=0.0)
            values[:, 4] = np.nan_to_num(zcr[indices], nan=0.0, posinf=0.0, neginf=0.0)
            return values, valid.astype(np.uint8)
        except Exception:
            # The WavLM representation remains usable if an auxiliary prosody
            # extractor is unavailable. Its mask makes the limitation explicit.
            return values, np.zeros(len(spans), dtype=np.uint8)


@dataclass
class _Detection:
    frame_index: int
    bbox: tuple[float, float, float, float]
    confidence: float


class VisionExtractor:
    def __init__(
        self,
        model_path: Path,
        device: Any,
        output_fps: float = 25.0,
        min_detection_confidence: float = 0.5,
        crop_margin: float = 0.20,
        track_iou_threshold: float = 0.20,
        track_max_gap_frames: int = 2,
    ):
        import torch
        import mediapipe as mp
        import cv2  # noqa: F401
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.torch = torch
        self.device = device
        self.output_fps = float(output_fps)
        self.min_detection_confidence = float(min_detection_confidence)
        self.crop_margin = float(crop_margin)
        self.track_iou_threshold = float(track_iou_threshold)
        self.track_max_gap_frames = int(track_max_gap_frames)
        self.mp = mp
        self.processor = AutoImageProcessor.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModelForImageClassification.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=_model_dtype(torch, device)
        )
        self.model.to(device).eval()
        if int(getattr(self.model.config, "hidden_size", 0)) != 768:
            raise RuntimeError("Vision output hidden size must be 768")
        expected_labels = ["sad", "disgust", "angry", "neutral", "fear", "surprise", "happy"]
        actual_labels = [self.model.config.id2label.get(str(i), self.model.config.id2label.get(i, "")) for i in range(7)]
        if actual_labels != expected_labels:
            raise RuntimeError(f"Unexpected vision emotion order: {actual_labels}")

    def extract(self, video_path: Path, duration: float) -> VisionFeatures:
        frames = self._decode_frames(video_path)
        decoded_frame_count = len(frames)
        decoded_indices = np.arange(decoded_frame_count, dtype=np.int32)
        decoded_centers = (decoded_indices.astype(np.float64) + 0.5) / self.output_fps
        decoded_spans = frame_spans_from_centers(decoded_centers, duration, self.output_fps)
        retained = (decoded_centers < float(duration)) & (decoded_spans[:, 1] > decoded_spans[:, 0])
        retained_indices = decoded_indices[retained]
        frames = [frame for frame, keep in zip(frames, retained.tolist()) if keep]
        frame_count = len(frames)
        discarded_frame_count = decoded_frame_count - frame_count
        if frame_count == 0:
            return VisionFeatures(
                np.zeros((0, 768), dtype=np.float32),
                np.zeros((0, 7), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float64),
                np.zeros(0, dtype=np.uint8),
                np.zeros(0, dtype=np.float32),
                np.zeros((0, 4), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.int32),
                0,
                {
                    "decoded_frames": decoded_frame_count,
                    "retained_frames": 0,
                    "discarded_out_of_timeline_frames": discarded_frame_count,
                    "face_frames": 0,
                    "track_switches": 0,
                },
            )
        detections = self._detect(frames)
        track = self._select_track(detections)
        features = np.zeros((frame_count, 768), dtype=np.float32)
        emotion_probs = np.zeros((frame_count, 7), dtype=np.float32)
        mask = np.zeros(frame_count, dtype=np.uint8)
        quality = np.zeros(frame_count, dtype=np.float32)
        face_bbox = np.zeros((frame_count, 4), dtype=np.float32)
        face_confidence = np.zeros(frame_count, dtype=np.float32)
        track_ids = np.full(frame_count, -1, dtype=np.int32)
        crops: list[np.ndarray] = []
        crop_indices: list[int] = []
        for index, frame in enumerate(frames):
            selected = track.get(index)
            if selected is None:
                continue
            crop = self._crop_face(frame, selected.bbox)
            if crop is None:
                continue
            crops.append(crop)
            crop_indices.append(index)
            face_bbox[index] = np.asarray(selected.bbox, dtype=np.float32)
            face_confidence[index] = selected.confidence
            track_ids[index] = 0
            area = max(0.0, selected.bbox[2] - selected.bbox[0]) * max(0.0, selected.bbox[3] - selected.bbox[1])
            quality[index] = float(np.clip(selected.confidence * np.sqrt(max(area, 0.0)), 0.0, 1.0))
        if crops:
            for start in range(0, len(crops), 32):
                batch_crops = crops[start : start + 32]
                inputs = self.processor(images=batch_crops, return_tensors="pt")
                parameter_dtype = next(self.model.parameters()).dtype
                model_inputs = {
                    key: value.to(self.device, dtype=parameter_dtype if value.is_floating_point() else None)
                    for key, value in inputs.items()
                }
                with self.torch.inference_mode():
                    outputs = self.model(**model_inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[-1][:, 0, :].detach().float().cpu().numpy()
                probs = self.torch.softmax(outputs.logits, dim=-1).detach().float().cpu().numpy()
                for offset, frame_index in enumerate(crop_indices[start : start + 32]):
                    features[frame_index] = hidden[offset]
                    emotion_probs[frame_index] = probs[offset]
                    mask[frame_index] = 1
        centers = (retained_indices.astype(np.float64) + 0.5) / self.output_fps
        spans = frame_spans_from_centers(centers, duration, self.output_fps)
        face_frames = int(mask.sum())
        return VisionFeatures(
            features,
            emotion_probs,
            spans,
            mask,
            quality,
            face_bbox,
            face_confidence,
            track_ids,
            retained_indices,
            frame_count,
            {
                "decoded_frames": decoded_frame_count,
                "retained_frames": frame_count,
                "discarded_out_of_timeline_frames": discarded_frame_count,
                "face_frames": face_frames,
                "face_detection_ratio": face_frames / max(frame_count, 1),
                "track_switches": 0,
            },
        )

    def _decode_frames(self, video_path: Path) -> list[np.ndarray]:
        import cv2

        process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-vf",
                f"fps={self.output_fps:g}",
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "-q:v",
                "2",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        frames: list[np.ndarray] = []
        buffer = b""
        assert process.stdout is not None
        while True:
            chunk = process.stdout.read(1024 * 1024)
            if not chunk:
                break
            buffer += chunk
            while True:
                start = buffer.find(b"\xff\xd8")
                if start < 0:
                    buffer = buffer[-1:]
                    break
                end = buffer.find(b"\xff\xd9", start + 2)
                if end < 0:
                    buffer = buffer[start:]
                    break
                jpeg = buffer[start : end + 2]
                buffer = buffer[end + 2 :]
                decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if decoded is not None:
                    frames.append(decoded)
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg video decode failed for {video_path}: {stderr[-1000:]}")
        return frames

    def _detect(self, frames: list[np.ndarray]) -> list[list[_Detection]]:
        import cv2

        detections: list[list[_Detection]] = []
        with self.mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=self.min_detection_confidence
        ) as detector:
            for frame_index, frame in enumerate(frames):
                result = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_detections: list[_Detection] = []
                for detection in result.detections or []:
                    location = detection.location_data.relative_bounding_box
                    x1 = float(np.clip(location.xmin, 0.0, 1.0))
                    y1 = float(np.clip(location.ymin, 0.0, 1.0))
                    x2 = float(np.clip(location.xmin + location.width, 0.0, 1.0))
                    y2 = float(np.clip(location.ymin + location.height, 0.0, 1.0))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    score = float(detection.score[0]) if detection.score else 0.0
                    frame_detections.append(_Detection(frame_index, (x1, y1, x2, y2), score))
                detections.append(frame_detections)
        return detections

    @staticmethod
    def _iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
        x1 = max(left[0], right[0])
        y1 = max(left[1], right[1])
        x2 = min(left[2], right[2])
        y2 = min(left[3], right[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
        area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
        return intersection / max(area_left + area_right - intersection, 1e-12)

    def _select_track(self, detections: list[list[_Detection]]) -> dict[int, _Detection]:
        tracks: list[list[_Detection]] = []
        for frame_detections in detections:
            frame_detections = sorted(frame_detections, key=lambda item: item.confidence, reverse=True)
            assigned: set[int] = set()
            for detection in frame_detections:
                best_index, best_iou = None, 0.0
                for track_index, track in enumerate(tracks):
                    if track_index in assigned:
                        continue
                    previous = track[-1]
                    if detection.frame_index - previous.frame_index > self.track_max_gap_frames:
                        continue
                    score = self._iou(previous.bbox, detection.bbox)
                    if score > best_iou:
                        best_index, best_iou = track_index, score
                if best_index is not None and best_iou >= self.track_iou_threshold:
                    tracks[best_index].append(detection)
                    assigned.add(best_index)
                else:
                    tracks.append([detection])
        if not tracks:
            return {}
        selected = max(
            tracks,
            key=lambda track: (len(track), float(np.mean([item.confidence for item in track])), track[0].frame_index),
        )
        return {item.frame_index: item for item in selected}

    def _crop_face(self, frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray | None:
        import cv2

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        left, right = x1 * width, x2 * width
        top, bottom = y1 * height, y2 * height
        center_x, center_y = (left + right) / 2.0, (top + bottom) / 2.0
        size = max(right - left, bottom - top) * (1.0 + self.crop_margin)
        if size <= 1:
            return None
        left = int(max(0, round(center_x - size / 2)))
        right = int(min(width, round(center_x + size / 2)))
        top = int(max(0, round(center_y - size / 2)))
        bottom = int(min(height, round(center_y + size / 2)))
        if right <= left or bottom <= top:
            return None
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return None
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
