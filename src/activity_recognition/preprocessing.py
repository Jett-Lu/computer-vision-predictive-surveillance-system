"""Reproducible frame, pose, crop, and backbone-feature preprocessing."""

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Iterable

import cv2
import numpy as np

from activity_recognition.dataset import ActivitySample

if TYPE_CHECKING:
    import torch


KEYPOINT_COUNT = 17
DEFAULT_FRAMES_PER_SAMPLE = 16
DEFAULT_CROP_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
S3D_MEAN = (0.43216, 0.394666, 0.37645)
S3D_STD = (0.22803, 0.22145, 0.216989)


def normalize_pose(
    landmarks: dict[int, tuple[float, float]],
    box: tuple[int, int, int, int] | None,
    frame_shape: tuple[int, ...],
) -> np.ndarray:
    """Return 17 bounding-box-relative x/y/visibility keypoint triples."""
    features = np.zeros((KEYPOINT_COUNT, 3), dtype=np.float32)
    if box is None:
        return features
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = box
    box_width = max(1.0, float(x2 - x1))
    box_height = max(1.0, float(y2 - y1))
    for index, (x_normalized, y_normalized) in landmarks.items():
        if not 0 <= int(index) < KEYPOINT_COUNT:
            continue
        x_pixels = x_normalized * width
        y_pixels = y_normalized * height
        features[int(index)] = (
            float(np.clip((x_pixels - x1) / box_width, 0.0, 1.0)),
            float(np.clip((y_pixels - y1) / box_height, 0.0, 1.0)),
            1.0,
        )
    return features


def crop_person(
    frame: np.ndarray,
    box: tuple[int, int, int, int] | None,
    output_size: int = DEFAULT_CROP_SIZE,
    padding_fraction: float = 0.08,
) -> np.ndarray:
    """Crop one person with padding, falling back to the full frame."""
    height, width = frame.shape[:2]
    if box is None:
        x1, y1, x2, y2 = 0, 0, width, height
    else:
        x1, y1, x2, y2 = box
        pad_x = int((x2 - x1) * padding_fraction)
        pad_y = int((y2 - y1) * padding_fraction)
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        crop = frame
    resized = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def sample_video_frames(
    video_path: Path,
    frame_count: int = DEFAULT_FRAMES_PER_SAMPLE,
) -> list[np.ndarray]:
    """Decode uniformly spaced frames in source order."""
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"Video contains no readable frames: {video_path}")
    indexes = np.linspace(0, len(frames) - 1, frame_count).round().astype(int)
    return [frames[int(index)] for index in indexes]


def cache_activity_samples(
    samples: Iterable[ActivitySample],
    *,
    frames_per_sample: int = DEFAULT_FRAMES_PER_SAMPLE,
    overwrite: bool = False,
    pose_analyzer: Any | None = None,
) -> dict[str, int]:
    """Run YOLO pose once and cache normalized poses plus person crops."""
    sample_list = list(samples)
    pending_samples = [
        sample
        for sample in sample_list
        if overwrite or not Path(sample.cache_path).is_file()
    ]
    skipped = len(sample_list) - len(pending_samples)
    if not pending_samples:
        return {"completed": 0, "skipped": skipped, "failed": 0}

    if pose_analyzer is None:
        from pose import PoseAnalyzer

        pose_analyzer = PoseAnalyzer()
    completed = failed = 0
    for position, sample in enumerate(pending_samples, start=1):
        cache_path = Path(sample.cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(f"{cache_path.suffix}.partial")
        started = perf_counter()
        try:
            frames = sample_video_frames(Path(sample.video_path), frames_per_sample)
            reset = getattr(pose_analyzer, "reset_tracking", None)
            if callable(reset):
                reset()
            poses: list[np.ndarray] = []
            crops: list[np.ndarray] = []
            detected_count = 0
            for frame in frames:
                pose_results = pose_analyzer.analyze(frame)
                selected = max(
                    pose_results,
                    key=lambda result: _box_area(result.box),
                    default=None,
                )
                box = None if selected is None else selected.box
                landmarks = {} if selected is None else selected.landmarks
                detected_count += int(selected is not None)
                poses.append(normalize_pose(landmarks, box, frame.shape))
                crops.append(crop_person(frame, box))
            with temporary_path.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    pose=np.stack(poses),
                    crops=np.stack(crops),
                    label=np.int64(sample.label_index),
                    split=np.array(sample.split),
                    source=np.array(sample.video_path),
                    pose_detection_rate=np.float32(detected_count / len(frames)),
                    extraction_seconds=np.float64(perf_counter() - started),
                )
            os.replace(temporary_path, cache_path)
            completed += 1
            print(
                f"[{position}/{len(pending_samples)}] cached "
                f"{Path(sample.video_path).name}"
            )
        except (OSError, RuntimeError, ValueError, cv2.error) as exc:
            failed += 1
            temporary_path.unlink(missing_ok=True)
            print(
                f"[{position}/{len(pending_samples)}] failed "
                f"{sample.video_path}: {exc}"
            )
    return {"completed": completed, "skipped": skipped, "failed": failed}


def load_cached_arrays(
    samples: Iterable[ActivitySample],
    field_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    values: list[np.ndarray] = []
    labels: list[int] = []
    for sample in samples:
        cache_path = Path(sample.cache_path)
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Activity cache missing: {cache_path}. Run prepare_activity_data.py."
            )
        with np.load(cache_path, allow_pickle=False) as payload:
            values.append(np.asarray(payload[field_name]))
            labels.append(int(payload["label"]))
    return np.stack(values), np.asarray(labels, dtype=np.int64)


def image_tensor(crops: np.ndarray) -> "torch.Tensor":
    """Convert RGB uint8 crops to normalized MobileNet tensors."""
    import torch

    tensor = torch.from_numpy(crops).permute(0, 3, 1, 2).float().div(255.0)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (tensor - mean) / std


def video_tensor(crops: np.ndarray) -> "torch.Tensor":
    """Convert RGB uint8 crops to one normalized S3D clip."""
    import torch

    tensor = torch.from_numpy(crops).permute(3, 0, 1, 2).float().div(255.0)
    mean = torch.tensor(S3D_MEAN).view(3, 1, 1, 1)
    std = torch.tensor(S3D_STD).view(3, 1, 1, 1)
    return ((tensor - mean) / std).unsqueeze(0)


def _box_area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])
