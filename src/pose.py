from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "yolov8n-pose.pt"
MIN_KEYPOINT_SCORE = 0.3
DEFAULT_SKELETON_COLOR = (0, 255, 0)


class MoveNetKeypoint(IntEnum):
    """COCO keypoint indexes shared by MoveNet and Ultralytics pose models."""

    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


POSE_CONNECTIONS = (
    (MoveNetKeypoint.LEFT_SHOULDER, MoveNetKeypoint.RIGHT_SHOULDER),
    (MoveNetKeypoint.LEFT_SHOULDER, MoveNetKeypoint.LEFT_ELBOW),
    (MoveNetKeypoint.LEFT_ELBOW, MoveNetKeypoint.LEFT_WRIST),
    (MoveNetKeypoint.RIGHT_SHOULDER, MoveNetKeypoint.RIGHT_ELBOW),
    (MoveNetKeypoint.RIGHT_ELBOW, MoveNetKeypoint.RIGHT_WRIST),
    (MoveNetKeypoint.LEFT_SHOULDER, MoveNetKeypoint.LEFT_HIP),
    (MoveNetKeypoint.RIGHT_SHOULDER, MoveNetKeypoint.RIGHT_HIP),
    (MoveNetKeypoint.LEFT_HIP, MoveNetKeypoint.RIGHT_HIP),
    (MoveNetKeypoint.LEFT_HIP, MoveNetKeypoint.LEFT_KNEE),
    (MoveNetKeypoint.LEFT_KNEE, MoveNetKeypoint.LEFT_ANKLE),
    (MoveNetKeypoint.RIGHT_HIP, MoveNetKeypoint.RIGHT_KNEE),
    (MoveNetKeypoint.RIGHT_KNEE, MoveNetKeypoint.RIGHT_ANKLE),
    (MoveNetKeypoint.NOSE, MoveNetKeypoint.LEFT_EYE),
    (MoveNetKeypoint.NOSE, MoveNetKeypoint.RIGHT_EYE),
    (MoveNetKeypoint.LEFT_EYE, MoveNetKeypoint.LEFT_EAR),
    (MoveNetKeypoint.RIGHT_EYE, MoveNetKeypoint.RIGHT_EAR),
)


@dataclass(frozen=True)
class PoseResult:
    track_id: int | None
    box: tuple[int, int, int, int]
    landmarks: dict[int, tuple[float, float]]


class PoseAnalyzer:
    """Ultralytics multi-person pose tracking and skeleton rendering."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))

    def analyze(self, frame: np.ndarray) -> list[PoseResult]:
        results = self.model.track(
            frame,
            conf=0.30,
            iou=0.45,
            verbose=False,
            classes=[0],
            persist=True,
            tracker="bytetrack.yaml",
        )
        result = results[0]
        boxes = result.boxes
        keypoints = result.keypoints
        if boxes is None or keypoints is None:
            return []

        height, width = frame.shape[:2]
        tracked_poses: list[PoseResult] = []
        for index in range(len(boxes)):
            track_id = None if boxes.id is None else int(boxes.id[index].item())
            landmarks = self.landmarks_from_keypoints(
                keypoints.xy[index],
                None if keypoints.conf is None else keypoints.conf[index],
                frame_width=width,
                frame_height=height,
            )
            tracked_poses.append(
                PoseResult(
                    track_id=track_id,
                    box=tuple(boxes.xyxy[index].int().tolist()),
                    landmarks=landmarks,
                )
            )

        return tracked_poses

    @staticmethod
    def landmarks_from_keypoints(
        keypoints: Any,
        confidence: Any,
        frame_width: int,
        frame_height: int,
    ) -> dict[int, tuple[float, float]]:
        points = _as_numpy(keypoints)
        scores = (
            np.ones(len(points), dtype=np.float32)
            if confidence is None
            else _as_numpy(confidence)
        )
        landmarks: dict[int, tuple[float, float]] = {}
        for index, ((x, y), score) in enumerate(zip(points, scores)):
            if float(score) >= MIN_KEYPOINT_SCORE:
                landmarks[index] = (
                    float(np.clip(x / frame_width, 0.0, 1.0)),
                    float(np.clip(y / frame_height, 0.0, 1.0)),
                )
        return landmarks

    def render(
        self,
        frame: np.ndarray,
        landmarks: dict[int, tuple[float, float]],
        color: tuple[int, int, int] = DEFAULT_SKELETON_COLOR,
    ) -> np.ndarray:
        annotated = frame.copy()
        self.draw_landmarks(annotated, landmarks, color)
        return annotated

    def draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: dict[int, tuple[float, float]],
        color: tuple[int, int, int] = DEFAULT_SKELETON_COLOR,
    ) -> None:
        height, width = frame.shape[:2]
        points = {
            index: (int(point[0] * width), int(point[1] * height))
            for index, point in landmarks.items()
        }

        for start, end in POSE_CONNECTIONS:
            if start in points and end in points:
                cv2.line(frame, points[start], points[end], color, 2)

        for point in points.values():
            cv2.circle(frame, point, 4, color, -1)

    def person_box(
        self,
        landmarks: dict[int, tuple[float, float]],
        frame: np.ndarray,
        padding_ratio: float = 0.08,
    ) -> tuple[int, int, int, int] | None:
        if not landmarks:
            return None

        height, width = frame.shape[:2]
        xs = [point[0] for point in landmarks.values()]
        ys = [point[1] for point in landmarks.values()]
        left = max(0.0, min(xs) - padding_ratio)
        top = max(0.0, min(ys) - padding_ratio)
        right = min(1.0, max(xs) + padding_ratio)
        bottom = min(1.0, max(ys) + padding_ratio)
        return (
            int(left * width),
            int(top * height),
            int(right * width),
            int(bottom * height),
        )


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)
