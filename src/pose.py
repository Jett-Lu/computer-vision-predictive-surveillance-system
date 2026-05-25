from __future__ import annotations

import os
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
if "MPLCONFIGDIR" not in os.environ:
    matplotlib_cache = Path.cwd() / ".tmp" / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)

import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "movenet_singlepose_lightning.tflite"
MOVENET_INPUT_SIZE = 192
MIN_KEYPOINT_SCORE = 0.3
DEFAULT_SKELETON_COLOR = (0, 255, 0)


class MoveNetKeypoint(IntEnum):
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


MOVENET_CONNECTIONS = (
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
    landmarks: dict[int, tuple[float, float]]
    person_detected: bool


class PoseAnalyzer:
    """MoveNet Lightning pose inference and overlay rendering for one person."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"MoveNet model not found: {model_path}")

        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def analyze(self, frame: np.ndarray) -> PoseResult:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints = self._infer_keypoints(rgb)
        landmarks = self._visible_landmarks(keypoints)
        return PoseResult(
            landmarks=landmarks,
            person_detected=bool(landmarks),
        )

    def render(
        self,
        frame: np.ndarray,
        landmarks: dict[int, tuple[float, float]],
        color: tuple[int, int, int] = DEFAULT_SKELETON_COLOR,
    ) -> np.ndarray:
        annotated = frame.copy()
        self._draw_landmarks(annotated, landmarks, color)
        return annotated

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

    def _infer_keypoints(self, frame: np.ndarray) -> np.ndarray:
        model_input, scale, pad_x, pad_y = self._prepare_input(frame)
        self.interpreter.set_tensor(self.input_details[0]["index"], model_input)
        self.interpreter.invoke()
        raw_keypoints = self.interpreter.get_tensor(self.output_details[0]["index"])[0, 0]

        height, width = frame.shape[:2]
        keypoints = np.zeros((len(MoveNetKeypoint), 3), dtype=np.float32)

        for index, (input_y, input_x, score) in enumerate(raw_keypoints):
            pixel_x = (float(input_x) * MOVENET_INPUT_SIZE - pad_x) / scale
            pixel_y = (float(input_y) * MOVENET_INPUT_SIZE - pad_y) / scale
            keypoints[index] = (
                float(np.clip(pixel_x / width, 0.0, 1.0)),
                float(np.clip(pixel_y / height, 0.0, 1.0)),
                float(score),
            )

        return keypoints

    def _prepare_input(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        height, width = frame.shape[:2]
        scale = min(MOVENET_INPUT_SIZE / width, MOVENET_INPUT_SIZE / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=interpolation)

        canvas = np.zeros((MOVENET_INPUT_SIZE, MOVENET_INPUT_SIZE, 3), dtype=np.uint8)
        pad_x = (MOVENET_INPUT_SIZE - resized_width) // 2
        pad_y = (MOVENET_INPUT_SIZE - resized_height) // 2
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        return np.expand_dims(canvas, axis=0), scale, pad_x, pad_y

    def _visible_landmarks(self, keypoints: np.ndarray) -> dict[int, tuple[float, float]]:
        visible: dict[int, tuple[float, float]] = {}
        for index, (x, y, score) in enumerate(keypoints):
            if score >= MIN_KEYPOINT_SCORE:
                visible[index] = (float(x), float(y))
        return visible

    def _draw_landmarks(
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

        for start, end in MOVENET_CONNECTIONS:
            if start in points and end in points:
                cv2.line(frame, points[start], points[end], color, 2)

        for point in points.values():
            cv2.circle(frame, point, 4, color, -1)
