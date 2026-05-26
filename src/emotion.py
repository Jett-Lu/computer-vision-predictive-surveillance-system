from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_DIR = PROJECT_ROOT / ".tmp"
(TMP_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP_DIR / "matplotlib"))

import cv2
import mediapipe as mp
from emotiefflib.facial_analysis import EmotiEffLibRecognizer
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


DEFAULT_FACE_MODEL_PATH = PROJECT_ROOT / "data" / "blaze_face_short_range.tflite"
FACE_PADDING_RATIO = 0.12


@dataclass(frozen=True)
class FaceEmotionResult:
    box: tuple[int, int, int, int]
    keypoints: tuple[tuple[int, int], ...]
    label: str
    confidence: float


class FaceEmotionAnalyzer:
    """Detect a face within the pose box and classify its visible expression."""

    def __init__(self, model_path: Path = DEFAULT_FACE_MODEL_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Face detector model not found: {model_path}")

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        detector_options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.5,
        )
        self.detector = vision.FaceDetector.create_from_options(detector_options)
        self.recognizer = EmotiEffLibRecognizer(
            engine="onnx",
            model_name="enet_b2_8",
            device="cpu",
        )

    def analyze(
        self,
        frame,
        person_box: tuple[int, int, int, int] | None,
    ) -> FaceEmotionResult | None:
        if person_box is None:
            return None

        x1, y1, x2, y2 = person_box
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        crop_height, crop_width = crop.shape[:2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
        detection_result = self.detector.detect(image)
        if not detection_result.detections:
            return None

        face = max(
            detection_result.detections,
            key=lambda detection: detection.bounding_box.width * detection.bounding_box.height,
        )
        bbox = face.bounding_box
        fx1 = max(0, bbox.origin_x)
        fy1 = max(0, bbox.origin_y)
        fx2 = min(crop_width, bbox.origin_x + bbox.width)
        fy2 = min(crop_height, bbox.origin_y + bbox.height)
        if fx2 <= fx1 or fy2 <= fy1:
            return None

        padded_box = self._padded_face_box((fx1, fy1, fx2, fy2), crop_width, crop_height)
        ex1, ey1, ex2, ey2 = padded_box
        face_rgb = crop_rgb[ey1:ey2, ex1:ex2]
        if face_rgb.size == 0:
            return None

        _, scores = self.recognizer.predict_emotions(face_rgb, logits=False)
        emotion_index = int(scores[0].argmax())
        label = self.recognizer.idx_to_emotion_class[emotion_index]
        confidence = float(scores[0][emotion_index])
        keypoints = tuple(
            (int(keypoint.x * crop_width) + x1, int(keypoint.y * crop_height) + y1)
            for keypoint in face.keypoints
        )

        return FaceEmotionResult(
            box=(fx1 + x1, fy1 + y1, fx2 + x1, fy2 + y1),
            keypoints=keypoints,
            label=label,
            confidence=confidence,
        )

    def close(self) -> None:
        self.detector.close()

    @staticmethod
    def _padded_face_box(
        face_box: tuple[int, int, int, int],
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = face_box
        pad_x = int((x2 - x1) * FACE_PADDING_RATIO)
        pad_y = int((y2 - y1) * FACE_PADDING_RATIO)
        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(width, x2 + pad_x),
            min(height, y2 + pad_y),
        )
