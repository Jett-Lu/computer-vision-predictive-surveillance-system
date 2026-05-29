from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DETECTOR_MODEL_PATH = PROJECT_ROOT / "data" / "face_detection_yunet_2023mar.onnx"
DEFAULT_RECOGNIZER_MODEL_PATH = PROJECT_ROOT / "data" / "face_recognition_sface_2021dec.onnx"
DEFAULT_DETECTOR_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
DEFAULT_RECOGNIZER_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
    "face_recognition_sface_2021dec.onnx"
)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
COSINE_MATCH_THRESHOLD = 0.363


@dataclass(frozen=True)
class DetectedFace:
    raw: np.ndarray
    box: tuple[int, int, int, int]
    confidence: float


@dataclass(frozen=True)
class KnownIdentity:
    name: str
    feature: np.ndarray
    image_path: Path


@dataclass(frozen=True)
class IdentityMatch:
    name: str
    score: float | None
    matched: bool


class OpenCVFaceIdentifier:
    """Dlib-free face detection and identity matching using OpenCV DNN models."""

    def __init__(
        self,
        detector_model_path: Path = DEFAULT_DETECTOR_MODEL_PATH,
        recognizer_model_path: Path = DEFAULT_RECOGNIZER_MODEL_PATH,
        cosine_threshold: float = COSINE_MATCH_THRESHOLD,
    ) -> None:
        ensure_model(detector_model_path, DEFAULT_DETECTOR_MODEL_URL)
        ensure_model(recognizer_model_path, DEFAULT_RECOGNIZER_MODEL_URL)

        self.cosine_threshold = cosine_threshold
        self.detector = cv2.FaceDetectorYN_create(
            str(detector_model_path),
            "",
            (320, 320),
            0.8,
            0.3,
            5000,
        )
        self.recognizer = cv2.FaceRecognizerSF_create(str(recognizer_model_path), "")

    def load_enrollments(self, enrollments_dir: Path) -> list[KnownIdentity]:
        if not enrollments_dir.exists():
            return []

        identities: list[KnownIdentity] = []
        for person_dir in sorted(path for path in enrollments_dir.iterdir() if path.is_dir()):
            for image_path in sorted(person_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                image = cv2.imread(str(image_path))
                if image is None:
                    print(f"Warning: could not read {image_path}, skipping")
                    continue

                face = self.detect_largest_face(image)
                if face is None:
                    print(f"Warning: no face found in {image_path}, skipping")
                    continue

                feature = self.extract_feature(image, face)
                if feature is None:
                    print(f"Warning: could not encode face in {image_path}, skipping")
                    continue

                identities.append(
                    KnownIdentity(
                        name=person_dir.name,
                        feature=feature,
                        image_path=image_path,
                    )
                )

        return identities

    def detect_largest_face(self, image: np.ndarray) -> DetectedFace | None:
        if image.size == 0:
            return None

        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image)
        if faces is None or len(faces) == 0:
            return None

        raw_face = max(faces, key=lambda face: face[2] * face[3])
        return DetectedFace(
            raw=raw_face,
            box=face_box(raw_face),
            confidence=float(raw_face[14]),
        )

    def extract_feature(
        self,
        image: np.ndarray,
        face: DetectedFace,
    ) -> np.ndarray | None:
        aligned = self.recognizer.alignCrop(image, face.raw)
        if aligned is None or aligned.size == 0:
            return None

        return self.recognizer.feature(aligned).copy()

    def identify(
        self,
        image: np.ndarray,
        face: DetectedFace,
        known_identities: list[KnownIdentity],
    ) -> IdentityMatch:
        feature = self.extract_feature(image, face)
        if feature is None:
            return IdentityMatch(name="Unknown", score=None, matched=False)

        best_name = "Unknown"
        best_score = -1.0
        for identity in known_identities:
            score = self.recognizer.match(
                feature,
                identity.feature,
                cv2.FaceRecognizerSF_FR_COSINE,
            )
            if score > best_score:
                best_score = float(score)
                best_name = identity.name

        matched = best_score >= self.cosine_threshold
        return IdentityMatch(
            name=best_name if matched else "Unknown",
            score=best_score,
            matched=matched,
        )


def face_box(raw_face: np.ndarray) -> tuple[int, int, int, int]:
    x, y, width, height = raw_face[:4]
    x1 = max(0, int(round(float(x))))
    y1 = max(0, int(round(float(y))))
    x2 = max(x1, int(round(float(x + width))))
    y2 = max(y1, int(round(float(y + height))))
    return x1, y1, x2, y2


def offset_box(
    box: tuple[int, int, int, int],
    offset_x: int,
    offset_y: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y


def ensure_model(model_path: Path, model_url: str) -> None:
    if model_path.exists():
        return

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading OpenCV face model: {model_path.name}")
    try:
        urlretrieve(model_url, model_path)
    except Exception as exc:
        if model_path.exists():
            model_path.unlink()
        raise RuntimeError(
            f"Could not download required OpenCV face model from {model_url}"
        ) from exc
