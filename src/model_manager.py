"""Reproducible, atomic model provisioning for first-run setup."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen
import os
import shutil
import time

from config import AppConfig
from logging_setup import get_logger


logger = get_logger("models")


class ModelUnavailableError(RuntimeError):
    """Raised when a required local model cannot be verified or downloaded."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    url: str
    sha256: str


FACE_DETECTOR_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
FACE_RECOGNIZER_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
POSE_MODEL_SHA256 = "c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212"
OPENCV_ZOO_REVISION = "47534e27c9851bb1128ccc0102f1145e27f23f98"


def identity_model_specs(config: AppConfig) -> tuple[ModelSpec, ModelSpec]:
    return (
        ModelSpec(
            "YuNet face detector",
            config.face_detector_model_path,
            f"https://raw.githubusercontent.com/opencv/opencv_zoo/{OPENCV_ZOO_REVISION}/"
            "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            FACE_DETECTOR_SHA256,
        ),
        ModelSpec(
            "SFace recognizer",
            config.face_recognizer_model_path,
            f"https://raw.githubusercontent.com/opencv/opencv_zoo/{OPENCV_ZOO_REVISION}/"
            "models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
            FACE_RECOGNIZER_SHA256,
        ),
    )


def pose_model_spec(config: AppConfig) -> ModelSpec:
    return ModelSpec(
        "YOLO pose model",
        config.pose_model_path,
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-pose.pt",
        POSE_MODEL_SHA256,
    )


def ensure_models(specs: tuple[ModelSpec, ...], config: AppConfig) -> None:
    for spec in specs:
        ensure_model(spec, config)


def ensure_model(spec: ModelSpec, config: AppConfig) -> Path:
    """Return a verified model path, downloading atomically when permitted."""
    if spec.path.exists() and file_sha256(spec.path) == spec.sha256:
        logger.info("Model ready: %s", spec.name)
        return spec.path

    if not config.allow_model_downloads:
        raise ModelUnavailableError(
            f"{spec.name} is missing or invalid at {spec.path}. "
            "Model downloads are disabled."
        )

    spec.path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, config.model_download_attempts + 1):
        temporary_path = spec.path.with_name(f"{spec.path.name}.download-{os.getpid()}")
        try:
            logger.info(
                "Downloading %s (attempt %s/%s)",
                spec.name,
                attempt,
                config.model_download_attempts,
            )
            request = Request(spec.url, headers={"User-Agent": "monitoring-poc/1.0"})
            with urlopen(request, timeout=config.model_download_timeout_seconds) as response:
                with temporary_path.open("wb") as destination:
                    shutil.copyfileobj(response, destination)

            actual_hash = file_sha256(temporary_path)
            if actual_hash != spec.sha256:
                raise ModelUnavailableError(
                    f"Checksum mismatch for {spec.name}: expected {spec.sha256}, "
                    f"received {actual_hash}."
                )

            os.replace(temporary_path, spec.path)
            logger.info("Model installed: %s", spec.path)
            return spec.path
        except Exception as exc:
            last_error = exc
            logger.warning("Could not provision %s: %s", spec.name, exc)
            if attempt < config.model_download_attempts:
                time.sleep(min(2 ** (attempt - 1), 4))
        finally:
            temporary_path.unlink(missing_ok=True)

    raise ModelUnavailableError(
        f"Could not prepare {spec.name} at {spec.path}. Last error: {last_error}"
    ) from last_error


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
