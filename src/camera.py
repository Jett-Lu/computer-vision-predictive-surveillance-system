"""Camera source helpers for OpenCV capture."""

from __future__ import annotations

import os

import cv2


def normalize_camera_source(source: str | int) -> int | str:
    """Convert numeric camera choices to OpenCV camera indexes."""
    if isinstance(source, int):
        return source

    stripped = source.strip()
    return int(stripped) if stripped.isdigit() else stripped


def prompt_camera_source(default: str = "1") -> int | str:
    """Ask for a camera index/path, defaulting to the usual external camera."""
    response = input(f"Camera source [{default}=external, 0=built-in]: ").strip()
    return normalize_camera_source(response or default)


def open_capture(source: str | int) -> cv2.VideoCapture:
    """Open a camera or video source with a Windows-friendly backend fallback."""
    normalized = normalize_camera_source(source)

    if isinstance(normalized, int) and os.name == "nt":
        capture = cv2.VideoCapture(normalized, cv2.CAP_DSHOW)
        if capture.isOpened():
            return capture
        capture.release()

    return cv2.VideoCapture(normalized)
