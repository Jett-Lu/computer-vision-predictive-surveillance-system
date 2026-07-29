"""Canonical activity labels shared by training, evaluation, and reporting."""

from __future__ import annotations


ACTIVITY_LABELS = ("walking", "running", "standing", "sitting")
LABEL_TO_INDEX = {label: index for index, label in enumerate(ACTIVITY_LABELS)}
HMDB51_TO_ACTIVITY = {
    "walk": "walking",
    "run": "running",
    "stand": "standing",
    "sit": "sitting",
}


def normalize_hmdb51_label(label: str) -> str:
    """Map one HMDB51 class name to the public four-class vocabulary."""
    try:
        return HMDB51_TO_ACTIVITY[label.strip().casefold()]
    except KeyError as exc:
        raise ValueError(f"Unsupported HMDB51 activity: {label}") from exc
