"""Operator-facing environment and model readiness checks."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import os
import platform
import sys

import cv2

from config import AppConfig
from model_manager import (
    ModelUnavailableError,
    ensure_model,
    file_sha256,
    identity_model_specs,
    pose_model_spec,
)


@dataclass(frozen=True)
class DiagnosticCheck:
    name: str
    passed: bool
    detail: str


def run_diagnostics(
    config: AppConfig,
    prepare_models: bool = False,
) -> list[DiagnosticCheck]:
    checks = [
        DiagnosticCheck(
            "Platform",
            True,
            f"{platform.system()} {platform.release()} | Python {sys.version.split()[0]}",
        ),
        _check_packages(),
        DiagnosticCheck(
            "OpenCV face APIs",
            hasattr(cv2, "FaceDetectorYN_create") and hasattr(cv2, "FaceRecognizerSF_create"),
            "YuNet and SFace APIs available"
            if hasattr(cv2, "FaceDetectorYN_create") and hasattr(cv2, "FaceRecognizerSF_create")
            else "Install opencv-contrib-python",
        ),
        _check_writable_directory(config.output_dir, "Output directory"),
        _check_writable_directory(config.data_dir, "Model directory"),
    ]

    specs = (*identity_model_specs(config), pose_model_spec(config))
    for spec in specs:
        if prepare_models:
            try:
                ensure_model(spec, config)
            except ModelUnavailableError as exc:
                checks.append(DiagnosticCheck(spec.name, False, str(exc)))
                continue
        if not spec.path.exists():
            checks.append(DiagnosticCheck(spec.name, False, f"missing: {spec.path}"))
            continue
        actual_hash = file_sha256(spec.path)
        checks.append(
            DiagnosticCheck(
                spec.name,
                actual_hash == spec.sha256,
                "ready" if actual_hash == spec.sha256 else "checksum mismatch",
            )
        )
    return checks


def print_diagnostics(checks: list[DiagnosticCheck]) -> None:
    for check in checks:
        print(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    print(
        f"\nOverall: {'READY' if all(check.passed for check in checks) else 'ACTION REQUIRED'}"
    )


def _check_packages() -> DiagnosticCheck:
    package_names = (
        "opencv-contrib-python",
        "numpy",
        "mediapipe",
        "emotiefflib",
        "ultralytics",
        "lap",
    )
    installed: list[str] = []
    missing: list[str] = []
    for package_name in package_names:
        try:
            installed.append(f"{package_name}={version(package_name)}")
        except PackageNotFoundError:
            missing.append(package_name)
    return DiagnosticCheck(
        "Python packages",
        not missing,
        ", ".join(installed) if not missing else f"missing: {', '.join(missing)}",
    )


def _check_writable_directory(path: Path, name: str) -> DiagnosticCheck:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".write-test-{os.getpid()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return DiagnosticCheck(name, True, str(path))
    except OSError as exc:
        return DiagnosticCheck(name, False, str(exc))
