from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import DEFAULT_CONFIG
from model_manager import ModelSpec, ModelUnavailableError, ensure_model


class ModelManagerTest(unittest.TestCase):
    def test_existing_model_must_match_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.bin"
            payload = b"verified model"
            model_path.write_bytes(payload)
            spec = ModelSpec(
                "test model",
                model_path,
                "https://invalid.example/model.bin",
                sha256(payload).hexdigest(),
            )
            config = replace(DEFAULT_CONFIG, allow_model_downloads=False)

            self.assertEqual(ensure_model(spec, config), model_path)

    def test_missing_model_fails_cleanly_when_downloads_are_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = ModelSpec(
                "test model",
                Path(temp_dir) / "missing.bin",
                "https://invalid.example/model.bin",
                "0" * 64,
            )
            config = replace(DEFAULT_CONFIG, allow_model_downloads=False)

            with self.assertRaises(ModelUnavailableError):
                ensure_model(spec, config)

    def test_unreadable_model_fails_cleanly_when_downloads_are_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.bin"
            model_path.write_bytes(b"model")
            spec = ModelSpec(
                "test model",
                model_path,
                "https://invalid.example/model.bin",
                "0" * 64,
            )
            config = replace(DEFAULT_CONFIG, allow_model_downloads=False)

            with patch("model_manager.file_sha256", side_effect=OSError("denied")):
                with self.assertRaises(ModelUnavailableError):
                    ensure_model(spec, config)


if __name__ == "__main__":
    unittest.main()
