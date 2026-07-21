from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import AppConfig


class EnvironmentConfigTest(unittest.TestCase):
    def test_invalid_boolean_preserves_the_default(self) -> None:
        with patch.dict(os.environ, {"MONITOR_ALLOW_MODEL_DOWNLOADS": "maybe"}):
            config = AppConfig.from_env()

        self.assertTrue(config.allow_model_downloads)

    def test_non_finite_number_preserves_the_default(self) -> None:
        with patch.dict(os.environ, {"MONITOR_EXPRESSION_CONFIDENCE": "nan"}):
            config = AppConfig.from_env()

        self.assertEqual(config.min_concern_expression_confidence, 0.65)


if __name__ == "__main__":
    unittest.main()
