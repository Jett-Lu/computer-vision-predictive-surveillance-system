from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotion import FaceEmotionAnalyzer


class FaceEmotionAnalyzerTest(unittest.TestCase):
    def test_padded_face_box_expands_and_clamps_to_crop(self) -> None:
        padded = FaceEmotionAnalyzer._padded_face_box((5, 5, 45, 45), 48, 48)

        self.assertEqual(padded, (1, 1, 48, 48))


if __name__ == "__main__":
    unittest.main()
