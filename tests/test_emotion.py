from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotion import FaceEmotionAnalyzer, display_emotion_label


class FaceEmotionAnalyzerTest(unittest.TestCase):
    def test_padded_face_box_expands_and_clamps_to_crop(self) -> None:
        padded = FaceEmotionAnalyzer._padded_face_box((5, 5, 45, 45), 48, 48)

        self.assertEqual(padded, (1, 1, 48, 48))

    def test_low_confidence_expression_is_marked_uncertain(self) -> None:
        self.assertEqual(display_emotion_label("Anger", 0.44, 0.45), "Uncertain")
        self.assertEqual(display_emotion_label("Anger", 0.45, 0.45), "Anger")


if __name__ == "__main__":
    unittest.main()
