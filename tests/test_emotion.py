from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotion import (
    EmotionSmoother,
    FaceEmotionAnalyzer,
    FaceEmotionResult,
)


class FaceEmotionAnalyzerTest(unittest.TestCase):
    def test_padded_face_box_expands_and_clamps_to_crop(self) -> None:
        padded = FaceEmotionAnalyzer._padded_face_box((5, 5, 45, 45), 48, 48)

        self.assertEqual(padded, (1, 1, 48, 48))

    def test_uncertain_expression_keeps_its_best_label(self) -> None:
        uncertain = FaceEmotionResult((0, 0, 10, 10), (), "Anger", 0.44, True)
        confident = FaceEmotionResult((0, 0, 10, 10), (), "Anger", 0.45)

        self.assertEqual(uncertain.display_label, "Anger?")
        self.assertEqual(confident.display_label, "Anger")

    def test_smoother_uses_a_weighted_rolling_vote(self) -> None:
        smoother = EmotionSmoother(window_seconds=0.8, uncertainty_threshold=0.45)
        neutral = FaceEmotionResult((0, 0, 10, 10), (), "Neutral", 0.40, True)
        happiness = FaceEmotionResult((0, 0, 10, 10), (), "Happiness", 0.35, True)

        smoother.update(neutral, 0.0)
        smoother.update(neutral, 0.2)
        smoothed = smoother.update(happiness, 0.4)

        self.assertEqual(smoothed.label, "Neutral")
        self.assertEqual(smoothed.display_label, "Neutral?")

    def test_smoother_clears_when_the_face_disappears(self) -> None:
        smoother = EmotionSmoother()
        result = FaceEmotionResult((0, 0, 10, 10), (), "Neutral", 0.70)

        smoother.update(result, 0.0)

        self.assertIsNone(smoother.update(None, 0.2))


if __name__ == "__main__":
    unittest.main()
