from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pose import PoseAnalyzer, _clamp_box


class PoseAnalyzerOverlayTest(unittest.TestCase):
    def test_person_box_is_clamped_to_frame_boundaries(self) -> None:
        self.assertEqual(_clamp_box((-10, 5, 120, 90), 100, 80), (0, 5, 100, 80))

    def test_landmarks_are_normalized_and_low_confidence_points_are_ignored(self) -> None:
        landmarks = PoseAnalyzer.landmarks_from_keypoints(
            np.array([[50.0, 25.0], [190.0, 90.0]], dtype=np.float32),
            np.array([0.90, 0.10], dtype=np.float32),
            frame_width=200,
            frame_height=100,
        )

        self.assertEqual(landmarks, {0: (0.25, 0.25)})


if __name__ == "__main__":
    unittest.main()
