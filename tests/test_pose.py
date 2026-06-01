from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pose import MoveNetKeypoint, PoseAnalyzer


class PoseAnalyzerOverlayTest(unittest.TestCase):
    def test_landmarks_are_normalized_and_low_confidence_points_are_ignored(self) -> None:
        landmarks = PoseAnalyzer.landmarks_from_keypoints(
            np.array([[50.0, 25.0], [190.0, 90.0]], dtype=np.float32),
            np.array([0.90, 0.10], dtype=np.float32),
            frame_width=200,
            frame_height=100,
        )

        self.assertEqual(landmarks, {0: (0.25, 0.25)})

    def test_person_box_wraps_visible_landmarks(self) -> None:
        analyzer = PoseAnalyzer.__new__(PoseAnalyzer)
        frame = np.zeros((100, 200, 3), dtype=np.uint8)

        box = analyzer.person_box(
            {
                MoveNetKeypoint.RIGHT_SHOULDER: (0.40, 0.25),
                MoveNetKeypoint.RIGHT_WRIST: (0.80, 0.55),
            },
            frame,
            padding_ratio=0.0,
        )

        self.assertEqual(box, (80, 25, 160, 55))

    def test_render_uses_supplied_alert_color(self) -> None:
        analyzer = PoseAnalyzer.__new__(PoseAnalyzer)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        annotated = analyzer.render(
            frame,
            {
                MoveNetKeypoint.LEFT_SHOULDER: (0.2, 0.2),
                MoveNetKeypoint.RIGHT_SHOULDER: (0.8, 0.2),
            },
            color=(0, 0, 255),
        )

        self.assertGreater(int(annotated[:, :, 2].sum()), 0)
        self.assertEqual(int(annotated[:, :, 1].sum()), 0)


if __name__ == "__main__":
    unittest.main()
