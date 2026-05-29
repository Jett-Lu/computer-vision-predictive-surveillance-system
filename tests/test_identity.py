from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from identity import face_box, offset_box


class IdentityHelpersTest(unittest.TestCase):
    def test_face_box_converts_yunet_width_height_to_corners(self) -> None:
        raw_face = np.array([10.2, 20.4, 30.3, 40.6, *([0.0] * 10), 0.95])

        self.assertEqual(face_box(raw_face), (10, 20, 40, 61))

    def test_offset_box_moves_crop_box_to_frame_coordinates(self) -> None:
        self.assertEqual(offset_box((5, 6, 30, 40), 100, 200), (105, 206, 130, 240))


if __name__ == "__main__":
    unittest.main()
