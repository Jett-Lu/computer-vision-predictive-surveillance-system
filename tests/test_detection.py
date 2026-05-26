from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from detection import _face_centre_inside_box


class DetectionAssociationTest(unittest.TestCase):
    def test_face_is_associated_when_its_centre_is_inside_person_box(self) -> None:
        self.assertTrue(_face_centre_inside_box((30, 20, 50, 40), (0, 0, 100, 100)))

    def test_face_is_not_associated_when_person_box_is_missing(self) -> None:
        self.assertFalse(_face_centre_inside_box((30, 20, 50, 40), None))


if __name__ == "__main__":
    unittest.main()
