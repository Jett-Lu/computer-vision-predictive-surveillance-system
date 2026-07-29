from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from activity_recognition.labels import (
    ACTIVITY_LABELS,
    LABEL_TO_INDEX,
    normalize_hmdb51_label,
)


class ActivityLabelTest(unittest.TestCase):
    def test_hmdb51_labels_map_to_stable_public_order(self) -> None:
        self.assertEqual(
            ACTIVITY_LABELS,
            ("walking", "running", "standing", "sitting"),
        )
        self.assertEqual(normalize_hmdb51_label("walk"), "walking")
        self.assertEqual(normalize_hmdb51_label("RUN"), "running")
        self.assertEqual(LABEL_TO_INDEX["sitting"], 3)

    def test_unknown_hmdb51_label_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_hmdb51_label("cartwheel")


if __name__ == "__main__":
    unittest.main()
