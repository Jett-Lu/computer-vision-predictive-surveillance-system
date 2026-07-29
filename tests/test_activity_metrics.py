from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from activity_recognition.metrics import (
    classification_metrics,
    confusion_matrix,
    write_predictions_csv,
)


class ActivityMetricsTest(unittest.TestCase):
    def test_metrics_are_computed_from_predictions(self) -> None:
        matrix = confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1], class_count=2)
        metrics = classification_metrics(matrix)

        self.assertEqual(matrix.tolist(), [[1, 1], [0, 2]])
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["macro_precision"], 5 / 6)
        self.assertAlmostEqual(metrics["macro_recall"], 0.75)
        self.assertAlmostEqual(metrics["macro_f1"], 11 / 15)

    def test_empty_confusion_matrix_has_zero_metrics(self) -> None:
        metrics = classification_metrics(confusion_matrix([], [], class_count=4))

        self.assertEqual(metrics["accuracy"], 0.0)
        self.assertEqual(metrics["macro_f1"], 0.0)
        self.assertEqual(metrics["sample_count"], 0)

    def test_confusion_matrix_rejects_malformed_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "counts must match"):
            confusion_matrix([0, 1], [0])
        with self.assertRaisesRegex(ValueError, "out of range"):
            confusion_matrix([0], [-1])
        with self.assertRaisesRegex(ValueError, "must be integers"):
            confusion_matrix([0], [1.5])

    def test_metrics_reject_negative_or_non_square_matrices(self) -> None:
        with self.assertRaisesRegex(ValueError, "square"):
            classification_metrics(np.zeros((2, 3), dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "must not be negative"):
            classification_metrics(np.array([[1, -1], [0, 1]]))

    def test_predictions_are_written_with_label_metadata(self) -> None:
        row = {
            "model": "mlp",
            "sample_key": "sample-1",
            "source_video": "walk_1.avi",
            "split": "test",
            "expected_index": 0,
            "expected_label": "walking",
            "predicted_index": 1,
            "predicted_label": "running",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            write_predictions_csv(path, [row])
            with path.open(newline="", encoding="utf-8") as stream:
                saved = list(csv.DictReader(stream))

        self.assertEqual(saved[0]["expected_label"], "walking")
        self.assertEqual(saved[0]["predicted_label"], "running")


if __name__ == "__main__":
    unittest.main()
