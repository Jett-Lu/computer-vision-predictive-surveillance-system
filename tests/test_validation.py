from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from validation import ValidationMetrics, _collect_snapshots, evaluate_metrics


class ValidationMetricsTest(unittest.TestCase):
    def test_empty_input_fails_validation(self) -> None:
        metrics = ValidationMetrics(0, 0, 0.0, 0.0, (), {})

        self.assertIn("no readable frames", evaluate_metrics({}, metrics)[0])

    def test_high_tier_is_counted_once_per_frame_with_multiple_people(self) -> None:
        tiers: Counter[str] = Counter()
        snapshots = [
            SimpleNamespace(
                identity_confirmed=False,
                identity_name="Person",
                tier_label="HIGH",
            ),
            SimpleNamespace(
                identity_confirmed=False,
                identity_name="Person",
                tier_label="HIGH",
            ),
        ]

        _collect_snapshots(snapshots, set(), tiers, 0, 0)

        self.assertEqual(tiers["HIGH"], 1)

    def test_reports_missing_identity_and_low_fps(self) -> None:
        metrics = ValidationMetrics(
            frames_processed=100,
            max_people=1,
            average_people=1.0,
            processing_fps=3.0,
            observed_identities=("Other",),
            tier_counts={"CLEAR": 100},
        )

        failures = evaluate_metrics(
            {
                "min_people": 1,
                "min_processing_fps": 5.0,
                "expected_identities": ["Alex"],
            },
            metrics,
        )

        self.assertEqual(len(failures), 2)
        self.assertIn("processing FPS", failures[0])
        self.assertIn("Alex", failures[1])


if __name__ == "__main__":
    unittest.main()
