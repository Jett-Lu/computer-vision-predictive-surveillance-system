from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gesture import WaveAlertState
from review import (
    CLEAR_COLOR,
    HIGH_COLOR,
    MONITOR_COLOR,
    REVIEW_COLOR,
    ReviewLevelMonitor,
)


def wave_state(count: int) -> WaveAlertState:
    return WaveAlertState(
        recent_wave_count=count,
        wave_detected=False,
    )


class ReviewLevelMonitorTest(unittest.TestCase):
    def test_alert_colors_follow_green_yellow_orange_red_ladder(self) -> None:
        monitor = ReviewLevelMonitor()

        self.assertEqual(monitor.update(wave_state(0)).color, CLEAR_COLOR)
        self.assertEqual(monitor.update(wave_state(3)).color, MONITOR_COLOR)
        self.assertEqual(monitor.update(wave_state(5)).color, REVIEW_COLOR)
        self.assertEqual(monitor.update(wave_state(7)).color, HIGH_COLOR)

    def test_concern_expression_strength_changes_gradually(self) -> None:
        monitor = ReviewLevelMonitor(concern_smoothing_alpha=0.50)

        monitor.observe_expression("Fear", 0.80)
        first = monitor.update(wave_state(0))
        monitor.observe_expression("Fear", 0.80)
        second = monitor.update(wave_state(0))

        self.assertAlmostEqual(first.concern_strength, 0.40)
        self.assertAlmostEqual(second.concern_strength, 0.60)
        self.assertEqual(second.tier_label, "CLEAR")

    def test_expression_context_does_not_change_behavior_tier(self) -> None:
        monitor = ReviewLevelMonitor(concern_smoothing_alpha=1.0)
        monitor.observe_expression("Sadness", 0.80)

        no_behavior = monitor.update(wave_state(0))
        repeated_behavior = monitor.update(wave_state(4))

        self.assertEqual(no_behavior.tier_label, "CLEAR")
        self.assertEqual(no_behavior.color, CLEAR_COLOR)
        self.assertEqual(repeated_behavior.score, 2.0)
        self.assertEqual(repeated_behavior.tier_label, "MONITOR")
        self.assertEqual(repeated_behavior.color, MONITOR_COLOR)

    def test_neutral_happy_and_surprise_do_not_add_concern_context(self) -> None:
        for label in ("Neutral", "Happiness", "Surprise"):
            monitor = ReviewLevelMonitor(concern_smoothing_alpha=1.0)
            monitor.observe_expression(label, 0.90)

            state = monitor.update(wave_state(4))

            self.assertEqual(state.concern_strength, 0.0)
            self.assertEqual(state.tier_label, "MONITOR")

    def test_concern_context_clears_immediately_when_face_is_not_visible(self) -> None:
        monitor = ReviewLevelMonitor(concern_smoothing_alpha=1.0)
        monitor.observe_expression("Anger", 0.90)
        monitor.observe_expression(None, None)

        state = monitor.update(wave_state(4))

        self.assertEqual(state.concern_strength, 0.0)
        self.assertIsNone(state.concern_label)
        self.assertEqual(state.tier_label, "MONITOR")

    def test_low_confidence_concern_label_is_ignored(self) -> None:
        monitor = ReviewLevelMonitor(
            concern_smoothing_alpha=1.0,
            min_concern_confidence=0.65,
        )
        monitor.observe_expression("Anger", 0.40)

        state = monitor.update(wave_state(4))

        self.assertEqual(state.concern_strength, 0.0)
        self.assertIsNone(state.concern_label)


if __name__ == "__main__":
    unittest.main()
