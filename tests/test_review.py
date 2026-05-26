from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gesture import GREEN, WaveAlertState
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
        tier_label="CLEAR",
        color=GREEN,
        wave_detected=False,
    )


class ReviewLevelMonitorTest(unittest.TestCase):
    def test_alert_colors_follow_green_yellow_orange_red_ladder(self) -> None:
        monitor = ReviewLevelMonitor()

        self.assertEqual(monitor.update(wave_state(0)).color, CLEAR_COLOR)
        self.assertEqual(monitor.update(wave_state(3)).color, MONITOR_COLOR)
        self.assertEqual(monitor.update(wave_state(5)).color, REVIEW_COLOR)
        self.assertEqual(monitor.update(wave_state(7)).color, HIGH_COLOR)

    def test_concern_expression_changes_multiplier_immediately_and_gradually(self) -> None:
        monitor = ReviewLevelMonitor(concern_smoothing_alpha=0.50)

        monitor.observe_expression("Fear", 0.80)
        first = monitor.update(wave_state(0))
        monitor.observe_expression("Fear", 0.80)
        second = monitor.update(wave_state(0))

        self.assertAlmostEqual(first.concern_strength, 0.40)
        self.assertAlmostEqual(first.expression_multiplier, 1.20)
        self.assertAlmostEqual(second.concern_strength, 0.60)
        self.assertAlmostEqual(second.expression_multiplier, 1.30)
        self.assertEqual(second.tier_label, "CLEAR")

    def test_modifier_strengthens_repeated_activity_but_not_expression_alone(self) -> None:
        monitor = ReviewLevelMonitor(concern_smoothing_alpha=1.0)
        monitor.observe_expression("Sadness", 0.80)

        no_behavior = monitor.update(wave_state(0))
        repeated_behavior = monitor.update(wave_state(4))

        self.assertEqual(no_behavior.tier_label, "CLEAR")
        self.assertEqual(no_behavior.color, CLEAR_COLOR)
        self.assertEqual(repeated_behavior.score, 2.8)
        self.assertEqual(repeated_behavior.tier_label, "REVIEW")
        self.assertEqual(repeated_behavior.color, REVIEW_COLOR)

    def test_neutral_happy_and_surprise_do_not_increase_multiplier(self) -> None:
        for label in ("Neutral", "Happiness", "Surprise"):
            monitor = ReviewLevelMonitor(concern_smoothing_alpha=1.0)
            monitor.observe_expression(label, 0.90)

            state = monitor.update(wave_state(4))

            self.assertEqual(state.expression_multiplier, 1.0)
            self.assertEqual(state.tier_label, "MONITOR")

    def test_modifier_clears_immediately_when_face_is_not_visible(self) -> None:
        monitor = ReviewLevelMonitor(concern_smoothing_alpha=1.0)
        monitor.observe_expression("Anger", 0.90)
        monitor.observe_expression(None, None)

        state = monitor.update(wave_state(4))

        self.assertEqual(state.expression_multiplier, 1.0)
        self.assertIsNone(state.concern_label)
        self.assertEqual(state.tier_label, "MONITOR")


if __name__ == "__main__":
    unittest.main()
