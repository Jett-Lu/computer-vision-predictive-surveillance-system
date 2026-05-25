from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gesture import GREEN, WaveAlertState
from review import ReviewLevelMonitor


def wave_state(count: int) -> WaveAlertState:
    return WaveAlertState(
        recent_wave_count=count,
        tier_label="CLEAR",
        color=GREEN,
        wave_detected=False,
    )


def count_sustained_concern(
    monitor: ReviewLevelMonitor,
    start_time: float,
    label: str = "Fear",
) -> None:
    monitor.observe_expression(label, 0.90, start_time)
    monitor.observe_expression(label, 0.90, start_time + 1.6)


class ReviewLevelMonitorTest(unittest.TestCase):
    def test_single_concerning_expression_period_does_not_raise_tier(self) -> None:
        monitor = ReviewLevelMonitor()
        count_sustained_concern(monitor, 0.0)

        state = monitor.update(wave_state(0), 2.0)

        self.assertEqual(state.recent_expression_event_count, 1)
        self.assertEqual(state.expression_points, 0)
        self.assertEqual(state.tier_label, "CLEAR")
        self.assertEqual(state.color, GREEN)

    def test_repeated_sustained_concern_can_raise_monitor_but_not_high_alone(self) -> None:
        monitor = ReviewLevelMonitor()
        for start_time in (0.0, 5.0, 10.0, 15.0):
            count_sustained_concern(monitor, start_time)

        state = monitor.update(wave_state(0), 17.0)

        self.assertEqual(state.expression_points, 2)
        self.assertEqual(state.tier_label, "MONITOR")

    def test_neutral_happy_and_surprise_do_not_add_expression_events(self) -> None:
        monitor = ReviewLevelMonitor()
        for index, label in enumerate(("Neutral", "Happiness", "Surprise")):
            count_sustained_concern(monitor, index * 5.0, label)

        state = monitor.update(wave_state(0), 16.0)

        self.assertEqual(state.recent_expression_event_count, 0)
        self.assertEqual(state.tier_label, "CLEAR")

    def test_expression_cues_can_increase_existing_wave_review_level(self) -> None:
        monitor = ReviewLevelMonitor()
        count_sustained_concern(monitor, 0.0, "Sadness")
        count_sustained_concern(monitor, 5.0, "Fear")
        count_sustained_concern(monitor, 10.0, "Anger")

        state = monitor.update(wave_state(5), 12.0)

        self.assertEqual(state.expression_points, 2)
        self.assertEqual(state.score, 5)
        self.assertEqual(state.tier_label, "HIGH")
        self.assertEqual(state.color, (0, 0, 255))

    def test_expression_events_expire_and_review_level_recovers(self) -> None:
        monitor = ReviewLevelMonitor(expression_window_seconds=5.0)
        count_sustained_concern(monitor, 0.0)
        count_sustained_concern(monitor, 5.0)

        state = monitor.update(wave_state(0), 12.0)

        self.assertEqual(state.recent_expression_event_count, 0)
        self.assertEqual(state.tier_label, "CLEAR")


if __name__ == "__main__":
    unittest.main()
