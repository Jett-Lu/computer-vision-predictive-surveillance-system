from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from events import EventRecorder, TrackSnapshot


def snapshot(tier: str = "CLEAR", waves: int = 0) -> TrackSnapshot:
    return TrackSnapshot(
        track_key=7,
        track_id=7,
        identity_name="Person",
        identity_score=None,
        identity_confirmed=False,
        tier_label=tier,
        review_score=float(waves),
        wave_count=waves,
        expression_label=None,
        expression_confidence=None,
        expression_context_strength=0.0,
    )


class EventRecorderTest(unittest.TestCase):
    def test_records_track_start_wave_and_tier_change(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "events.jsonl"
            recorder = EventRecorder(path)
            recorder.record(snapshot(), 0.0, 0)
            recorder.record(snapshot("MONITOR", 3), 1.0, 1)
            recorder.close()

            events = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual(
            [event["event_type"] for event in events],
            ["track_started", "tier_changed", "wave_counted"],
        )


if __name__ == "__main__":
    unittest.main()
