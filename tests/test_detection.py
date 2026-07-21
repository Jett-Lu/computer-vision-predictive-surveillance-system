from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from detection import (
    MonitoringProcessor,
    PersonRuntime,
    _apply_demo_review_override,
    _discard_stale_tracks,
    _face_centre_inside_box,
    _parse_demo_high_review_names,
    _track_key,
)
from config import DEFAULT_CONFIG
from events import EventRecorder
from identity import IdentityConsensus
from pose import PoseResult
from review import CLEAR_COLOR, HIGH_COLOR, ReviewState


class DetectionAssociationTest(unittest.TestCase):
    def test_tracker_id_is_used_as_person_state_key(self) -> None:
        self.assertEqual(_track_key(42, 0), 42)
        self.assertEqual(_track_key(None, 0, frame_number=0), -10001)
        self.assertNotEqual(
            _track_key(None, 0, frame_number=0),
            _track_key(None, 0, frame_number=1),
        )

    def test_stale_person_state_and_identity_name_are_removed(self) -> None:
        person_states = {
            3: PersonRuntime(None, None, IdentityConsensus(), last_seen_frame=0),
            4: PersonRuntime(None, None, IdentityConsensus(), last_seen_frame=100),
        }

        stale = _discard_stale_tracks(person_states, frame_count=100)

        self.assertNotIn(3, person_states)
        self.assertIn(4, person_states)
        self.assertEqual(stale, [3])

    def test_face_is_associated_when_its_centre_is_inside_person_box(self) -> None:
        self.assertTrue(_face_centre_inside_box((30, 20, 50, 40), (0, 0, 100, 100)))

    def test_face_is_not_associated_when_person_box_is_missing(self) -> None:
        self.assertFalse(_face_centre_inside_box((30, 20, 50, 40), None))

    def test_demo_high_review_names_are_parsed_case_insensitively(self) -> None:
        self.assertEqual(
            _parse_demo_high_review_names("Alex Chen; Priya Shah,  "),
            {"alex chen", "priya shah"},
        )

    def test_demo_high_review_override_only_changes_named_identity(self) -> None:
        state = ReviewState("CLEAR", CLEAR_COLOR, 0.0, 0, 1.0, 0.0, None)

        unchanged = _apply_demo_review_override(state, "Jordan Lee", {"alex chen"})
        overridden = _apply_demo_review_override(state, "Alex Chen", {"alex chen"})

        self.assertIs(unchanged, state)
        self.assertEqual(overridden.tier_label, "HIGH")
        self.assertEqual(overridden.color, HIGH_COLOR)
        self.assertEqual(overridden.recent_wave_count, 5)


class FakePoseAnalyzer:
    def analyze(self, frame):
        return [
            PoseResult(
                track_id=7,
                box=(10, 10, 80, 90),
                landmarks={},
            )
        ]

    def draw_landmarks(self, frame, landmarks, color):
        return None


class MonitoringProcessorIntegrationTest(unittest.TestCase):
    def test_frame_pipeline_runs_with_optional_models_disabled(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            allow_model_downloads=False,
            event_logging_enabled=False,
            expression_interval_frames=1,
        )
        processor = MonitoringProcessor(
            config,
            pose_analyzer=FakePoseAnalyzer(),
            emotion_analyzer=None,
            identity_matcher=None,
            known_identities=[],
            event_recorder=EventRecorder(None),
        )
        try:
            annotated = processor.process_frame(
                np.zeros((100, 100, 3), dtype=np.uint8),
                timestamp=0.0,
            )
        finally:
            processor.close()

        self.assertEqual(annotated.shape, (100, 100, 3))
        self.assertEqual(len(processor.last_snapshots), 1)
        self.assertEqual(processor.last_snapshots[0].track_id, 7)
        self.assertEqual(processor.last_snapshots[0].identity_name, "Person")


if __name__ == "__main__":
    unittest.main()
