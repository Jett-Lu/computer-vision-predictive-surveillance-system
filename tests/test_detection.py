from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from detection import (
    PersonRuntime,
    _apply_demo_review_override,
    _discard_stale_tracks,
    _face_centre_inside_box,
    _parse_demo_high_review_names,
    _track_key,
)
from review import CLEAR_COLOR, HIGH_COLOR, ReviewState


class DetectionAssociationTest(unittest.TestCase):
    def test_tracker_id_is_used_as_person_state_key(self) -> None:
        self.assertEqual(_track_key(42, 0), 42)
        self.assertEqual(_track_key(None, 0), -1)

    def test_stale_person_state_and_identity_name_are_removed(self) -> None:
        person_states = {
            3: PersonRuntime(None, None, last_seen_frame=0),
            4: PersonRuntime(None, None, last_seen_frame=100),
        }
        names = {3: "Old", 4: "Current"}

        _discard_stale_tracks(person_states, names, frame_count=100)

        self.assertNotIn(3, person_states)
        self.assertNotIn(3, names)
        self.assertIn(4, person_states)
        self.assertIn(4, names)

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


if __name__ == "__main__":
    unittest.main()
