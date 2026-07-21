from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enrollment import (
    EnrollmentSession,
    MenuState,
    handle_delete_state,
    handle_enrollment_state,
    sanitize_enrollment_label,
    validate_enrollment_frame,
    _remove_enrollment_folder,
)
from identity import DetectedFace

import numpy as np


class DeleteSessionStub:
    def __init__(self, names: list[str]) -> None:
        self.names = names
        self.page_number = 0
        self.selected_name = ""


class EnrollmentInputTest(unittest.TestCase):
    def test_sanitize_enrollment_label_removes_path_characters(self) -> None:
        self.assertEqual(sanitize_enrollment_label("../Jane:Doe?"), "JaneDoe")

    def test_sanitize_enrollment_label_rejects_empty_or_dot_only_names(self) -> None:
        self.assertEqual(sanitize_enrollment_label("..."), "")

    def test_sanitize_enrollment_label_preserves_spaces(self) -> None:
        self.assertEqual(sanitize_enrollment_label("Taylor Brooks"), "Taylor Brooks")

    def test_sanitize_enrollment_label_rejects_reserved_windows_names(self) -> None:
        self.assertEqual(sanitize_enrollment_label("CON.txt"), "")

    def test_enrollment_state_preserves_the_display_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = EnrollmentSession()

            state, error = handle_enrollment_state(
                MenuState.ENROLL_GET_NAME,
                "Taylor Brooks",
                session,
                Path(temp_dir),
            )

        self.assertEqual(state, MenuState.ENROLL_GET_COUNT)
        self.assertEqual(error, "")
        self.assertEqual(session.label, "Taylor Brooks")


class DeleteInputTest(unittest.TestCase):
    def test_delete_selection_rejects_index_equal_to_length(self) -> None:
        session = DeleteSessionStub(["one"])

        with patch("enrollment.enrollment_folders", return_value=["one"]):
            state, error = handle_delete_state(MenuState.DELETE_CHOOSE, "1", session)

        self.assertEqual(state, MenuState.DELETE_CHOOSE)
        self.assertIn("out of range", error)

    def test_delete_rejects_a_folder_outside_the_enrollment_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "enrollments"
            outside = Path(temp_dir) / "outside"
            root.mkdir()
            outside.mkdir()

            with self.assertRaises(ValueError):
                _remove_enrollment_folder(outside, root)

            self.assertTrue(outside.exists())


class EnrollmentQualityIdentifierStub:
    def __init__(self, faces, usable=True):
        self.faces = faces
        self.usable = usable

    def detect_faces(self, frame):
        return self.faces

    def is_face_usable(self, face):
        return self.usable


class EnrollmentQualityTest(unittest.TestCase):
    def test_rejects_multiple_faces(self) -> None:
        face = DetectedFace(np.zeros(15), (0, 0, 100, 100), 0.95)
        accepted, message = validate_enrollment_frame(
            EnrollmentQualityIdentifierStub([face, face]),
            np.zeros((100, 100, 3), dtype=np.uint8),
        )

        self.assertFalse(accepted)
        self.assertIn("More than one", message)

    def test_accepts_one_usable_face(self) -> None:
        face = DetectedFace(np.zeros(15), (0, 0, 100, 100), 0.95)
        accepted, _ = validate_enrollment_frame(
            EnrollmentQualityIdentifierStub([face]),
            np.zeros((100, 100, 3), dtype=np.uint8),
        )

        self.assertTrue(accepted)


if __name__ == "__main__":
    unittest.main()
