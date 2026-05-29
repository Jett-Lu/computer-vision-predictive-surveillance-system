from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enrollment import MenuState, handle_delete_state, sanitize_enrollment_label


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


class DeleteInputTest(unittest.TestCase):
    def test_delete_selection_rejects_index_equal_to_length(self) -> None:
        session = DeleteSessionStub(["one"])

        with patch("enrollment.enrollment_folders", return_value=["one"]):
            state, error = handle_delete_state(MenuState.DELETE_CHOOSE, "1", session)

        self.assertEqual(state, MenuState.DELETE_CHOOSE)
        self.assertIn("out of range", error)


if __name__ == "__main__":
    unittest.main()
