from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
import datetime
import math
import os
import shutil

import cv2

from camera import open_capture, prompt_camera_source


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENROLLMENTS_DIR = PROJECT_ROOT / "enrollments"
PAGE_SIZE = 5
INVALID_LABEL_CHARS = set('<>:"/\\|?*')
RED_TEXT = "\033[91m"
RESET_TEXT = "\033[0m"


class MenuState(Enum):
    MENU = auto()
    HELP = auto()
    ENROLL_GET_NAME = auto()
    ENROLL_DUPLICATE = auto()
    ENROLL_GET_COUNT = auto()
    ENROLL_CAPTURE = auto()
    ENROLL_COMPLETE = auto()
    ENROLL_ABORT = auto()
    DELETE_CHOOSE = auto()
    DELETE_CONFIRM = auto()
    DELETE_COMPLETE = auto()
    DELETE_ABORT = auto()
    DETECT = auto()


ENROLLMENT_STATES = {
    MenuState.ENROLL_GET_NAME,
    MenuState.ENROLL_DUPLICATE,
    MenuState.ENROLL_GET_COUNT,
    MenuState.ENROLL_CAPTURE,
    MenuState.ENROLL_COMPLETE,
    MenuState.ENROLL_ABORT,
}

DELETE_STATES = {
    MenuState.DELETE_CHOOSE,
    MenuState.DELETE_CONFIRM,
    MenuState.DELETE_COMPLETE,
    MenuState.DELETE_ABORT,
}


@dataclass
class EnrollmentSession:
    label: str = ""
    folder: Path | None = None
    target_image_count: int = 0
    saved_count: int = 0

    def reset(self) -> None:
        self.label = ""
        self.folder = None
        self.target_image_count = 0
        self.saved_count = 0


@dataclass
class DeleteSession:
    names: list[str]
    page_number: int = 0
    selected_name: str = ""


def enrollment_folders() -> list[str]:
    ENROLLMENTS_DIR.mkdir(exist_ok=True)
    return sorted(path.name for path in ENROLLMENTS_DIR.iterdir() if path.is_dir())


def sanitize_enrollment_label(value: str) -> str:
    label = "".join(char for char in value.strip() if char not in INVALID_LABEL_CHARS)
    return label.strip(". ")


def clear_terminal() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def prompt_user(error_message: str = "") -> str:
    if error_message:
        print(f"{RED_TEXT}{error_message}{RESET_TEXT}")
    return input(">> ").strip().lower().replace(" ", "")


def display_state(state: MenuState, delete_session: DeleteSession) -> None:
    clear_terminal()
    if state == MenuState.MENU:
        print("Integrated Live Demo")
        print("Type 'detect' to start live monitoring.")
        print("Commands: detect | enroll | delete | help | q")
    elif state == MenuState.HELP:
        print("----- List of Commands -----")
        print("[q] - Quit the program")
        print("[help] - Access the help screen")
        print("[menu] - Go back to the main menu")
        print("[enroll] - Enroll a person into the database")
        print("[delete] - Delete a currently enrolled person")
        print("[detect] - Start live monitoring")
    elif state == MenuState.ENROLL_GET_NAME:
        print("----- Enter a name for the enrolled person -----")
        print("Enter 'exit' to abort enrolling")
    elif state == MenuState.ENROLL_DUPLICATE:
        print("That person already exists. Overwrite or add more photos?")
        print("[a] - Overwrite")
        print("[b] - Add more photos")
        print("Enter 'exit' to abort enrolling")
    elif state == MenuState.ENROLL_GET_COUNT:
        print("----- Enter the number of pictures you want to take -----")
    elif state == MenuState.ENROLL_CAPTURE:
        print("Opening camera...")
    elif state == MenuState.ENROLL_COMPLETE:
        print("----- Enrollment is complete. Enter 'exit' to return to the main menu -----")
    elif state == MenuState.ENROLL_ABORT:
        print("----- Enrollment aborted -----")
        print("Do you want to resume or exit?")
        print("[a] - Resume")
        print("[b] - Exit")
    elif state == MenuState.DELETE_CHOOSE:
        print("----- Choose an enrollment to delete -----")
        print("Enter 'exit' to return to main menu")
        display_delete_page(delete_session)
    elif state == MenuState.DELETE_CONFIRM:
        print(f"Delete '{delete_session.selected_name}'?")
        print("[y/n] - yes/no")
    elif state == MenuState.DELETE_COMPLETE:
        print("----- Successfully deleted -----")
        print("Enter 'exit' to return to main menu")
    elif state == MenuState.DELETE_ABORT:
        print("----- Delete aborted -----")
        print("Enter 'exit' to return to main menu")


def display_delete_page(delete_session: DeleteSession) -> None:
    if not delete_session.names:
        print("No enrolled person. Enroll someone first.")
        return

    offset = delete_session.page_number * PAGE_SIZE
    for index, name in enumerate(delete_session.names[offset : offset + PAGE_SIZE], start=offset):
        print(f"[{index}] - {name}")

    if len(delete_session.names) > PAGE_SIZE:
        print("----- [n/p] - next/prev -----")


def capture_enrollment_images(session: EnrollmentSession) -> MenuState:
    if session.folder is None:
        raise RuntimeError("Enrollment folder has not been selected.")

    camera_source = prompt_camera_source()
    capture = open_capture(camera_source)
    if not capture.isOpened():
        print(f"Could not open camera source {camera_source}.")
        input("Press Enter to continue...")
        return MenuState.MENU

    count = session.saved_count
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Failed to grab frame. Returning to menu.")
                input("Press Enter to continue...")
                return MenuState.MENU

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s") and count < session.target_image_count:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = session.folder / f"{session.label}_{count}_{timestamp}.jpg"
                if cv2.imwrite(str(image_path), frame):
                    count += 1
                    if count >= session.target_image_count:
                        session.saved_count = 0
                        return MenuState.ENROLL_COMPLETE
                else:
                    print("Failed to write image to disk.")

            cv2.putText(
                frame,
                f"Images: {count}/{session.target_image_count}. Press 's' to save, 'q' to stop",
                (10, 24),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 0, 0),
                1,
            )
            cv2.imshow("Enrollment Capture", frame)

            if key == ord("q") and count < session.target_image_count:
                session.saved_count = count
                return MenuState.ENROLL_ABORT
    finally:
        capture.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def handle_enrollment_state(
    state: MenuState,
    command: str,
    session: EnrollmentSession,
) -> tuple[MenuState, str]:
    if command == "exit":
        return MenuState.MENU, ""

    if state == MenuState.ENROLL_GET_NAME:
        if command == "q":
            return state, "Enter a name for the person you want to enroll."

        label = sanitize_enrollment_label(command)
        if not label:
            return state, "Enter a valid name using letters or numbers."

        session.label = label
        session.folder = ENROLLMENTS_DIR / label
        if not session.folder.exists():
            session.folder.mkdir(parents=True)
            return MenuState.ENROLL_GET_COUNT, ""
        return MenuState.ENROLL_DUPLICATE, ""

    if state == MenuState.ENROLL_DUPLICATE:
        if session.folder is None:
            return MenuState.MENU, "Enrollment folder is missing. Returning to menu."

        if command == "a":
            shutil.rmtree(session.folder)
            session.folder.mkdir(parents=True, exist_ok=True)
        elif command != "b":
            return state, "Invalid command. Choose 'a' or 'b'."
        return MenuState.ENROLL_GET_COUNT, ""

    if state == MenuState.ENROLL_GET_COUNT:
        try:
            target_count = int(command)
        except ValueError:
            return state, "Invalid input. Please enter an integer number."

        if target_count <= 0:
            return state, "Enter a value greater than 0."

        session.target_image_count = target_count
        return MenuState.ENROLL_CAPTURE, ""

    if state == MenuState.ENROLL_ABORT:
        return (MenuState.ENROLL_CAPTURE if command == "a" else MenuState.MENU), ""

    return MenuState.MENU, ""


def handle_delete_state(
    state: MenuState,
    command: str,
    delete_session: DeleteSession,
) -> tuple[MenuState, str]:
    delete_session.names = enrollment_folders()
    if command == "q":
        return state, "Enter 'exit' instead."
    if command == "exit":
        return MenuState.MENU, ""

    if state == MenuState.DELETE_CHOOSE:
        max_page = max(0, math.ceil(len(delete_session.names) / PAGE_SIZE) - 1)
        if command == "n":
            delete_session.page_number = min(max_page, delete_session.page_number + 1)
            return state, ""
        if command == "p":
            delete_session.page_number = max(0, delete_session.page_number - 1)
            return state, ""

        try:
            index = int(command)
        except ValueError:
            return state, "Input is not an integer value."

        if 0 <= index < len(delete_session.names):
            delete_session.selected_name = delete_session.names[index]
            return MenuState.DELETE_CONFIRM, ""
        return state, "Input is out of range. Choose from the provided list."

    if state == MenuState.DELETE_CONFIRM:
        if command == "y":
            selected_path = ENROLLMENTS_DIR / delete_session.selected_name
            if not selected_path.exists():
                return MenuState.DELETE_ABORT, "Enrollment folder no longer exists."
            shutil.rmtree(selected_path)
            return MenuState.DELETE_COMPLETE, ""
        if command == "n":
            return MenuState.DELETE_ABORT, ""
        return state, "Invalid command. Enter either 'n' or 'y'."

    return MenuState.MENU, ""


def main() -> None:
    state = MenuState.MENU
    error = ""
    enrollment_session = EnrollmentSession()
    delete_session = DeleteSession(names=[])

    while True:
        if state == MenuState.ENROLL_CAPTURE:
            state = capture_enrollment_images(enrollment_session)
            continue

        if state == MenuState.DETECT:
            from detection import run_detection

            run_detection(source=prompt_camera_source())
            state = MenuState.MENU
            continue

        display_state(state, delete_session)
        command = prompt_user(error)
        error = ""

        if state in ENROLLMENT_STATES:
            state, error = handle_enrollment_state(state, command, enrollment_session)
            continue

        if state in DELETE_STATES:
            state, error = handle_delete_state(state, command, delete_session)
            continue

        if command == "q" and state in {MenuState.MENU, MenuState.HELP}:
            break
        if command == "help":
            state = MenuState.HELP
        elif command == "menu":
            state = MenuState.MENU
        elif command in {"enroll", "enrol"}:
            enrollment_session.reset()
            state = MenuState.ENROLL_GET_NAME
        elif command == "delete":
            delete_session = DeleteSession(names=enrollment_folders())
            state = MenuState.DELETE_CHOOSE
        elif command == "detect":
            state = MenuState.DETECT
        else:
            error = "Invalid command. Go to 'help' to see the list of commands."


if __name__ == "__main__":
    main()
