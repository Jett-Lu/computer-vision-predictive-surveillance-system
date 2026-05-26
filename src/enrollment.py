# Enrollment
import os
import shutil
import cv2
import datetime
from enum import Enum, auto
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENROLLMENTS_DIR = PROJECT_ROOT / "enrollments"

# States for the state machine
class States(Enum):
    MENU = auto()
    HELP = auto()
    ENROL_START = auto()
    ENROL_GET_NAME = auto()
    ENROL_DUPLICATE_PEOPLE = auto()
    ENROL_GET_NUM_OF_PICS = auto()
    ENROL_TAKE_PICS = auto()
    ENROL_COMPLETE = auto()
    ENROL_ABORT = auto()
    DELETE_START = auto()
    DELETE_CHOOSE_ENROLLED = auto()
    DELETE_CONFIRM = auto()
    DELETE_COMPLETE = auto()
    DELETE_ABORT = auto()
    DETECTION_START = auto()  # NEW

ENROL_STATES = {
    States.ENROL_START,
    States.ENROL_GET_NAME,
    States.ENROL_DUPLICATE_PEOPLE,
    States.ENROL_GET_NUM_OF_PICS,
    States.ENROL_TAKE_PICS,
    States.ENROL_COMPLETE,
    States.ENROL_ABORT,
}

DELETE_STATES = {
    States.DELETE_START,
    States.DELETE_CHOOSE_ENROLLED,
    States.DELETE_CONFIRM,
    States.DELETE_COMPLETE,
    States.DELETE_ABORT
}

RED_TEXT = '\033[91m'
RESET_TEXT = '\033[0m'

PER_PAGES = 5

def displayText(state, enrol_list=None, page_num=0):
    enrol_list = enrol_list or []
    # Clear the terminal
    os.system('cls' if os.name == 'nt' else 'clear')
    if state == States.MENU:
        print("""
 █████  ███    ██  ██████  ███    ███  █████  ██      ██    ██     ██████  ███████ ████████ ███████  ██████ ████████ ██  ██████  ███    ██ 
██   ██ ████   ██ ██    ██ ████  ████ ██   ██ ██       ██  ██      ██   ██ ██         ██    ██      ██         ██    ██ ██    ██ ████   ██ 
███████ ██ ██  ██ ██    ██ ██ ████ ██ ███████ ██        ████       ██   ██ █████      ██    █████   ██         ██    ██ ██    ██ ██ ██  ██ 
██   ██ ██  ██ ██ ██    ██ ██  ██  ██ ██   ██ ██         ██        ██   ██ ██         ██    ██      ██         ██    ██ ██    ██ ██  ██ ██ 
██   ██ ██   ████  ██████  ██      ██ ██   ██ ███████    ██        ██████  ███████    ██    ███████  ██████    ██    ██  ██████  ██   ████
              """)
        print("Enter 'help' to view available commands.")
    elif state == States.HELP:
        print("————— List of Commands —————")
        print("[q] — Quit the program")
        print("[help] — Access the help screen")
        print("[menu] — Go back to the main menu")
        print("[enrol] — Enrol a person into the database")
        print("[delete] — Delete a currently enrolled person")
        print("[detect] — Start live face recognition")  # NEW LINE
    elif state == States.ENROL_GET_NAME:
        print("————— Enter a name for the enrolled person —————")
        print("Enter 'exit' to abort enrolling")
    elif state == States.ENROL_DUPLICATE_PEOPLE:
        print("That person already exists. Do you want to overwrite or use pre-existing and add more photos?")
        print("[a] — Overwrite")
        print("[b] — Use pre-existing")
        print("Enter 'exit' to abort enrolling")
    elif state == States.ENROL_GET_NUM_OF_PICS:
        print("————— Enter the number of pictures you want to take —————")
    elif state == States.ENROL_TAKE_PICS:
        print("Opening Camera...")
    elif state == States.ENROL_COMPLETE:
        print("————— Enrollment is complete! Enter 'exit' to return to the main menu —————")
    elif state == States.ENROL_ABORT:
        print("————— Enrolment aborted! —————")
        print("Do you want to resume or exit?")
        print("[a] — Resume")
        print("[b] — Exit")
    elif state == States.DELETE_CHOOSE_ENROLLED:
        print("————— Choose an Enrolment to delete —————")
        print("Enter 'exit' to return to main menu")
        if len(enrol_list) > 0:
            # Pagination logic
            result_per_page = PER_PAGES
            offset = page_num * result_per_page
            sub_list = []
            if (len(enrol_list) - offset) > 5:
                sub_list = enrol_list[offset:offset+5]
            else:
                sub_list = enrol_list[offset:]

            for i in range(len(sub_list)):
                print(f"[{i + offset}] — {sub_list[i]}")
            print("————— [n/p] — next/prev —————")
    elif state == States.DELETE_CONFIRM:
        print("————— Are you sure? —————")
        print("[y/n] — yes/no")
    elif state == States.DELETE_COMPLETE:
        print("————— Successfully Deleted! —————")
        print("Enter 'exit' to return to main menu")
    elif state == States.DELETE_ABORT:
        print("————— Delete aborted! —————")
        print("Enter 'exit' to return to main menu")
    else:
        print("No enrolled person. Enrol someone first!")
        print("Enter 'exit' to return to main menu")


def getUserResponse(errorMsg):
    if errorMsg:
        print(f"{RED_TEXT}{errorMsg}{RESET_TEXT}")
    user_input = input('>> ').lower()
    user_input = user_input.replace(" ", "")
    return user_input

def main():
    # Initialize to begin at the MENU
    curr_state = States.MENU
    error = ""

    # For Enrollments
    path: Path | None = None
    enrolled_label = ""
    num_of_pics = 0
    saved_count = 0

    # For Deletion
    enrol_list = []
    page_num = 0
    chosen_enrolled = ""

    while True:

        # State Entry Actions
        if curr_state == States.ENROL_START:
            curr_state = States.ENROL_GET_NAME
            continue
        elif curr_state == States.ENROL_TAKE_PICS:
            count = saved_count
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                error = "Could not open webcam. Exited to main menu"
                curr_state = States.MENU
                continue

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame... Exiting")
                    break
                
                key = cv2.waitKey(1) & 0xFF

                if key == ord('s') and count < num_of_pics:
                    timestamp = datetime.datetime.now().strftime("Y%YM%mD%d_H%HM%MS%S")
                    img_name = f"{enrolled_label}_{count}_{timestamp}.jpg"
                    full_path = path / img_name
                    if cv2.imwrite(str(full_path), frame):
                        count += 1
                        if count >= num_of_pics:
                            curr_state = States.ENROL_COMPLETE
                            saved_count = 0
                            break
                    else:
                        print("Failed to write to disk")

                cv2.putText(frame, f"Number of Images: {count}. Press 'q' to quit early", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
                cv2.imshow("Take Photo", frame)

                if key == ord('q') and count < num_of_pics:
                    curr_state = States.ENROL_ABORT
                    saved_count = count
                    break

            cap.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            continue
        elif curr_state == States.DELETE_START:
            curr_state = States.DELETE_CHOOSE_ENROLLED
            continue
        elif curr_state == States.DETECTION_START:  # NEW BLOCK
            # Imported locally rather than at the top of the file so that loading
            # YOLO and face_recognition only happens when the user actually runs detection.
            # Keeps the menu's startup fast.
            from detection import run_detection
            run_detection()
            curr_state = States.MENU
            continue

        displayText(curr_state, enrol_list=enrol_list, page_num=page_num)
        user_res = getUserResponse(errorMsg=error)

        error = ""

        # Handle Enrolling states
        if curr_state in ENROL_STATES:
            if user_res == 'exit':
                curr_state = States.MENU
                continue
            
            # Handle States
            if curr_state == States.ENROL_GET_NAME:
                if user_res == 'q':
                    error = "Enter a name for the person you want to enrol."
                else:
                    enrolled_label = user_res
                    path = ENROLLMENTS_DIR / enrolled_label
                    if not os.path.exists(path):
                        os.makedirs(path)
                        curr_state = States.ENROL_GET_NUM_OF_PICS
                    else:
                        curr_state = States.ENROL_DUPLICATE_PEOPLE
            elif curr_state == States.ENROL_DUPLICATE_PEOPLE:
                if user_res == 'a':
                    try:
                        # Delete the directory
                        shutil.rmtree(path)

                        # Re-create it
                        os.makedirs(path, exist_ok=True)
                    except OSError:
                        error = "Failed to delete path... Returning to main menu"
                        curr_state = States.MENU
                        continue

                elif user_res != 'b':
                    error = "Invalid Command. Enter the correct option as listed before"
                    continue
                curr_state = States.ENROL_GET_NUM_OF_PICS
            elif curr_state == States.ENROL_GET_NUM_OF_PICS:
                try:
                    num_of_pics = int(user_res)

                    if num_of_pics <= 0:
                        error = "Enter a value greater than 0"
                        continue

                    curr_state = States.ENROL_TAKE_PICS
                except ValueError:
                    error = "Invalid input. Please enter an integer number"
            elif curr_state == States.ENROL_ABORT:
                if user_res == 'a':
                    curr_state = States.ENROL_TAKE_PICS
                else:
                    curr_state = States.MENU
            continue
        
        if curr_state in DELETE_STATES:
            ENROLLMENTS_DIR.mkdir(exist_ok=True)
            enrol_list = os.listdir(ENROLLMENTS_DIR)
            if user_res == 'q':
                error = "Enter 'exit' instead"
            elif user_res == 'exit':
                curr_state = States.MENU
                continue
            elif curr_state == States.DELETE_CHOOSE_ENROLLED:
                max_num_pages = math.ceil(len(enrol_list) / PER_PAGES) - 1
                if user_res == 'n':
                    if page_num >= 0 and page_num < max_num_pages:
                        page_num += 1
                elif user_res == 'p':
                    if page_num > 0 and page_num <= max_num_pages:
                        page_num -= 1
                else:
                    try:
                        index = int(user_res);
                        if index >= 0 and index <= len(enrol_list):
                            chosen_enrolled = enrol_list[index]
                            curr_state = States.DELETE_CONFIRM
                        else:
                            error = "Input is out of range. Choose from the provided list"
                    except ValueError:
                        error = "Input isn't an integer value."
            elif curr_state == States.DELETE_CONFIRM:
                if user_res == 'y':
                    chosen_path = ENROLLMENTS_DIR / chosen_enrolled
                    if not chosen_path.exists():
                        error = "Error: path to enrolled person doesn't exist"
                        curr_state = States.DELETE_ABORT
                    else:
                        shutil.rmtree(chosen_path)
                        curr_state = States.DELETE_COMPLETE
                elif user_res == 'n':
                    curr_state = States.DELETE_ABORT
                else:
                    error = "Invalid command. Enter either 'n' or 'y'"            
            continue

        # Handle user input
        if user_res == 'q':
            # Only exit the program if we are not in enrol/delete states
            if curr_state == States.MENU or curr_state == States.HELP:
                break
            else:
                continue
        elif user_res == 'help':
            curr_state = States.HELP
        elif user_res == 'menu':
            curr_state = States.MENU
        elif user_res == 'enrol':
            curr_state = States.ENROL_START

            # Reset all enrolment related variables
            path = None
            enrolled_label = ""
            saved_count = 0
            num_of_pics = 0
        elif user_res == 'delete':
            curr_state = States.DELETE_START
            ENROLLMENTS_DIR.mkdir(exist_ok=True)
            enrol_list = os.listdir(ENROLLMENTS_DIR)
            page_num = 0
        elif user_res == 'detect':  # NEW BLOCK
            curr_state = States.DETECTION_START
        else:
            error = "Invalid Command. Go to 'help' to see list of commands"
            continue

if __name__ == "__main__":
    main()
