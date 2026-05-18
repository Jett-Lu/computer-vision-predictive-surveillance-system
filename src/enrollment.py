import os
import shutil
import cv2
import datetime
from enum import Enum, auto

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

ENROL_STATES = {
    States.ENROL_START,
    States.ENROL_GET_NAME,
    States.ENROL_DUPLICATE_PEOPLE,
    States.ENROL_GET_NUM_OF_PICS,
    States.ENROL_TAKE_PICS,
    States.ENROL_COMPLETE,
    States.ENROL_ABORT,
}

RED_TEXT = '\033[91m'
RESET_TEXT = '\033[0m'

def displayText(state):
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
        print("————— Enter 'help' to see options —————")
    elif state == States.HELP:
        print("————— List of Commands —————")
        print("[q] — Quit the program")
        print("[help] — Access the help screen")
        print("[menu] — Go back to the main menu")
        print("[enrol] — Enrol a person into the database")
        print("[delete] — Delete a currently enrolled person")
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
    path = ""
    enrolled_label = ""
    num_of_pics = 0
    saved_count = 0

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
                    full_path = path + img_name
                    if cv2.imwrite(full_path, frame):
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

        displayText(curr_state)
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
                    path = f'../enrollments/{enrolled_label}/'
                    if not os.path.exists(path):
                        os.makedirs(path)
                        curr_state = States.ENROL_GET_NUM_OF_PICS
                    else:
                        curr_state = States.ENROL_DUPLICATE_PEOPLE
            elif curr_state == States.ENROL_DUPLICATE_PEOPLE:
                if user_res == 'a':
                    # Delete the directory
                    try:
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
            path = ""
            enrolled_label = ""
            saved_count = 0
            num_of_pics = 0
        else:
            error = "Invalid Command. Go to 'help' to see list of commands"
            continue

main()