import os
import cv2
import datetime

state = ['menu', 'help', 'enrol_get_name', 'enrol_get_num_of_pics', 'enrol_get_pictures', 'enrol_complete', 'enrol_aborted']
current_state = state[0]
num_error = False
label = ""
num_of_pics = 0
enrol_aborted = False
img_count = 0

def display(curr_state):
    os.system('cls' if os.name == 'nt' else 'clear')
    if curr_state == state[0]:
        print("""
     █████  ███    ██  ██████  ███    ███  █████  ██      ██    ██     ██████  ███████ ████████ ███████  ██████ ████████ ██  ██████  ███    ██ 
    ██   ██ ████   ██ ██    ██ ████  ████ ██   ██ ██       ██  ██      ██   ██ ██         ██    ██      ██         ██    ██ ██    ██ ████   ██ 
    ███████ ██ ██  ██ ██    ██ ██ ████ ██ ███████ ██        ████       ██   ██ █████      ██    █████   ██         ██    ██ ██    ██ ██ ██  ██ 
    ██   ██ ██  ██ ██ ██    ██ ██  ██  ██ ██   ██ ██         ██        ██   ██ ██         ██    ██      ██         ██    ██ ██    ██ ██  ██ ██ 
    ██   ██ ██   ████  ██████  ██      ██ ██   ██ ███████    ██        ██████  ███████    ██    ███████  ██████    ██    ██  ██████  ██   ████
    """)
        print("——— Enter 'help' to see options ———")
    elif curr_state == state[1]:
        print("——— List of Commands ———")
        print("[q] — Quit the program")
        print("[help] — Access the help screen")
        print("[menu] — Go back to the main menu")
        print("[enrol] — Enrol a person into the database")
    elif curr_state == state[2]:
        print("——— Enter a name ———")
        print("Enter 'exit' to exit outside of enrolment")
    elif curr_state == state[3]:
        print("——— Enter the amount of pictures you want to take ———")
        print("Enter 'exit' to exit outside of enrolment")
        if num_error:
            print("Invalid Input. Please enter a valid integer value")
    elif curr_state == state[5]:
        print("——— Enrolment complete! Enter 'exit' to return to main menu ———")
    elif curr_state == state[6]:
        print("——— Enrolment was aborted! Enter 'exit' to return to main menu or enter 'again' to re-enrol ———")
    return

while True:
    display(curr_state=current_state)
    if current_state == state[4]:
        print("Opening camera...")
        cap = cv2.VideoCapture(0)

        path = f"../enrollments/{label}/"
        if not os.path.exists(path):
            enrol_aborted = False
            os.makedirs(path)
        else:
            print("Path already exists")

        if not enrol_aborted:
            img_count = 0

        if not cap.isOpened():
            raise RuntimeError("Error — Can't open the webcam. Check for available camera.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame, exiting...")
                break
            
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') and img_count < num_of_pics:
                timestamp = datetime.datetime.now().strftime("Y%YM%mD%d_H%HM%MS%S")
                img_name = f"{label}_{img_count}_{timestamp}.png"
                full_path = path + img_name
                if cv2.imwrite(full_path, frame) == True:
                    img_count += 1
                    if (img_count >= num_of_pics):
                        enrol_aborted = False
                        break
                else:
                    print("Could not write to file")

            cv2.putText(frame, f"Number of Images: {img_count}. Press 'q' to quit early", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
            cv2.imshow("Enrollment", frame)
            if key == ord('q'):
                enrol_aborted = True
                print('Exiting...')
                break
        
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        if enrol_aborted:
            current_state = state[6]
        else:
            current_state = state[5]
        display(curr_state=current_state)

    user_input = input('>> ').lower()

    # Sanitize user input
    user_input = user_input.replace(" ", "")

    if (user_input == 'q' and current_state.find('enrol') == -1):
        os.system('cls' if os.name == 'nt' else 'clear')
        break
    elif (user_input == 'menu'):
        current_state = state[0]
    elif (user_input == 'help'):
        current_state = state[1]
    elif (user_input == 'enrol'):
        current_state = state[2]
    elif (current_state.find('enrol') != -1):
        if (user_input == 'exit'):
            current_state = state[0]
        elif current_state == state[2] and user_input != 'q':
            label = user_input
            current_state = state[3]
        elif current_state == state[3]:
            try:
                num_of_pics = int(user_input)
                current_state = state[4]
                num_error = False
            except ValueError:
                num_error = True
        elif current_state == state[6] and user_input == 'again':
            current_state = state[4]

    else:
        print("Invalid command... try entering 'help' to see list of commands.")

