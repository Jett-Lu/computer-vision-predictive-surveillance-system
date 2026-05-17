import os
import cv2

state = ['menu', 'help', 'enrol_get_name', 'enrol_get_num_of_pics', 'enrol_get_pictures', 'enrol_complete']
current_state = state[0]
num_error = False
label = ""
num_of_pics = 0

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
        pass
    elif curr_state == state[3]:
        print("——— Enter the amount of pictures you want to take ———")
        print("Enter 'exit' to exit outside of enrolment")
        if num_error:
            print("Invalid Input. Please enter a valid integer value")
    elif curr_state == state[5]:
        print("——— Enrolment complete! Enter 'exit' to return to main menu ———")
    return

while True:
    display(curr_state=current_state)
    if current_state == state[4]:
        print("Opening camera...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise RuntimeError("Error — Can't open the webcam. Check for available camera.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame, exiting...")
                break

            cv2.imshow("Enrollment", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Exiting...')
                break
        
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
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
        elif current_state == state[2]:
            label = user_input
            current_state = state[3]
        elif current_state == state[3]:
            try:
                num_of_pics = int(user_input)
                current_state = state[4]
                num_error = False
            except ValueError:
                num_error = True

    else:
        print("Invalid command... try entering 'help' to see list of commands.")

