import cv2
from ultralytics import YOLO
import numpy as np
import face_recognition

# Initialize the model
model = YOLO('../data/yolov8n.pt')

# Start screen capturing
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # This is where the YOLO model is used
    results = model.track(frame, conf=0.30, iou=0.45, verbose=False, classes=[0], persist=True, tracker="bytetrack.yaml")
    annotated = results[0].plot()
    
    # Extract the bounding box
    boxes = results[0].boxes

    # Extract the bounding box's coords
    # [x-min, y-min, x-max, y-max]
    for i in range(len(boxes)):
        # Crop out the person to detect the faces
        x1, y1, x2, y2 = boxes.xyxy[i].int().tolist()
        cropped_img = frame[y1:y2, x1:x2]

        """
        # Optional - display the cropped img and coordinates
        cv2.circle(annotated, (x1, y1), radius=2, color=(0,0,255), thickness=2)
        cv2.circle(annotated, (x2, y2), radius=2, color=(0,0,255), thickness=2)
        cv2.imshow('Cropped image', cropped_img)
        """

        # OpenCV uses BGR but face_recognition uses RGB so we need to correct it
        # corrected_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        # face_location = face_recognition.face_locations(corrected_img, number_of_times_to_upsample=1, model='hog')
        # encodings = face_recognition.face_encodings(corrected_img, face_location, num_jitters=1, model='small')

    cv2.imshow('Anomaly Detection', annotated)
    # cv2.imshow('Detected boxes', boxes)

    # Exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()