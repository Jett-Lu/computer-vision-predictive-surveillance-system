import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize the model
model = YOLO('../data/yolov8n.pt')

MODEL_PATH = '../data/blaze_face_short_range.tflite'

# Create the model
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

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
        cropped_img = annotated[y1:y2, x1:x2]

        """
        # Optional - display the cropped img and coordinates
        cv2.circle(annotated, (x1, y1), radius=2, color=(0,0,255), thickness=2)
        cv2.circle(annotated, (x2, y2), radius=2, color=(0,0,255), thickness=2)
        cv2.imshow('Cropped image', cropped_img)
        """

        # OpenCV uses BGR but MediaPipe uses RGB so we need to correct it
        corrected_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

        # Convert np.ndarray to mp.image
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=corrected_img)

        # Detect the faces
        detection_result = detector.detect(img)

        #iterate through all the detections and visualize it
        face_results = detection_result.detections
        # print("Results: ")
        # print(face_results)
        # print("")
        for detection in face_results:
            confidence = round(detection.categories[0].score, 2)

            # Get the bounding box coords
            box_x1 = detection.bounding_box.origin_x + x1
            box_y1 = detection.bounding_box.origin_y + y1
            box_x2 = box_x1 + detection.bounding_box.width
            box_y2 = box_y1 + detection.bounding_box.height

            cv2.rectangle(annotated, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), 1)
            cv2.putText(annotated, f"({confidence})", (box_x1 + 10, box_y1 + 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

            # Get the landmarks
            for landmark in detection.keypoints:
                """
                Denormalize the landmarks. The landmarks given are in the range 0 to 1
                relative to the image's height and width passed into the detection
                """
                px = (landmark.x * (x2 - x1)) + x1
                py = (landmark.y * (y2 - y1)) + y1

                cv2.circle(annotated, (int(px), int(py)), 1, (0, 255, 0), 2)

    cv2.imshow('Anomaly Detection', annotated)
    # cv2.imshow('Detected boxes', boxes)

    # Exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()