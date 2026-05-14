import cv2
from ultralytics import YOLO

# Import the pre-trained model
model = YOLO("../data/yolov8n.pt")

cap = cv2.VideoCapture(0)   # 0 = default webcam; try 1, 2, ... for others
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Run inference on this single frame & add a track id
    results = model.track(frame, conf=0.75, iou=0.45, verbose=False, classes=[0], persist=True, tracker="bytetrack.yaml")

    # results[0].plot() returns a numpy array (BGR) with boxes + labels drawn
    annotated = results[0].plot()

    cv2.imshow("YOLOv8n Webcam", annotated)
    
    # Finish the program if the window is focused and 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()