import cv2
from ultralytics import YOLO

model = YOLO("../data/yolov8n.pt")

cap = cv2.VideoCapture(0)   # 0 = default webcam; try 1, 2, ... for others
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Run inference on this single frame
    results = model(frame, conf=0.25, iou=0.45, verbose=False, classes=[0])

    # results[0].plot() returns a numpy array (BGR) with boxes + labels drawn
    annotated = results[0].plot()

    # Example: print detected class names for this frame
    names = results[0].names
    for cls_id in results[0].boxes.cls.tolist():
        print(names[int(cls_id)])

    cv2.imshow("YOLOv8n Webcam", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()