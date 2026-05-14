import urllib.request
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

# Import the pre-trained YOLO model
model = YOLO("../data/yolov8n.pt")

# Download the MediaPipe Face Detector model file (one-time, cached locally)
MODEL_PATH = "../data/blaze_face_short_range.tflite"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"

if not os.path.exists(MODEL_PATH):
    print(f"Downloading MediaPipe face detector model to {MODEL_PATH}...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")

# Create the MediaPipe face detector ONCE outside the loop (expensive to instantiate)
# blaze_face_short_range is optimized for faces within ~2m (good for person crops)
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
face_detector_options = vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.5,
)
face_detector = vision.FaceDetector.create_from_options(face_detector_options)

cap = cv2.VideoCapture(0)   # 0 = default webcam; try 1, 2, ... for others
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Run inference on this single frame & add a track id
    results = model.track(frame, conf=0.60, iou=0.45, verbose=False, classes=[0], persist=True, tracker="bytetrack.yaml")

    # results[0].plot() returns a numpy array (BGR) with boxes + labels drawn
    annotated = results[0].plot()

    # Crop out the bounding box(es)
    frame_height, frame_width = frame.shape[:2]
    boxes = results[0].boxes

    # Track IDs may be None on early frames before tracking is established
    track_ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)

    person_crops = []  # list of (track_id, crop, offset) tuples for downstream stages
    for box, track_id in zip(boxes.xyxy, track_ids):
        # Convert tensor coords to ints
        x1, y1, x2, y2 = box.int().tolist()

        # Clamp to frame bounds (YOLO can return coords slightly outside the frame)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_width, x2)
        y2 = min(frame_height, y2)

        # Skip degenerate (zero-area) boxes
        if x2 <= x1 or y2 <= y1:
            continue

        # Crop from the ORIGINAL frame, not the annotated one
        crop = frame[y1:y2, x1:x2].copy()
        person_crops.append((track_id, crop, (x1, y1)))

    # Run face detection on each person crop
    face_results = {}  # {track_id: (face_box_in_crop, keypoints_in_crop)}

    for track_id, crop, (offset_x, offset_y) in person_crops:
        crop_h, crop_w = crop.shape[:2]

        # MediaPipe expects RGB; OpenCV gives BGR
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # Wrap in a MediaPipe Image object (required by Tasks API)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)

        # Run detection
        detection_result = face_detector.detect(mp_image)

        # No faces detected — normal outcome, just skip
        if not detection_result.detections:
            continue

        # If multiple faces detected, pick the largest by box area
        def face_area(detection):
            bbox = detection.bounding_box
            return bbox.width * bbox.height

        best_detection = max(detection_result.detections, key=face_area)
        bbox = best_detection.bounding_box

        # Tasks API gives box coords in PIXELS (relative to the input image, i.e. the crop)
        fx1 = bbox.origin_x
        fy1 = bbox.origin_y
        fx2 = bbox.origin_x + bbox.width
        fy2 = bbox.origin_y + bbox.height

        # Clamp face box to crop bounds (can be slightly OOB)
        fx1 = max(0, fx1)
        fy1 = max(0, fy1)
        fx2 = min(crop_w, fx2)
        fy2 = min(crop_h, fy2)

        if fx2 <= fx1 or fy2 <= fy1:
            continue

        # Keypoints: list of NormalizedKeypoint with .x and .y in 0-1 (normalized to crop dims)
        # Order: right_eye, left_eye, nose_tip, mouth_center, right_ear_tragion, left_ear_tragion
        keypoints = [
            (int(kp.x * crop_w), int(kp.y * crop_h))
            for kp in best_detection.keypoints
        ]

        face_results[track_id] = ((fx1, fy1, fx2, fy2), keypoints)

        # Translate face box back to original-frame coords for drawing
        draw_x1 = fx1 + offset_x
        draw_y1 = fy1 + offset_y
        draw_x2 = fx2 + offset_x
        draw_y2 = fy2 + offset_y

        # Draw the face box on the annotated frame (cyan in BGR)
        cv2.rectangle(annotated, (draw_x1, draw_y1), (draw_x2, draw_y2), (255, 255, 0), 2)

        # Draw the 6 keypoints (yellow in BGR)
        for kx, ky in keypoints:
            cv2.circle(annotated, (kx + offset_x, ky + offset_y), 2, (0, 255, 255), -1)

    cv2.imshow("YOLOv8n Webcam", annotated)

    # Finish the program if the window is focused and 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
face_detector.close()