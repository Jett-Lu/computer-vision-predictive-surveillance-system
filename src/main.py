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

# Create the MediaPipe face detector ONCE outside the loop
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
face_detector_options = vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.5,
)
face_detector = vision.FaceDetector.create_from_options(face_detector_options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

# How often to run face detection (1 = every frame, 5 = every 5th frame)
RECOGNITION_INTERVAL = 5
frame_count = 0

# Persistent across frames — keyed by track_id, stores face box + keypoints in crop coords
face_results = {}  # {track_id: ((fx1, fy1, fx2, fy2), [(kx, ky), ...])}

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame_height, frame_width = frame.shape[:2]

    # YOLO + ByteTrack run EVERY frame so track IDs stay current
    results = model.track(frame, conf=0.75, iou=0.45, verbose=False, classes=[0], persist=True, tracker="bytetrack.yaml")
    annotated = results[0].plot()

    boxes = results[0].boxes
    track_ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)

    # Build this frame's track_id → person box mapping (cheap, done every frame)
    # Used both for face detection input AND for drawing cached face boxes
    current_person_boxes = {}  # {track_id: (x1, y1, x2, y2)}
    for box, track_id in zip(boxes.xyxy, track_ids):
        if track_id is None:
            continue
        x1, y1, x2, y2 = box.int().tolist()
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_width, x2)
        y2 = min(frame_height, y2)
        if x2 > x1 and y2 > y1:
            current_person_boxes[track_id] = (x1, y1, x2, y2)

    # Cull face_results entries for tracks that no longer exist
    # (person left frame, or ByteTrack dropped the track)
    face_results = {tid: data for tid, data in face_results.items() if tid in current_person_boxes}

    # Run face detection only every Nth frame
    if frame_count % RECOGNITION_INTERVAL == 0:
        for track_id, (x1, y1, x2, y2) in current_person_boxes.items():
            # Crop from the ORIGINAL frame (not annotated)
            crop = frame[y1:y2, x1:x2].copy()
            crop_h, crop_w = crop.shape[:2]

            # MediaPipe expects RGB; OpenCV gives BGR
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)

            detection_result = face_detector.detect(mp_image)

            # No faces detected — leave any prior face_results entry intact
            # (don't delete; the cached box stays useful until a new detection succeeds
            #  or the track disappears)
            if not detection_result.detections:
                continue

            # If multiple faces, pick the largest by area
            best_detection = max(
                detection_result.detections,
                key=lambda d: d.bounding_box.width * d.bounding_box.height,
            )
            bbox = best_detection.bounding_box

            # Tasks API gives box coords in PIXELS relative to the crop
            fx1 = max(0, bbox.origin_x)
            fy1 = max(0, bbox.origin_y)
            fx2 = min(crop_w, bbox.origin_x + bbox.width)
            fy2 = min(crop_h, bbox.origin_y + bbox.height)

            if fx2 <= fx1 or fy2 <= fy1:
                continue

            # Keypoints are normalized 0-1 relative to crop dims
            keypoints = [
                (int(kp.x * crop_w), int(kp.y * crop_h))
                for kp in best_detection.keypoints
            ]

            face_results[track_id] = ((fx1, fy1, fx2, fy2), keypoints)

    # Draw cached face boxes EVERY frame using the CURRENT person-box offset
    # This is what eliminates the flicker between recognition frames
    for track_id, ((fx1, fy1, fx2, fy2), keypoints) in face_results.items():
        px1, py1, _, _ = current_person_boxes[track_id]

        # Translate face box from crop coords to original-frame coords
        draw_x1 = fx1 + px1
        draw_y1 = fy1 + py1
        draw_x2 = fx2 + px1
        draw_y2 = fy2 + py1

        # Cyan face box (BGR: 255, 255, 0 renders as cyan)
        cv2.rectangle(annotated, (draw_x1, draw_y1), (draw_x2, draw_y2), (255, 255, 0), 2)

        # Yellow keypoints (BGR: 0, 255, 255 renders as yellow)
        for kx, ky in keypoints:
            cv2.circle(annotated, (kx + px1, ky + py1), 2, (0, 255, 255), -1)

    cv2.imshow("YOLOv8n Webcam", annotated)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
face_detector.close()