import os
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_DIR = PROJECT_ROOT / ".tmp"
TMP_DIR.mkdir(exist_ok=True)
(TMP_DIR / "ultralytics").mkdir(exist_ok=True)
(TMP_DIR / "matplotlib").mkdir(exist_ok=True)

os.environ.setdefault("YOLO_CONFIG_DIR", str(TMP_DIR / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(TMP_DIR / "matplotlib"))

import cv2
import mediapipe as mp
import numpy as np
from emotiefflib.facial_analysis import EmotiEffLibRecognizer
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

# Import the pre-trained YOLO model
YOLO_MODEL_PATH = PROJECT_ROOT / "data" / "yolov8n.pt"
model = YOLO(str(YOLO_MODEL_PATH))

# Download the MediaPipe Face Detector model file (one-time, cached locally)
FACE_DETECTOR_MODEL_PATH = PROJECT_ROOT / "data" / "blaze_face_short_range.tflite"
FACE_DETECTOR_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)

if not FACE_DETECTOR_MODEL_PATH.exists():
    print(f"Downloading MediaPipe face detector model to {FACE_DETECTOR_MODEL_PATH}...")
    os.makedirs(FACE_DETECTOR_MODEL_PATH.parent, exist_ok=True)
    urllib.request.urlretrieve(FACE_DETECTOR_MODEL_URL, FACE_DETECTOR_MODEL_PATH)
    print("Done.")

# Create the detectors ONCE outside the loop
base_options = python.BaseOptions(model_asset_path=str(FACE_DETECTOR_MODEL_PATH))
face_detector_options = vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.5,
)
face_detector = vision.FaceDetector.create_from_options(face_detector_options)

# Accuracy-first 8-class model:
# Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
emotion_recognizer = EmotiEffLibRecognizer(
    engine="onnx",
    model_name="enet_b2_8",
    device="cpu",
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

# How often to run face / emotion recognition (1 = every frame, 5 = every 5th frame)
RECOGNITION_INTERVAL = 5
EMOTION_SMOOTHING_ALPHA = 0.35
FACE_PADDING_RATIO = 0.12
frame_count = 0

# Persistent across frames - keyed by track_id
face_results = {}  # {track_id: ((fx1, fy1, fx2, fy2), [(kx, ky), ...])}
emotion_results = {}  # {track_id: np.ndarray probabilities}


def padded_face_box(
    face_box: tuple[int, int, int, int],
    crop_width: int,
    crop_height: int,
) -> tuple[int, int, int, int]:
    fx1, fy1, fx2, fy2 = face_box
    pad_x = int((fx2 - fx1) * FACE_PADDING_RATIO)
    pad_y = int((fy2 - fy1) * FACE_PADDING_RATIO)
    return (
        max(0, fx1 - pad_x),
        max(0, fy1 - pad_y),
        min(crop_width, fx2 + pad_x),
        min(crop_height, fy2 + pad_y),
    )


while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame_height, frame_width = frame.shape[:2]

    # YOLO + ByteTrack run EVERY frame so track IDs stay current
    results = model.track(
        frame,
        conf=0.75,
        iou=0.45,
        verbose=False,
        classes=[0],
        persist=True,
        tracker="bytetrack.yaml",
    )
    annotated = results[0].plot()

    boxes = results[0].boxes
    track_ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)

    # Build this frame's track_id -> person box mapping
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

    # Cull cached entries for tracks that no longer exist
    face_results = {tid: data for tid, data in face_results.items() if tid in current_person_boxes}
    emotion_results = {
        tid: data for tid, data in emotion_results.items() if tid in current_person_boxes
    }

    # Run face detection and emotion recognition only every Nth frame
    if frame_count % RECOGNITION_INTERVAL == 0:
        for track_id, (x1, y1, x2, y2) in current_person_boxes.items():
            # Crop from the ORIGINAL frame (not annotated)
            crop = frame[y1:y2, x1:x2].copy()
            crop_h, crop_w = crop.shape[:2]

            # MediaPipe expects RGB; OpenCV gives BGR
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
            detection_result = face_detector.detect(mp_image)

            if not detection_result.detections:
                continue

            # If multiple faces appear inside the person crop, use the largest one
            best_detection = max(
                detection_result.detections,
                key=lambda detection: detection.bounding_box.width * detection.bounding_box.height,
            )
            bbox = best_detection.bounding_box

            fx1 = max(0, bbox.origin_x)
            fy1 = max(0, bbox.origin_y)
            fx2 = min(crop_w, bbox.origin_x + bbox.width)
            fy2 = min(crop_h, bbox.origin_y + bbox.height)

            if fx2 <= fx1 or fy2 <= fy1:
                continue

            keypoints = [
                (int(keypoint.x * crop_w), int(keypoint.y * crop_h))
                for keypoint in best_detection.keypoints
            ]
            face_results[track_id] = ((fx1, fy1, fx2, fy2), keypoints)

            ex1, ey1, ex2, ey2 = padded_face_box((fx1, fy1, fx2, fy2), crop_w, crop_h)
            face_rgb = crop_rgb[ey1:ey2, ex1:ex2]
            if face_rgb.size == 0:
                continue

            _, emotion_scores = emotion_recognizer.predict_emotions(face_rgb, logits=False)
            current_scores = emotion_scores[0]

            previous_scores = emotion_results.get(track_id)
            if previous_scores is None:
                emotion_results[track_id] = current_scores
            else:
                emotion_results[track_id] = (
                    EMOTION_SMOOTHING_ALPHA * current_scores
                    + (1 - EMOTION_SMOOTHING_ALPHA) * previous_scores
                )

    # Draw cached face boxes and emotion labels every frame
    for track_id, ((fx1, fy1, fx2, fy2), keypoints) in face_results.items():
        px1, py1, _, _ = current_person_boxes[track_id]

        draw_x1 = fx1 + px1
        draw_y1 = fy1 + py1
        draw_x2 = fx2 + px1
        draw_y2 = fy2 + py1

        # Cyan face box
        cv2.rectangle(annotated, (draw_x1, draw_y1), (draw_x2, draw_y2), (255, 255, 0), 2)

        # Yellow facial keypoints
        for kx, ky in keypoints:
            cv2.circle(annotated, (kx + px1, ky + py1), 2, (0, 255, 255), -1)

        scores = emotion_results.get(track_id)
        if scores is not None:
            emotion_index = int(np.argmax(scores))
            emotion_label = emotion_recognizer.idx_to_emotion_class[emotion_index]
            emotion_confidence = float(scores[emotion_index])
            text = f"{emotion_label} {emotion_confidence:.2f}"

            text_y = max(20, draw_y1 - 10)
            cv2.putText(
                annotated,
                text,
                (draw_x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    cv2.imshow("YOLOv8n Webcam", annotated)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
face_detector.close()
