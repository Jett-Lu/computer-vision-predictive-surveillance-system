import cv2
from ultralytics import YOLO
import numpy as np
import face_recognition
import os

# ===== Load known encodings (your existing enrolment walk, unchanged) =====
model = YOLO('../data/yolov8n.pt')

if not os.path.exists("../enrollments"):
    raise RuntimeError("Error: enrollments directory doesn't exist")

knownFaces = []
knownEncodings = []

ENROLLMENTS_DIR = "../enrollments"
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

for root, dirs, files in os.walk(ENROLLMENTS_DIR):
    if root == ENROLLMENTS_DIR:
        continue

    person_name = os.path.basename(root)

    for filename in files:
        if not filename.lower().endswith(IMAGE_EXTENSIONS):
            continue

        full_path = os.path.join(root, filename)

        img = cv2.imread(full_path)
        if img is None:
            print(f"Warning: could not read {full_path}, skipping")
            continue

        corrected = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        encodings = face_recognition.face_encodings(corrected)
        if len(encodings) == 0:
            print(f"Warning: no face found in {full_path}, skipping")
            continue

        knownEncodings.append(encodings[0])
        knownFaces.append(person_name)

print(f"Loaded {len(knownEncodings)} encodings for {len(set(knownFaces))} people")

# ===== Recognition setup =====

# Cache: track_id -> recognized name. Populated the first time we identify a tracked person,
# then reused on subsequent frames so we don't re-encode every frame.
# This is the per-track-ID caching that makes recognition feasible on CPU.
track_id_to_name = {}

# Tolerance for the smallest face_distance to count as a match.
# 0.6 is the library default. Lower = stricter. Tune empirically by observing
# the actual distances you get for "same person" vs "different person".
TOLERANCE = 0.6

# ===== Webcam loop =====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # YOLO person detection + tracking
    results = model.track(
        frame, conf=0.30, iou=0.45, verbose=False, classes=[0],
        persist=True, tracker="bytetrack.yaml"
    )
    boxes = results[0].boxes

    # Iterate every detected person
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].int().tolist()

        # Get this person's track ID. boxes.id can be None on frames where
        # tracking hasn't initialized or all tracks were dropped.
        if boxes.id is None:
            track_id = None
        else:
            track_id = int(boxes.id[i].item())

        # If we already know who this track is, skip the expensive recognition
        # work and just reuse the cached name.
        if track_id is not None and track_id in track_id_to_name:
            name = track_id_to_name[track_id]
        else:
            # Recognise this person for the first time (or every frame, if no track_id).
            cropped_img = frame[y1:y2, x1:x2]

            # Guard against zero-area crops, which would crash face_recognition.
            if cropped_img.size == 0:
                name = "Unknown"
            else:
                # BGR -> RGB for face_recognition
                corrected = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

                face_locs = face_recognition.face_locations(
                    corrected, number_of_times_to_upsample=1, model='hog'
                )
                face_encs = face_recognition.face_encodings(
                    corrected, face_locs, num_jitters=1, model='small'
                )

                if len(face_encs) == 0:
                    # No face found in this person crop — don't cache, so we'll
                    # try again next frame in case the angle improves.
                    name = "Unknown"
                else:
                    # Use the first face found in the crop (typically only one,
                    # since the crop is one person).
                    face_enc = face_encs[0]

                    # face_distance against ALL known encodings (all photos of all people).
                    # np.argmin gives the index of the smallest distance.
                    distances = face_recognition.face_distance(knownEncodings, face_enc)

                    if len(distances) == 0:
                        name = "Unknown"
                    else:
                        best_index = np.argmin(distances)
                        if distances[best_index] <= TOLERANCE:
                            name = knownFaces[best_index]
                        else:
                            name = "Unknown"

                    # Cache this result so subsequent frames for this track ID skip the work.
                    # Only cache when we actually had something to recognise (a face was found).
                    if track_id is not None:
                        track_id_to_name[track_id] = name

        # Draw the person box and the recognized name on the original frame.
        # If you prefer YOLO's overlay (with its own boxes/labels), swap `frame` for
        # `results[0].plot()` when drawing — but be aware that re-plotting every iteration
        # would redraw YOLO's boxes too.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.rectangle(frame, (x1, y2 - 30), (x2, y2), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (x1 + 6, y2 - 8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)  # macOS window-teardown pump