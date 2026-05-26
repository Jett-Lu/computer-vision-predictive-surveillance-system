# Detection
import cv2
from ultralytics import YOLO
import numpy as np
import face_recognition
import os

ENROLLMENTS_DIR = "../enrollments"
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

# Tolerance for the smallest face_distance to count as a match.
# 0.6 is the library default. Lower = stricter. Tune empirically by observing
# the actual distances you get for "same person" vs "different person".
TOLERANCE = 0.6


def load_known_encodings():
    """Walk the enrollments directory and build parallel name/encoding lists.

    Returns:
        (known_faces, known_encodings) — two parallel lists. By construction,
        len(known_faces) == len(known_encodings) and index i in both refers to
        the same enrolled photo.
    """
    if not os.path.exists(ENROLLMENTS_DIR):
        raise RuntimeError("Error: enrollments directory doesn't exist")

    known_faces = []
    known_encodings = []

    for root, dirs, files in os.walk(ENROLLMENTS_DIR):
        # Skip the top-level enrollments/ directory itself — only process person subfolders.
        if root == ENROLLMENTS_DIR:
            continue

        # The person's name is the folder name, not parsed from filenames.
        person_name = os.path.basename(root)

        for filename in files:
            # Filter to image files only — skips .DS_Store and any other stray files.
            if not filename.lower().endswith(IMAGE_EXTENSIONS):
                continue

            full_path = os.path.join(root, filename)

            # cv2.imread returns None on failure (corrupt file, unsupported format, etc.)
            # rather than raising. Must be checked before use.
            img = cv2.imread(full_path)
            if img is None:
                print(f"Warning: could not read {full_path}, skipping")
                continue

            # face_recognition expects RGB; OpenCV loads as BGR.
            corrected = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # face_encodings returns a list — empty if no face was detected.
            encodings = face_recognition.face_encodings(corrected)
            if len(encodings) == 0:
                print(f"Warning: no face found in {full_path}, skipping")
                continue

            # Append to BOTH lists together — invariant that prevents desync.
            known_encodings.append(encodings[0])
            known_faces.append(person_name)

    return known_faces, known_encodings


def run_detection():
    """Run the live face recognition loop. Returns when the user presses 'q'."""
    print("Loading enrolments...")
    try:
        known_faces, known_encodings = load_known_encodings()
    except RuntimeError as e:
        print(e)
        input("Press Enter to return to menu...")
        return

    if len(known_encodings) == 0:
        print("No enrolled faces found. Enrol someone first.")
        input("Press Enter to return to menu...")
        return

    print(f"Loaded {len(known_encodings)} encodings for {len(set(known_faces))} people")

    model = YOLO('../data/yolov8n.pt')

    # Cache: track_id -> recognized name. Populated the first time we identify a
    # tracked person, then reused on subsequent frames so we don't re-encode every frame.
    # This is the per-track-ID caching that makes recognition feasible on CPU.
    track_id_to_name = {}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Returning to menu.")
        input("Press Enter to continue...")
        return

    print("Detection running. Press 'q' in the camera window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting detection.")
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
                        face_enc = face_encs[0]
                        distances = face_recognition.face_distance(known_encodings, face_enc)

                        if len(distances) == 0:
                            name = "Unknown"
                        else:
                            best_index = np.argmin(distances)
                            if distances[best_index] <= TOLERANCE:
                                name = known_faces[best_index]
                            else:
                                name = "Unknown"

                        # Cache only when a face was actually found.
                        if track_id is not None:
                            track_id_to_name[track_id] = name

            # Draw the person box and the recognized name on the frame.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(frame, (x1, y2 - 30), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 8),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting detection.")
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # macOS window-teardown pump

if __name__ == "__main__":
    run_detection()