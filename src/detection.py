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

def _face_centre_inside_box(face_box, person_box):
    """Return True if the centre of face_box falls inside person_box.

    Both boxes are (x1, y1, x2, y2) in the same coordinate space (full frame).
    Used to attach the pose-derived emotion to the face-recognition box that
    corresponds to the same person.
    """
    if person_box is None:
        return False
    fx1, fy1, fx2, fy2 = face_box
    px1, py1, px2, py2 = person_box
    cx = (fx1 + fx2) // 2
    cy = (fy1 + fy2) // 2
    return px1 <= cx <= px2 and py1 <= cy <= py2


def run_detection():
    """Run live face recognition combined with pose, gesture, and emotion analysis."""
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

    # Imports are local so menu startup doesn't pay these costs.
    import time
    from pose import PoseAnalyzer, DEFAULT_MODEL_PATH
    from gesture import RightHandWaveMonitor
    from review import ReviewLevelMonitor
    from emotion import FaceEmotionAnalyzer, FaceEmotionResult

    print("Loading YOLO, pose, and emotion models...")
    model = YOLO('../data/yolov8n.pt')
    pose_analyzer = PoseAnalyzer(model_path=DEFAULT_MODEL_PATH)
    wave_monitor = RightHandWaveMonitor()
    review_monitor = ReviewLevelMonitor()
    emotion_analyzer = FaceEmotionAnalyzer()

    # Recognition cache: track_id -> recognized name.
    track_id_to_name = {}

    # Emotion analysis is throttled to keep the loop manageable on CPU.
    EXPRESSION_INTERVAL_FRAMES = 5
    EXPRESSION_SCORE_ALPHA = 0.35
    cached_emotion: FaceEmotionResult | None = None
    frame_count = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Returning to menu.")
        emotion_analyzer.close()
        input("Press Enter to continue...")
        return

    print("Detection running. Press 'q' in the camera window to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Exiting detection.")
                break

            timestamp = time.monotonic()

            # ===== Pose analysis (full frame, single person) =====
            pose_result = pose_analyzer.analyze(frame)
            wave_state = wave_monitor.update(pose_result.landmarks, timestamp)
            person_box_from_pose = pose_analyzer.person_box(pose_result.landmarks, frame)

            # ===== Emotion analysis (throttled, tied to pose's person) =====
            expression_event_counted = False
            if frame_count % EXPRESSION_INTERVAL_FRAMES == 0:
                detected_emotion = emotion_analyzer.analyze(frame, person_box_from_pose)
                expression_event_counted = review_monitor.observe_expression(
                    detected_emotion.label if detected_emotion is not None else None,
                    detected_emotion.confidence if detected_emotion is not None else None,
                    timestamp,
                )
                if detected_emotion is None:
                    cached_emotion = None
                elif (
                    cached_emotion is not None
                    and cached_emotion.label == detected_emotion.label
                ):
                    smoothed_confidence = (
                        EXPRESSION_SCORE_ALPHA * detected_emotion.confidence
                        + (1 - EXPRESSION_SCORE_ALPHA) * cached_emotion.confidence
                    )
                    cached_emotion = FaceEmotionResult(
                        box=detected_emotion.box,
                        keypoints=detected_emotion.keypoints,
                        label=detected_emotion.label,
                        confidence=smoothed_confidence,
                    )
                else:
                    cached_emotion = detected_emotion

            review_state = review_monitor.update(wave_state, timestamp)

            # Render pose skeleton; review color reflects wave/expression state.
            annotated = pose_analyzer.render(frame, pose_result.landmarks, review_state.color)

            # ===== Draw pose-derived person box + review status =====
            if person_box_from_pose is not None:
                px1, py1, px2, py2 = person_box_from_pose
                cv2.rectangle(annotated, (px1, py1), (px2, py2), review_state.color, 3)
                status = (
                    f"Review level: {review_state.tier_label} "
                    f"| waves: {review_state.recent_wave_count} "
                    f"| expression cues: {review_state.recent_expression_event_count}"
                )
                cv2.putText(
                    annotated, status,
                    (px1, max(25, py1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, review_state.color, 2,
                )
                if wave_state.wave_detected:
                    cv2.putText(
                        annotated, "Right-hand wave counted",
                        (px1, min(frame.shape[0] - 15, py2 + 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, review_state.color, 2,
                    )
                elif expression_event_counted:
                    cv2.putText(
                        annotated, "Sustained concern expression counted",
                        (px1, min(frame.shape[0] - 15, py2 + 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, review_state.color, 2,
                    )

            # NOTE: Removed — the separate yellow emotion face box and BlazeFace keypoint
            # circles. The emotion is now drawn on the face-recognition box below.

            # ===== Face recognition per YOLO-tracked person =====
            results = model.track(
                frame, conf=0.30, iou=0.45, verbose=False, classes=[0],
                persist=True, tracker="bytetrack.yaml"
            )
            boxes = results[0].boxes

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].int().tolist()

                if boxes.id is None:
                    track_id = None
                else:
                    track_id = int(boxes.id[i].item())

                cropped_img = frame[y1:y2, x1:x2]
                if cropped_img.size == 0:
                    continue

                corrected = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

                face_locs = face_recognition.face_locations(
                    corrected, number_of_times_to_upsample=1, model='hog'
                )
                if len(face_locs) == 0:
                    continue

                top, right, bottom, left = face_locs[0]
                face_x1 = left + x1
                face_y1 = top + y1
                face_x2 = right + x1
                face_y2 = bottom + y1

                # Recognise (with caching by track ID).
                if track_id is not None and track_id in track_id_to_name:
                    name = track_id_to_name[track_id]
                else:
                    face_encs = face_recognition.face_encodings(
                        corrected, face_locs, num_jitters=1, model='small'
                    )
                    if len(face_encs) == 0:
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
                        if track_id is not None:
                            track_id_to_name[track_id] = name

                # If this face matches the person pose locked onto, attach the
                # emotion label to this person's name. Otherwise show name only.
                face_box = (face_x1, face_y1, face_x2, face_y2)
                if (
                    cached_emotion is not None
                    and _face_centre_inside_box(face_box, person_box_from_pose)
                ):
                    label_text = f"{name} | {cached_emotion.label} {cached_emotion.confidence:.2f}"
                else:
                    label_text = name

                # Draw face box + combined label.
                cv2.rectangle(annotated, (face_x1, face_y1), (face_x2, face_y2), (0, 0, 255), 2)
                cv2.rectangle(annotated, (face_x1, face_y2 - 30), (face_x2, face_y2), (0, 0, 255), cv2.FILLED)
                cv2.putText(annotated, label_text, (face_x1 + 6, face_y2 - 8),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('Face Recognition + Pose/Emotion', annotated)
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting detection.")
                break
    finally:
        cap.release()
        emotion_analyzer.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

if __name__ == "__main__":
    run_detection()