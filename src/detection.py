"""Live integrated monitoring with optional enrolled-person identification."""

from pathlib import Path
import os
import time
from collections import defaultdict
from contextlib import contextmanager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_DIR = PROJECT_ROOT / ".tmp"
(TMP_DIR / "ultralytics").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(TMP_DIR / "ultralytics"))

import cv2
import numpy as np

try:
    import face_recognition
except ModuleNotFoundError:
    face_recognition = None

ENROLLMENTS_DIR = PROJECT_ROOT / "enrollments"
PERSON_MODEL_PATH = PROJECT_ROOT / "data" / "yolov8n.pt"
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

# Tolerance for the smallest face_distance to count as a match.
# 0.6 is the library default. Lower = stricter. Tune empirically by observing
# the actual distances you get for "same person" vs "different person".
TOLERANCE = 0.6

# Performance tuning controls for the live camera loop.
FACE_DETECT_DOWNSCALE = 2
YOLO_INTERVAL_FRAMES = 2
DEBUG_TIMING = False


class StageTimer:
    """Accumulate and periodically report per-stage frame processing time."""

    def __init__(self, print_every: int = 30):
        self.print_every = print_every
        self._totals = defaultdict(float)
        self._counts = defaultdict(int)
        self._frames = 0

    @contextmanager
    def __call__(self, stage_name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self._totals[stage_name] += time.perf_counter() - start
            self._counts[stage_name] += 1

    def tick(self):
        """Print timing averages after each measurement window."""
        self._frames += 1
        if self._frames % self.print_every != 0:
            return

        lines = [f"\n--- timing over last {self.print_every} frames ---"]
        rows = [
            (stage, (total / self._counts[stage]) * 1000, self._counts[stage])
            for stage, total in self._totals.items()
        ]
        rows.sort(key=lambda row: row[1], reverse=True)
        for stage, average_ms, calls in rows:
            lines.append(f"  {stage:<20} {average_ms:6.1f} ms  ({calls} calls)")

        per_frame_total_ms = sum(
            total * 1000 / self.print_every for total in self._totals.values()
        )
        if per_frame_total_ms > 0:
            lines.append(
                f"  -- sum: {per_frame_total_ms:.1f} ms/frame "
                f"(~{1000 / per_frame_total_ms:.1f} FPS upper bound)"
            )
        print("\n".join(lines))
        self._totals.clear()
        self._counts.clear()


class NoOpTimer:
    """Timer-compatible context manager used when diagnostics are disabled."""

    @contextmanager
    def __call__(self, stage_name: str):
        yield

    def tick(self):
        pass


def load_known_encodings():
    """Walk the enrollments directory and build parallel name/encoding lists.

    Returns:
        (known_faces, known_encodings) — two parallel lists. By construction,
        len(known_faces) == len(known_encodings) and index i in both refers to
        the same enrolled photo.
    """
    if face_recognition is None or not ENROLLMENTS_DIR.exists():
        return [], []

    known_faces = []
    known_encodings = []

    for root, _, files in os.walk(ENROLLMENTS_DIR):
        # Skip the top-level enrollments/ directory itself — only process person subfolders.
        if Path(root) == ENROLLMENTS_DIR:
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
    known_faces, known_encodings = load_known_encodings()
    if face_recognition is None:
        print("Identity matching is unavailable. Continuing with anonymous monitoring.")
    elif len(known_encodings) == 0:
        print("No enrolled faces found. Continuing with anonymous monitoring.")
    else:
        print(f"Loaded {len(known_encodings)} encodings for {len(set(known_faces))} people")

    # Imports are local so menu startup doesn't pay these costs.
    from pose import PoseAnalyzer, DEFAULT_MODEL_PATH
    from gesture import RightHandWaveMonitor
    from review import ReviewLevelMonitor
    from emotion import FaceEmotionAnalyzer, FaceEmotionResult
    from ultralytics import YOLO

    print("Loading YOLO, pose, and emotion models...")
    model = YOLO(str(PERSON_MODEL_PATH))
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
    cached_boxes = None
    timer = StageTimer(print_every=30) if DEBUG_TIMING else NoOpTimer()

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
            with timer("pose"):
                pose_result = pose_analyzer.analyze(frame)
                wave_state = wave_monitor.update(pose_result.landmarks, timestamp)
                person_box_from_pose = pose_analyzer.person_box(pose_result.landmarks, frame)

            # ===== Emotion analysis (throttled, tied to pose's person) =====
            expression_event_counted = False
            if frame_count % EXPRESSION_INTERVAL_FRAMES == 0:
                with timer("emotion"):
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
            with timer("render_pose"):
                annotated = pose_analyzer.render(
                    frame, pose_result.landmarks, review_state.color,
                )

            # ===== Draw pose-derived person box + review status =====
            if person_box_from_pose is not None:
                px1, py1, px2, py2 = person_box_from_pose
                cv2.rectangle(annotated, (px1, py1), (px2, py2), review_state.color, 3)
                status = (
                    f"Review level: {review_state.tier_label} "
                    f"| waves: {review_state.recent_wave_count} "
                    f"| modifier: x{review_state.expression_multiplier:.2f}"
                )
                cv2.putText(
                    annotated, status,
                    (px1, max(25, py1 - 34)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, review_state.color, 2,
                )
                concern_text = "Concern influence: none"
                if review_state.concern_expression_active:
                    concern_text = (
                        f"Concern influence: {review_state.concern_label} "
                        f"{review_state.concern_strength:.0%}"
                    )
                cv2.putText(
                    annotated, concern_text,
                    (px1, max(47, py1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, review_state.color, 2,
                )
                if wave_state.wave_detected:
                    cv2.putText(
                        annotated, "Right-hand wave counted",
                        (px1, min(frame.shape[0] - 15, py2 + 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, review_state.color, 2,
                    )
                elif expression_event_counted:
                    cv2.putText(
                        annotated, "Expression modifier active",
                        (px1, min(frame.shape[0] - 15, py2 + 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, review_state.color, 2,
                    )

            # NOTE: Removed — the separate yellow emotion face box and BlazeFace keypoint
            # circles. The emotion is now drawn on the face-recognition box below.

            # ===== YOLO person detection + tracking (throttled) =====
            if frame_count % YOLO_INTERVAL_FRAMES == 0 or cached_boxes is None:
                with timer("yolo_track"):
                    results = model.track(
                        frame, conf=0.30, iou=0.45, verbose=False, classes=[0],
                        persist=True, tracker="bytetrack.yaml"
                    )
                    cached_boxes = results[0].boxes
            boxes = cached_boxes

            # ===== Face recognition per YOLO-tracked person =====
            with timer("face_recog_loop"):
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].int().tolist()

                    if boxes.id is None:
                        track_id = None
                    else:
                        track_id = int(boxes.id[i].item())

                    cropped_img = frame[y1:y2, x1:x2]
                    if cropped_img.size == 0:
                        continue

                    if face_recognition is None:
                        face_box = (x1, y1, x2, y2)
                        corrected = None
                        face_locs = None
                    else:
                        crop_h, crop_w = cropped_img.shape[:2]
                        small_crop = cv2.resize(
                            cropped_img,
                            (
                                max(1, crop_w // FACE_DETECT_DOWNSCALE),
                                max(1, crop_h // FACE_DETECT_DOWNSCALE),
                            ),
                        )
                        corrected = cv2.cvtColor(small_crop, cv2.COLOR_BGR2RGB)

                        face_locs = face_recognition.face_locations(
                            corrected, number_of_times_to_upsample=1, model='hog'
                        )
                        if len(face_locs) == 0:
                            continue

                        top, right, bottom, left = face_locs[0]
                        face_box = (
                            left * FACE_DETECT_DOWNSCALE + x1,
                            top * FACE_DETECT_DOWNSCALE + y1,
                            right * FACE_DETECT_DOWNSCALE + x1,
                            bottom * FACE_DETECT_DOWNSCALE + y1,
                        )

                    face_x1, face_y1, face_x2, face_y2 = face_box

                    # Recognise only when identities have been enrolled.
                    if not known_encodings or face_recognition is None:
                        name = "Person"
                    elif track_id is not None and track_id in track_id_to_name:
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

                    # Attach emotion to the face that matches pose's person.
                    follows_pose_subject = _face_centre_inside_box(
                        face_box,
                        person_box_from_pose,
                    )
                    if (
                        face_recognition is None
                        and cached_emotion is not None
                        and follows_pose_subject
                    ):
                        face_box = cached_emotion.box
                        face_x1, face_y1, face_x2, face_y2 = face_box
                    if cached_emotion is not None and follows_pose_subject:
                        label_text = (
                            f"{name} | {cached_emotion.label} "
                            f"{cached_emotion.confidence:.2f}"
                        )
                    else:
                        label_text = name

                    box_color = review_state.color if follows_pose_subject else (0, 255, 0)
                    cv2.rectangle(annotated, (face_x1, face_y1), (face_x2, face_y2), box_color, 2)
                    cv2.rectangle(
                        annotated,
                        (face_x1, face_y2 - 30),
                        (face_x2, face_y2),
                        box_color,
                        cv2.FILLED,
                    )
                    cv2.putText(
                        annotated,
                        label_text,
                        (face_x1 + 6, face_y2 - 8),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                    )

            with timer("display"):
                cv2.imshow('Face Recognition + Pose/Emotion', annotated)
                frame_count += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting detection.")
                    break

            timer.tick()
    finally:
        cap.release()
        emotion_analyzer.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

if __name__ == "__main__":
    run_detection()
