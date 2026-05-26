# Detection
import cv2
from ultralytics import YOLO
import numpy as np
import face_recognition
import os
import time
from collections import defaultdict
from contextlib import contextmanager

ENROLLMENTS_DIR = "../enrollments"
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

# Tolerance for the smallest face_distance to count as a match.
# 0.6 is the library default. Lower = stricter. Tune empirically by observing
# the actual distances you get for "same person" vs "different person".
TOLERANCE = 0.6

# Downscale factor for face detection. HOG cost scales roughly with pixel area,
# so 2x downscale ~= 4x speedup. Coordinates are scaled back up before drawing
# on the full-resolution frame. Increase to push speed further at the cost of
# missing smaller faces; decrease (down to 1, i.e. no downscale) for accuracy.
FACE_DETECT_DOWNSCALE = 2

# Run YOLO tracking every Nth frame, reuse boxes between. ByteTrack handles
# this gracefully because tracking is designed to be robust to occasional
# update gaps. Tradeoff: between updates, box coordinates lag behind motion.
# Tune higher for more speed, lower for more responsive bounding boxes.
YOLO_INTERVAL_FRAMES = 2

# Toggle timing instrumentation. When False, the timer becomes a no-op with
# negligible overhead — the `with timer(...)` blocks throughout run_detection
# remain in place but do nothing. Flip to True to diagnose performance.
DEBUG_TIMING = False


class StageTimer:
    """Accumulate per-stage durations across frames and print periodically.

    Designed to be easy to add/remove without restructuring the loop:
    just wrap stages in `with timer("stage_name"):` and call `timer.tick()`
    once per frame.

    Output (every print_every frames):
        --- timing over last 30 frames ---
          face_recog_loop       45.2 ms  (30 calls)
          yolo_track            28.7 ms  (30 calls)
          ...
          -- sum: 103.6 ms/frame (~9.6 FPS upper bound)

    Stages are sorted by average ms descending so the bottleneck is at the top.
    Note: stages that don't run every frame (throttled or skipped) report
    fewer calls; the per-frame impact in the FPS sum is correctly weighted.
    """

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
            elapsed = time.perf_counter() - start
            self._totals[stage_name] += elapsed
            self._counts[stage_name] += 1

    def tick(self):
        """Call once per frame. Prints aggregated averages every print_every frames."""
        self._frames += 1
        if self._frames % self.print_every != 0:
            return

        lines = [f"\n--- timing over last {self.print_every} frames ---"]
        rows = []
        for stage, total in self._totals.items():
            avg_ms_per_call = (total / self._counts[stage]) * 1000
            rows.append((stage, avg_ms_per_call, self._counts[stage]))
        rows.sort(key=lambda r: r[1], reverse=True)
        for stage, avg_ms_per_call, calls in rows:
            lines.append(f"  {stage:<20} {avg_ms_per_call:6.1f} ms  ({calls} calls)")

        # Per-frame impact = total stage time / frames in this window.
        # This correctly accounts for stages that don't run every frame
        # (throttled stages contribute less to per-frame cost).
        per_frame_total_ms = sum(
            total * 1000 / self.print_every
            for total in self._totals.values()
        )
        if per_frame_total_ms > 0:
            implied_fps = 1000 / per_frame_total_ms
            lines.append(
                f"  -- sum: {per_frame_total_ms:.1f} ms/frame "
                f"(~{implied_fps:.1f} FPS upper bound)"
            )
        print("\n".join(lines))

        self._totals.clear()
        self._counts.clear()


class NoOpTimer:
    """Drop-in replacement for StageTimer with zero measurement overhead.

    Implements the same interface (callable context manager + tick method)
    so the surrounding code is identical whether timing is enabled or not.
    """

    @contextmanager
    def __call__(self, stage_name: str):
        yield

    def tick(self):
        pass


def load_known_encodings():
    """Walk the enrollments directory and build parallel name/encoding lists."""
    if not os.path.exists(ENROLLMENTS_DIR):
        raise RuntimeError("Error: enrollments directory doesn't exist")

    known_faces = []
    known_encodings = []

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

    # Holds the most recent YOLO boxes between throttled updates.
    cached_boxes = None

    # Per-stage timing instrumentation, gated by the DEBUG_TIMING flag.
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

            # ===== Pose skeleton rendering =====
            with timer("render_pose"):
                annotated = pose_analyzer.render(
                    frame, pose_result.landmarks, review_state.color,
                )

            # ===== Pose-derived person box + review status overlays =====
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

            # ===== YOLO person detection + tracking (throttled) =====
            # Only run on every Nth frame; reuse the previous boxes between.
            # Per-track-ID caching downstream means cached boxes incur no
            # extra recognition cost — names stay attached to their IDs.
            # The `cached_boxes is None` guard ensures the very first frame
            # always runs YOLO regardless of the modulo check.
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

                    # Downscale the person crop before face detection.
                    # HOG cost scales roughly with pixel area, so this is
                    # where the face-detection speed win lives.
                    crop_h, crop_w = cropped_img.shape[:2]
                    small_w = max(1, crop_w // FACE_DETECT_DOWNSCALE)
                    small_h = max(1, crop_h // FACE_DETECT_DOWNSCALE)
                    small_crop = cv2.resize(cropped_img, (small_w, small_h))

                    # face_recognition expects RGB; OpenCV gives BGR.
                    corrected = cv2.cvtColor(small_crop, cv2.COLOR_BGR2RGB)

                    # Detect on the downscaled crop.
                    face_locs = face_recognition.face_locations(
                        corrected, number_of_times_to_upsample=1, model='hog'
                    )
                    if len(face_locs) == 0:
                        continue

                    # Scale face_locs coordinates back up to original crop space,
                    # then offset to full-frame coordinates for drawing.
                    # face_locations returns (top, right, bottom, left).
                    top, right, bottom, left = face_locs[0]
                    top *= FACE_DETECT_DOWNSCALE
                    right *= FACE_DETECT_DOWNSCALE
                    bottom *= FACE_DETECT_DOWNSCALE
                    left *= FACE_DETECT_DOWNSCALE

                    face_x1 = left + x1
                    face_y1 = top + y1
                    face_x2 = right + x1
                    face_y2 = bottom + y1

                    # Recognise (with caching by track ID).
                    if track_id is not None and track_id in track_id_to_name:
                        name = track_id_to_name[track_id]
                    else:
                        # face_encodings must use the same image and coordinates
                        # that face_locations produced — both at downscaled resolution.
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
                    face_box = (face_x1, face_y1, face_x2, face_y2)
                    if (
                        cached_emotion is not None
                        and _face_centre_inside_box(face_box, person_box_from_pose)
                    ):
                        label_text = f"{name} | {cached_emotion.label} {cached_emotion.confidence:.2f}"
                    else:
                        label_text = name

                    # Draw face box + combined label on the full-resolution frame.
                    cv2.rectangle(annotated, (face_x1, face_y1), (face_x2, face_y2), (0, 0, 255), 2)
                    cv2.rectangle(annotated, (face_x1, face_y2 - 30), (face_x2, face_y2), (0, 0, 255), cv2.FILLED)
                    cv2.putText(annotated, label_text, (face_x1 + 6, face_y2 - 8),
                                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            # ===== Display + frame bookkeeping =====
            with timer("display"):
                cv2.imshow('Face Recognition + Pose/Emotion', annotated)
                frame_count += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting detection.")
                    break

            # Report averaged timings every print_every frames (no-op when timing disabled).
            timer.tick()

    finally:
        cap.release()
        emotion_analyzer.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


if __name__ == "__main__":
    run_detection()