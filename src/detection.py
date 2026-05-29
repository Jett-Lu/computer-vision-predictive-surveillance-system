"""Live integrated monitoring with optional enrolled-person identification."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence
import os
import time

import cv2

from camera import open_capture
from identity import KnownIdentity, OpenCVFaceIdentifier, offset_box
from review import ReviewState


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_DIR = PROJECT_ROOT / ".tmp"
(TMP_DIR / "ultralytics").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(TMP_DIR / "ultralytics"))

ENROLLMENTS_DIR = PROJECT_ROOT / "enrollments"
PERSON_MODEL_PATH = PROJECT_ROOT / "data" / "yolov8n.pt"

YOLO_INTERVAL_FRAMES = 2
DEBUG_TIMING = False


class StageTimer:
    """Accumulate and periodically report per-stage frame processing time."""

    def __init__(self, print_every: int = 30) -> None:
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

    def tick(self) -> None:
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

    def tick(self) -> None:
        pass


def _face_centre_inside_box(
    face_box: tuple[int, int, int, int],
    person_box: tuple[int, int, int, int] | None,
) -> bool:
    """Return True if the centre of face_box falls inside person_box."""
    if person_box is None:
        return False
    fx1, fy1, fx2, fy2 = face_box
    px1, py1, px2, py2 = person_box
    cx = (fx1 + fx2) // 2
    cy = (fy1 + fy2) // 2
    return px1 <= cx <= px2 and py1 <= cy <= py2


def run_detection(source: int | str = 0) -> None:
    """Run live monitoring with pose, expression, tracking, and identity overlays."""
    cap = open_capture(source)
    if not cap.isOpened():
        print(f"Could not open camera source {source}. Returning to menu.")
        input("Press Enter to continue...")
        return

    print("Loading identity models and enrollments...")
    identity_matcher = OpenCVFaceIdentifier()
    known_identities = identity_matcher.load_enrollments(ENROLLMENTS_DIR)
    if not known_identities:
        print("Identity names: OFF")
        print("Reason: no enrolled faces were found.")
        print(
            "Action: live monitoring will run anonymously. "
            "Use 'enroll' from the menu to add people."
        )
    else:
        names = {identity.name for identity in known_identities}
        print(f"Identity names: ON ({len(known_identities)} encodings for {len(names)} people)")

    from emotion import FaceEmotionAnalyzer, FaceEmotionResult
    from gesture import RightHandWaveMonitor
    from pose import DEFAULT_MODEL_PATH, PoseAnalyzer
    from review import ReviewLevelMonitor
    from ultralytics import YOLO

    print("Loading YOLO, pose, and emotion models...")
    model = YOLO(str(PERSON_MODEL_PATH))
    pose_analyzer = PoseAnalyzer(model_path=DEFAULT_MODEL_PATH)
    wave_monitor = RightHandWaveMonitor()
    review_monitor = ReviewLevelMonitor()
    emotion_analyzer = FaceEmotionAnalyzer()

    expression_interval_frames = 5
    expression_score_alpha = 0.35
    cached_emotion: FaceEmotionResult | None = None
    cached_boxes = None
    frame_count = 0
    track_id_to_name: dict[int, str] = {}
    timer = StageTimer(print_every=30) if DEBUG_TIMING else NoOpTimer()

    print("Detection running. Press 'q' in the camera window to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Exiting detection.")
                break

            timestamp = time.monotonic()

            with timer("pose"):
                pose_result = pose_analyzer.analyze(frame)
                wave_state = wave_monitor.update(pose_result.landmarks, timestamp)
                person_box_from_pose = pose_analyzer.person_box(
                    pose_result.landmarks,
                    frame,
                )

            expression_event_counted = False
            if frame_count % expression_interval_frames == 0:
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
                            expression_score_alpha * detected_emotion.confidence
                            + (1 - expression_score_alpha) * cached_emotion.confidence
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

            with timer("render_pose"):
                annotated = pose_analyzer.render(
                    frame,
                    pose_result.landmarks,
                    review_state.color,
                )

            if person_box_from_pose is not None:
                _draw_review_overlay(
                    annotated,
                    person_box_from_pose,
                    review_state,
                    wave_state.wave_detected,
                    expression_event_counted,
                )

            if frame_count % YOLO_INTERVAL_FRAMES == 0 or cached_boxes is None:
                with timer("yolo_track"):
                    results = model.track(
                        frame,
                        conf=0.30,
                        iou=0.45,
                        verbose=False,
                        classes=[0],
                        persist=True,
                        tracker="bytetrack.yaml",
                    )
                    cached_boxes = results[0].boxes

            with timer("identity_loop"):
                _draw_identity_overlays(
                    frame=frame,
                    annotated=annotated,
                    boxes=cached_boxes,
                    identity_matcher=identity_matcher,
                    known_identities=known_identities,
                    track_id_to_name=track_id_to_name,
                    person_box_from_pose=person_box_from_pose,
                    cached_emotion=cached_emotion,
                    review_color=review_state.color,
                )

            with timer("display"):
                cv2.imshow("Live Monitoring", annotated)
                frame_count += 1
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Exiting detection.")
                    break

            timer.tick()
    finally:
        cap.release()
        emotion_analyzer.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def _draw_review_overlay(
    frame: Any,
    person_box: tuple[int, int, int, int],
    review_state: ReviewState,
    wave_detected: bool,
    expression_event_counted: bool,
) -> None:
    px1, py1, px2, py2 = person_box
    cv2.rectangle(frame, (px1, py1), (px2, py2), review_state.color, 3)
    status = (
        f"Review level: {review_state.tier_label} "
        f"| waves: {review_state.recent_wave_count} "
        f"| modifier: x{review_state.expression_multiplier:.2f}"
    )
    cv2.putText(
        frame,
        status,
        (px1, max(25, py1 - 34)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        review_state.color,
        2,
    )
    concern_text = "Concern influence: none"
    if review_state.concern_expression_active:
        concern_text = (
            f"Concern influence: {review_state.concern_label} "
            f"{review_state.concern_strength:.0%}"
        )
    cv2.putText(
        frame,
        concern_text,
        (px1, max(47, py1 - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        review_state.color,
        2,
    )
    if wave_detected:
        message = "Right-hand wave counted"
    elif expression_event_counted:
        message = "Expression modifier active"
    else:
        return

    cv2.putText(
        frame,
        message,
        (px1, min(frame.shape[0] - 15, py2 + 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        review_state.color,
        2,
    )


def _draw_identity_overlays(
    frame: Any,
    annotated: Any,
    boxes: Any,
    identity_matcher: OpenCVFaceIdentifier,
    known_identities: Sequence[KnownIdentity],
    track_id_to_name: dict[int, str],
    person_box_from_pose: tuple[int, int, int, int] | None,
    cached_emotion: Any,
    review_color: tuple[int, int, int],
) -> None:
    if boxes is None:
        return

    for index in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[index].int().tolist()
        cropped_img = frame[y1:y2, x1:x2]
        if cropped_img.size == 0:
            continue

        track_id = None if boxes.id is None else int(boxes.id[index].item())
        detected_face = identity_matcher.detect_largest_face(cropped_img)
        face_box = (x1, y1, x2, y2) if detected_face is None else offset_box(
            detected_face.box,
            x1,
            y1,
        )

        if not known_identities or detected_face is None:
            name = "Person"
        elif track_id is not None and track_id in track_id_to_name:
            name = track_id_to_name[track_id]
        else:
            match = identity_matcher.identify(cropped_img, detected_face, known_identities)
            name = match.name
            if track_id is not None and match.matched:
                track_id_to_name[track_id] = name

        follows_pose_subject = _face_centre_inside_box(face_box, person_box_from_pose)
        if cached_emotion is not None and follows_pose_subject and detected_face is None:
            face_box = cached_emotion.box

        label_text = name
        if cached_emotion is not None and follows_pose_subject:
            label_text = f"{name} | {cached_emotion.label} {cached_emotion.confidence:.2f}"

        box_color = review_color if follows_pose_subject else (0, 255, 0)
        _draw_label_box(annotated, face_box, label_text, box_color)


def _draw_label_box(
    frame: Any,
    box: tuple[int, int, int, int],
    label_text: str,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(frame, (x1, y2 - 30), (x2, y2), color, cv2.FILLED)
    cv2.putText(
        frame,
        label_text,
        (x1 + 6, y2 - 8),
        cv2.FONT_HERSHEY_DUPLEX,
        0.6,
        (255, 255, 255),
        1,
    )


if __name__ == "__main__":
    run_detection()
