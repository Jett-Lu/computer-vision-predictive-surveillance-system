"""Live multi-person monitoring with optional enrolled-person identification."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
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
EXPRESSION_INTERVAL_FRAMES = 5
EXPRESSION_SCORE_ALPHA = 0.35
STALE_TRACK_FRAMES = 90
DEBUG_TIMING = False


@dataclass
class PersonRuntime:
    """State that must stay isolated for each tracked person."""

    wave_monitor: Any
    review_monitor: Any
    cached_emotion: Any = None
    last_seen_frame: int = 0


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


def _track_key(track_id: int | None, detection_index: int) -> int:
    """Return a persistent tracker ID or a frame-local fallback ID."""
    return track_id if track_id is not None else -(detection_index + 1)


def run_detection(source: int | str = 0) -> None:
    """Run live multi-person pose, expression, tracking, and identity overlays."""
    cap = open_capture(source)
    if not cap.isOpened():
        print(f"Could not open camera source {source}. Returning to menu.")
        try:
            input("Press Enter to continue...")
        except EOFError:
            pass
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

    print("Loading multi-person pose and emotion models...")
    pose_analyzer = PoseAnalyzer(model_path=DEFAULT_MODEL_PATH)
    emotion_analyzer = FaceEmotionAnalyzer()
    person_states: dict[int, PersonRuntime] = {}
    track_id_to_name: dict[int, str] = {}
    frame_count = 0
    timer = StageTimer(print_every=30) if DEBUG_TIMING else NoOpTimer()

    print("Detection running. Press 'q' in the camera window to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Exiting detection.")
                break

            timestamp = time.monotonic()
            annotated = frame.copy()

            with timer("pose_track"):
                tracked_poses = pose_analyzer.analyze(frame)

            for detection_index, pose_result in enumerate(tracked_poses):
                track_key = _track_key(pose_result.track_id, detection_index)
                runtime = person_states.get(track_key)
                if runtime is None:
                    runtime = PersonRuntime(
                        wave_monitor=RightHandWaveMonitor(),
                        review_monitor=ReviewLevelMonitor(),
                    )
                    person_states[track_key] = runtime
                runtime.last_seen_frame = frame_count

                wave_state = runtime.wave_monitor.update(
                    pose_result.landmarks,
                    timestamp,
                )

                expression_event_counted = False
                if frame_count % EXPRESSION_INTERVAL_FRAMES == 0:
                    with timer("emotion"):
                        detected_emotion = emotion_analyzer.analyze(
                            frame,
                            pose_result.box,
                        )
                    expression_event_counted = runtime.review_monitor.observe_expression(
                        detected_emotion.label if detected_emotion is not None else None,
                        detected_emotion.confidence if detected_emotion is not None else None,
                        timestamp,
                    )
                    runtime.cached_emotion = _smooth_emotion(
                        runtime.cached_emotion,
                        detected_emotion,
                        FaceEmotionResult,
                    )

                review_state = runtime.review_monitor.update(wave_state, timestamp)
                with timer("render_pose"):
                    pose_analyzer.draw_landmarks(
                        annotated,
                        pose_result.landmarks,
                        review_state.color,
                    )
                    _draw_review_overlay(
                        annotated,
                        pose_result.box,
                        review_state,
                        wave_state.wave_detected,
                        expression_event_counted,
                    )

                with timer("identity"):
                    _draw_identity_overlay(
                        frame=frame,
                        annotated=annotated,
                        person_box=pose_result.box,
                        track_id=pose_result.track_id,
                        identity_matcher=identity_matcher,
                        known_identities=known_identities,
                        track_id_to_name=track_id_to_name,
                        cached_emotion=runtime.cached_emotion,
                        review_color=review_state.color,
                    )

            _discard_stale_tracks(person_states, track_id_to_name, frame_count)

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


def _smooth_emotion(
    cached_emotion: Any,
    detected_emotion: Any,
    emotion_result_type: Any,
) -> Any:
    if detected_emotion is None:
        return None
    if cached_emotion is None or cached_emotion.label != detected_emotion.label:
        return detected_emotion

    smoothed_confidence = (
        EXPRESSION_SCORE_ALPHA * detected_emotion.confidence
        + (1 - EXPRESSION_SCORE_ALPHA) * cached_emotion.confidence
    )
    return emotion_result_type(
        box=detected_emotion.box,
        keypoints=detected_emotion.keypoints,
        label=detected_emotion.label,
        confidence=smoothed_confidence,
    )


def _discard_stale_tracks(
    person_states: dict[int, PersonRuntime],
    track_id_to_name: dict[int, str],
    frame_count: int,
) -> None:
    stale_keys = [
        track_key
        for track_key, runtime in person_states.items()
        if frame_count - runtime.last_seen_frame > STALE_TRACK_FRAMES
    ]
    for track_key in stale_keys:
        del person_states[track_key]
        track_id_to_name.pop(track_key, None)


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


def _draw_identity_overlay(
    frame: Any,
    annotated: Any,
    person_box: tuple[int, int, int, int],
    track_id: int | None,
    identity_matcher: OpenCVFaceIdentifier,
    known_identities: Sequence[KnownIdentity],
    track_id_to_name: dict[int, str],
    cached_emotion: Any,
    review_color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = person_box
    cropped_img = frame[y1:y2, x1:x2]
    if cropped_img.size == 0:
        return

    detected_face = identity_matcher.detect_largest_face(cropped_img)
    face_box = person_box if detected_face is None else offset_box(
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

    if cached_emotion is not None and detected_face is None:
        face_box = cached_emotion.box

    label_text = name
    if cached_emotion is not None:
        label_text = f"{name} | {cached_emotion.label} {cached_emotion.confidence:.2f}"

    _draw_label_box(annotated, face_box, label_text, review_color)


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
