"""Live multi-person monitoring with optional enrolled-person identification."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence
import os
import time

import cv2

from camera import open_capture
from identity import KnownIdentity, OpenCVFaceIdentifier, offset_box
from review import HIGH_COLOR, ReviewState


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_DIR = PROJECT_ROOT / ".tmp"
(TMP_DIR / "ultralytics").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(TMP_DIR / "ultralytics"))

ENROLLMENTS_DIR = PROJECT_ROOT / "enrollments"
EXPRESSION_INTERVAL_FRAMES = 5
EXPRESSION_SCORE_ALPHA = 0.35
STALE_TRACK_FRAMES = 90
DEBUG_TIMING = False
DEMO_HIGH_REVIEW_ENV = "DEMO_HIGH_REVIEW_NAMES"


@dataclass
class PersonRuntime:
    """State that must stay isolated for each tracked person."""

    wave_monitor: Any
    review_monitor: Any
    cached_emotion: Any = None
    last_seen_frame: int = 0


@dataclass(frozen=True)
class IdentityOverlay:
    """Resolved identity label details for one tracked person."""

    name: str
    face_box: tuple[int, int, int, int]
    label_text: str


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


def _parse_demo_high_review_names(raw_names: str) -> set[str]:
    """Parse optional demo-only names that should be displayed as HIGH."""
    return {
        name.strip().casefold()
        for name in raw_names.replace(";", ",").split(",")
        if name.strip()
    }


def _apply_demo_review_override(
    review_state: ReviewState,
    identity_name: str,
    demo_high_review_names: set[str],
) -> ReviewState:
    """Force named enrolled demo identities to HIGH without changing normal logic."""
    if identity_name.casefold() not in demo_high_review_names:
        return review_state

    return replace(
        review_state,
        tier_label="HIGH",
        color=HIGH_COLOR,
        score=max(review_state.score, 5.0),
        recent_wave_count=max(review_state.recent_wave_count, 5),
    )


class MonitoringProcessor:
    """Apply the monitoring pipeline to frames from a live or recorded source."""

    def __init__(self, debug_timing: bool = DEBUG_TIMING) -> None:
        print("Loading identity models and enrollments...")
        self.identity_matcher = OpenCVFaceIdentifier()
        self.known_identities = self.identity_matcher.load_enrollments(ENROLLMENTS_DIR)
        if not self.known_identities:
            print("Identity names: OFF")
            print("Reason: no enrolled faces were found.")
            print(
                "Action: monitoring will run anonymously. "
                "Use 'enroll' from the menu to add people."
            )
        else:
            names = {identity.name for identity in self.known_identities}
            print(
                f"Identity names: ON "
                f"({len(self.known_identities)} encodings for {len(names)} people)"
            )

        from emotion import FaceEmotionAnalyzer, FaceEmotionResult
        from gesture import RightHandWaveMonitor
        from pose import DEFAULT_MODEL_PATH, PoseAnalyzer
        from review import ReviewLevelMonitor

        print("Loading multi-person pose and emotion models...")
        self.pose_analyzer = PoseAnalyzer(model_path=DEFAULT_MODEL_PATH)
        self.emotion_analyzer = FaceEmotionAnalyzer()
        self.emotion_result_type = FaceEmotionResult
        self.wave_monitor_type = RightHandWaveMonitor
        self.review_monitor_type = ReviewLevelMonitor
        self.person_states: dict[int, PersonRuntime] = {}
        self.track_id_to_name: dict[int, str] = {}
        self.demo_high_review_names = _parse_demo_high_review_names(
            os.environ.get(DEMO_HIGH_REVIEW_ENV, "")
        )
        if self.demo_high_review_names:
            names = ", ".join(sorted(self.demo_high_review_names))
            print(f"Demo review override: HIGH for {names}")
        self.frame_count = 0
        self.timer = StageTimer(print_every=30) if debug_timing else NoOpTimer()

    def process_frame(
        self,
        frame: Any,
        timestamp: float | None = None,
    ) -> Any:
        """Return one annotated frame while preserving per-person tracking state."""
        if timestamp is None:
            timestamp = time.monotonic()
        annotated = frame.copy()

        with self.timer("pose_track"):
            tracked_poses = self.pose_analyzer.analyze(frame)

        for detection_index, pose_result in enumerate(tracked_poses):
            track_key = _track_key(pose_result.track_id, detection_index)
            runtime = self.person_states.get(track_key)
            if runtime is None:
                runtime = PersonRuntime(
                    wave_monitor=self.wave_monitor_type(),
                    review_monitor=self.review_monitor_type(),
                )
                self.person_states[track_key] = runtime
            runtime.last_seen_frame = self.frame_count

            wave_state = runtime.wave_monitor.update(
                pose_result.landmarks,
                timestamp,
            )

            expression_event_counted = False
            if self.frame_count % EXPRESSION_INTERVAL_FRAMES == 0:
                with self.timer("emotion"):
                    detected_emotion = self.emotion_analyzer.analyze(
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
                    self.emotion_result_type,
                )

            review_state = runtime.review_monitor.update(wave_state, timestamp)
            with self.timer("identity"):
                identity_overlay = _resolve_identity_overlay(
                    frame=frame,
                    person_box=pose_result.box,
                    track_id=pose_result.track_id,
                    identity_matcher=self.identity_matcher,
                    known_identities=self.known_identities,
                    track_id_to_name=self.track_id_to_name,
                    cached_emotion=runtime.cached_emotion,
                )
            review_state = _apply_demo_review_override(
                review_state,
                identity_overlay.name,
                self.demo_high_review_names,
            )
            with self.timer("render_pose"):
                self.pose_analyzer.draw_landmarks(
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
                _draw_identity_overlay(
                    annotated=annotated,
                    identity_overlay=identity_overlay,
                    review_color=review_state.color,
                )

        _discard_stale_tracks(
            self.person_states,
            self.track_id_to_name,
            self.frame_count,
        )
        self.frame_count += 1
        self.timer.tick()
        return annotated

    def close(self) -> None:
        """Release model resources owned by the processor."""
        self.emotion_analyzer.close()


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

    processor: MonitoringProcessor | None = None
    try:
        processor = MonitoringProcessor()
        print("Detection running. Press 'q' in the camera window to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Exiting detection.")
                break

            annotated = processor.process_frame(frame)
            cv2.imshow("Live Monitoring", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Exiting detection.")
                break
    finally:
        cap.release()
        if processor is not None:
            processor.close()
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
    status_lines = [
        review_state.tier_label,
        f"w:{review_state.recent_wave_count} | x{review_state.expression_multiplier:.2f}",
    ]
    if review_state.concern_expression_active:
        status_lines.append(
            f"{review_state.concern_label} {review_state.concern_strength:.0%}"
        )
    _draw_status_panel(frame, person_box, status_lines, review_state.color)

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


def _draw_status_panel(
    frame: Any,
    person_box: tuple[int, int, int, int],
    lines: list[str],
    color: tuple[int, int, int],
) -> None:
    """Draw a compact readable status panel within one person's bounding box."""
    x1, y1, x2, y2 = person_box
    available_width = max(1, x2 - x1 - 8)
    line_height = 15
    panel_bottom = min(y2, y1 + 5 + line_height * len(lines))
    cv2.rectangle(frame, (x1, y1), (x2, panel_bottom), color, cv2.FILLED)
    for index, text in enumerate(lines):
        preferred_scale = 0.48 if index == 0 else 0.40
        text_width = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            preferred_scale,
            1,
        )[0][0]
        scale = (
            preferred_scale
            if text_width <= available_width
            else preferred_scale * available_width / max(1, text_width)
        )
        cv2.putText(
            frame,
            text,
            (x1 + 4, y1 + 13 + line_height * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            1,
        )


def _resolve_identity_overlay(
    frame: Any,
    person_box: tuple[int, int, int, int],
    track_id: int | None,
    identity_matcher: OpenCVFaceIdentifier,
    known_identities: Sequence[KnownIdentity],
    track_id_to_name: dict[int, str],
    cached_emotion: Any,
) -> IdentityOverlay:
    x1, y1, x2, y2 = person_box
    cropped_img = frame[y1:y2, x1:x2]
    if cropped_img.size == 0:
        return IdentityOverlay("Person", person_box, "Person")

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

    return IdentityOverlay(name, face_box, label_text)


def _draw_identity_overlay(
    annotated: Any,
    identity_overlay: IdentityOverlay,
    review_color: tuple[int, int, int],
) -> None:
    _draw_label_box(
        annotated,
        identity_overlay.face_box,
        identity_overlay.label_text,
        review_color,
    )


def _draw_label_box(
    frame: Any,
    box: tuple[int, int, int, int],
    label_text: str,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    frame_height, frame_width = frame.shape[:2]
    preferred_scale = 0.6
    thickness = 1
    text_width, text_height = cv2.getTextSize(
        label_text,
        cv2.FONT_HERSHEY_DUPLEX,
        preferred_scale,
        thickness,
    )[0]
    max_width = max(80, frame_width - 12)
    scale = (
        preferred_scale
        if text_width + 12 <= max_width
        else preferred_scale * max_width / max(1, text_width + 12)
    )
    text_width, text_height = cv2.getTextSize(
        label_text,
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        thickness,
    )[0]
    label_width = min(frame_width, text_width + 12)
    label_height = max(24, text_height + 14)
    label_x1 = min(max(0, x1), max(0, frame_width - label_width))
    label_y1 = y2 - label_height
    if label_y1 < 0:
        label_y1 = min(max(0, y1), max(0, frame_height - label_height))
    label_x2 = label_x1 + label_width
    label_y2 = min(frame_height, label_y1 + label_height)

    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, cv2.FILLED)
    cv2.putText(
        frame,
        label_text,
        (label_x1 + 6, label_y2 - 8),
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        (255, 255, 255),
        thickness,
    )


if __name__ == "__main__":
    run_detection()
