from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from emotion import FaceEmotionAnalyzer, FaceEmotionResult
from gesture import RightHandWaveMonitor
from pose import DEFAULT_MODEL_PATH, PoseAnalyzer
from review import ReviewLevelMonitor


EXPRESSION_INTERVAL_FRAMES = 5
EXPRESSION_SCORE_ALPHA = 0.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the combined pose, wave activity, face, and emotion demo.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index, video path, or RTSP/HTTP camera URL.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the MoveNet Lightning TFLite model.",
    )
    return parser.parse_args()


def normalize_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def run_pose_viewer(source: str, model_path: Path) -> None:
    analyzer = PoseAnalyzer(model_path=model_path)
    wave_monitor = RightHandWaveMonitor()
    review_monitor = ReviewLevelMonitor()
    emotion_analyzer = FaceEmotionAnalyzer()
    cached_emotion: FaceEmotionResult | None = None
    frame_count = 0

    capture = cv2.VideoCapture(normalize_source(source))
    if not capture.isOpened():
        emotion_analyzer.close()
        raise RuntimeError(f"Could not open video source: {source}")

    try:
        print("Combined live camera demo is running.")
        print("Press q to quit.")

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            timestamp = time.monotonic()
            result = analyzer.analyze(frame)
            wave_state = wave_monitor.update(result.landmarks, timestamp)

            person_box = analyzer.person_box(result.landmarks, frame)
            expression_event_counted = False
            if frame_count % EXPRESSION_INTERVAL_FRAMES == 0:
                detected_emotion = emotion_analyzer.analyze(frame, person_box)
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
                    confidence = (
                        EXPRESSION_SCORE_ALPHA * detected_emotion.confidence
                        + (1 - EXPRESSION_SCORE_ALPHA) * cached_emotion.confidence
                    )
                    cached_emotion = FaceEmotionResult(
                        box=detected_emotion.box,
                        keypoints=detected_emotion.keypoints,
                        label=detected_emotion.label,
                        confidence=confidence,
                    )
                else:
                    cached_emotion = detected_emotion

            review_state = review_monitor.update(wave_state, timestamp)
            annotated = analyzer.render(frame, result.landmarks, review_state.color)

            if person_box is not None:
                x1, y1, x2, y2 = person_box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), review_state.color, 3)
                status = (
                    f"Review level: {review_state.tier_label} "
                    f"| waves: {review_state.recent_wave_count} "
                    f"| expression cues: {review_state.recent_expression_event_count}"
                )
                cv2.putText(
                    annotated,
                    status,
                    (x1, max(25, y1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    review_state.color,
                    2,
                )

                if wave_state.wave_detected:
                    cv2.putText(
                        annotated,
                        "Right-hand wave counted",
                        (x1, min(frame.shape[0] - 15, y2 + 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        review_state.color,
                        2,
                    )
                elif expression_event_counted:
                    cv2.putText(
                        annotated,
                        "Sustained concern expression counted",
                        (x1, min(frame.shape[0] - 15, y2 + 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        review_state.color,
                        2,
                    )

            if cached_emotion is not None:
                fx1, fy1, fx2, fy2 = cached_emotion.box
                cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (255, 255, 0), 2)
                for keypoint in cached_emotion.keypoints:
                    cv2.circle(annotated, keypoint, 2, (0, 255, 255), -1)
                cv2.putText(
                    annotated,
                    f"Expression: {cached_emotion.label} {cached_emotion.confidence:.2f}",
                    (fx1, max(48, fy1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (255, 255, 0),
                    2,
                )

            cv2.imshow("Integrated Live Demo", annotated)
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        emotion_analyzer.close()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    run_pose_viewer(source=args.source, model_path=args.model)


if __name__ == "__main__":
    main()

"""
Entry point for the application.
The menu (enrollment.py) is the main interaction loop; detection runs as one of its states.
"""

from enrollment import main

if __name__ == "__main__":
    main()