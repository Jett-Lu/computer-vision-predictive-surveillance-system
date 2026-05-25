from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from gesture import RightHandWaveMonitor
from pose import DEFAULT_MODEL_PATH, PoseAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw a MoveNet Lightning body skeleton over a webcam or video feed.",
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
    capture = cv2.VideoCapture(normalize_source(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    analyzer = PoseAnalyzer(model_path=model_path)
    wave_monitor = RightHandWaveMonitor()

    try:
        print("MoveNet repeated-wave activity viewer is running.")
        print("Press q to quit.")

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            result = analyzer.analyze(frame)
            alert_state = wave_monitor.update(result.landmarks, time.monotonic())
            annotated = analyzer.render(frame, result.landmarks, alert_state.color)

            person_box = analyzer.person_box(result.landmarks, frame)
            if person_box is not None:
                x1, y1, x2, y2 = person_box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), alert_state.color, 3)
                status = (
                    f"Wave activity: {alert_state.tier_label} "
                    f"| recent waves: {alert_state.recent_wave_count}"
                )
                cv2.putText(
                    annotated,
                    status,
                    (x1, max(25, y1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    alert_state.color,
                    2,
                )

                if alert_state.wave_detected:
                    cv2.putText(
                        annotated,
                        "Right-hand wave counted",
                        (x1, min(frame.shape[0] - 15, y2 + 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        alert_state.color,
                        2,
                    )

            cv2.imshow("Body Recognition", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    run_pose_viewer(source=args.source, model_path=args.model)


if __name__ == "__main__":
    main()
