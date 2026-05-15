from __future__ import annotations

import argparse
from pathlib import Path

import cv2

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

    try:
        print("MoveNet skeleton viewer is running.")
        print("Press q to quit.")

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            annotated = analyzer.analyze(frame)
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

