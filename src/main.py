"""Application entrypoint for the integrated monitoring demo."""

from __future__ import annotations

import argparse

from camera import normalize_camera_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrated live monitoring demo.")
    parser.add_argument(
        "--detect",
        action="store_true",
        help="Start live monitoring directly instead of opening the menu.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index, video path, or stream URL used with --detect.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.detect:
        from detection import run_detection

        run_detection(source=normalize_camera_source(args.source))
        return

    from enrollment import main as menu_main

    menu_main()


if __name__ == "__main__":
    main()
