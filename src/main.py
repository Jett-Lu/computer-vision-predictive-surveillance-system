"""Application entrypoint for the integrated monitoring demo."""

from __future__ import annotations

import argparse

from camera import normalize_camera_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrated live monitoring demo.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--detect",
        action="store_true",
        help="Start live monitoring directly instead of opening the menu.",
    )
    mode.add_argument(
        "--process-media",
        action="store_true",
        help="Write annotated photos or videos from the input folder to the output folder.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index, video path, or stream URL used with --detect.",
    )
    parser.add_argument(
        "--input",
        default="input",
        help="Photo, video, or folder used with --process-media. Default: input",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Folder for annotated files created with --process-media. Default: output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.process_media:
        from pathlib import Path

        from media_export import export_media

        export_media(Path(args.input), Path(args.output_dir))
        return
    if args.detect:
        from detection import run_detection

        run_detection(source=normalize_camera_source(args.source))
        return

    from enrollment import main as menu_main

    menu_main()


if __name__ == "__main__":
    main()
