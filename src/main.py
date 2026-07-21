"""Application entrypoint for the integrated monitoring demo."""

from __future__ import annotations

import argparse
from dataclasses import replace

from camera import normalize_camera_source
from config import AppConfig


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
    mode.add_argument(
        "--validate",
        action="store_true",
        help="Run recorded validation scenarios from a JSON manifest.",
    )
    mode.add_argument(
        "--doctor",
        action="store_true",
        help="Check package, model, and filesystem readiness.",
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
    parser.add_argument(
        "--manifest",
        default="validation/manifest.json",
        help="Validation manifest used with --validate.",
    )
    parser.add_argument(
        "--report",
        default="validation/reports/latest.json",
        help="Validation report path used with --validate.",
    )
    parser.add_argument(
        "--prepare-models",
        action="store_true",
        help="Download and verify models while running --doctor.",
    )
    parser.add_argument(
        "--no-model-downloads",
        action="store_true",
        help="Run without downloading missing models.",
    )
    parser.add_argument(
        "--no-events",
        action="store_true",
        help="Disable JSONL operator event reports.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Override the MONITOR_LOG_LEVEL setting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig.from_env()
    config = replace(
        config,
        allow_model_downloads=(
            False if args.no_model_downloads else config.allow_model_downloads
        ),
        event_logging_enabled=(
            False if args.no_events else config.event_logging_enabled
        ),
        log_level=args.log_level or config.log_level,
    )
    if args.doctor:
        from diagnostics import print_diagnostics, run_diagnostics

        checks = run_diagnostics(config, prepare_models=args.prepare_models)
        print_diagnostics(checks)
        return
    if args.validate:
        from pathlib import Path

        from validation import print_validation_summary, run_validation

        results = run_validation(Path(args.manifest), Path(args.report), config)
        print_validation_summary(results)
        if not all(result.passed for result in results):
            raise SystemExit(1)
        return
    if args.process_media:
        from pathlib import Path

        from media_export import export_media

        export_media(Path(args.input), Path(args.output_dir), config=config)
        return
    if args.detect:
        from detection import run_detection

        run_detection(source=normalize_camera_source(args.source), config=config)
        return

    from enrollment import main as menu_main

    menu_main()


if __name__ == "__main__":
    main()
