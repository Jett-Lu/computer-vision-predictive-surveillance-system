"""Export annotated monitoring results for photos and recorded videos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "input"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})
VIDEO_EXTENSIONS = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4"})
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
DEFAULT_VIDEO_FPS = 24.0


@dataclass(frozen=True)
class ExportResult:
    """Describe one successfully generated annotated media file."""

    input_path: Path
    output_path: Path
    media_type: str
    frames_processed: int


def discover_media_files(input_path: Path) -> list[Path]:
    """Return supported input files from one file path or one directory."""
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in SUPPORTED_EXTENSIONS else []
    if not input_path.exists():
        input_path.mkdir(parents=True, exist_ok=True)
    return sorted(
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def annotated_output_path(input_path: Path, output_dir: Path) -> Path:
    """Build a clear output filename for a photo or video."""
    suffix = (
        input_path.suffix.lower()
        if input_path.suffix.lower() in IMAGE_EXTENSIONS
        else ".mp4"
    )
    return output_dir / f"{input_path.stem}_annotated{suffix}"


def export_media(
    input_path: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[ExportResult]:
    """Process supported media inputs and return the successfully generated files."""
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    media_files = discover_media_files(input_path)
    if not media_files:
        print(f"No supported photos or videos found in {input_path}")
        print(f"Add a file and run the command again. Output will appear in {output_dir}")
        return []

    results: list[ExportResult] = []
    for media_path in media_files:
        output_path = annotated_output_path(media_path, output_dir)
        print(f"\nProcessing {media_path.name}...")
        try:
            if media_path.suffix.lower() in IMAGE_EXTENSIONS:
                result = _export_image(media_path, output_path)
            else:
                result = _export_video(media_path, output_path)
        except Exception as exc:
            print(f"Failed to process {media_path.name}: {exc}")
            continue
        results.append(result)
        print(f"Created {result.output_path}")

    print(f"\nCompleted {len(results)} of {len(media_files)} media files.")
    return results


def _export_image(input_path: Path, output_path: Path) -> ExportResult:
    image = cv2.imread(str(input_path))
    if image is None:
        raise RuntimeError("OpenCV could not read the image.")

    processor = _create_processor()
    try:
        annotated = processor.process_frame(image, timestamp=0.0)
    finally:
        processor.close()

    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError("OpenCV could not write the annotated image.")
    return ExportResult(input_path, output_path, "image", frames_processed=1)


def _export_video(input_path: Path, output_path: Path) -> ExportResult:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open the video.")

    writer: Any = None
    processor: Any = None
    frames_processed = 0
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = DEFAULT_VIDEO_FPS
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise RuntimeError("The video does not report a valid frame size.")

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not create the annotated MP4 output.")

        processor = _create_processor()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frames_processed / fps
            writer.write(processor.process_frame(frame, timestamp=timestamp))
            frames_processed += 1
            if frames_processed % 30 == 0:
                print(f"Processed {frames_processed} frames...")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if processor is not None:
            processor.close()

    if frames_processed == 0:
        if output_path.exists():
            output_path.unlink()
        raise RuntimeError("The video did not contain readable frames.")
    return ExportResult(input_path, output_path, "video", frames_processed)


def _create_processor() -> Any:
    from detection import MonitoringProcessor

    return MonitoringProcessor()
