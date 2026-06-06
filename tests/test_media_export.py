from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from media_export import annotated_output_path, discover_media_files


class MediaExportHelpersTest(unittest.TestCase):
    def test_discover_media_files_returns_supported_files_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            (folder / "notes.txt").touch()
            (folder / "clip.mp4").touch()
            (folder / "photo.jpg").touch()

            self.assertEqual(
                discover_media_files(folder),
                [folder / "clip.mp4", folder / "photo.jpg"],
            )

    def test_annotated_output_path_keeps_image_format(self) -> None:
        self.assertEqual(
            annotated_output_path(Path("input/photo.png"), Path("output")),
            Path("output/photo_annotated.png"),
        )

    def test_annotated_output_path_converts_video_to_mp4(self) -> None:
        self.assertEqual(
            annotated_output_path(Path("input/clip.mov"), Path("output")),
            Path("output/clip_annotated.mp4"),
        )


if __name__ == "__main__":
    unittest.main()
