import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.annotation_model import Box, VideoAnnotation
from src.nonaccident_pipeline import repair_annotation_metadata, validate_annotation_integrity


class IntegrityTests(unittest.TestCase):
    def make_video(self, directory: Path, name: str = "demo.mp4") -> Path:
        path = directory / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
        for _ in range(3):
            writer.write(255 * __import__("numpy").ones((48, 64, 3), dtype="uint8"))
        writer.release()
        return path

    def make_annotation(self, video: Path) -> VideoAnnotation:
        return VideoAnnotation(
            "demo", str(video), "normal", 64, 48, 99, 25.0,
            boxes=[Box(0, [1, 1, 20, 20], 0), Box(1, [25, 1, 45, 20], 0)],
            status="normal",
        )

    def test_integrity_reports_media_metadata_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            video = self.make_video(Path(directory))
            errors = validate_annotation_integrity(self.make_annotation(video))
            self.assertTrue(any("frame_count" in error for error in errors))
            self.assertTrue(any("fps" in error for error in errors))

    def test_repair_metadata_updates_media_fields_only(self):
        with tempfile.TemporaryDirectory() as directory:
            video = self.make_video(Path(directory))
            annotation = self.make_annotation(video)
            before_boxes = json.dumps([box.__dict__ for box in annotation.boxes], sort_keys=True)
            before_status = annotation.status
            changed = repair_annotation_metadata(annotation)
            self.assertTrue(changed)
            self.assertEqual((annotation.width, annotation.height, annotation.frame_count), (64, 48, 3))
            self.assertAlmostEqual(annotation.fps, 10.0, places=2)
            self.assertEqual(json.dumps([box.__dict__ for box in annotation.boxes], sort_keys=True), before_boxes)
            self.assertEqual(annotation.status, before_status)
            self.assertEqual(validate_annotation_integrity(annotation), [])

    def test_repair_metadata_clamps_out_of_range_reference_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            video = self.make_video(Path(directory))
            annotation = self.make_annotation(video)
            annotation.boxes[0].reference_frame = 99
            repair_annotation_metadata(annotation)
            self.assertEqual(annotation.boxes[0].reference_frame, 2)
            self.assertEqual(validate_annotation_integrity(annotation), [])


if __name__ == "__main__":
    unittest.main()
