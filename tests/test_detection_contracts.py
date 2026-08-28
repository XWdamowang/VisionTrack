"""检测与跟踪接入契约的单元测试。"""

import unittest

import numpy as np

from src.detection import Detection, DetectionConfig, FrameDetections
from src.tracking import DEFAULT_TRACKER_CONFIG, TRACK_CSV_FIELDS


class DetectionContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((8, 8, 3), dtype=np.uint8)
        self.detection = Detection(
            class_id=0,
            class_name="person",
            confidence=0.80,
            bbox_xyxy=(1.0, 2.0, 3.0, 4.0),
        )

    def test_frame_detections_contains_tracking_inputs(self) -> None:
        frame_result = FrameDetections(
            frame_number=1,
            timestamp_seconds=0.0,
            source_fps=29.97,
            frame=self.frame,
            annotated_frame=self.frame.copy(),
            detections=(self.detection,),
        )

        self.assertEqual(frame_result.frame_number, 1)
        self.assertEqual(frame_result.detections, (self.detection,))
        self.assertEqual(frame_result.frame.shape, (8, 8, 3))

    def test_invalid_detection_box_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "x1 < x2"):
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.80,
                bbox_xyxy=(3.0, 2.0, 1.0, 4.0),
            )

    def test_invalid_detection_threshold_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "置信度阈值"):
            DetectionConfig(confidence_threshold=1.01)

    def test_tracking_output_contract_is_stable(self) -> None:
        self.assertEqual(DEFAULT_TRACKER_CONFIG, "bytetrack.yaml")
        self.assertEqual(
            TRACK_CSV_FIELDS,
            (
                "frame_number",
                "timestamp_seconds",
                "track_id",
                "class_id",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
            ),
        )


if __name__ == "__main__":
    unittest.main()
