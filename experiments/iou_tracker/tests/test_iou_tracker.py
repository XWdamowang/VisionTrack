"""轻量 IoU Tracker 的单元测试。"""

import unittest

from src.detection import Detection
from experiments.iou_tracker import (
    DEFAULT_TRACKER_CONFIG,
    TRACK_CSV_FIELDS,
    IouTracker,
    IouTrackerConfig,
    calculate_iou,
)


def make_detection(x1: float, x2: float, class_id: int = 0) -> Detection:
    return Detection(class_id, f"class-{class_id}", 0.9, (x1, 0.0, x2, 10.0))


class IouTrackerTests(unittest.TestCase):
    def test_tracking_output_contract_is_stable(self) -> None:
        self.assertEqual(DEFAULT_TRACKER_CONFIG, "iou")
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

    def test_iou_value(self) -> None:
        self.assertAlmostEqual(
            calculate_iou((0.0, 0.0, 10.0, 10.0), (5.0, 0.0, 15.0, 10.0)),
            1.0 / 3.0,
        )

    def test_overlapping_detection_keeps_id(self) -> None:
        tracker = IouTracker(IouTrackerConfig(iou_threshold=0.3))
        first = tracker.update((make_detection(0.0, 10.0),))
        second = tracker.update((make_detection(2.0, 12.0),))
        self.assertEqual(first[0].track_id, second[0].track_id)

    def test_fast_motion_creates_new_id(self) -> None:
        tracker = IouTracker(IouTrackerConfig(iou_threshold=0.3))
        first = tracker.update((make_detection(0.0, 10.0),))
        second = tracker.update((make_detection(20.0, 30.0),))
        self.assertNotEqual(first[0].track_id, second[0].track_id)

    def test_short_miss_keeps_track_available(self) -> None:
        tracker = IouTracker(IouTrackerConfig(iou_threshold=0.3, max_missed_frames=2))
        first = tracker.update((make_detection(0.0, 10.0),))
        tracker.update(())
        tracker.update(())
        resumed = tracker.update((make_detection(1.0, 11.0),))
        self.assertEqual(first[0].track_id, resumed[0].track_id)

    def test_class_aware_matching_does_not_cross_classes(self) -> None:
        tracker = IouTracker(IouTrackerConfig(class_aware=True))
        first = tracker.update((make_detection(0.0, 10.0, class_id=0),))
        second = tracker.update((make_detection(0.0, 10.0, class_id=1),))
        self.assertNotEqual(first[0].track_id, second[0].track_id)


if __name__ == "__main__":
    unittest.main()
