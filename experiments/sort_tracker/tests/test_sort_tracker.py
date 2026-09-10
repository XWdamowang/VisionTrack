import unittest

from experiments.sort_tracker import SortTrackResult, SortTracker, SortTrackerConfig
from src.detection import BBoxXYXY, Detection


def _detection(name: str, bbox: BBoxXYXY) -> Detection:
    return Detection(0, name, 0.90, bbox)


def _observed_ids(results: tuple[SortTrackResult, ...]) -> dict[str, int]:
    return {
        result.detection.class_name: result.track_id
        for result in results
        if result.detection is not None
    }


class SortTrackerTests(unittest.TestCase):
    def test_sort_id_creation_retention_and_deletion(self) -> None:
        tracker = SortTracker(
            SortTrackerConfig(iou_threshold=0.30, max_missed_frames=2)
        )

        first = tracker.update((_detection("A", (0.0, 0.0, 20.0, 20.0)),))
        a_id = _observed_ids(first)["A"]
        second = tracker.update((_detection("A", (2.0, 0.0, 22.0, 20.0)),))
        self.assertEqual(_observed_ids(second)["A"], a_id)

        third = tracker.update(
            (
                _detection("A", (4.0, 0.0, 24.0, 20.0)),
                _detection("B", (60.0, 0.0, 80.0, 20.0)),
            )
        )
        b_id = _observed_ids(third)["B"]
        self.assertNotEqual(b_id, a_id)

        missed_once = tracker.update(
            (_detection("B", (62.0, 0.0, 82.0, 20.0)),)
        )
        predicted_a = next(item for item in missed_once if item.track_id == a_id)
        self.assertTrue(predicted_a.is_predicted)
        self.assertEqual(predicted_a.missed_frames, 1)

        recovered = tracker.update(
            (
                _detection("A", (8.0, 0.0, 28.0, 20.0)),
                _detection("B", (64.0, 0.0, 84.0, 20.0)),
            )
        )
        self.assertEqual(_observed_ids(recovered)["A"], a_id)

        tracker.update((_detection("A", (10.0, 0.0, 30.0, 20.0)),))
        tracker.update((_detection("A", (12.0, 0.0, 32.0, 20.0)),))
        self.assertIn(b_id, tracker.active_track_ids)
        tracker.update((_detection("A", (14.0, 0.0, 34.0, 20.0)),))
        self.assertNotIn(b_id, tracker.active_track_ids)

    def test_iou_gating_creates_a_new_id(self) -> None:
        tracker = SortTracker(SortTrackerConfig(iou_threshold=0.50))
        first_id = tracker.update(
            (_detection("A", (0.0, 0.0, 10.0, 10.0)),)
        )[0].track_id
        results = tracker.update(
            (_detection("A", (100.0, 0.0, 110.0, 10.0)),)
        )
        observed_id = _observed_ids(results)["A"]
        self.assertNotEqual(observed_id, first_id)

    def test_reset_restarts_track_ids(self) -> None:
        tracker = SortTracker()
        first = tracker.update((_detection("A", (0.0, 0.0, 10.0, 10.0)),))
        self.assertEqual(first[0].track_id, 1)
        tracker.reset()
        reset_first = tracker.update(
            (_detection("B", (20.0, 0.0, 30.0, 10.0)),)
        )
        self.assertEqual(reset_first[0].track_id, 1)

    def test_seven_dimensional_state_predicts_scale(self) -> None:
        tracker = SortTracker(
            SortTrackerConfig(iou_threshold=0.10, max_age=2, min_hits=2)
        )
        tracker.update((_detection("A", (0.0, 0.0, 10.0, 10.0)),))
        tracker.update((_detection("A", (0.0, 0.0, 12.0, 12.0)),))
        third = tracker.update((_detection("A", (0.0, 0.0, 14.0, 14.0)),))[0]

        self.assertGreater(third.v_area, 0.0)
        self.assertTrue(third.is_confirmed)
        missed = tracker.update(())[0]
        self.assertGreater(missed.area, third.area)
        self.assertEqual(missed.bbox_xyxy, missed.predicted_bbox_xyxy)
        self.assertGreater(
            missed.bbox_xyxy[2] - missed.bbox_xyxy[0],
            third.bbox_xyxy[2] - third.bbox_xyxy[0],
        )

    def test_standard_lifecycle_fields_and_max_age(self) -> None:
        tracker = SortTracker(SortTrackerConfig(min_hits=3, max_age=1))
        first = tracker.update((_detection("A", (0.0, 0.0, 10.0, 10.0)),))[0]
        self.assertEqual((first.age, first.hits, first.hit_streak), (1, 1, 1))
        self.assertFalse(first.is_confirmed)

        second = tracker.update((_detection("A", (1.0, 0.0, 11.0, 10.0)),))[0]
        self.assertEqual((second.age, second.hits, second.hit_streak), (2, 2, 2))
        self.assertFalse(second.is_confirmed)
        third = tracker.update((_detection("A", (2.0, 0.0, 12.0, 10.0)),))[0]
        self.assertTrue(third.is_confirmed)
        self.assertEqual(tracker.confirmed_track_ids, (third.track_id,))

        missed = tracker.update(())[0]
        self.assertEqual(missed.time_since_update, 1)
        self.assertEqual(missed.hit_streak, 0)
        self.assertEqual(missed.missed_frames, missed.time_since_update)
        self.assertEqual(tracker.update(()), ())

    def test_boundary_configs_are_valid(self) -> None:
        configs = (
            SortTrackerConfig(iou_threshold=0.0),
            SortTrackerConfig(iou_threshold=1.0),
            SortTrackerConfig(max_missed_frames=0),
        )
        for config in configs:
            with self.subTest(config=config):
                SortTracker(config)

    def test_invalid_config_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "IoU"):
            SortTrackerConfig(iou_threshold=1.1)
        with self.assertRaisesRegex(ValueError, "丢失"):
            SortTrackerConfig(max_missed_frames=-1)
        with self.assertRaisesRegex(ValueError, "dt"):
            SortTrackerConfig(dt=0.0)
        with self.assertRaisesRegex(ValueError, "max_age"):
            SortTrackerConfig(max_age=-1)
        with self.assertRaisesRegex(ValueError, "min_hits"):
            SortTrackerConfig(min_hits=0)


if __name__ == "__main__":
    unittest.main()
