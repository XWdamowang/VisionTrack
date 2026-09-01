"""Hungarian 分配和 IoU gating 测试。"""

import unittest

import numpy as np

from experiments.hungarian_assignment import (
    HungarianAssignmentConfig,
    apply_iou_gating,
    associate_by_iou,
    solve_hungarian,
)


class HungarianAssignmentTests(unittest.TestCase):
    def test_rectangular_assignment_reports_extra_detection(self) -> None:
        details = associate_by_iou(
            ((0.0, 0.0, 10.0, 10.0), (20.0, 0.0, 30.0, 10.0)),
            (
                (0.0, 0.0, 10.0, 10.0),
                (20.0, 0.0, 30.0, 10.0),
                (40.0, 0.0, 50.0, 10.0),
            ),
        )

        self.assertEqual(details.result.matches, ((0, 0), (1, 1)))
        self.assertEqual(details.result.unmatched_tracks, ())
        self.assertEqual(details.result.unmatched_detections, (2,))

    def test_empty_inputs_are_supported(self) -> None:
        no_tracks = associate_by_iou((), ((0.0, 0.0, 10.0, 10.0),))
        no_detections = associate_by_iou(((0.0, 0.0, 10.0, 10.0),), ())

        self.assertEqual(no_tracks.result.unmatched_detections, (0,))
        self.assertEqual(no_detections.result.unmatched_tracks, (0,))

    def test_gating_rejects_low_iou_match_on_both_sides(self) -> None:
        result = apply_iou_gating(
            ((0, 0),),
            np.array([[0.20]], dtype=np.float64),
            iou_threshold=0.30,
        )

        self.assertEqual(result.matches, ())
        self.assertEqual(result.unmatched_tracks, (0,))
        self.assertEqual(result.unmatched_detections, (0,))

    def test_hungarian_beats_local_greedy_example(self) -> None:
        details = associate_by_iou(
            ((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 8.0, 10.0)),
            ((0.0, 0.0, 9.0, 10.0), (2.0, 0.0, 10.0, 10.0)),
        )
        greedy_matches = ((0, 0), (1, 1))
        greedy_total_iou = sum(details.iou_matrix[pair] for pair in greedy_matches)
        hungarian_total_iou = sum(details.iou_matrix[pair] for pair in details.raw_matches)

        self.assertEqual(details.raw_matches, ((0, 1), (1, 0)))
        self.assertGreater(hungarian_total_iou, greedy_total_iou)

    def test_solver_rejects_non_matrix_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "二维"):
            solve_hungarian(np.array([1.0, 2.0]))

    def test_invalid_gating_threshold_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "阈值"):
            HungarianAssignmentConfig(iou_threshold=1.1)


if __name__ == "__main__":
    unittest.main()
