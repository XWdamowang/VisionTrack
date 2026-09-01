"""IoU 与代价矩阵构造测试。"""

import unittest

import numpy as np

from experiments.hungarian_assignment import build_iou_cost_matrices


class CostMatrixTests(unittest.TestCase):
    def test_cost_is_one_minus_iou(self) -> None:
        iou_matrix, cost_matrix = build_iou_cost_matrices(
            ((0.0, 0.0, 10.0, 10.0), (20.0, 0.0, 30.0, 10.0)),
            ((0.0, 0.0, 10.0, 10.0), (5.0, 0.0, 15.0, 10.0)),
        )

        np.testing.assert_allclose(
            iou_matrix,
            np.array([[1.0, 1.0 / 3.0], [0.0, 0.0]]),
        )
        np.testing.assert_allclose(cost_matrix, 1.0 - iou_matrix)

    def test_empty_inputs_keep_rectangular_shapes(self) -> None:
        no_tracks_iou, no_tracks_cost = build_iou_cost_matrices(
            (),
            ((0.0, 0.0, 10.0, 10.0),),
        )
        no_detections_iou, no_detections_cost = build_iou_cost_matrices(
            ((0.0, 0.0, 10.0, 10.0),),
            (),
        )

        self.assertEqual(no_tracks_iou.shape, (0, 1))
        self.assertEqual(no_tracks_cost.shape, (0, 1))
        self.assertEqual(no_detections_iou.shape, (1, 0))
        self.assertEqual(no_detections_cost.shape, (1, 0))

    def test_invalid_bbox_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "x1 < x2"):
            build_iou_cost_matrices(((10.0, 0.0, 0.0, 10.0),), ())


if __name__ == "__main__":
    unittest.main()
