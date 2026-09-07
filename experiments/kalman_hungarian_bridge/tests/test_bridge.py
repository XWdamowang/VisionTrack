"""Kalman predicted bbox 双路关联测试。"""

import unittest

import numpy as np

from experiments.kalman_cv_2d.kalman_filter import (
    ConstantVelocityKalmanFilter,
    KalmanFilterConfig,
)
from experiments.kalman_hungarian_bridge import (
    ExistingTrack,
    compare_association_routes,
    predicted_bbox_from_state,
)


def build_track(
    track_id: int = 1,
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 10.0, 20.0),
    state: tuple[float, float, float, float] = (5.0, 10.0, 2.0, 3.0),
) -> ExistingTrack:
    return ExistingTrack(
        track_id=track_id,
        last_bbox=bbox,
        kalman_filter=ConstantVelocityKalmanFilter(
            np.array(state, dtype=np.float64),
            KalmanFilterConfig(dt=1.0),
        ),
    )


class KalmanHungarianBridgeTests(unittest.TestCase):
    def test_predicted_bbox_coordinate_conversion_keeps_size(self) -> None:
        bbox = predicted_bbox_from_state(
            (10.0, 20.0, 30.0, 50.0),
            np.array([100.0, 200.0, 7.0, -3.0], dtype=np.float64),
        )

        self.assertEqual(bbox, (90.0, 185.0, 110.0, 215.0))

    def test_empty_detections(self) -> None:
        comparison = compare_association_routes((build_track(),), (), ())

        self.assertEqual(comparison.last_bbox_details.iou_matrix.shape, (1, 0))
        self.assertEqual(comparison.kalman_details.result.unmatched_tracks, (0,))

    def test_empty_tracks(self) -> None:
        comparison = compare_association_routes(
            (),
            (101,),
            ((0.0, 0.0, 10.0, 10.0),),
        )

        self.assertEqual(comparison.kalman_details.iou_matrix.shape, (0, 1))
        self.assertEqual(comparison.last_bbox_details.result.unmatched_detections, (0,))

    def test_consecutive_predict_during_missing_detections(self) -> None:
        track = build_track()

        predicted_bbox = track.predict_bbox(steps=4)

        self.assertEqual(predicted_bbox, (8.0, 12.0, 18.0, 32.0))
        np.testing.assert_allclose(track.kalman_filter.state, [13.0, 22.0, 2.0, 3.0])

    def test_gating_rejects_inaccurate_high_speed_prediction(self) -> None:
        track = build_track(
            bbox=(0.0, 0.0, 10.0, 10.0),
            state=(5.0, 5.0, 20.0, 0.0),
        )
        comparison = compare_association_routes(
            (track,),
            (101,),
            ((35.0, 0.0, 45.0, 10.0),),
        )

        self.assertEqual(comparison.kalman_details.raw_matches, ((0, 0),))
        self.assertEqual(comparison.kalman_details.result.matches, ())
        self.assertEqual(comparison.kalman_details.result.unmatched_tracks, (0,))
        self.assertEqual(comparison.kalman_details.result.unmatched_detections, (0,))

    def test_last_bbox_fails_while_kalman_prediction_succeeds(self) -> None:
        track = build_track(
            bbox=(0.0, 0.0, 10.0, 10.0),
            state=(5.0, 5.0, 8.0, 0.0),
        )
        comparison = compare_association_routes(
            (track,),
            (101,),
            ((8.0, 0.0, 18.0, 10.0),),
        )

        self.assertAlmostEqual(comparison.last_bbox_details.iou_matrix[0, 0], 1.0 / 9.0)
        self.assertAlmostEqual(comparison.kalman_details.iou_matrix[0, 0], 1.0)
        self.assertEqual(comparison.last_bbox_details.result.matches, ())
        self.assertEqual(comparison.kalman_details.result.matches, ((0, 0),))


if __name__ == "__main__":
    unittest.main()
