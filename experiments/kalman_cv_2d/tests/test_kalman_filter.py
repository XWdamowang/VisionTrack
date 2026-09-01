"""二维匀速卡尔曼滤波器的单元测试。"""

import unittest

import numpy as np

from experiments.kalman_cv_2d.kalman_filter import (
    ConstantVelocityKalmanFilter,
    KalmanFilterConfig,
)
from experiments.kalman_cv_2d.run_experiment import (
    TrackMeasurement,
    select_longest_complete_track,
)


class ConstantVelocityKalmanFilterTests(unittest.TestCase):
    def test_complete_track_selection_rejects_tracks_with_gaps(self) -> None:
        tracks = {
            1: [
                TrackMeasurement(1, 1, 0.0, 0.0),
                TrackMeasurement(2, 1, 1.0, 0.0),
                TrackMeasurement(3, 1, 2.0, 0.0),
                TrackMeasurement(4, 1, 3.0, 0.0),
                TrackMeasurement(6, 1, 5.0, 0.0),
                TrackMeasurement(7, 1, 6.0, 0.0),
            ],
            2: [
                TrackMeasurement(frame_id, 2, float(frame_id), 0.0)
                for frame_id in range(10, 15)
            ],
        }

        selected = select_longest_complete_track(tracks)

        self.assertEqual(selected[0].track_id, 2)

    def test_transition_and_observation_matrices(self) -> None:
        kalman_filter = ConstantVelocityKalmanFilter(
            np.array([10.0, 20.0, 3.0, -2.0]),
            KalmanFilterConfig(dt=0.5),
        )

        np.testing.assert_allclose(
            kalman_filter.transition_matrix,
            np.array(
                [
                    [1.0, 0.0, 0.5, 0.0],
                    [0.0, 1.0, 0.0, 0.5],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )
        np.testing.assert_allclose(
            kalman_filter.observation_matrix,
            np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        )

    def test_predict_follows_constant_velocity(self) -> None:
        kalman_filter = ConstantVelocityKalmanFilter(
            np.array([10.0, 20.0, 3.0, -2.0]),
            KalmanFilterConfig(dt=0.5),
        )
        predicted = kalman_filter.predict()
        np.testing.assert_allclose(predicted, np.array([11.5, 19.0, 3.0, -2.0]))

    def test_update_moves_prediction_toward_measurement(self) -> None:
        kalman_filter = ConstantVelocityKalmanFilter(
            np.array([0.0, 0.0, 1.0, 0.0]),
            KalmanFilterConfig(dt=1.0),
        )
        prediction = kalman_filter.predict()
        updated = kalman_filter.update(np.array([3.0, 0.0]))
        self.assertGreater(updated[0], prediction[0])
        self.assertLess(updated[0], 3.0)

    def test_missing_observation_can_use_prediction_without_update(self) -> None:
        kalman_filter = ConstantVelocityKalmanFilter(
            np.array([5.0, 7.0, 2.0, 1.0]),
            KalmanFilterConfig(dt=0.25),
        )
        first_missing_prediction = kalman_filter.predict()
        second_missing_prediction = kalman_filter.predict()
        np.testing.assert_allclose(first_missing_prediction[:2], [5.5, 7.25])
        np.testing.assert_allclose(second_missing_prediction[:2], [6.0, 7.5])

    def test_invalid_dt_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "dt"):
            KalmanFilterConfig(dt=0.0)


if __name__ == "__main__":
    unittest.main()
