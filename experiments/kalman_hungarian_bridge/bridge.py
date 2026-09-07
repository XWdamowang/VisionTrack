"""对比 last bbox 与 Kalman predicted bbox 的 Hungarian 关联输入。"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from experiments.hungarian_assignment import (
    AssociationDetails,
    HungarianAssignmentConfig,
    associate_by_iou,
)
from experiments.kalman_cv_2d.kalman_filter import ConstantVelocityKalmanFilter
from src.detection import BBoxXYXY


FloatArray = NDArray[np.float64]


def predicted_bbox_from_state(
    last_bbox: BBoxXYXY,
    predicted_state: FloatArray,
) -> BBoxXYXY:
    """保留 last bbox 尺寸，并以预测状态的中心坐标重建 xyxy 框。"""
    state = np.asarray(predicted_state, dtype=np.float64)
    if state.shape != (4,):
        raise ValueError("预测状态必须是形状为 (4,) 的 [x, y, vx, vy]。")
    x1, y1, x2, y2 = last_bbox
    if x1 >= x2 or y1 >= y2:
        raise ValueError("last bbox 必须满足 x1 < x2 且 y1 < y2。")
    half_width = (x2 - x1) / 2.0
    half_height = (y2 - y1) / 2.0
    center_x, center_y = float(state[0]), float(state[1])
    return (
        center_x - half_width,
        center_y - half_height,
        center_x + half_width,
        center_y + half_height,
    )


@dataclass
class ExistingTrack:
    """实验开始前已经存在的固定轨迹，不包含生命周期管理。"""

    track_id: int
    last_bbox: BBoxXYXY
    kalman_filter: ConstantVelocityKalmanFilter

    def predict_bbox(self, steps: int = 1) -> BBoxXYXY:
        if steps < 1:
            raise ValueError("预测步数必须至少为 1。")
        predicted_state = self.kalman_filter.state.copy()
        for _ in range(steps):
            predicted_state = self.kalman_filter.predict()
        return predicted_bbox_from_state(self.last_bbox, predicted_state)


@dataclass(frozen=True)
class BridgeComparison:
    """同一批轨迹和检测在两条关联路线上的完整结果。"""

    track_ids: tuple[int, ...]
    detection_ids: tuple[int, ...]
    last_bboxes: tuple[BBoxXYXY, ...]
    predicted_bboxes: tuple[BBoxXYXY, ...]
    last_bbox_details: AssociationDetails
    kalman_details: AssociationDetails


def compare_association_routes(
    tracks: Sequence[ExistingTrack],
    detection_ids: Sequence[int],
    detection_bboxes: Sequence[BBoxXYXY],
    *,
    prediction_steps: int = 1,
    iou_threshold: float = 0.30,
) -> BridgeComparison:
    """对完全相同的检测分别执行 last bbox 和 Kalman bbox 关联。"""
    if len(detection_ids) != len(detection_bboxes):
        raise ValueError("detection_ids 与 detection_bboxes 数量必须一致。")
    config = HungarianAssignmentConfig(iou_threshold=iou_threshold)
    last_bboxes = tuple(track.last_bbox for track in tracks)
    predicted_bboxes = tuple(
        track.predict_bbox(prediction_steps) for track in tracks
    )
    return BridgeComparison(
        track_ids=tuple(track.track_id for track in tracks),
        detection_ids=tuple(detection_ids),
        last_bboxes=last_bboxes,
        predicted_bboxes=predicted_bboxes,
        last_bbox_details=associate_by_iou(last_bboxes, detection_bboxes, config),
        kalman_details=associate_by_iou(predicted_bboxes, detection_bboxes, config),
    )
