"""根据预测轨迹框和检测框构造 IoU 与代价矩阵。"""

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from experiments.iou_tracker.iou_tracker import calculate_iou
from src.detection import BBoxXYXY


FloatMatrix: TypeAlias = NDArray[np.float64]
BBoxInput: TypeAlias = Sequence[BBoxXYXY] | FloatMatrix


def _as_bbox_matrix(bboxes: BBoxInput, input_name: str) -> FloatMatrix:
    matrix = np.asarray(bboxes, dtype=np.float64)
    if matrix.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 4:
        raise ValueError(f"{input_name} 必须是形状为 (N, 4) 的 xyxy 边界框集合。")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{input_name} 不能包含 NaN 或无穷大。")
    if np.any(matrix[:, 0] >= matrix[:, 2]) or np.any(matrix[:, 1] >= matrix[:, 3]):
        raise ValueError(f"{input_name} 中的边界框必须满足 x1 < x2 且 y1 < y2。")
    return matrix


def build_iou_cost_matrices(
    predicted_tracks_bboxes: BBoxInput,
    detection_bboxes: BBoxInput,
) -> tuple[FloatMatrix, FloatMatrix]:
    """返回形状为 Tracks×Detections 的 IoU 矩阵和 `1 - IoU` 代价矩阵。"""
    tracks = _as_bbox_matrix(predicted_tracks_bboxes, "predicted_tracks_bboxes")
    detections = _as_bbox_matrix(detection_bboxes, "detection_bboxes")
    iou_matrix = np.empty((len(tracks), len(detections)), dtype=np.float64)
    for track_index, track_bbox in enumerate(tracks):
        for detection_index, detection_bbox in enumerate(detections):
            iou_matrix[track_index, detection_index] = calculate_iou(
                tuple(float(value) for value in track_bbox),
                tuple(float(value) for value in detection_bbox),
            )
    cost_matrix = 1.0 - iou_matrix
    return iou_matrix, cost_matrix
