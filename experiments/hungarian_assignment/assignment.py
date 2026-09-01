"""使用 SciPy 匈牙利算法和 IoU gating 关联轨迹与检测。"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .cost_matrix import BBoxInput, FloatMatrix, build_iou_cost_matrices


Match = tuple[int, int]


@dataclass(frozen=True)
class HungarianAssignmentConfig:
    """Hungarian 原始分配完成后的 IoU 门控配置。"""

    iou_threshold: float = 0.30

    def __post_init__(self) -> None:
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("IoU gating 阈值必须在 0 到 1 之间。")


@dataclass(frozen=True)
class AssociationResult:
    """门控后的统一关联结果，索引均对应输入矩阵的行或列。"""

    matches: tuple[Match, ...]
    unmatched_tracks: tuple[int, ...]
    unmatched_detections: tuple[int, ...]


@dataclass(frozen=True)
class AssociationDetails:
    """用于学习和可视化一次关联过程的完整中间结果。"""

    iou_matrix: FloatMatrix
    cost_matrix: FloatMatrix
    raw_matches: tuple[Match, ...]
    result: AssociationResult


def solve_hungarian(cost_matrix: FloatMatrix) -> tuple[Match, ...]:
    """求解矩形代价矩阵的全局最小总代价一对一分配。"""
    matrix = np.asarray(cost_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("cost_matrix 必须是二维矩阵。")
    if not np.isfinite(matrix).all():
        raise ValueError("cost_matrix 不能包含 NaN 或无穷大。")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return ()
    track_indices, detection_indices = linear_sum_assignment(matrix)
    return tuple(
        (int(track_index), int(detection_index))
        for track_index, detection_index in zip(track_indices, detection_indices)
    )


def apply_iou_gating(
    raw_matches: tuple[Match, ...],
    iou_matrix: FloatMatrix,
    iou_threshold: float,
) -> AssociationResult:
    """拒绝低 IoU 原始匹配，并把双方重新放入未匹配集合。"""
    config = HungarianAssignmentConfig(iou_threshold=iou_threshold)
    matrix = np.asarray(iou_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("iou_matrix 必须是二维矩阵。")

    valid_matches = tuple(
        (track_index, detection_index)
        for track_index, detection_index in raw_matches
        if matrix[track_index, detection_index] >= config.iou_threshold
    )
    matched_tracks = {track_index for track_index, _ in valid_matches}
    matched_detections = {detection_index for _, detection_index in valid_matches}
    return AssociationResult(
        matches=valid_matches,
        unmatched_tracks=tuple(
            index for index in range(matrix.shape[0]) if index not in matched_tracks
        ),
        unmatched_detections=tuple(
            index for index in range(matrix.shape[1]) if index not in matched_detections
        ),
    )


def associate_by_iou(
    predicted_tracks_bboxes: BBoxInput,
    detection_bboxes: BBoxInput,
    config: HungarianAssignmentConfig | None = None,
) -> AssociationDetails:
    """构造代价、执行 Hungarian，并用可配置 IoU 阈值过滤结果。"""
    assignment_config = config or HungarianAssignmentConfig()
    iou_matrix, cost_matrix = build_iou_cost_matrices(
        predicted_tracks_bboxes,
        detection_bboxes,
    )
    raw_matches = solve_hungarian(cost_matrix)
    result = apply_iou_gating(
        raw_matches,
        iou_matrix,
        assignment_config.iou_threshold,
    )
    return AssociationDetails(
        iou_matrix=iou_matrix,
        cost_matrix=cost_matrix,
        raw_matches=raw_matches,
        result=result,
    )
