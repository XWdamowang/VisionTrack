"""匈牙利算法与 IoU 代价矩阵学习实验。"""

from .assignment import (
    AssociationDetails,
    AssociationResult,
    HungarianAssignmentConfig,
    apply_iou_gating,
    associate_by_iou,
    solve_hungarian,
)
from .cost_matrix import build_iou_cost_matrices

__all__ = [
    "AssociationDetails",
    "AssociationResult",
    "HungarianAssignmentConfig",
    "apply_iou_gating",
    "associate_by_iou",
    "build_iou_cost_matrices",
    "solve_hungarian",
]
