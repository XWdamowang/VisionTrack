"""Kalman 预测框与 last bbox 的 Hungarian 关联对照实验。"""

from .bridge import (
    BridgeComparison,
    ExistingTrack,
    compare_association_routes,
    predicted_bbox_from_state,
)

__all__ = [
    "BridgeComparison",
    "ExistingTrack",
    "compare_association_routes",
    "predicted_bbox_from_state",
]
