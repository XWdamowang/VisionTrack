"""目标跟踪模块的公共契约，跟踪算法将在后续阶段实现。"""

from .contracts import DEFAULT_TRACKER_CONFIG, TRACK_CSV_FIELDS

__all__ = ["DEFAULT_TRACKER_CONFIG", "TRACK_CSV_FIELDS"]
