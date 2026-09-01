"""IoU Tracker 学习实验的输出契约。"""

from typing import Final


DEFAULT_TRACKER_CONFIG: Final[str] = "iou"
TRACK_CSV_FIELDS: Final[tuple[str, ...]] = (
    "frame_number",
    "timestamp_seconds",
    "track_id",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
)
