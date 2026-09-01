"""轻量 IoU 目标跟踪学习实验。"""

from .contracts import DEFAULT_TRACKER_CONFIG, TRACK_CSV_FIELDS
from .iou_tracker import (
    IouTracker,
    IouTrackerConfig,
    TrackedDetection,
    calculate_iou,
)
from .video_tracker import TrackingSummary, draw_tracks, run_video_tracking

__all__ = [
    "DEFAULT_TRACKER_CONFIG",
    "TRACK_CSV_FIELDS",
    "IouTracker",
    "IouTrackerConfig",
    "TrackedDetection",
    "TrackingSummary",
    "calculate_iou",
    "draw_tracks",
    "run_video_tracking",
]
