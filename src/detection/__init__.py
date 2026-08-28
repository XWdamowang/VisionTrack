"""目标检测模块。"""

from .models import BBoxXYXY, Detection, FrameDetections, ImageArray
from .video_detector import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DetectionConfig,
    DetectionSummary,
    VideoDetector,
    iter_video_detections,
    run_video_detection,
)

__all__ = [
    "BBoxXYXY",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_IOU_THRESHOLD",
    "Detection",
    "DetectionConfig",
    "DetectionSummary",
    "FrameDetections",
    "ImageArray",
    "VideoDetector",
    "iter_video_detections",
    "run_video_detection",
]
