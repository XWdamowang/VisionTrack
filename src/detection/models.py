"""检测模块与下游模块共享的数据结构。"""

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


ImageArray: TypeAlias = NDArray[np.uint8]
BBoxXYXY: TypeAlias = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    """单个目标在一帧中的检测结果。"""

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: BBoxXYXY

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox_xyxy
        if self.class_id < 0:
            raise ValueError("类别 ID 不能为负数。")
        if not self.class_name:
            raise ValueError("类别名称不能为空。")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("检测置信度必须在 0 到 1 之间。")
        if x1 >= x2 or y1 >= y2:
            raise ValueError("检测框必须满足 x1 < x2 且 y1 < y2。")


@dataclass(frozen=True)
class FrameDetections:
    """一帧图像及其全部检测结果。"""

    frame_number: int
    timestamp_seconds: float
    source_fps: float
    frame: ImageArray
    annotated_frame: ImageArray
    detections: tuple[Detection, ...]

    def __post_init__(self) -> None:
        if self.frame_number < 1:
            raise ValueError("帧号必须从 1 开始。")
        if self.timestamp_seconds < 0:
            raise ValueError("时间戳不能为负数。")
        if self.source_fps <= 0:
            raise ValueError("视频帧率必须大于 0。")
