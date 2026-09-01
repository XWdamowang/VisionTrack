"""仅使用边界框 IoU 的轻量多目标跟踪学习实现。"""

from dataclasses import dataclass

from src.detection import BBoxXYXY, Detection


@dataclass(frozen=True)
class IouTrackerConfig:
    """IoU 关联和轨迹生命周期配置。"""

    iou_threshold: float = 0.30
    max_missed_frames: int = 2
    class_aware: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("跟踪 IoU 阈值必须在 0 到 1 之间。")
        if self.max_missed_frames < 0:
            raise ValueError("最大丢失帧数不能为负数。")


@dataclass(frozen=True)
class TrackedDetection:
    """已分配轨迹 ID 的当前帧检测结果。"""

    track_id: int
    detection: Detection


@dataclass
class _TrackState:
    track_id: int
    detection: Detection
    missed_frames: int = 0


def calculate_iou(first: BBoxXYXY, second: BBoxXYXY) -> float:
    """计算两个 xyxy 边界框的交并比。"""
    first_x1, first_y1, first_x2, first_y2 = first
    second_x1, second_y1, second_x2, second_y2 = second
    intersection_width = max(0.0, min(first_x2, second_x2) - max(first_x1, second_x1))
    intersection_height = max(0.0, min(first_y2, second_y2) - max(first_y1, second_y1))
    intersection_area = intersection_width * intersection_height
    if intersection_area == 0.0:
        return 0.0

    first_area = (first_x2 - first_x1) * (first_y2 - first_y1)
    second_area = (second_x2 - second_x1) * (second_y2 - second_y1)
    return intersection_area / (first_area + second_area - intersection_area)


class IouTracker:
    """通过当前检测框与上一轨迹框的 IoU 贪心关联目标。"""

    def __init__(self, config: IouTrackerConfig | None = None) -> None:
        self.config = config or IouTrackerConfig()
        self._tracks: dict[int, _TrackState] = {}
        self._next_track_id = 1

    def update(self, detections: tuple[Detection, ...]) -> tuple[TrackedDetection, ...]:
        """关联一帧检测结果，并按检测顺序返回轨迹 ID。"""
        candidates: list[tuple[float, int, int]] = []
        for track_id, state in self._tracks.items():
            for detection_index, detection in enumerate(detections):
                if self.config.class_aware and (
                    state.detection.class_id != detection.class_id
                ):
                    continue
                iou = calculate_iou(
                    state.detection.bbox_xyxy,
                    detection.bbox_xyxy,
                )
                if iou >= self.config.iou_threshold:
                    candidates.append((iou, track_id, detection_index))

        matched_track_ids: set[int] = set()
        matched_detection_indices: set[int] = set()
        assignments: dict[int, int] = {}
        for _, track_id, detection_index in sorted(candidates, reverse=True):
            if track_id in matched_track_ids:
                continue
            if detection_index in matched_detection_indices:
                continue
            matched_track_ids.add(track_id)
            matched_detection_indices.add(detection_index)
            assignments[detection_index] = track_id

        for track_id, state in list(self._tracks.items()):
            if track_id in matched_track_ids:
                continue
            state.missed_frames += 1
            if state.missed_frames > self.config.max_missed_frames:
                del self._tracks[track_id]

        tracked_detections: list[TrackedDetection] = []
        for detection_index, detection in enumerate(detections):
            track_id = assignments.get(detection_index)
            if track_id is None:
                track_id = self._next_track_id
                self._next_track_id += 1
                self._tracks[track_id] = _TrackState(track_id, detection)
            else:
                state = self._tracks[track_id]
                state.detection = detection
                state.missed_frames = 0
            tracked_detections.append(TrackedDetection(track_id, detection))

        return tuple(tracked_detections)

    def reset(self) -> None:
        """清空全部轨迹并从 ID 1 重新开始。"""
        self._tracks.clear()
        self._next_track_id = 1
