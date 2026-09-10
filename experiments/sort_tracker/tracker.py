"""复用现有 Hungarian 与 IoU gating 的标准 SORT 核心跟踪器。"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from experiments.hungarian_assignment import (
    HungarianAssignmentConfig,
    associate_by_iou,
)
from src.detection import BBoxXYXY, Detection


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SortTrackerConfig:
    """SORT 的关联、轨迹保留与运动模型配置。"""

    iou_threshold: float = 0.30
    max_missed_frames: int | None = None
    dt: float = 1.0
    max_age: int = 1
    min_hits: int = 3

    def __post_init__(self) -> None:
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("SORT IoU 阈值必须在 0 到 1 之间。")
        if self.max_missed_frames is not None and self.max_missed_frames < 0:
            raise ValueError("SORT 最大丢失帧数不能为负数。")
        if self.dt <= 0.0:
            raise ValueError("SORT 时间步长 dt 必须大于 0。")
        if self.max_age < 0:
            raise ValueError("SORT max_age 不能为负数。")
        if self.min_hits < 1:
            raise ValueError("SORT min_hits 必须至少为 1。")

    @property
    def effective_max_age(self) -> int:
        """兼容旧参数；显式 max_missed_frames 优先于 max_age。"""
        if self.max_missed_frames is not None:
            return self.max_missed_frames
        return self.max_age


@dataclass(frozen=True)
class SortTrackResult:
    """一帧更新后仍处于活动状态的轨迹快照。"""

    track_id: int
    bbox_xyxy: BBoxXYXY
    detection: Detection | None
    age: int
    hits: int
    hit_streak: int
    time_since_update: int
    area: float
    aspect_ratio: float
    v_area: float
    predicted_bbox_xyxy: BBoxXYXY
    is_confirmed: bool

    @property
    def missed_frames(self) -> int:
        """兼容旧接口，等价于标准 SORT 的 time_since_update。"""
        return self.time_since_update

    @property
    def is_predicted(self) -> bool:
        """当前框是否来自漏检后的纯 Kalman 预测。"""
        return self.detection is None


def _bbox_to_sort_measurement(bbox: BBoxXYXY) -> tuple[float, float, float, float]:
    """将 xyxy 框转换为 SORT 常用的 [cx, cy, area, aspect_ratio]。"""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    if width <= 0.0 or height <= 0.0:
        raise ValueError("SORT 边界框必须满足 x1 < x2 且 y1 < y2。")
    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
        width * height,
        width / height,
    )


def _sort_measurement_to_bbox(
    center_x: float,
    center_y: float,
    area: float,
    aspect_ratio: float,
) -> BBoxXYXY:
    """将 [cx, cy, area, aspect_ratio] 转回 xyxy 框。"""
    if area <= 0.0 or aspect_ratio <= 0.0:
        raise ValueError("SORT 的 area 和 aspect_ratio 必须大于 0。")
    width = float(np.sqrt(area * aspect_ratio))
    height = area / width
    return (
        center_x - width / 2.0,
        center_y - height / 2.0,
        center_x + width / 2.0,
        center_y + height / 2.0,
    )


class _SortBBoxKalmanFilter:
    """状态为 [cx, cy, area, ratio, vx, vy, v_area] 的线性 Kalman。"""

    def __init__(self, measurement: tuple[float, float, float, float], dt: float) -> None:
        self.state: FloatArray = np.zeros(7, dtype=np.float64)
        self.state[:4] = np.asarray(measurement, dtype=np.float64)

        self.transition_matrix: FloatArray = np.eye(7, dtype=np.float64)
        self.transition_matrix[0, 4] = dt
        self.transition_matrix[1, 5] = dt
        self.transition_matrix[2, 6] = dt
        self.observation_matrix: FloatArray = np.zeros((4, 7), dtype=np.float64)
        self.observation_matrix[:4, :4] = np.eye(4, dtype=np.float64)

        self.covariance: FloatArray = np.eye(7, dtype=np.float64)
        self.covariance[4:, 4:] *= 1000.0
        self.covariance *= 10.0
        self.measurement_noise_covariance: FloatArray = np.eye(4, dtype=np.float64)
        self.measurement_noise_covariance[2:, 2:] *= 10.0
        self.process_noise_covariance: FloatArray = np.eye(7, dtype=np.float64)
        self.process_noise_covariance[4:, 4:] *= 0.01
        self.process_noise_covariance[6, 6] *= 0.01

    def predict(self) -> FloatArray:
        """预测中心、面积及对应速度，并阻止面积变为非正数。"""
        if self.state[2] + self.transition_matrix[2, 6] * self.state[6] <= 0.0:
            self.state[6] = 0.0
        self.state = self.transition_matrix @ self.state
        self.covariance = (
            self.transition_matrix @ self.covariance @ self.transition_matrix.T
            + self.process_noise_covariance
        )
        return self.state.copy()

    def update(self, measurement: tuple[float, float, float, float]) -> FloatArray:
        """使用 [cx, cy, area, ratio] 检测观测修正状态。"""
        observation = np.asarray(measurement, dtype=np.float64)
        innovation = observation - self.observation_matrix @ self.state
        innovation_covariance = (
            self.observation_matrix @ self.covariance @ self.observation_matrix.T
            + self.measurement_noise_covariance
        )
        kalman_gain = np.linalg.solve(
            innovation_covariance,
            self.observation_matrix @ self.covariance,
        ).T
        self.state = self.state + kalman_gain @ innovation

        identity = np.eye(7, dtype=np.float64)
        correction = identity - kalman_gain @ self.observation_matrix
        self.covariance = (
            correction @ self.covariance @ correction.T
            + kalman_gain
            @ self.measurement_noise_covariance
            @ kalman_gain.T
        )
        return self.state.copy()


@dataclass
class _SortTrack:
    track_id: int
    kalman_filter: _SortBBoxKalmanFilter
    bbox_xyxy: BBoxXYXY
    predicted_bbox_xyxy: BBoxXYXY
    last_detection: Detection
    age: int = 1
    hits: int = 1
    hit_streak: int = 1
    time_since_update: int = 0

    @classmethod
    def create(
        cls,
        track_id: int,
        detection: Detection,
        dt: float,
    ) -> "_SortTrack":
        measurement = _bbox_to_sort_measurement(detection.bbox_xyxy)
        kalman_filter = _SortBBoxKalmanFilter(measurement, dt)
        return cls(
            track_id=track_id,
            kalman_filter=kalman_filter,
            bbox_xyxy=detection.bbox_xyxy,
            predicted_bbox_xyxy=detection.bbox_xyxy,
            last_detection=detection,
        )

    def predict(self) -> BBoxXYXY:
        predicted_state = self.kalman_filter.predict()
        self.predicted_bbox_xyxy = _sort_measurement_to_bbox(
            float(predicted_state[0]),
            float(predicted_state[1]),
            float(predicted_state[2]),
            float(predicted_state[3]),
        )
        self.bbox_xyxy = self.predicted_bbox_xyxy
        self.age += 1
        self.time_since_update += 1
        return self.bbox_xyxy

    def update(self, detection: Detection) -> None:
        measurement = _bbox_to_sort_measurement(detection.bbox_xyxy)
        updated_state = self.kalman_filter.update(measurement)
        self.bbox_xyxy = _sort_measurement_to_bbox(
            float(updated_state[0]),
            float(updated_state[1]),
            float(updated_state[2]),
            float(updated_state[3]),
        )
        self.last_detection = detection
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0

    def mark_missed(self) -> None:
        """结束连续命中，并保留当前预测状态等待后续帧。"""
        self.hit_streak = 0

    def snapshot(self, min_hits: int) -> SortTrackResult:
        state = self.kalman_filter.state
        return SortTrackResult(
            track_id=self.track_id,
            bbox_xyxy=self.bbox_xyxy,
            detection=(
                self.last_detection if self.time_since_update == 0 else None
            ),
            age=self.age,
            hits=self.hits,
            hit_streak=self.hit_streak,
            time_since_update=self.time_since_update,
            area=float(state[2]),
            aspect_ratio=float(state[3]),
            v_area=float(state[6]),
            predicted_bbox_xyxy=self.predicted_bbox_xyxy,
            is_confirmed=self.hits >= min_hits,
        )


class SortTracker:
    """以每帧 predict-associate-update-manage 顺序运行的最小 SORT。"""

    def __init__(self, config: SortTrackerConfig | None = None) -> None:
        self.config = config or SortTrackerConfig()
        self._association_config = HungarianAssignmentConfig(
            iou_threshold=self.config.iou_threshold
        )
        self._tracks: dict[int, _SortTrack] = {}
        self._next_track_id = 1

    @property
    def active_track_ids(self) -> tuple[int, ...]:
        """按 ID 返回当前尚未超时的轨迹。"""
        return tuple(sorted(self._tracks))

    @property
    def confirmed_track_ids(self) -> tuple[int, ...]:
        """返回累计命中次数达到 min_hits 的活动轨迹。"""
        return tuple(
            track_id
            for track_id, track in sorted(self._tracks.items())
            if track.hits >= self.config.min_hits
        )

    def update(
        self,
        detections: tuple[Detection, ...],
    ) -> tuple[SortTrackResult, ...]:
        """处理一帧检测，并返回匹配、预测或新建的全部活动轨迹。"""
        tracks = list(self._tracks.values())
        predicted_bboxes = tuple(track.predict() for track in tracks)
        detection_bboxes = tuple(detection.bbox_xyxy for detection in detections)
        association = associate_by_iou(
            predicted_bboxes,
            detection_bboxes,
            self._association_config,
        ).result

        for track_index, detection_index in association.matches:
            tracks[track_index].update(detections[detection_index])

        for track_index in association.unmatched_tracks:
            tracks[track_index].mark_missed()

        expired_track_ids = tuple(
            tracks[track_index].track_id
            for track_index in association.unmatched_tracks
            if tracks[track_index].time_since_update
            > self.config.effective_max_age
        )
        for track_id in expired_track_ids:
            del self._tracks[track_id]

        for detection_index in association.unmatched_detections:
            detection = detections[detection_index]
            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = _SortTrack.create(
                track_id,
                detection,
                self.config.dt,
            )

        return tuple(
            self._tracks[track_id].snapshot(self.config.min_hits)
            for track_id in sorted(self._tracks)
        )

    def reset(self) -> None:
        """清空轨迹，并让下一条新轨迹重新从 ID 1 开始。"""
        self._tracks.clear()
        self._next_track_id = 1
