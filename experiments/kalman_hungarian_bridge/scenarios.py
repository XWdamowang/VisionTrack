"""Kalman/Hungarian 桥接实验的确定性合成场景。"""

from dataclasses import dataclass

import numpy as np

from experiments.kalman_cv_2d.kalman_filter import (
    ConstantVelocityKalmanFilter,
    KalmanFilterConfig,
)
from src.detection import BBoxXYXY

from .bridge import ExistingTrack


@dataclass(frozen=True)
class TrackSeed:
    track_id: int
    last_bbox: BBoxXYXY
    velocity: tuple[float, float]


@dataclass(frozen=True)
class DetectionSeed:
    detection_id: int
    bbox: BBoxXYXY


@dataclass(frozen=True)
class Scenario:
    name: str
    tracks: tuple[TrackSeed, ...]
    detections: tuple[DetectionSeed, ...]
    prediction_steps: int
    note: str

    def build_tracks(self) -> tuple[ExistingTrack, ...]:
        config = KalmanFilterConfig(dt=1.0)
        return tuple(
            ExistingTrack(
                track_id=seed.track_id,
                last_bbox=seed.last_bbox,
                kalman_filter=ConstantVelocityKalmanFilter(
                    np.array(
                        [
                            (seed.last_bbox[0] + seed.last_bbox[2]) / 2.0,
                            (seed.last_bbox[1] + seed.last_bbox[3]) / 2.0,
                            seed.velocity[0],
                            seed.velocity[1],
                        ],
                        dtype=np.float64,
                    ),
                    config,
                ),
            )
            for seed in self.tracks
        )


def build_scenarios() -> tuple[Scenario, ...]:
    """返回覆盖正常、交叉、漏检与门控失败的六个场景。"""
    return (
        Scenario(
            name="正常匀速",
            tracks=(
                TrackSeed(1, (0.0, 0.0, 10.0, 10.0), (5.0, 0.0)),
                TrackSeed(2, (40.0, 0.0, 50.0, 10.0), (-4.0, 0.0)),
            ),
            detections=(
                DetectionSeed(101, (5.0, 0.0, 15.0, 10.0)),
                DetectionSeed(102, (36.0, 0.0, 46.0, 10.0)),
            ),
            prediction_steps=1,
            note="匀速模型准确，预测框应与当前检测重合。",
        ),
        Scenario(
            name="两目标交叉",
            tracks=(
                TrackSeed(1, (0.0, 0.0, 10.0, 10.0), (15.0, 0.0)),
                TrackSeed(2, (20.0, 0.0, 30.0, 10.0), (-15.0, 0.0)),
            ),
            detections=(
                DetectionSeed(101, (15.0, 0.0, 25.0, 10.0)),
                DetectionSeed(102, (5.0, 0.0, 15.0, 10.0)),
            ),
            prediction_steps=1,
            note="last bbox 倾向匹配交叉后的另一目标，预测框保留运动方向。",
        ),
        Scenario(
            name="漏检1帧",
            tracks=(
                TrackSeed(1, (0.0, 0.0, 10.0, 10.0), (6.0, 0.0)),
                TrackSeed(2, (40.0, 0.0, 50.0, 10.0), (-5.0, 0.0)),
            ),
            detections=(
                DetectionSeed(101, (12.0, 0.0, 22.0, 10.0)),
                DetectionSeed(102, (30.0, 0.0, 40.0, 10.0)),
            ),
            prediction_steps=2,
            note="中间漏检一帧，因此从 last bbox 到当前检测共执行两次 predict。",
        ),
        Scenario(
            name="漏检3帧",
            tracks=(
                TrackSeed(1, (0.0, 0.0, 10.0, 10.0), (4.0, 0.0)),
                TrackSeed(2, (50.0, 0.0, 60.0, 10.0), (-4.0, 0.0)),
            ),
            detections=(
                DetectionSeed(101, (16.0, 0.0, 26.0, 10.0)),
                DetectionSeed(102, (34.0, 0.0, 44.0, 10.0)),
            ),
            prediction_steps=4,
            note="连续漏检三帧后，当前关联前累计执行四次 predict。",
        ),
        Scenario(
            name="高速运动导致gating拒绝",
            tracks=(
                TrackSeed(1, (0.0, 0.0, 10.0, 10.0), (20.0, 0.0)),
                TrackSeed(2, (70.0, 0.0, 80.0, 10.0), (0.0, 0.0)),
            ),
            detections=(
                DetectionSeed(101, (35.0, 0.0, 45.0, 10.0)),
                DetectionSeed(102, (70.0, 0.0, 80.0, 10.0)),
            ),
            prediction_steps=1,
            note="目标1实际位移超出匀速预测，两个输入框都低于 gating 阈值。",
        ),
        Scenario(
            name="last bbox失败但Kalman成功",
            tracks=(
                TrackSeed(1, (0.0, 0.0, 10.0, 10.0), (8.0, 0.0)),
                TrackSeed(2, (40.0, 0.0, 50.0, 10.0), (0.0, 0.0)),
            ),
            detections=(
                DetectionSeed(101, (8.0, 0.0, 18.0, 10.0)),
                DetectionSeed(102, (40.0, 0.0, 50.0, 10.0)),
            ),
            prediction_steps=1,
            note="目标1的 last bbox IoU 低于 0.30，预测框与检测重合。",
        ),
    )
