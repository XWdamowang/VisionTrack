"""用于学习匀速运动模型的二维卡尔曼滤波器。"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class KalmanFilterConfig:
    """二维匀速卡尔曼滤波器的噪声和初始不确定度配置。"""

    dt: float
    process_acceleration_std: float = 50.0
    measurement_position_std: float = 5.0
    initial_position_std: float = 5.0
    initial_velocity_std: float = 100.0

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("时间步长 dt 必须大于 0。")
        for name, value in (
            ("过程加速度标准差", self.process_acceleration_std),
            ("观测位置标准差", self.measurement_position_std),
            ("初始位置标准差", self.initial_position_std),
            ("初始速度标准差", self.initial_velocity_std),
        ):
            if value <= 0.0:
                raise ValueError(f"{name}必须大于 0。")


class ConstantVelocityKalmanFilter:
    """状态为 [x, y, vx, vy]、观测为 [x, y] 的简化滤波器。"""

    def __init__(
        self,
        initial_state: FloatArray,
        config: KalmanFilterConfig,
    ) -> None:
        state = np.asarray(initial_state, dtype=np.float64)
        if state.shape != (4,):
            raise ValueError("初始状态必须是形状为 (4,) 的 [x, y, vx, vy]。")

        self.config = config
        self.state = state.copy()
        dt = config.dt
        self.transition_matrix: FloatArray = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.observation_matrix: FloatArray = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        self.process_noise_covariance = self._build_process_noise_covariance()
        self.measurement_noise_covariance: FloatArray = np.eye(2, dtype=np.float64) * (
            config.measurement_position_std**2
        )
        self.covariance: FloatArray = np.diag(
            [
                config.initial_position_std**2,
                config.initial_position_std**2,
                config.initial_velocity_std**2,
                config.initial_velocity_std**2,
            ]
        ).astype(np.float64)

    def _build_process_noise_covariance(self) -> FloatArray:
        """用帧间恒定的随机加速度构造状态过程噪声 Q。"""
        dt = self.config.dt
        acceleration_variance = self.config.process_acceleration_std**2
        noise_mapping = np.array(
            [
                [0.5 * dt**2, 0.0],
                [0.0, 0.5 * dt**2],
                [dt, 0.0],
                [0.0, dt],
            ],
            dtype=np.float64,
        )
        return acceleration_variance * noise_mapping @ noise_mapping.T

    @property
    def position(self) -> tuple[float, float]:
        return float(self.state[0]), float(self.state[1])

    def predict(self) -> FloatArray:
        """使用匀速模型预测下一帧状态及其不确定度。"""
        self.state = self.transition_matrix @ self.state
        self.covariance = (
            self.transition_matrix @ self.covariance @ self.transition_matrix.T
            + self.process_noise_covariance
        )
        return self.state.copy()

    def update(self, measurement: FloatArray) -> FloatArray:
        """使用位置观测修正预测状态。"""
        observation = np.asarray(measurement, dtype=np.float64)
        if observation.shape != (2,):
            raise ValueError("观测必须是形状为 (2,) 的 [x, y]。")

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

        identity = np.eye(4, dtype=np.float64)
        correction = identity - kalman_gain @ self.observation_matrix
        self.covariance = (
            correction @ self.covariance @ correction.T
            + kalman_gain
            @ self.measurement_noise_covariance
            @ kalman_gain.T
        )
        return self.state.copy()
