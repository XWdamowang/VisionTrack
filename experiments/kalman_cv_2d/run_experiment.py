"""从 IoU Tracker 轨迹 CSV 运行二维匀速卡尔曼滤波实验。"""

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import numpy as np

from experiments.kalman_cv_2d.kalman_filter import (
    ConstantVelocityKalmanFilter,
    KalmanFilterConfig,
)


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_TRACK_CSV: Final[Path] = (
    PROJECT_ROOT / "outputs" / "csv" / "test_track_iou_0.50_miss_2.csv"
)
DEFAULT_VIDEO: Final[Path] = PROJECT_ROOT / "data" / "videos" / "test.mp4"
DEFAULT_OUTPUT_CSV: Final[Path] = (
    PROJECT_ROOT / "outputs" / "tracking" / "kalman_test.csv"
)
DEFAULT_OUTPUT_IMAGE: Final[Path] = (
    PROJECT_ROOT / "outputs" / "images" / "tracking" / "kalman_trajectory.png"
)
DEFAULT_OUTPUT_LOG: Final[Path] = (
    PROJECT_ROOT / "outputs" / "logs" / "kalman_experiment.md"
)
RESULT_FIELDS: Final[tuple[str, ...]] = (
    "frame_id",
    "track_id",
    "observation_available",
    "measured_x",
    "measured_y",
    "predicted_x",
    "predicted_y",
    "filtered_x",
    "filtered_y",
)


@dataclass(frozen=True)
class TrackMeasurement:
    frame_id: int
    track_id: int
    center_x: float
    center_y: float


@dataclass(frozen=True)
class KalmanResult:
    frame_id: int
    track_id: int
    observation_available: bool
    measured_x: float | None
    measured_y: float | None
    predicted_x: float
    predicted_y: float
    filtered_x: float
    filtered_y: float


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_video_fps(video_path: Path) -> float:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开源视频以读取 FPS：{video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if fps <= 0.0:
        raise RuntimeError(f"源视频 FPS 无效：{fps}")
    return fps


def load_track_measurements(csv_path: Path) -> dict[int, list[TrackMeasurement]]:
    required_fields = {"frame_number", "track_id", "x1", "y1", "x2", "y2"}
    tracks: dict[int, list[TrackMeasurement]] = {}
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            missing_fields = required_fields.difference(reader.fieldnames or ())
            if missing_fields:
                missing_text = ", ".join(sorted(missing_fields))
                raise RuntimeError(f"轨迹 CSV 缺少字段：{missing_text}")
            for row in reader:
                track_id = int(row["track_id"])
                x1, y1, x2, y2 = (
                    float(row[field]) for field in ("x1", "y1", "x2", "y2")
                )
                tracks.setdefault(track_id, []).append(
                    TrackMeasurement(
                        frame_id=int(row["frame_number"]),
                        track_id=track_id,
                        center_x=(x1 + x2) / 2.0,
                        center_y=(y1 + y2) / 2.0,
                    )
                )
    except OSError as error:
        raise RuntimeError(f"无法读取轨迹 CSV：{csv_path}") from error
    if not tracks:
        raise RuntimeError("轨迹 CSV 中没有可用记录。")
    for measurements in tracks.values():
        measurements.sort(key=lambda item: item.frame_id)
    return tracks


def select_longest_complete_track(
    tracks: dict[int, list[TrackMeasurement]],
) -> list[TrackMeasurement]:
    complete_tracks = [
        measurements
        for measurements in tracks.values()
        if len(measurements) >= 5
        and all(
            current.frame_id == previous.frame_id + 1
            for previous, current in zip(measurements, measurements[1:])
        )
    ]
    if not complete_tracks:
        raise RuntimeError("没有长度至少为 5 帧且全程逐帧连续的完整轨迹。")
    return max(complete_tracks, key=len)


def run_kalman_filter(
    measurements: list[TrackMeasurement],
    config: KalmanFilterConfig,
    missing_frame_ids: set[int],
) -> tuple[np.ndarray, list[KalmanResult]]:
    first = measurements[0]
    initial_state = np.array(
        [
            first.center_x,
            first.center_y,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    kalman_filter = ConstantVelocityKalmanFilter(initial_state, config)
    results = [
        KalmanResult(
            frame_id=first.frame_id,
            track_id=first.track_id,
            observation_available=True,
            measured_x=first.center_x,
            measured_y=first.center_y,
            predicted_x=first.center_x,
            predicted_y=first.center_y,
            filtered_x=first.center_x,
            filtered_y=first.center_y,
        )
    ]

    for measurement in measurements[1:]:
        predicted_state = kalman_filter.predict()
        observation_available = measurement.frame_id not in missing_frame_ids
        if observation_available:
            filtered_state = kalman_filter.update(
                np.array([measurement.center_x, measurement.center_y])
            )
            measured_x: float | None = measurement.center_x
            measured_y: float | None = measurement.center_y
        else:
            filtered_state = predicted_state
            measured_x = None
            measured_y = None
        results.append(
            KalmanResult(
                frame_id=measurement.frame_id,
                track_id=measurement.track_id,
                observation_available=observation_available,
                measured_x=measured_x,
                measured_y=measured_y,
                predicted_x=float(predicted_state[0]),
                predicted_y=float(predicted_state[1]),
                filtered_x=float(filtered_state[0]),
                filtered_y=float(filtered_state[1]),
            )
        )
    return initial_state, results


def save_results(output_path: Path, results: list[KalmanResult]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "frame_id": result.frame_id,
                    "track_id": result.track_id,
                    "observation_available": int(result.observation_available),
                    "measured_x": "" if result.measured_x is None else f"{result.measured_x:.4f}",
                    "measured_y": "" if result.measured_y is None else f"{result.measured_y:.4f}",
                    "predicted_x": f"{result.predicted_x:.4f}",
                    "predicted_y": f"{result.predicted_y:.4f}",
                    "filtered_x": f"{result.filtered_x:.4f}",
                    "filtered_y": f"{result.filtered_y:.4f}",
                }
            )


def save_trajectory_plot(output_path: Path, results: list[KalmanResult]) -> None:
    matplotlib_cache = PROJECT_ROOT / "outputs" / "tracking" / ".matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    measured_x = [result.measured_x for result in results]
    measured_y = [result.measured_y for result in results]
    predicted_x = [result.predicted_x for result in results]
    predicted_y = [result.predicted_y for result in results]
    filtered_x = [result.filtered_x for result in results]
    filtered_y = [result.filtered_y for result in results]
    missing_results = [result for result in results if not result.observation_available]

    figure, axis = plt.subplots(figsize=(11, 7))
    axis.plot(measured_x, measured_y, "o-", markersize=2.5, linewidth=1.0, label="Measured")
    axis.plot(predicted_x, predicted_y, "--", linewidth=1.2, label="Kalman predicted")
    axis.plot(filtered_x, filtered_y, linewidth=1.8, label="Kalman filtered")
    axis.scatter(
        [result.predicted_x for result in missing_results],
        [result.predicted_y for result in missing_results],
        marker="x",
        s=80,
        linewidths=2.0,
        color="red",
        label="Prediction during missing observations",
        zorder=5,
    )
    axis.set_title("2D Constant-Velocity Kalman Filter")
    axis.set_xlabel("center_x (pixel)")
    axis.set_ylabel("center_y (pixel)")
    axis.invert_yaxis()
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def calculate_errors(
    measurements: list[TrackMeasurement],
    results: list[KalmanResult],
    missing_frame_ids: set[int],
) -> tuple[float, float, float, float]:
    measurement_by_frame = {item.frame_id: item for item in measurements}
    normal_prediction_errors: list[float] = []
    normal_filter_errors: list[float] = []
    missing_prediction_errors: list[float] = []
    for result in results[1:]:
        hidden_measurement = measurement_by_frame[result.frame_id]
        prediction_error = float(
            np.hypot(
                result.predicted_x - hidden_measurement.center_x,
                result.predicted_y - hidden_measurement.center_y,
            )
        )
        if result.frame_id in missing_frame_ids:
            missing_prediction_errors.append(prediction_error)
        else:
            normal_prediction_errors.append(prediction_error)
            normal_filter_errors.append(
                float(
                    np.hypot(
                        result.filtered_x - hidden_measurement.center_x,
                        result.filtered_y - hidden_measurement.center_y,
                    )
                )
            )
    return (
        float(np.mean(normal_prediction_errors)),
        float(np.mean(normal_filter_errors)),
        float(np.mean(missing_prediction_errors)),
        float(np.max(missing_prediction_errors)),
    )


def save_experiment_log(
    output_path: Path,
    source_csv: Path,
    video_path: Path,
    measurements: list[TrackMeasurement],
    fps: float,
    config: KalmanFilterConfig,
    initial_state: np.ndarray,
    missing_frame_ids: set[int],
    errors: tuple[float, float, float, float],
) -> None:
    normal_prediction_error, normal_filter_error, missing_mean_error, missing_max_error = errors
    missing_text = ", ".join(str(frame_id) for frame_id in sorted(missing_frame_ids))
    content = f"""# 二维匀速卡尔曼滤波实验记录

## 数据与参数

- IoU Tracker 输出：`{source_csv}`
- 源视频：`{video_path}`
- 使用的 track_id：`{measurements[0].track_id}`
- 连续轨迹范围：第 {measurements[0].frame_id}–{measurements[-1].frame_id} 帧，共 {len(measurements)} 帧
- FPS：`{fps:.6f}`
- dt：`{config.dt:.8f} s`
- 初始状态 `[x, y, vx, vy]`：`[{initial_state[0]:.4f}, {initial_state[1]:.4f}, {initial_state[2]:.4f}, {initial_state[3]:.4f}]`
- 初始化方式：首帧检测框中心作为位置；没有先验速度，因此 `vx=0、vy=0`
- 过程加速度标准差：`{config.process_acceleration_std:.4f} pixel/s²`
- 观测位置标准差：`{config.measurement_position_std:.4f} pixel`
- 初始位置标准差：`{config.initial_position_std:.4f} pixel`
- 初始速度标准差：`{config.initial_velocity_std:.4f} pixel/s`

## 矩阵含义

- `F`（状态转移矩阵）：把 `[x, y, vx, vy]` 按匀速模型推进 `dt`，即位置增加 `速度 × dt`。
- `H`（观测矩阵）：从四维状态中取出 `x、y`，与二维中心点观测比较。
- `Q`（过程噪声协方差）：描述未建模加速度造成的不确定性，由加速度标准差和 `dt` 构造。
- `R`（观测噪声协方差）：描述检测框中心点测量噪声，本实验假设 x、y 独立且同方差。
- `P`（状态协方差）：记录当前对位置和速度估计的不确定程度，会在预测时增大、更新后减小。
- `K`（卡尔曼增益）：根据 `P` 与 `R` 动态平衡模型预测和当前观测。

## 正常轨迹表现

- 正常观测帧的预测位置平均误差：`{normal_prediction_error:.4f} pixel`
- 正常观测帧的滤波位置平均误差：`{normal_filter_error:.4f} pixel`
- 滤波轨迹相对原始检测中心更平滑；预测是更新前结果，滤波是融合当前观测后的结果。

## 短暂漏检实验

- 人为隐藏观测帧：`{missing_text}`
- 漏检帧只执行预测，不执行更新；CSV 中对应的 `measured_x/measured_y` 留空。
- 漏检期间预测位置相对被隐藏原始观测的平均误差：`{missing_mean_error:.4f} pixel`
- 漏检期间预测位置相对被隐藏原始观测的最大误差：`{missing_max_error:.4f} pixel`
- 结论：短暂漏检时滤波器仍能按已有位置和速度连续输出位置，但漏检越久，协方差与潜在误差会继续增大。
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行独立的二维匀速卡尔曼滤波实验。")
    parser.add_argument("--tracks", type=Path, default=DEFAULT_TRACK_CSV, help="IoU Tracker 轨迹 CSV。")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="用于读取 FPS 的源视频。")
    parser.add_argument("--missing-count", type=int, choices=(1, 2, 3), default=3, help="中段模拟连续漏检帧数。")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-image", type=Path, default=DEFAULT_OUTPUT_IMAGE)
    parser.add_argument("--output-log", type=Path, default=DEFAULT_OUTPUT_LOG)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    track_csv = resolve_path(args.tracks)
    video_path = resolve_path(args.video)
    if not track_csv.is_file():
        print(f"错误：未找到 IoU Tracker 轨迹 CSV：{track_csv}")
        return 1
    if not video_path.is_file():
        print(f"错误：未找到源视频：{video_path}")
        return 1

    try:
        fps = read_video_fps(video_path)
        measurements = select_longest_complete_track(
            load_track_measurements(track_csv)
        )
        missing_start = len(measurements) // 2
        missing_frame_ids = {
            item.frame_id
            for item in measurements[
                missing_start : missing_start + args.missing_count
            ]
        }
        config = KalmanFilterConfig(dt=1.0 / fps)
        initial_state, results = run_kalman_filter(
            measurements,
            config,
            missing_frame_ids,
        )
        output_csv = resolve_path(args.output_csv)
        output_image = resolve_path(args.output_image)
        output_log = resolve_path(args.output_log)
        save_results(output_csv, results)
        save_trajectory_plot(output_image, results)
        errors = calculate_errors(measurements, results, missing_frame_ids)
        save_experiment_log(
            output_log,
            track_csv,
            video_path,
            measurements,
            fps,
            config,
            initial_state,
            missing_frame_ids,
            errors,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"实验失败：{error}")
        return 1

    print(f"实验完成：track_id={measurements[0].track_id}，共 {len(results)} 帧。")
    print(f"FPS={fps:.6f}，dt={config.dt:.8f} 秒。")
    print(f"模拟漏检帧：{', '.join(str(value) for value in sorted(missing_frame_ids))}")
    print(f"结果 CSV：{output_csv}")
    print(f"轨迹对比图：{output_image}")
    print(f"实验记录：{output_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
