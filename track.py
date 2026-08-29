"""单视频 YOLO11 检测与轻量 IoU 跟踪入口。"""

import argparse
from pathlib import Path
from time import perf_counter
from typing import Final, Sequence

from src.detection import DetectionConfig
from src.tracking import IouTrackerConfig, run_video_tracking


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
DEFAULT_INPUT: Final[Path] = PROJECT_ROOT / "data" / "videos" / "test.mp4"


def threshold_value(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("阈值必须是数字。") from error
    if not 0.0 <= result <= 1.0:
        raise argparse.ArgumentTypeError("阈值必须在 0 到 1 之间。")
    return result


def non_negative_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("最大丢失帧数必须是整数。") from error
    if result < 0:
        raise argparse.ArgumentTypeError("最大丢失帧数不能为负数。")
    return result


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 YOLO11 + 轻量 IoU Tracker。")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="输入视频路径。")
    parser.add_argument("--conf", type=threshold_value, default=0.35, help="检测置信度阈值。")
    parser.add_argument("--det-iou", type=threshold_value, default=0.50, help="检测 NMS IoU 阈值。")
    parser.add_argument("--track-iou", type=threshold_value, default=0.30, help="轨迹关联 IoU 阈值。")
    parser.add_argument("--max-missed", type=non_negative_integer, default=2, help="轨迹最多保留的丢失帧数。")
    parser.add_argument("--output-video", type=Path, help="跟踪视频路径。")
    parser.add_argument("--output-csv", type=Path, help="轨迹 CSV 路径。")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = resolve_path(args.input)
    tag = f"track_iou_{args.track_iou:.2f}_miss_{args.max_missed}"
    output_video = resolve_path(args.output_video) if args.output_video else (
        PROJECT_ROOT / "outputs" / "videos" / f"{input_path.stem}_{tag}.mp4"
    )
    output_csv = resolve_path(args.output_csv) if args.output_csv else (
        PROJECT_ROOT / "outputs" / "csv" / f"{input_path.stem}_{tag}.csv"
    )
    if not input_path.is_file():
        print(f"错误：未找到输入视频：{input_path}")
        return 1

    print(
        f"开始跟踪：关联 IoU={args.track_iou:.2f}，"
        f"最大丢失帧数={args.max_missed}。"
    )
    start_time = perf_counter()
    try:
        summary = run_video_tracking(
            input_path=input_path,
            output_video_path=output_video,
            output_csv_path=output_csv,
            detection_config=DetectionConfig(
                confidence_threshold=args.conf,
                iou_threshold=args.det_iou,
            ),
            tracker_config=IouTrackerConfig(
                iou_threshold=args.track_iou,
                max_missed_frames=args.max_missed,
            ),
        )
    except (ImportError, RuntimeError, ValueError) as error:
        print(f"处理失败：{error}")
        return 1

    elapsed = perf_counter() - start_time
    print(f"处理完成：{summary.processed_frames} 帧，共出现 {summary.created_tracks} 个轨迹 ID。")
    print(f"使用设备：{summary.device_name}；耗时：{elapsed:.2f} 秒。")
    print(f"跟踪视频：{output_video}")
    print(f"轨迹 CSV：{output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
