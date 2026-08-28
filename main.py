"""单视频 YOLO11 目标检测基线入口。"""

import argparse
from pathlib import Path
from time import perf_counter
from typing import Final, Sequence

from src.detection import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DetectionConfig,
    run_video_detection,
)


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
INPUT_VIDEO: Final[Path] = PROJECT_ROOT / "data" / "videos" / "test.mp4"


def threshold_value(value: str) -> float:
    """将命令行参数转换为 0 到 1 之间的阈值。"""
    try:
        threshold = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("阈值必须是数字。") from error

    if not 0.0 <= threshold <= 1.0:
        raise argparse.ArgumentTypeError("阈值必须在 0 到 1 之间。")
    return threshold


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析检测入口的命令行参数。"""
    parser = argparse.ArgumentParser(description="运行单视频 YOLO11 目标检测。")
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_VIDEO,
        help="输入视频路径，默认使用 data/videos/test.mp4。",
    )
    parser.add_argument(
        "--conf",
        type=threshold_value,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="置信度阈值，范围为 0 到 1，默认值为 0.35。",
    )
    parser.add_argument(
        "--iou",
        type=threshold_value,
        default=DEFAULT_IOU_THRESHOLD,
        help="IoU 阈值，范围为 0 到 1，默认值为 0.50。",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        help="结果视频路径；不提供时根据输入文件名和阈值自动生成。",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="检测结果 CSV 路径；不提供时根据输入文件名和阈值自动生成。",
    )
    return parser.parse_args(argv)


def resolve_project_path(path: Path) -> Path:
    """将相对路径按项目根目录解析。"""
    return path if path.is_absolute() else PROJECT_ROOT / path


def default_output_paths(
    input_path: Path,
    confidence_threshold: float,
    iou_threshold: float,
) -> tuple[Path, Path]:
    """根据输入名称和阈值生成互不覆盖的默认输出路径。"""
    parameter_tag = f"conf_{confidence_threshold:.2f}_iou_{iou_threshold:.2f}"
    output_video = (
        PROJECT_ROOT
        / "outputs"
        / "videos"
        / f"{input_path.stem}_detected_{parameter_tag}.mp4"
    )
    output_csv = (
        PROJECT_ROOT
        / "outputs"
        / "csv"
        / f"{input_path.stem}_detections_{parameter_tag}.csv"
    )
    return output_video, output_csv


def main(argv: Sequence[str] | None = None) -> int:
    """检查输入并运行目标检测。"""
    args = parse_args(argv)
    input_path = resolve_project_path(args.input)
    default_video, default_csv = default_output_paths(
        input_path,
        args.conf,
        args.iou,
    )
    output_video = (
        resolve_project_path(args.output_video)
        if args.output_video is not None
        else default_video
    )
    output_csv = (
        resolve_project_path(args.output_csv)
        if args.output_csv is not None
        else default_csv
    )

    if not input_path.is_file():
        print(
            "错误：未找到测试视频。\n"
            f"请检查输入路径：{input_path}"
        )
        return 1

    print(f"置信度阈值：{args.conf:.2f}")
    print(f"IoU 阈值：{args.iou:.2f}")
    start_time = perf_counter()
    try:
        summary = run_video_detection(
            input_path,
            output_video,
            output_csv,
            DetectionConfig(
                confidence_threshold=args.conf,
                iou_threshold=args.iou,
            ),
        )
    except (ImportError, RuntimeError, ValueError) as error:
        print(f"处理失败：{error}")
        return 1

    elapsed_seconds = perf_counter() - start_time
    frame_count = summary.processed_frames
    processing_fps = frame_count / elapsed_seconds if elapsed_seconds > 0 else 0.0
    print(f"使用设备：{summary.device_name}")
    print(f"处理完成，共处理 {frame_count} 帧。")
    print(f"运行时间：{elapsed_seconds:.2f} 秒。")
    print(f"平均处理速度：{processing_fps:.2f} FPS。")
    print(f"结果视频保存位置：{output_video}")
    print(f"检测结果 CSV 保存位置：{output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
