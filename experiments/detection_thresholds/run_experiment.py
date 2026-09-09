"""使用相同输入和 IoU 阈值运行多组检测置信度实验。"""

import argparse
from pathlib import Path
from typing import Final, Sequence

from main import INPUT_VIDEO, main as run_single_detection, threshold_value
from src.detection import DEFAULT_IOU_THRESHOLD


CONFIDENCE_THRESHOLDS: Final[tuple[float, ...]] = (0.20, 0.35, 0.60)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析阈值对比实验参数。"""
    parser = argparse.ArgumentParser(description="运行多组 YOLO 检测置信度对比实验。")
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_VIDEO,
        help="输入视频路径，默认使用 data/videos/test.mp4。",
    )
    parser.add_argument(
        "--iou",
        type=threshold_value,
        default=DEFAULT_IOU_THRESHOLD,
        help="各组共用的 IoU 阈值，默认值为 0.50。",
    )
    parser.add_argument(
        "--confidences",
        type=threshold_value,
        nargs="+",
        default=CONFIDENCE_THRESHOLDS,
        help="要依次运行的置信度阈值，默认值为 0.20 0.35 0.60。",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """依次调用正式单视频检测入口并保留各组独立输出。"""
    args = parse_args(argv)
    total = len(args.confidences)
    for index, confidence_threshold in enumerate(args.confidences, start=1):
        print(
            f"\n开始阈值实验 {index}/{total}："
            f"conf={confidence_threshold:.2f}，iou={args.iou:.2f}"
        )
        exit_code = run_single_detection(
            [
                "--input",
                str(args.input),
                "--conf",
                f"{confidence_threshold:.2f}",
                "--iou",
                f"{args.iou:.2f}",
            ]
        )
        if exit_code != 0:
            return exit_code

    print(f"\n阈值对比实验完成，共运行 {total} 组。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
