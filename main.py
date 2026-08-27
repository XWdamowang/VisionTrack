"""单视频 YOLO11 目标检测基线入口。"""

from pathlib import Path
from typing import Final


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
INPUT_VIDEO: Final[Path] = PROJECT_ROOT / "data" / "videos" / "test.mp4"
OUTPUT_VIDEO: Final[Path] = (
    PROJECT_ROOT / "outputs" / "videos" / "test_detected.mp4"
)
MODEL_NAME: Final[str] = "yolo11n.pt"
CONFIDENCE_THRESHOLD: Final[float] = 0.35
IOU_THRESHOLD: Final[float] = 0.50


def run_detection(input_path: Path, output_path: Path) -> int:
    """逐帧检测视频、保存可视化结果，并返回成功处理的帧数。"""
    # 延迟导入第三方库，使输入路径错误可以优先得到清晰提示。
    import cv2
    import torch
    from ultralytics import YOLO

    device: str | int = 0 if torch.cuda.is_available() else "cpu"
    device_name: str = "CUDA" if device == 0 else "CPU"
    print(f"使用设备：{device_name}")

    model = YOLO(MODEL_NAME)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开输入视频：{input_path}")

    fps: float = capture.get(cv2.CAP_PROP_FPS)
    width: int = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("无法读取输入视频的帧率或分辨率信息。")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"无法创建输出视频：{output_path}")

    processed_frames: int = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            # Ultralytics 的 plot() 返回包含类别、置信度和检测框的 BGR 图像。
            result = model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                device=device,
                verbose=False,
            )[0]
            writer.write(result.plot())
            processed_frames += 1
    finally:
        capture.release()
        writer.release()

    return processed_frames


def main() -> int:
    """检查输入并运行目标检测。"""
    if not INPUT_VIDEO.is_file():
        print(
            "错误：未找到测试视频。\n"
            f"请将视频放置在：{INPUT_VIDEO}"
        )
        return 1

    try:
        frame_count: int = run_detection(INPUT_VIDEO, OUTPUT_VIDEO)
    except (ImportError, RuntimeError) as error:
        print(f"处理失败：{error}")
        return 1

    print(f"处理完成，共处理 {frame_count} 帧。")
    print(f"结果保存位置：{OUTPUT_VIDEO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
