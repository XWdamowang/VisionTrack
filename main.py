"""单视频 YOLO11 目标检测基线入口。"""

import csv
from pathlib import Path
from typing import Final, TextIO


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
INPUT_VIDEO: Final[Path] = PROJECT_ROOT / "data" / "videos" / "test.mp4"
OUTPUT_VIDEO: Final[Path] = (
    PROJECT_ROOT / "outputs" / "videos" / "test_detected.mp4"
)
OUTPUT_CSV: Final[Path] = (
    PROJECT_ROOT / "outputs" / "csv" / "test_detections.csv"
)
MODEL_NAME: Final[str] = "yolo11n.pt"
CONFIDENCE_THRESHOLD: Final[float] = 0.35
IOU_THRESHOLD: Final[float] = 0.50
CSV_FIELDS: Final[tuple[str, ...]] = (
    "frame_number",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
)


def run_detection(input_path: Path, output_path: Path, csv_path: Path) -> int:
    """逐帧检测视频，保存可视化视频和检测结果 CSV。"""
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

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file: TextIO | None = None
    processed_frames: int = 0
    try:
        try:
            csv_file = csv_path.open("w", encoding="utf-8-sig", newline="")
        except OSError as error:
            raise RuntimeError(f"无法创建检测结果 CSV：{csv_path}") from error

        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        csv_writer.writeheader()

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

            boxes = result.boxes
            if boxes is None:
                continue

            coordinates = boxes.xyxy.cpu().tolist()
            class_ids = boxes.cls.cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            for coordinates_xyxy, class_id_value, confidence in zip(
                coordinates,
                class_ids,
                confidences,
            ):
                class_id = int(class_id_value)
                x1, y1, x2, y2 = coordinates_xyxy
                csv_writer.writerow(
                    {
                        "frame_number": processed_frames,
                        "class_id": class_id,
                        "class_name": result.names[class_id],
                        "confidence": f"{confidence:.6f}",
                        "x1": f"{x1:.2f}",
                        "y1": f"{y1:.2f}",
                        "x2": f"{x2:.2f}",
                        "y2": f"{y2:.2f}",
                    }
                )
    finally:
        if csv_file is not None:
            csv_file.close()
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
        frame_count: int = run_detection(INPUT_VIDEO, OUTPUT_VIDEO, OUTPUT_CSV)
    except (ImportError, RuntimeError) as error:
        print(f"处理失败：{error}")
        return 1

    print(f"处理完成，共处理 {frame_count} 帧。")
    print(f"结果视频保存位置：{OUTPUT_VIDEO}")
    print(f"检测结果 CSV 保存位置：{OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
