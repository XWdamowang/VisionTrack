"""视频目标检测及结果导出。"""

import csv
from collections.abc import Generator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TextIO

from src.utils.video_writer import H264VideoWriter, create_video_writer

from .models import Detection, FrameDetections, ImageArray


DEFAULT_MODEL_NAME: Final[str] = "yolo11n.pt"
DEFAULT_CONFIDENCE_THRESHOLD: Final[float] = 0.35
DEFAULT_IOU_THRESHOLD: Final[float] = 0.50
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


@dataclass(frozen=True)
class DetectionConfig:
    """单次目标检测所需的模型和阈值配置。"""

    model_name: str = DEFAULT_MODEL_NAME
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    iou_threshold: float = DEFAULT_IOU_THRESHOLD

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("置信度阈值必须在 0 到 1 之间。")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("IoU 阈值必须在 0 到 1 之间。")


@dataclass(frozen=True)
class DetectionSummary:
    """单次视频检测的运行摘要。"""

    processed_frames: int
    device_name: str


class VideoDetector:
    """将模型推理结果转换为稳定的帧级检测数据。"""

    def __init__(self, config: DetectionConfig) -> None:
        import torch
        from ultralytics import YOLO

        self.config = config
        self.device: str | int = 0 if torch.cuda.is_available() else "cpu"
        self.device_name = "CUDA" if self.device == 0 else "CPU"
        self.model = YOLO(config.model_name)

    def detect_frame(
        self,
        frame: ImageArray,
        frame_number: int,
        timestamp_seconds: float,
        source_fps: float,
    ) -> FrameDetections:
        """检测单帧并返回与具体跟踪算法无关的数据对象。"""
        result = self.model.predict(
            source=frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            device=self.device,
            verbose=False,
        )[0]

        detections: list[Detection] = []
        boxes = result.boxes
        if boxes is not None:
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
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=result.names[class_id],
                        confidence=float(confidence),
                        bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    )
                )

        return FrameDetections(
            frame_number=frame_number,
            timestamp_seconds=timestamp_seconds,
            source_fps=source_fps,
            frame=frame,
            annotated_frame=result.plot(),
            detections=tuple(detections),
        )


def iter_video_detections(
    input_path: Path,
    detector: VideoDetector,
) -> Generator[FrameDetections, None, None]:
    """按帧生成检测结果，并在结束或提前停止时释放视频资源。"""
    import cv2

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开输入视频：{input_path}")

    fps: float = capture.get(cv2.CAP_PROP_FPS)
    width: int = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("无法读取输入视频的帧率或分辨率信息。")

    frame_number = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            frame_number += 1
            yield detector.detect_frame(
                frame=frame,
                frame_number=frame_number,
                timestamp_seconds=(frame_number - 1) / fps,
                source_fps=fps,
            )
    finally:
        capture.release()


def run_video_detection(
    input_path: Path,
    output_video_path: Path,
    output_csv_path: Path,
    config: DetectionConfig,
) -> DetectionSummary:
    """消费帧级检测接口，并保存可视化视频和检测结果 CSV。"""
    detector = VideoDetector(config)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    csv_file: TextIO | None = None
    writer: H264VideoWriter | None = None
    processed_frames = 0
    try:
        try:
            csv_file = output_csv_path.open(
                "w",
                encoding="utf-8-sig",
                newline="",
            )
        except OSError as error:
            raise RuntimeError(
                f"无法创建检测结果 CSV：{output_csv_path}"
            ) from error

        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        csv_writer.writeheader()

        with closing(iter_video_detections(input_path, detector)) as frames:
            for frame_result in frames:
                if writer is None:
                    height, width = frame_result.annotated_frame.shape[:2]
                    writer = create_video_writer(
                        output_video_path,
                        frame_result.source_fps,
                        (width, height),
                    )

                writer.write(frame_result.annotated_frame)
                processed_frames = frame_result.frame_number
                for detection in frame_result.detections:
                    x1, y1, x2, y2 = detection.bbox_xyxy
                    csv_writer.writerow(
                        {
                            "frame_number": frame_result.frame_number,
                            "class_id": detection.class_id,
                            "class_name": detection.class_name,
                            "confidence": f"{detection.confidence:.6f}",
                            "x1": f"{x1:.2f}",
                            "y1": f"{y1:.2f}",
                            "x2": f"{x2:.2f}",
                            "y2": f"{y2:.2f}",
                        }
                    )
    finally:
        if csv_file is not None:
            csv_file.close()
        if writer is not None:
            writer.release()

    if processed_frames == 0:
        raise RuntimeError("输入视频中没有可处理的画面。")

    return DetectionSummary(
        processed_frames=processed_frames,
        device_name=detector.device_name,
    )
