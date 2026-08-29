"""在逐帧检测结果上运行 IoU 跟踪并导出结果。"""

import csv
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from src.detection import (
    DetectionConfig,
    ImageArray,
    VideoDetector,
    iter_video_detections,
)

from .contracts import TRACK_CSV_FIELDS
from .iou_tracker import IouTracker, IouTrackerConfig, TrackedDetection


@dataclass(frozen=True)
class TrackingSummary:
    """单次视频跟踪的运行摘要。"""

    processed_frames: int
    created_tracks: int
    device_name: str


def _track_color(track_id: int) -> tuple[int, int, int]:
    return (
        64 + (track_id * 47) % 192,
        64 + (track_id * 89) % 192,
        64 + (track_id * 137) % 192,
    )


def draw_tracks(
    frame: ImageArray,
    tracks: tuple[TrackedDetection, ...],
) -> ImageArray:
    """在画面上绘制轨迹框、类别和 ID。"""
    import cv2

    annotated = frame.copy()
    for tracked in tracks:
        detection = tracked.detection
        x1, y1, x2, y2 = (int(value) for value in detection.bbox_xyxy)
        color = _track_color(tracked.track_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = (
            f"ID {tracked.track_id} {detection.class_name} "
            f"{detection.confidence:.2f}"
        )
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def run_video_tracking(
    input_path: Path,
    output_video_path: Path,
    output_csv_path: Path,
    detection_config: DetectionConfig,
    tracker_config: IouTrackerConfig,
) -> TrackingSummary:
    """检测并跟踪单个视频，保存带 ID 的视频和轨迹 CSV。"""
    import cv2

    detector = VideoDetector(detection_config)
    tracker = IouTracker(tracker_config)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    csv_file: TextIO | None = None
    video_writer = None
    processed_frames = 0
    observed_track_ids: set[int] = set()
    try:
        try:
            csv_file = output_csv_path.open("w", encoding="utf-8-sig", newline="")
        except OSError as error:
            raise RuntimeError(f"无法创建轨迹 CSV：{output_csv_path}") from error
        csv_writer = csv.DictWriter(csv_file, fieldnames=TRACK_CSV_FIELDS)
        csv_writer.writeheader()

        with closing(iter_video_detections(input_path, detector)) as frames:
            for frame_result in frames:
                tracks = tracker.update(frame_result.detections)
                annotated_frame = draw_tracks(frame_result.frame, tracks)
                if video_writer is None:
                    height, width = annotated_frame.shape[:2]
                    video_writer = cv2.VideoWriter(
                        str(output_video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        frame_result.source_fps,
                        (width, height),
                    )
                    if not video_writer.isOpened():
                        raise RuntimeError(f"无法创建跟踪视频：{output_video_path}")

                video_writer.write(annotated_frame)
                processed_frames = frame_result.frame_number
                for tracked in tracks:
                    observed_track_ids.add(tracked.track_id)
                    detection = tracked.detection
                    x1, y1, x2, y2 = detection.bbox_xyxy
                    csv_writer.writerow(
                        {
                            "frame_number": frame_result.frame_number,
                            "timestamp_seconds": f"{frame_result.timestamp_seconds:.6f}",
                            "track_id": tracked.track_id,
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
        if video_writer is not None:
            video_writer.release()

    if processed_frames == 0:
        raise RuntimeError("输入视频中没有可处理的画面。")
    return TrackingSummary(
        processed_frames=processed_frames,
        created_tracks=len(observed_track_ids),
        device_name=detector.device_name,
    )
