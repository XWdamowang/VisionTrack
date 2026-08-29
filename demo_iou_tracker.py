"""生成可复现的 IoU Tracker 遮挡、交叉和高速运动演示。"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np

from src.detection import BBoxXYXY, Detection, ImageArray
from src.tracking import IouTracker, IouTrackerConfig


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
OUTPUT_VIDEO: Final[Path] = PROJECT_ROOT / "outputs" / "videos" / "iou_tracker_scenarios.mp4"
OUTPUT_IMAGE: Final[Path] = PROJECT_ROOT / "outputs" / "images" / "iou_tracker_scenarios.jpg"
OUTPUT_CSV: Final[Path] = PROJECT_ROOT / "outputs" / "csv" / "iou_tracker_scenarios.csv"
FRAME_SIZE: Final[tuple[int, int]] = (720, 400)
FPS: Final[float] = 20.0
SCENE_FRAMES: Final[int] = 60


@dataclass(frozen=True)
class ScenarioFrame:
    boxes: tuple[tuple[str, BBoxXYXY], ...]
    note: str


def _detection(box: BBoxXYXY) -> Detection:
    return Detection(0, "object", 0.95, box)


def _occlusion_frame(frame_index: int) -> ScenarioFrame:
    x = 70.0 + frame_index * 4.0
    box = (x, 165.0, x + 90.0, 235.0)
    if 27 <= frame_index <= 29:
        return ScenarioFrame((), "Detection missing: track is kept for 3 frames")
    note = "Same ID after short occlusion" if frame_index > 29 else "Normal overlap matching"
    return ScenarioFrame((("A", box),), note)


def _crossing_frame(frame_index: int) -> ScenarioFrame:
    first_x = 60.0 + frame_index * 4.5
    second_x = 510.0 - frame_index * 4.5
    boxes = (
        ("A", (first_x, 135.0, first_x + 120.0, 245.0)),
        ("B", (second_x, 135.0, second_x + 120.0, 245.0)),
    )
    if frame_index >= 50:
        boxes = tuple(reversed(boxes))
    note = "Ambiguous overlap can swap IDs" if 42 <= frame_index <= 54 else "Two same-class objects"
    return ScenarioFrame(boxes, note)


def _fast_motion_frame(frame_index: int) -> ScenarioFrame:
    x = 45.0 + (frame_index % 8) * 82.0
    box = (x, 165.0, x + 55.0, 225.0)
    return ScenarioFrame((("A", box),), "No overlap -> a new ID on every frame")


def _draw_frame(
    title: str,
    scene_frame: ScenarioFrame,
    assignments: tuple[tuple[str, int, BBoxXYXY], ...],
    local_frame_number: int,
) -> ImageArray:
    width, height = FRAME_SIZE
    frame = np.full((height, width, 3), (24, 27, 34), dtype=np.uint8)
    cv2.putText(frame, title, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(frame, scene_frame.note, (24, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 210, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Frame {local_frame_number + 1}/{SCENE_FRAMES}", (560, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)

    if title.startswith("1."):
        cv2.rectangle(frame, (175, 110), (290, 280), (60, 60, 65), -1)
        cv2.putText(frame, "OCCLUDER", (189, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (145, 145, 145), 1, cv2.LINE_AA)

    palette = {"A": (90, 210, 110), "B": (240, 150, 70)}
    for assignment_index, (identity, track_id, box) in enumerate(assignments):
        x1, y1, x2, y2 = (int(value) for value in box)
        color = palette[identity]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, f"ID {track_id}", (x1 + 5, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"GT {identity} -> ID {track_id}", (24, 340 + assignment_index * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2, cv2.LINE_AA)
    return frame


def main() -> int:
    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(OUTPUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), FPS, FRAME_SIZE)
    if not writer.isOpened():
        print(f"生成失败：无法创建演示视频：{OUTPUT_VIDEO}")
        return 1

    scenarios = (
        ("1. Short occlusion", _occlusion_frame),
        ("2. Crossing objects", _crossing_frame),
        ("3. Fast motion", _fast_motion_frame),
    )
    csv_rows: list[dict[str, str | int]] = []
    preview_frames: list[ImageArray] = []
    try:
        for title, make_frame in scenarios:
            tracker = IouTracker(IouTrackerConfig(iou_threshold=0.30, max_missed_frames=3))
            for frame_index in range(SCENE_FRAMES):
                scene_frame = make_frame(frame_index)
                detections = tuple(_detection(box) for _, box in scene_frame.boxes)
                tracked = tracker.update(detections)
                assignments = tuple(
                    (identity, item.track_id, box)
                    for (identity, box), item in zip(scene_frame.boxes, tracked)
                )
                rendered = _draw_frame(title, scene_frame, assignments, frame_index)
                writer.write(rendered)
                if frame_index == 52:
                    preview_frames.append(rendered)
                for identity, track_id, _ in assignments:
                    csv_rows.append(
                        {
                            "scenario": title,
                            "frame_number": frame_index + 1,
                            "ground_truth_object": identity,
                            "track_id": track_id,
                        }
                    )
    finally:
        writer.release()

    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as csv_file:
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=("scenario", "frame_number", "ground_truth_object", "track_id"),
        )
        csv_writer.writeheader()
        csv_writer.writerows(csv_rows)

    if len(preview_frames) == len(scenarios):
        preview = cv2.vconcat([cv2.resize(frame, (540, 300)) for frame in preview_frames])
        cv2.imwrite(str(OUTPUT_IMAGE), preview)

    print("演示生成完成。")
    print(f"演示视频：{OUTPUT_VIDEO}")
    print(f"对比预览：{OUTPUT_IMAGE}")
    print(f"逐帧 ID：{OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
