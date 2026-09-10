"""运行覆盖 SORT ID 与轨迹生命周期的确定性小实验。"""

from src.detection import BBoxXYXY, Detection

from .tracker import SortTrackResult, SortTracker, SortTrackerConfig


def _detection(name: str, bbox: BBoxXYXY) -> Detection:
    class_id = 0 if name == "A" else 1
    return Detection(class_id, name, 0.90, bbox)


def _observed_ids(results: tuple[SortTrackResult, ...]) -> dict[str, int]:
    return {
        result.detection.class_name: result.track_id
        for result in results
        if result.detection is not None
    }


def _format_bbox(bbox: BBoxXYXY) -> str:
    return "(" + ", ".join(f"{value:.1f}" for value in bbox) + ")"


def main() -> int:
    tracker = SortTracker(
        SortTrackerConfig(iou_threshold=0.30, max_age=2, min_hits=2)
    )
    frames = (
        (_detection("A", (0.0, 0.0, 20.0, 20.0)),),
        (_detection("A", (1.0, -1.0, 23.0, 21.0)),),
        (
            _detection("A", (2.0, -2.0, 26.0, 22.0)),
            _detection("B", (60.0, 0.0, 80.0, 20.0)),
        ),
        (_detection("B", (62.0, 0.0, 82.0, 20.0)),),
        (
            _detection("A", (4.0, -4.0, 32.0, 24.0)),
            _detection("B", (64.0, 0.0, 84.0, 20.0)),
        ),
        (_detection("A", (5.0, -5.0, 35.0, 25.0)),),
        (_detection("A", (6.0, -6.0, 38.0, 26.0)),),
        (_detection("A", (7.0, -7.0, 41.0, 27.0)),),
    )

    histories: list[tuple[SortTrackResult, ...]] = []
    print(
        "frame | id | obs | confirmed | age hits streak tsu | "
        "area    v_area | predicted bbox             | filtered bbox"
    )
    for frame_number, detections in enumerate(frames, start=1):
        results = tracker.update(detections)
        histories.append(results)
        for item in results:
            observed_name = (
                item.detection.class_name if item.detection is not None else "-"
            )
            print(
                f"{frame_number:>5} | {item.track_id:>2} | {observed_name:^3} | "
                f"{str(item.is_confirmed):^9} | {item.age:>3} {item.hits:>4} "
                f"{item.hit_streak:>6} {item.time_since_update:>3} | "
                f"{item.area:>7.2f} {item.v_area:>8.2f} | "
                f"{_format_bbox(item.predicted_bbox_xyxy):<26} | "
                f"{_format_bbox(item.bbox_xyxy)}"
            )

    a_ids = tuple(
        observed["A"]
        for results in histories
        if "A" in (observed := _observed_ids(results))
    )
    assert len(set(a_ids)) == 1, "单目标连续/短暂丢失后的 ID 应保持稳定。"
    assert _observed_ids(histories[2])["B"] != a_ids[0], "新目标应创建新 ID。"
    assert _observed_ids(histories[4])["A"] == a_ids[0], "短暂丢失后应恢复原 ID。"
    b_id = _observed_ids(histories[4])["B"]
    assert b_id not in tracker.active_track_ids, "超过最大丢失帧数的轨迹应被删除。"
    assert any(item.v_area > 0.0 for item in histories[2]), "尺度速度应可观测。"
    print("\n四项 SORT 生命周期检查和尺度预测检查全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
