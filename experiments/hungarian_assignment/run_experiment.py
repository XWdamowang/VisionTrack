"""打印七种 IoU 代价矩阵与 Hungarian 匹配场景。"""

from dataclasses import dataclass

import numpy as np

from experiments.hungarian_assignment import (
    AssociationDetails,
    HungarianAssignmentConfig,
    associate_by_iou,
)
from src.detection import BBoxXYXY


@dataclass(frozen=True)
class Scenario:
    name: str
    predicted_tracks_bboxes: tuple[BBoxXYXY, ...]
    detection_bboxes: tuple[BBoxXYXY, ...]
    note: str


def _greedy_matches(iou_matrix: np.ndarray) -> tuple[tuple[int, int], ...]:
    candidates = [
        (float(iou_matrix[track_index, detection_index]), track_index, detection_index)
        for track_index in range(iou_matrix.shape[0])
        for detection_index in range(iou_matrix.shape[1])
    ]
    matched_tracks: set[int] = set()
    matched_detections: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, track_index, detection_index in sorted(candidates, reverse=True):
        if track_index in matched_tracks or detection_index in matched_detections:
            continue
        matches.append((track_index, detection_index))
        matched_tracks.add(track_index)
        matched_detections.add(detection_index)
    return tuple(sorted(matches))


def _total_iou(
    iou_matrix: np.ndarray,
    matches: tuple[tuple[int, int], ...],
) -> float:
    return sum(float(iou_matrix[track, detection]) for track, detection in matches)


def _print_scenario(scenario: Scenario, threshold: float) -> None:
    details: AssociationDetails = associate_by_iou(
        scenario.predicted_tracks_bboxes,
        scenario.detection_bboxes,
        HungarianAssignmentConfig(iou_threshold=threshold),
    )
    print("=" * 72)
    print(scenario.name)
    print(f"说明：{scenario.note}")
    print("IoU 矩阵：")
    print(np.array2string(details.iou_matrix, precision=4, suppress_small=True))
    print("cost 矩阵（1 - IoU）：")
    print(np.array2string(details.cost_matrix, precision=4, suppress_small=True))
    print(f"Hungarian 原始结果：{details.raw_matches}")
    print(f"gating 后 matches：{details.result.matches}")
    print(f"unmatched_tracks：{details.result.unmatched_tracks}")
    print(f"unmatched_detections：{details.result.unmatched_detections}")
    if scenario.name.startswith("G."):
        greedy_matches = _greedy_matches(details.iou_matrix)
        print(
            "局部贪心结果："
            f"{greedy_matches}，总 IoU={_total_iou(details.iou_matrix, greedy_matches):.4f}"
        )
        print(
            "Hungarian 总 IoU："
            f"{_total_iou(details.iou_matrix, details.raw_matches):.4f}"
        )


def build_scenarios() -> tuple[Scenario, ...]:
    return (
        Scenario(
            "A. 正常 3 Tracks / 3 Detections",
            ((0, 0, 10, 10), (20, 0, 30, 10), (40, 0, 50, 10)),
            ((1, 0, 11, 10), (19, 0, 29, 10), (41, 0, 51, 10)),
            "三个目标均有高 IoU 的一对一检测。",
        ),
        Scenario(
            "B. 3 Tracks / 4 Detections",
            ((0, 0, 10, 10), (20, 0, 30, 10), (40, 0, 50, 10)),
            ((1, 0, 11, 10), (19, 0, 29, 10), (41, 0, 51, 10), (70, 0, 80, 10)),
            "多出的 detection 应进入 unmatched_detections。",
        ),
        Scenario(
            "C. 4 Tracks / 2 Detections",
            ((0, 0, 10, 10), (20, 0, 30, 10), (40, 0, 50, 10), (60, 0, 70, 10)),
            ((1, 0, 11, 10), (39, 0, 49, 10)),
            "没有检测对应的 tracks 应进入 unmatched_tracks。",
        ),
        Scenario(
            "D. 没有 Tracks",
            (),
            ((0, 0, 10, 10), (20, 0, 30, 10)),
            "所有 detections 都未匹配。",
        ),
        Scenario(
            "E. 没有 Detections",
            ((0, 0, 10, 10), (20, 0, 30, 10)),
            (),
            "所有 tracks 都未匹配。",
        ),
        Scenario(
            "F. 所有 IoU 都很低",
            ((0, 0, 10, 10), (20, 0, 30, 10)),
            ((100, 0, 110, 10), (120, 0, 130, 10)),
            "Hungarian 仍给出原始配对，但 gating 应全部拒绝。",
        ),
        Scenario(
            "G. 贪心匹配不是全局最优",
            ((0, 0, 10, 10), (0, 0, 8, 10)),
            ((0, 0, 9, 10), (2, 0, 10, 10)),
            "贪心先取最高 IoU，会损害另一个 track 的整体分配。",
        ),
    )


def main() -> int:
    threshold = 0.30
    print(f"IoU gating 阈值：{threshold:.2f}")
    for scenario in build_scenarios():
        _print_scenario(scenario, threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
