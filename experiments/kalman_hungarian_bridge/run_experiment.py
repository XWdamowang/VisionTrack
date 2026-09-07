"""运行 last bbox 与 Kalman predicted bbox 的双路 Hungarian 对照。"""

import csv
from pathlib import Path
from typing import Final

import numpy as np

from .bridge import BridgeComparison, compare_association_routes
from .scenarios import Scenario, build_scenarios


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_CSV: Final[Path] = (
    PROJECT_ROOT
    / "experiments"
    / "kalman_hungarian_bridge"
    / "results"
    / "association_comparison.csv"
)
CSV_FIELDS: Final[tuple[str, ...]] = (
    "scenario",
    "track_id",
    "detection_id",
    "last_bbox_iou",
    "predicted_bbox_iou",
    "last_bbox_match_valid",
    "kalman_match_valid",
)


def _format_matrix(matrix: np.ndarray) -> str:
    return np.array2string(matrix, precision=4, suppress_small=True)


def _id_matches(
    comparison: BridgeComparison,
    matches: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (comparison.track_ids[track_index], comparison.detection_ids[detection_index])
        for track_index, detection_index in matches
    )


def _print_route(
    route_name: str,
    comparison: BridgeComparison,
    *,
    use_kalman: bool,
) -> None:
    details = (
        comparison.kalman_details if use_kalman else comparison.last_bbox_details
    )
    print(f"{route_name} IoU matrix：")
    print(_format_matrix(details.iou_matrix))
    print(f"{route_name} cost matrix（1 - IoU）：")
    print(_format_matrix(details.cost_matrix))
    print(f"{route_name} Hungarian 原始匹配：{_id_matches(comparison, details.raw_matches)}")
    print(f"{route_name} gating 后 matches：{_id_matches(comparison, details.result.matches)}")
    print(
        f"{route_name} unmatched tracks："
        f"{tuple(comparison.track_ids[index] for index in details.result.unmatched_tracks)}"
    )
    print(
        f"{route_name} unmatched detections："
        f"{tuple(comparison.detection_ids[index] for index in details.result.unmatched_detections)}"
    )


def run_scenario(scenario: Scenario, iou_threshold: float = 0.30) -> BridgeComparison:
    comparison = compare_association_routes(
        scenario.build_tracks(),
        tuple(item.detection_id for item in scenario.detections),
        tuple(item.bbox for item in scenario.detections),
        prediction_steps=scenario.prediction_steps,
        iou_threshold=iou_threshold,
    )
    print("=" * 80)
    print(f"场景：{scenario.name}")
    print(f"说明：{scenario.note}")
    print(f"预测步数：{scenario.prediction_steps}")
    _print_route("方案A（last bbox）", comparison, use_kalman=False)
    _print_route("方案B（Kalman predicted bbox）", comparison, use_kalman=True)
    return comparison


def save_comparisons(
    output_path: Path,
    comparisons: list[tuple[Scenario, BridgeComparison]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for scenario, comparison in comparisons:
            last_matches = set(comparison.last_bbox_details.result.matches)
            kalman_matches = set(comparison.kalman_details.result.matches)
            for track_index, track_id in enumerate(comparison.track_ids):
                for detection_index, detection_id in enumerate(comparison.detection_ids):
                    pair = (track_index, detection_index)
                    writer.writerow(
                        {
                            "scenario": scenario.name,
                            "track_id": track_id,
                            "detection_id": detection_id,
                            "last_bbox_iou": f"{comparison.last_bbox_details.iou_matrix[pair]:.6f}",
                            "predicted_bbox_iou": f"{comparison.kalman_details.iou_matrix[pair]:.6f}",
                            "last_bbox_match_valid": pair in last_matches,
                            "kalman_match_valid": pair in kalman_matches,
                        }
                    )


def main() -> int:
    threshold = 0.30
    print(f"IoU gating 阈值：{threshold:.2f}")
    comparisons = [
        (scenario, run_scenario(scenario, threshold))
        for scenario in build_scenarios()
    ]
    save_comparisons(DEFAULT_OUTPUT_CSV, comparisons)
    print(f"\n对比结果已保存：{DEFAULT_OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
