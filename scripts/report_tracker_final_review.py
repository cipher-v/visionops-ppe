"""Summarize the bounded final tracker review and create comparison contact sheets."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results/tracker_final_review"
RUNS = (
    "baseline_crowded",
    "low011_buffer60_crowded",
    "low011_buffer90_crowded",
    "low020_buffer60_crowded",
    "low025_buffer60_crowded",
    "match090_crowded",
    "gmc_none_crowded",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--sheet-count", type=int, default=5)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _track_groups(path: Path) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in _read_jsonl(path):
        groups[int(record["track_id"])].append(record)
    return dict(groups)


def _run_summary(run_dir: Path) -> dict[str, Any]:
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    tracker_metadata = metadata["tracking"]["tracker"]
    groups = _track_groups(run_dir / "tracks.jsonl")
    fps = float(metadata["source"]["reported_fps"])
    durations = {
        str(track_id): (
            float(records[-1]["timestamp_seconds"])
            - float(records[0]["timestamp_seconds"])
            + 1 / fps
        )
        for track_id, records in groups.items()
    }
    observations = {str(track_id): len(records) for track_id, records in groups.items()}
    diagnostics = metadata["tracking"]["diagnostics"]
    association = metadata["ppe_association"]["diagnostics"]
    temporal = metadata["temporal_compliance"]["diagnostics"]
    return {
        "tracker_configuration": tracker_metadata["configuration"],
        "reportable_person_threshold": tracker_metadata["person_confidence_threshold"],
        "tracking_confidence_floor": tracker_metadata.get(
            "tracking_confidence_floor",
            tracker_metadata["configuration"]["track_low_thresh"],
        ),
        "unique_tracks_created": diagnostics["unique_tracks_created"],
        "lost_transitions": diagnostics["lost_transitions"],
        "recovered_transitions": diagnostics["recovered_transitions"],
        "expired_tracks": diagnostics["expired_tracks"],
        "median_track_length_frames": diagnostics["median_track_length_frames"],
        "mean_track_length_frames": diagnostics["mean_track_length_frames"],
        "track_observations": diagnostics["track_observations"],
        "tracks_under_0_5_seconds": sum(value < 0.5 for value in durations.values()),
        "tracks_under_1_second": sum(value < 1.0 for value in durations.values()),
        "observations_per_track": observations,
        "duration_seconds_per_track": durations,
        "ppe_association": {
            "total_person_observations": association["total_person_observations"],
            "ppe_detections_assigned": association["ppe_detections_assigned"],
            "ppe_detections_unassigned": association["ppe_detections_unassigned"],
            "frame_level_equipment_states": association["frame_level_equipment_states"],
        },
        "temporal_compliance": {
            "incidents_opened": temporal["incidents_opened"],
            "incidents_closed": temporal["incidents_closed"],
            "incidents_active": temporal["incidents_active"],
            "temporal_state_observations": temporal["temporal_state_observations"],
        },
    }


def _center_distance(first: list[float], second: list[float]) -> float:
    ax = (first[0] + first[2]) / 2
    ay = (first[1] + first[3]) / 2
    bx = (second[0] + second[2]) / 2
    by = (second[1] + second[3]) / 2
    scale = max((first[2] - first[0] + second[2] - second[0]) / 2, 1.0)
    return math.hypot(ax - bx, ay - by) / scale


def _iou(first: list[float], second: list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _fragmentation_candidates(run_dir: Path, limit: int) -> list[dict[str, Any]]:
    groups = _track_groups(run_dir / "tracks.jsonl")
    candidates = []
    for new_id, new_records in groups.items():
        first = new_records[0]
        first_frame = int(first["frame_index"])
        for old_id, old_records in groups.items():
            if old_id == new_id:
                continue
            last = old_records[-1]
            gap = first_frame - int(last["frame_index"])
            if not 1 <= gap <= 30:
                continue
            overlap = _iou(last["bbox_xyxy"], first["bbox_xyxy"])
            distance = _center_distance(last["bbox_xyxy"], first["bbox_xyxy"])
            if overlap < 0.15 and distance > 0.75:
                continue
            first_box = first["bbox_xyxy"]
            last_box = last["bbox_xyxy"]
            x1 = min(first_box[0], last_box[0])
            y1 = min(first_box[1], last_box[1])
            x2 = max(first_box[2], last_box[2])
            y2 = max(first_box[3], last_box[3])
            width, height = x2 - x1, y2 - y1
            candidates.append(
                {
                    "frame": first_frame,
                    "timestamp_seconds": first["timestamp_seconds"],
                    "previous_track_id": old_id,
                    "new_track_id": new_id,
                    "gap_frames": gap,
                    "last_to_first_iou": overlap,
                    "normalized_center_distance": distance,
                    "review_bbox_xyxy": [
                        max(0.0, x1 - width),
                        max(0.0, y1 - height * 0.35),
                        min(3840.0, x2 + width),
                        min(2160.0, y2 + height * 0.35),
                    ],
                    "ranking_score": overlap + max(0.0, 1.0 - distance) - gap / 100,
                }
            )
    candidates.sort(key=lambda item: item["ranking_score"], reverse=True)
    selected = []
    for candidate in candidates:
        if any(abs(candidate["frame"] - item["frame"]) < 30 for item in selected):
            continue
        selected.append(candidate)
        if len(selected) == limit:
            break
    return selected


def _read_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(frame_index, 0))
    ok, image = capture.read()
    if not ok:
        raise RuntimeError(f"Unable to read annotated frame {frame_index}")
    return image


def _tile(image: np.ndarray, label: str, width: int = 384) -> np.ndarray:
    height = round(image.shape[0] * width / image.shape[1])
    tile = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (width, 34), (12, 18, 28), -1)
    cv2.putText(
        tile,
        label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def _crop_source_region(image: np.ndarray, bbox: list[float]) -> np.ndarray:
    scale_x = image.shape[1] / 3840
    scale_y = image.shape[0] / 2160
    x1, y1, x2, y2 = (
        round(bbox[0] * scale_x),
        round(bbox[1] * scale_y),
        round(bbox[2] * scale_x),
        round(bbox[3] * scale_y),
    )
    return image[
        max(0, y1) : min(image.shape[0], y2), max(0, x1) : min(image.shape[1], x2)
    ]


def _write_contact_sheets(
    results_root: Path, candidates: list[dict[str, Any]]
) -> list[str]:
    output_dir = results_root / "contact_sheets"
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = cv2.VideoCapture(str(results_root / RUNS[0] / "annotated.mp4"))
    buffer_candidate = cv2.VideoCapture(
        str(results_root / "low011_buffer60_crowded" / "annotated.mp4")
    )
    match_candidate = cv2.VideoCapture(
        str(results_root / "match090_crowded" / "annotated.mp4")
    )
    if (
        not baseline.isOpened()
        or not buffer_candidate.isOpened()
        or not match_candidate.isOpened()
    ):
        raise RuntimeError("Unable to open annotated videos for contact sheets")
    outputs = []
    try:
        for index, candidate in enumerate(candidates, start=1):
            center = int(candidate["frame"])
            bbox = candidate["review_bbox_xyxy"]
            frame_indices = [max(center + offset, 0) for offset in (-10, -3, 0, 3, 10)]
            baseline_row = np.hstack(
                [
                    _tile(
                        _crop_source_region(_read_frame(baseline, frame), bbox),
                        f"baseline | frame {frame}",
                    )
                    for frame in frame_indices
                ]
            )
            buffer_row = np.hstack(
                [
                    _tile(
                        _crop_source_region(_read_frame(buffer_candidate, frame), bbox),
                        f"buffer60 | frame {frame}",
                    )
                    for frame in frame_indices
                ]
            )
            match_row = np.hstack(
                [
                    _tile(
                        _crop_source_region(_read_frame(match_candidate, frame), bbox),
                        f"match090 | frame {frame}",
                    )
                    for frame in frame_indices
                ]
            )
            sheet = np.vstack((baseline_row, buffer_row, match_row))
            output = output_dir / f"fragmentation_window_{index:02d}_f{center}.jpg"
            if not cv2.imwrite(str(output), sheet):
                raise RuntimeError(f"Unable to write {output}")
            outputs.append(str(output.relative_to(results_root)))
    finally:
        baseline.release()
        buffer_candidate.release()
        match_candidate.release()
    return outputs


def _comparison_sheet(
    baseline_path: Path,
    candidate_path: Path,
    frame_indices: tuple[int, ...],
    output: Path,
    source_size: tuple[int, int],
    source_crop: list[float] | None = None,
) -> str:
    captures = (
        ("production", cv2.VideoCapture(str(baseline_path))),
        ("match090", cv2.VideoCapture(str(candidate_path))),
    )
    if any(not capture.isOpened() for _, capture in captures):
        raise RuntimeError(f"Unable to open regression videos for {output.name}")
    rows = []
    try:
        for label, capture in captures:
            tiles = []
            for frame_index in frame_indices:
                image = _read_frame(capture, frame_index)
                if source_crop is not None:
                    source_width, source_height = source_size
                    scale_x = image.shape[1] / source_width
                    scale_y = image.shape[0] / source_height
                    x1, y1, x2, y2 = (
                        round(source_crop[0] * scale_x),
                        round(source_crop[1] * scale_y),
                        round(source_crop[2] * scale_x),
                        round(source_crop[3] * scale_y),
                    )
                    image = image[y1:y2, x1:x2]
                tiles.append(_tile(image, f"{label} | frame {frame_index}"))
            rows.append(np.hstack(tiles))
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), np.vstack(rows)):
            raise RuntimeError(f"Unable to write {output}")
    finally:
        for _, capture in captures:
            capture.release()
    return str(output)


def _write_regression_sheets(results_root: Path) -> list[str]:
    robustness_root = ROOT / "results/tracker_robustness/botsort"
    output_root = results_root / "regression_sheets"
    outputs = [
        _comparison_sheet(
            robustness_root / "walking_multiworker/annotated.mp4",
            results_root / "match090_walking_multiworker/annotated.mp4",
            (252, 268, 269, 284, 300),
            output_root / "walking_crossing.jpg",
            (1920, 1080),
        ),
        _comparison_sheet(
            robustness_root / "helmet_transition/annotated.mp4",
            results_root / "match090_helmet_transition/annotated.mp4",
            (104, 106, 108, 110, 112),
            output_root / "helmet_duplicate.jpg",
            (3840, 2160),
            [1400, 0, 3500, 2160],
        ),
    ]
    return [str(Path(output).relative_to(results_root)) for output in outputs]


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Final bounded tracker review",
        "",
        "All runs use the frozen YOLO26n @ 960 detector and unchanged "
        "PPE/compliance settings.",
        "",
        "| Run | Tracks | Observations | Median frames | <0.5s | <1s | Lost | "
        "Recovered | Expired | Incidents |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in report["runs"].items():
        lines.append(
            f"| {name} | {summary['unique_tracks_created']} | "
            f"{summary['track_observations']} | "
            f"{summary['median_track_length_frames']} | "
            f"{summary['tracks_under_0_5_seconds']} | "
            f"{summary['tracks_under_1_second']} | "
            f"{summary['lost_transitions']} | "
            f"{summary['recovered_transitions']} | "
            f"{summary['expired_tracks']} | "
            f"{summary['temporal_compliance']['incidents_opened']} |"
        )
    lines.extend(
        [
            "",
            "## Automated review windows",
            "",
            "These are spatially plausible short-gap transitions for human review; "
            "they are not treated as ground truth.",
            "",
        ]
    )
    for candidate, sheet in zip(
        report["candidate_fragmentation_windows"], report["contact_sheets"], strict=True
    ):
        lines.append(
            f"- frame {candidate['frame']}: ID {candidate['previous_track_id']} -> "
            f"{candidate['new_track_id']}, gap {candidate['gap_frames']} frames; "
            f"`{sheet}`"
        )
    lines.extend(
        [
            "",
            "## Root cause and visual result",
            "",
            report["root_cause"],
            "",
            report["visual_result"],
            "",
            "## Regression comparison",
            "",
            "| Clip | Variant | Tracks | Observations | Lost | Recovered | Expired | "
            "Incidents |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for clip, variants in report["regressions"].items():
        for variant, summary in variants.items():
            lines.append(
                f"| {clip} | {variant} | {summary['unique_tracks_created']} | "
                f"{summary['track_observations']} | {summary['lost_transitions']} | "
                f"{summary['recovered_transitions']} | {summary['expired_tracks']} | "
                f"{summary['temporal_compliance']['incidents_opened']} |"
            )
    selected = report["selection"]["configuration"]
    lines.extend(
        [
            "",
            "## Selection",
            "",
            f"Selected: `{report['selection']['name']}`.",
            "",
            f"- track_high_thresh: `{selected['track_high_thresh']}`",
            f"- track_low_thresh: `{selected['track_low_thresh']}`",
            f"- new_track_thresh: `{selected['new_track_thresh']}`",
            f"- track_buffer: `{selected['track_buffer']}`",
            f"- match_thresh: `{selected['match_thresh']}`",
            f"- gmc_method: `{selected['gmc_method']}`",
            f"- with_reid: `{selected['with_reid']}`",
            "",
            report["known_limitation"],
            "",
            "Best candidate video: `match090_crowded/annotated.mp4`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    report = {
        "runs": {name: _run_summary(results_root / name) for name in RUNS},
        "candidate_fragmentation_windows": _fragmentation_candidates(
            results_root / RUNS[0], args.sheet_count
        ),
    }
    baseline_root = ROOT / "results/tracker_robustness/botsort"
    regression_names = (
        "compliant_multiworker",
        "helmet_transition",
        "walking_multiworker",
    )
    report["regressions"] = {
        name: {
            "production_baseline": _run_summary(baseline_root / name),
            "selected_match090": _run_summary(results_root / f"match090_{name}"),
        }
        for name in regression_names
    }
    report["root_cause"] = (
        "The dominant failure is detector-box instability during dense overlap and "
        "partial occlusion. Low-confidence Person boxes below the reportable 0.41 "
        "threshold provide useful second-stage continuity evidence; raising their "
        "floor increased losses. Buffer 90 and disabling GMC were neutral, while a "
        "more permissive association-cost gate reconnected more short gaps without "
        "a visible transfer in the reviewed clips."
    )
    report["visual_result"] = (
        "The selected candidate visibly preserves several workers through short "
        "occlusions, including the gray-shirt/yellow-trouser worker around frames "
        "182-202. The predefined walking crossing keeps the red-vest and "
        "dark-clothed workers distinct. The known helmet-transition duplicate "
        "remains bounded and is not materially worse."
    )
    report["selection"] = {
        "name": "BoT-SORT low011 buffer60 match090",
        "configuration": report["runs"]["match090_crowded"]["tracker_configuration"],
        "production_config": "configs/production_tracker.yaml",
    }
    report["known_limitation"] = (
        "ID churn is materially reduced, not eliminated. Dense multi-person overlap "
        "and weak or merged detector boxes can still fragment identities; aggregate "
        "IDs are not a ground-truth MOT score."
    )
    report["contact_sheets"] = _write_contact_sheets(
        results_root, report["candidate_fragmentation_windows"]
    )
    report["regression_sheets"] = _write_regression_sheets(results_root)
    (results_root / "metrics_comparison.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_markdown(results_root / "metrics_comparison.md", report)
    print(json.dumps(report["candidate_fragmentation_windows"], indent=2))


if __name__ == "__main__":
    main()
