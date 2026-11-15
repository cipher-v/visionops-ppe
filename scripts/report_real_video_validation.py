"""Build reproducible Checkpoint 13 metrics and compact visual review artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIDEO_ROOT = ROOT / "data/demo_videos"
RESULT_ROOT = ROOT / "results/real_video_validation"
DOC_ROOT = ROOT / "docs/experiments/real_video_validation"
REVIEW_ROOT = RESULT_ROOT / "review_artifacts"
CONFIG_PATHS = (
    ROOT / "configs/production_detector.yaml",
    ROOT / "configs/bytetrack_person.yaml",
    ROOT / "configs/ppe_association.yaml",
    ROOT / "configs/compliance_temporal.yaml",
)
PPE_CLASSES = ("helmet", "vest", "no_helmet", "no_vest")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL artifact, including valid empty files."""
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha256(path: Path) -> str:
    """Return the uppercase SHA-256 for a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def percentile(values: list[float], q: float) -> float | None:
    """Compute a deterministic percentile or None for an empty sample."""
    return float(np.percentile(values, q)) if values else None


def _video_properties(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    values = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": frame_count / fps,
    }
    capture.release()
    return values


def _frame(path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(frame_index, 0))
    ok, image = capture.read()
    capture.release()
    if not ok or image is None:
        raise RuntimeError(f"Could not read frame {frame_index} from {path}")
    return image


def _tile(image: np.ndarray, label: str, width: int = 480) -> np.ndarray:
    scale = width / image.shape[1]
    resized = cv2.resize(image, (width, max(1, round(image.shape[0] * scale))))
    canvas = cv2.copyMakeBorder(resized, 38, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(
        canvas,
        label,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _contact_sheet(
    video_path: Path,
    selections: list[tuple[int, str]],
    output_path: Path,
) -> None:
    tiles = [_tile(_frame(video_path, index), label) for index, label in selections]
    target_height = max(tile.shape[0] for tile in tiles)
    padded = [
        cv2.copyMakeBorder(
            tile,
            0,
            target_height - tile.shape[0],
            0,
            0,
            cv2.BORDER_CONSTANT,
        )
        for tile in tiles
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), cv2.hconcat(padded)):
        raise RuntimeError(f"Could not write contact sheet: {output_path}")


def _track_contact_sheet(
    video_path: Path,
    selections: list[tuple[int, str, list[float]]],
    output_path: Path,
) -> None:
    tiles = []
    for frame_index, label, bbox in selections:
        image = _frame(video_path, frame_index)
        x1, y1, x2, y2 = (round(value) for value in bbox)
        margin_x = round((x2 - x1) * 0.08)
        margin_y = round((y2 - y1) * 0.08)
        crop = image[
            max(0, y1 - margin_y) : min(image.shape[0], y2 + margin_y),
            max(0, x1 - margin_x) : min(image.shape[1], x2 + margin_x),
        ]
        tiles.append(_tile(crop, label))
    target_height = max(tile.shape[0] for tile in tiles)
    padded = [
        cv2.copyMakeBorder(
            tile,
            0,
            target_height - tile.shape[0],
            0,
            0,
            cv2.BORDER_CONSTANT,
        )
        for tile in tiles
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), cv2.hconcat(padded)):
        raise RuntimeError(f"Could not write track contact sheet: {output_path}")


def _track_stats(
    tracks: list[dict[str, Any]], fps: float
) -> tuple[dict[str, Any], dict[int, list[int]]]:
    frames_by_track: dict[int, list[int]] = defaultdict(list)
    for track in tracks:
        frames_by_track[int(track["track_id"])].append(int(track["frame_index"]))
    lengths = [len(frames) for frames in frames_by_track.values()]
    return (
        {
            "observations_per_track": {
                str(key): len(value) for key, value in sorted(frames_by_track.items())
            },
            "median_track_length_frames": statistics.median(lengths) if lengths else 0,
            "longest_track_frames": max(lengths, default=0),
            "singleton_tracks": sum(length == 1 for length in lengths),
            "very_short_tracks_under_one_second": sum(
                length < fps for length in lengths
            ),
        },
        frames_by_track,
    )


def _merge_incidents(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    incidents: dict[str, dict[str, Any]] = {}
    for event in events:
        incident = event["incident"]
        incident_id = incident["incident_id"]
        merged = incidents.setdefault(incident_id, {})
        merged.update(incident)
    return [incidents[key] for key in sorted(incidents)]


def _primary_detection(equipment: dict[str, Any], state: str) -> dict[str, Any] | None:
    if state == "present":
        return equipment.get("positive_detection")
    if state == "violation":
        return equipment.get("negative_detection")
    return equipment.get("positive_detection") or equipment.get("negative_detection")


def _sample_candidates(
    clip: str,
    associations: list[dict[str, Any]],
    detections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    desired = {
        "helmet_association": ("helmet", "present"),
        "vest_association": ("vest", "present"),
        "no_helmet_association": ("helmet", "violation"),
        "no_vest_association": ("vest", "violation"),
        "conflict": (None, "conflict"),
        "unknown": (None, "unknown"),
    }
    for review_type, (equipment_name, state) in desired.items():
        found = None
        for record in associations:
            names = (equipment_name,) if equipment_name else ("helmet", "vest")
            for name in names:
                equipment = record[name]
                if equipment["state"] == state:
                    found = (record, name, _primary_detection(equipment, state))
                    break
            if found:
                break
        if found:
            record, name, detection = found
            samples.append(
                {
                    "clip": clip,
                    "frame_index": record["frame_index"],
                    "timestamp_seconds": record["timestamp_seconds"],
                    "track_id": record["track_id"],
                    "review_type": review_type,
                    "predicted_state": f"{name}:{state}",
                    "ppe_class": detection.get("ppe_class_name") if detection else "",
                    "association_score": (
                        detection.get("association_score") if detection else ""
                    ),
                    "geometry": detection or {},
                    "notes": "Requires human visual review; no ground-truth verdict.",
                    "human_verdict": "",
                }
            )

    for record in associations:
        ambiguous = None
        equipment_name = ""
        for name in ("helmet", "vest"):
            equipment = record[name]
            candidates = (
                equipment["positive_candidates"] + equipment["negative_candidates"]
            )
            ambiguous = next((item for item in candidates if item["ambiguous"]), None)
            if ambiguous:
                equipment_name = name
                break
        if ambiguous:
            samples.append(
                {
                    "clip": clip,
                    "frame_index": record["frame_index"],
                    "timestamp_seconds": record["timestamp_seconds"],
                    "track_id": record["track_id"],
                    "review_type": "ambiguous_association",
                    "predicted_state": (
                        f"{equipment_name}:{record[equipment_name]['state']}"
                    ),
                    "ppe_class": ambiguous["ppe_class_name"],
                    "association_score": ambiguous["association_score"],
                    "geometry": ambiguous,
                    "notes": "Multiple candidate people; requires human review.",
                    "human_verdict": "",
                }
            )
            break

    associations_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in associations:
        associations_by_frame[int(record["frame_index"])].append(record)
    detections_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for detection in detections:
        detections_by_frame[int(detection["frame_index"])].append(detection)
    for frame_index, frame_detections in sorted(detections_by_frame.items()):
        assigned: set[int] = set()
        for record in associations_by_frame.get(frame_index, []):
            for name in ("helmet", "vest"):
                for candidate in (
                    record[name]["positive_candidates"]
                    + record[name]["negative_candidates"]
                ):
                    assigned.add(int(candidate["detection_index"]))
        for index, detection in enumerate(frame_detections):
            if detection["class_name"] in PPE_CLASSES and index not in assigned:
                samples.append(
                    {
                        "clip": clip,
                        "frame_index": frame_index,
                        "timestamp_seconds": detection["timestamp_seconds"],
                        "track_id": "",
                        "review_type": "unassigned_ppe",
                        "predicted_state": "unassigned",
                        "ppe_class": detection["class_name"],
                        "association_score": "",
                        "geometry": {"ppe_bbox_xyxy": detection["bbox_xyxy"]},
                        "notes": "Detected PPE had no accepted Person association.",
                        "human_verdict": "",
                    }
                )
                return samples
    return samples


def _write_review_image(sample: dict[str, Any]) -> str:
    clip = sample["clip"]
    frame_index = int(sample["frame_index"])
    annotated = RESULT_ROOT / clip / "annotated.mp4"
    name = f"{clip}_{sample['review_type']}_f{frame_index}.jpg"
    output = REVIEW_ROOT / "associations" / name
    output.parent.mkdir(parents=True, exist_ok=True)
    image = _frame(annotated, frame_index)
    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"Could not write review frame: {output}")
    return output.relative_to(ROOT).as_posix()


def _write_manifest(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "clip",
        "frame_or_time",
        "track_id",
        "review_type",
        "predicted_state",
        "ppe_class",
        "association_score",
        "geometry_diagnostics",
        "artifact",
        "notes",
        "human_verdict",
    ]
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    with (DOC_ROOT / "review_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Generate metrics, selections, contact sheets, and review manifest."""
    videos = sorted(VIDEO_ROOT.glob("*.mp4"))
    if not videos:
        raise RuntimeError(f"No MP4 videos found under {VIDEO_ROOT}")
    metrics: dict[str, Any] = {
        "purpose": "observational real-video system validation",
        "accuracy_claim": False,
        "provenance": {
            "source": "locally supplied by project owner",
            "original_urls": None,
            "license": None,
            "redistribution_allowed": None,
        },
        "configuration_freeze": {
            path.relative_to(ROOT).as_posix(): sha256(path) for path in CONFIG_PATHS
        },
        "clips": {},
    }
    aggregate_detector: list[float] = []
    aggregate_pipeline: list[float] = []
    aggregate_frames = 0
    aggregate_wall = 0.0
    aggregate_total_wall = 0.0
    manifest_samples: list[dict[str, Any]] = []

    for video in videos:
        clip = video.stem
        result_dir = RESULT_ROOT / clip
        metadata = json.loads(
            (result_dir / "run_metadata.json").read_text(encoding="utf-8")
        )
        if metadata["processing"]["frame_stride"] != 1:
            raise RuntimeError(f"{clip} was not processed at frame stride 1")
        if not metadata["temporal_compliance"]["real_video_temporal_validation"]:
            raise RuntimeError(f"{clip} is not marked as real-video validation")
        detections = read_jsonl(result_dir / "detections.jsonl")
        tracks = read_jsonl(result_dir / "tracks.jsonl")
        associations = read_jsonl(result_dir / "ppe_associations.jsonl")
        compliance = read_jsonl(result_dir / "compliance_states.jsonl")
        incident_events = read_jsonl(result_dir / "incidents.jsonl")
        properties = _video_properties(video)
        track_stats, frames_by_track = _track_stats(tracks, properties["fps"])
        detection_counts = Counter(item["class_name"] for item in detections)
        association_diagnostics = metadata["ppe_association"]["diagnostics"]
        assigned_by_class = association_diagnostics["per_class_assignment_counts"]
        assignment_rates = {
            name: (
                assigned_by_class.get(name, 0) / detection_counts.get(name, 0)
                if detection_counts.get(name, 0)
                else None
            )
            for name in PPE_CLASSES
        }
        incidents = _merge_incidents(incident_events)
        temporal_counts = Counter()
        for record in compliance:
            temporal_counts[record["helmet"]["temporal_state"]] += 1
            temporal_counts[record["vest"]["temporal_state"]] += 1
        clip_metrics = {
            "video": {
                "filename": video.name,
                "sha256": sha256(video),
                **properties,
                "source": "locally supplied; original URL not provided",
                "license": None,
            },
            "processing": metadata["processing"],
            "performance": {
                "processing_wall_clock_seconds": metadata["timing"][
                    "processing_wall_clock_seconds"
                ],
                "total_wall_clock_seconds": metadata["timing"][
                    "total_wall_clock_seconds"
                ],
                "effective_fps": metadata["timing"]["effective_pipeline_fps"],
                "detector_latency": metadata["timing"]["detector_latency"],
                "end_to_end_latency": metadata["timing"]["end_to_end_frame_latency"],
            },
            "detections": dict(sorted(detection_counts.items())),
            "tracking": {
                **metadata["tracking"]["diagnostics"],
                **track_stats,
            },
            "association": {
                **association_diagnostics,
                "assignment_rate_by_class": assignment_rates,
            },
            "temporal": {
                "state_counts": dict(sorted(temporal_counts.items())),
                **metadata["temporal_compliance"]["diagnostics"],
            },
            "incidents": incidents,
        }
        metrics["clips"][clip] = clip_metrics
        samples = metadata["timing"].get("latency_samples_ms")
        if not samples:
            raise RuntimeError(f"{clip} has no aggregate latency samples")
        aggregate_detector.extend(float(value) for value in samples["detector"])
        aggregate_pipeline.extend(float(value) for value in samples["end_to_end"])
        aggregate_frames += int(metadata["processing"]["frames_processed"])
        aggregate_wall += float(metadata["timing"]["processing_wall_clock_seconds"])
        aggregate_total_wall += float(metadata["timing"]["total_wall_clock_seconds"])

        longest = sorted(
            frames_by_track.items(), key=lambda item: (-len(item[1]), item[0])
        )[:2]
        selected_track_ids = [track_id for track_id, _ in longest]
        gap_ranked = sorted(
            frames_by_track.items(),
            key=lambda item: (
                -max(
                    (
                        current - previous
                        for previous, current in zip(item[1], item[1][1:])
                    ),
                    default=1,
                ),
                item[0],
            ),
        )
        for track_id, frames in gap_ranked[:2]:
            if any(b - a > 1 for a, b in zip(frames, frames[1:])):
                selected_track_ids.append(track_id)
        selected_track_ids.extend(int(item["track_id"]) for item in incidents)
        track_lookup = {
            (int(item["track_id"]), int(item["frame_index"])): item for item in tracks
        }
        for track_id in dict.fromkeys(selected_track_ids):
            frames = frames_by_track[track_id]
            selected = [frames[0], frames[len(frames) // 2], frames[-1]]
            output = REVIEW_ROOT / "tracks" / f"{clip}_track_{track_id}.jpg"
            _track_contact_sheet(
                result_dir / "annotated.mp4",
                [
                    (
                        index,
                        f"Track {track_id} | frame {index}",
                        track_lookup[(track_id, index)]["bbox_xyxy"],
                    )
                    for index in selected
                ],
                output,
            )
            manifest_samples.append(
                {
                    "clip": clip,
                    "frame_index": selected[1],
                    "timestamp_seconds": selected[1] / properties["fps"],
                    "track_id": track_id,
                    "review_type": "track_continuity",
                    "predicted_state": "same_track_id_across_time",
                    "ppe_class": "",
                    "association_score": "",
                    "geometry": {},
                    "artifact": output.relative_to(ROOT).as_posix(),
                    "notes": (
                        "Contact sheet selected from longest tracks; "
                        "human review required."
                    ),
                    "human_verdict": "",
                }
            )
        manifest_samples.extend(_sample_candidates(clip, associations, detections))

        for incident in incidents:
            confirmed = float(incident["confirmed_at_seconds"])
            times = [max(0.0, confirmed - 1.0), confirmed, confirmed + 1.0]
            selections = [
                (
                    min(
                        round(value * properties["fps"]), properties["frame_count"] - 1
                    ),
                    f"{value:.2f}s",
                )
                for value in times
            ]
            output = REVIEW_ROOT / "incidents" / f"{incident['incident_id']}.jpg"
            _contact_sheet(result_dir / "annotated.mp4", selections, output)
            manifest_samples.append(
                {
                    "clip": clip,
                    "frame_index": round(confirmed * properties["fps"]),
                    "timestamp_seconds": confirmed,
                    "track_id": incident["track_id"],
                    "review_type": "incident",
                    "predicted_state": f"{incident['equipment']}:confirmed_violation",
                    "ppe_class": incident["equipment"],
                    "association_score": "",
                    "geometry": {},
                    "artifact": output.relative_to(ROOT).as_posix(),
                    "notes": (
                        f"started={incident['started_at_seconds']}; "
                        f"confirmed={confirmed}; "
                        f"cleared={incident['cleared_at_seconds']}; "
                        f"close_reason={incident['close_reason']}"
                    ),
                    "human_verdict": "",
                }
            )

    metrics["aggregate"] = {
        "total_frames": aggregate_frames,
        "total_processing_wall_clock_seconds": aggregate_wall,
        "total_wall_clock_seconds": aggregate_total_wall,
        "weighted_effective_fps": aggregate_frames / aggregate_wall,
        "detector_latency_ms": {
            "p50": percentile(aggregate_detector, 50),
            "p95": percentile(aggregate_detector, 95),
        },
        "end_to_end_latency_ms": {
            "p50": percentile(aggregate_pipeline, 50),
            "p95": percentile(aggregate_pipeline, 95),
        },
        "hardware": "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
        "software": {
            "python": metadata["environment"]["python"],
            "torch": metadata["environment"]["torch"],
            "cuda": metadata["environment"]["torch_cuda"],
            "opencv": metadata["environment"]["opencv"],
            "ultralytics": metadata["tracking"]["tracker"]["ultralytics_version"],
        },
    }

    rows = []
    for sample in manifest_samples:
        if "artifact" not in sample:
            sample["artifact"] = _write_review_image(sample)
        rows.append(
            {
                "clip": sample["clip"],
                "frame_or_time": (
                    f"frame {sample['frame_index']} / "
                    f"{float(sample['timestamp_seconds']):.3f}s"
                ),
                "track_id": sample["track_id"],
                "review_type": sample["review_type"],
                "predicted_state": sample["predicted_state"],
                "ppe_class": sample["ppe_class"],
                "association_score": sample["association_score"],
                "geometry_diagnostics": json.dumps(sample["geometry"], sort_keys=True),
                "artifact": sample["artifact"],
                "notes": sample["notes"],
                "human_verdict": "",
            }
        )
    _write_manifest(rows)
    (DOC_ROOT / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics["aggregate"], indent=2))


if __name__ == "__main__":
    main()
