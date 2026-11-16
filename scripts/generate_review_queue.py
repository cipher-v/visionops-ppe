"""Build a bounded human-review queue from existing real-video pipeline outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from visionops.review.schema import ReviewCandidate
from visionops.review.selection import (
    ReviewConfig,
    perceptual_hash,
    priority_score,
    select_candidates,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "cp18-real-v1"
OUTPUT = ROOT / "results/review_queue" / RUN_ID
PIPELINE = ROOT / "results/deployment_optimization/video/pytorch"
DRIFT = ROOT / "results/robustness_drift/video"
VIDEOS = ROOT / "data/demo_videos"
CLIPS = ("compliant_multiworker", "helmet_transition", "walking_multiworker")
THRESHOLDS = {
    "helmet": 0.32,
    "vest": 0.19,
    "Person": 0.41,
    "no_helmet": 0.27,
    "no_vest": 0.11,
}
CONFIG = ReviewConfig()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return (
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if path.is_file()
        else []
    )


def make_candidate(
    source: str,
    frame: int,
    timestamp: float,
    candidate_type: str,
    reasons: list[str],
    **values: object,
) -> ReviewCandidate:
    equipment = values.get("equipment")
    track_id = values.get("track_id")
    identity = (
        f"{source}|{frame}|{candidate_type}|{track_id}|{equipment}|"
        f"{'|'.join(sorted(reasons))}"
    )
    candidate_id = hashlib.sha256(identity.encode()).hexdigest()[:20]
    margin = values.pop("confidence_margin", None)
    return ReviewCandidate(
        candidate_id,
        source,
        frame,
        timestamp,
        candidate_type,
        priority_score(
            reasons,
            CONFIG.reason_weights,
            margin if isinstance(margin, float) else None,
        ),
        reasons,
        **values,
    )


def collect_clip(clip: str) -> tuple[list[ReviewCandidate], list[float]]:
    folder = PIPELINE / clip
    source = clip
    candidates: list[ReviewCandidate] = []
    timings: list[float] = []
    detections = read_jsonl(folder / "detections.jsonl")
    detections_by_frame: dict[int, list[dict[str, object]]] = defaultdict(list)
    for detection in detections:
        detections_by_frame[int(detection["frame_index"])].append(detection)
        started = time.perf_counter()
        name, confidence = str(detection["class_name"]), float(detection["confidence"])
        margin = confidence - THRESHOLDS[name]
        if 0 <= margin <= 0.08:
            candidates.append(
                make_candidate(
                    source,
                    int(detection["frame_index"]),
                    float(detection["timestamp_seconds"]),
                    "detector_uncertainty",
                    ["low_confidence"],
                    equipment=name,
                    confidences={name: confidence},
                    confidence_margin=margin,
                )
            )
        timings.append((time.perf_counter() - started) * 1000)
    associations = read_jsonl(folder / "ppe_associations.jsonl")
    assigned_by_frame: dict[int, set[int]] = defaultdict(set)
    for record in associations:
        started = time.perf_counter()
        for equipment in ("helmet", "vest"):
            evidence = record[equipment]
            state = evidence["state"]
            common = {"track_id": int(record["track_id"]), "equipment": equipment}
            if state == "conflict":
                candidates.append(
                    make_candidate(
                        source,
                        int(record["frame_index"]),
                        float(record["timestamp_seconds"]),
                        "frame_conflict",
                        ["frame_conflict"],
                        temporal_state="conflict",
                        **common,
                    )
                )
            elif state == "unknown":
                candidates.append(
                    make_candidate(
                        source,
                        int(record["frame_index"]),
                        float(record["timestamp_seconds"]),
                        "temporal_uncertainty",
                        ["repeated_unknown"],
                        temporal_state="unknown",
                        **common,
                    )
                )
            for key in ("positive_detection", "negative_detection"):
                item = evidence.get(key)
                if not item:
                    continue
                assigned_by_frame[int(record["frame_index"])].add(
                    int(item["detection_index"])
                )
                if (
                    item.get("ambiguous")
                    or int(item.get("candidate_person_count", 1)) > 1
                ):
                    candidates.append(
                        make_candidate(
                            source,
                            int(record["frame_index"]),
                            float(record["timestamp_seconds"]),
                            "association_uncertainty",
                            ["association_ambiguous"],
                            association_score=float(item["association_score"]),
                            confidences={
                                str(item["ppe_class_name"]): float(
                                    item["ppe_confidence"]
                                )
                            },
                            **common,
                        )
                    )
                elif float(item["association_score"]) <= 0.60:
                    candidates.append(
                        make_candidate(
                            source,
                            int(record["frame_index"]),
                            float(record["timestamp_seconds"]),
                            "association_uncertainty",
                            ["association_near_gate"],
                            association_score=float(item["association_score"]),
                            confidences={
                                str(item["ppe_class_name"]): float(
                                    item["ppe_confidence"]
                                )
                            },
                            **common,
                        )
                    )
        timings.append((time.perf_counter() - started) * 1000)
    for frame_index, frame_detections in detections_by_frame.items():
        for detection_index, detection in enumerate(frame_detections):
            name = str(detection["class_name"])
            confidence = float(detection["confidence"])
            if (
                name != "Person"
                and detection_index not in assigned_by_frame[frame_index]
                and confidence >= THRESHOLDS[name] + 0.05
            ):
                candidates.append(
                    make_candidate(
                        source,
                        frame_index,
                        float(detection["timestamp_seconds"]),
                        "association_uncertainty",
                        ["high_confidence_unassigned"],
                        equipment=name,
                        confidences={name: confidence},
                    )
                )
    for record in read_jsonl(folder / "compliance_states.jsonl"):
        started = time.perf_counter()
        for equipment in ("helmet", "vest"):
            state = record[equipment]["temporal_state"]
            if state in {"suspected_violation", "confirmed_violation"}:
                reason = (
                    "confirmed_incident"
                    if state == "confirmed_violation"
                    else "suspected_violation"
                )
                candidates.append(
                    make_candidate(
                        source,
                        int(record["frame_index"]),
                        float(record["timestamp_seconds"]),
                        "temporal_uncertainty"
                        if state.startswith("suspected")
                        else "incident_context",
                        [reason],
                        track_id=int(record["track_id"]),
                        equipment=equipment,
                        temporal_state=str(state),
                        incident_id=record[equipment].get("active_incident_id"),
                    )
                )
        timings.append((time.perf_counter() - started) * 1000)
    tracks = read_jsonl(folder / "tracks.jsonl")
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in tracks:
        grouped[int(record["track_id"])].append(record)
        if record.get("track_state") in {"lost", "recovered"}:
            reason = f"track_{record['track_state']}"
            candidates.append(
                make_candidate(
                    source,
                    int(record["frame_index"]),
                    float(record["timestamp_seconds"]),
                    "tracking_event",
                    [reason],
                    track_id=int(record["track_id"]),
                )
            )
    fps = float(
        json.loads((folder / "run_metadata.json").read_text(encoding="utf-8"))[
            "source"
        ]["reported_fps"]
        or 30
    )
    for track_id, records in grouped.items():
        if len(records) < fps:
            record = records[len(records) // 2]
            candidates.append(
                make_candidate(
                    source,
                    int(record["frame_index"]),
                    float(record["timestamp_seconds"]),
                    "tracking_event",
                    ["short_track"],
                    track_id=track_id,
                )
            )
    for event in read_jsonl(folder / "incidents.jsonl"):
        incident = event["incident"]
        reason = (
            "incident_closed"
            if event["event_type"] == "incident_closed"
            else "confirmed_incident"
        )
        timestamp = float(event["event_timestamp_seconds"])
        candidates.append(
            make_candidate(
                source,
                round(timestamp * fps),
                timestamp,
                "incident_context",
                [reason],
                track_id=int(incident["track_id"]),
                equipment=str(incident["equipment"]),
                temporal_state=str(incident["status"]),
                incident_id=str(incident["incident_id"]),
            )
        )
    for window in read_jsonl(DRIFT / clip / "drift_windows.jsonl"):
        level = str(window["overall_level"])
        if level in {"warning", "high"}:
            timestamp = (
                float(window["window_start_seconds"])
                + float(window["window_end_seconds"])
            ) / 2
            candidates.append(
                make_candidate(
                    source,
                    round(timestamp * fps),
                    timestamp,
                    "drift_telemetry",
                    [f"drift_{level}"],
                    drift_level=level,
                )
            )
    return candidates, timings


def attach_hashes(clip: str, candidates: list[ReviewCandidate]) -> None:
    wanted = {item.frame_index for item in candidates if item.source_id == clip}
    capture = cv2.VideoCapture(str(VIDEOS / f"{clip}.mp4"))
    index = 0
    while wanted:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            value = perceptual_hash(frame)
            for item in candidates:
                if item.source_id == clip and item.frame_index == index:
                    item.visual_hash = value
            wanted.remove(index)
        index += 1
    capture.release()


def save_artifacts(selected: list[ReviewCandidate]) -> tuple[float, int]:
    started = time.perf_counter()
    image_dir = OUTPUT / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for clip in CLIPS:
        wanted = {item.frame_index: item for item in selected if item.source_id == clip}
        capture = cv2.VideoCapture(str(VIDEOS / f"{clip}.mp4"))
        index = 0
        while wanted:
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                item = wanted.pop(index)
                path = image_dir / f"{item.candidate_id}.jpg"
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                item.artifact_reference = str(path.relative_to(ROOT))
            index += 1
        capture.release()
    size = sum(path.stat().st_size for path in image_dir.glob("*.jpg"))
    return (time.perf_counter() - started) * 1000, size


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw: list[ReviewCandidate] = []
    event_timings: list[float] = []
    for clip in CLIPS:
        items, timings = collect_clip(clip)
        raw.extend(items)
        event_timings.extend(timings)
        attach_hashes(clip, items)
    selection_started = time.perf_counter()
    selected, stages = select_candidates(raw, CONFIG)
    selection_ms = (time.perf_counter() - selection_started) * 1000
    artifact_ms, artifact_bytes = save_artifacts(selected)
    candidates_path = OUTPUT / "candidates.jsonl"
    candidates_path.write_text(
        "".join(json.dumps(item.to_dict()) + "\n" for item in selected),
        encoding="utf-8",
    )
    fields = [
        "candidate_id",
        "source_id",
        "timestamp_seconds",
        "track_id",
        "candidate_type",
        "reason_codes",
        "priority",
        "equipment",
        "confidences",
        "association_score",
        "incident_id",
        "drift_level",
        "human_verdict",
        "human_review_label",
        "review_notes",
    ]
    with (OUTPUT / "review_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in selected:
            values = item.to_dict()
            writer.writerow(
                {
                    field: json.dumps(values[field])
                    if field in {"reason_codes", "confidences"}
                    else values[field]
                    for field in fields
                }
            )
    reason_counts = Counter(reason for item in selected for reason in item.reason_codes)
    summary = {
        "run_id": RUN_ID,
        "stages": stages,
        "dedup_reduction_percent": round(
            (1 - stages["selected"] / stages["raw"]) * 100, 3
        )
        if stages["raw"]
        else 0,
        "by_type": Counter(item.candidate_type for item in selected),
        "by_equipment": Counter(item.equipment or "none" for item in selected),
        "by_clip": Counter(item.source_id for item in selected),
        "reason_distribution": reason_counts,
        "priority": {
            "min": min(item.priority for item in selected),
            "mean": statistics.fmean(item.priority for item in selected),
            "max": max(item.priority for item in selected),
        },
        "incident_associated_percent": 100
        * sum(item.incident_id is not None for item in selected)
        / len(selected),
        "drift_associated_percent": 100
        * sum(item.drift_level is not None for item in selected)
        / len(selected),
        "artifact_bytes": artifact_bytes,
        "performance": {
            "event_mean_ms": statistics.fmean(event_timings),
            "event_p50_ms": statistics.median(event_timings),
            "event_p95_ms": sorted(event_timings)[
                round((len(event_timings) - 1) * 0.95)
            ],
            "selection_total_ms": selection_ms,
            "artifact_generation_ms": artifact_ms,
        },
        "human_fields_blank": all(
            item.human_verdict is None
            and item.human_review_label is None
            and item.review_notes is None
            for item in selected
        ),
        "drift_diagnostic_only": True,
        "accuracy_failure_predictor": False,
    }
    (OUTPUT / "queue_summary.json").write_text(
        json.dumps(summary, indent=2, default=dict), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=dict))


if __name__ == "__main__":
    main()
