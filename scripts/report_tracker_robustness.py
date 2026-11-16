"""Report the bounded ByteTrack versus BoT-SORT robustness experiment."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
DOC_ROOT = ROOT / "docs/experiments/tracker_robustness"
ARTIFACT_ROOT = ROOT / "results/tracker_robustness/review_artifacts"
TRACKERS = {
    "bytetrack": ROOT / "results/real_video_validation",
    "botsort": ROOT / "results/tracker_robustness/botsort",
    "botsort_reid": ROOT / "results/tracker_robustness/botsort_reid",
}
CLIPS = ("compliant_multiworker", "helmet_transition", "walking_multiworker")
WINDOWS = {
    "helmet_duplicate": {
        "clip": "helmet_transition",
        "start": 104,
        "end": 112,
        "frames": (104, 106, 108, 110, 112),
        "fps": 25.0,
        "crop": (1400, 0, 3500, 2160),
        "question": (
            "Is the three-frame duplicate avoided while one primary ID remains?"
        ),
    },
    "walking_crossing": {
        "clip": "walking_multiworker",
        "start": 252,
        "end": 300,
        "frames": (252, 268, 269, 284, 300),
        "fps": 23.976023976023978,
        "crop": (0, 0, 1920, 1080),
        "question": "Do dark-clothed and red-vest workers retain distinct IDs?",
    },
}
REVIEW_FINDINGS = {
    ("bytetrack", "helmet_duplicate"): {
        "observed_behavior": (
            "Track 2 overlaps Track 1 at frame 106, replaces it for frames 108-109, "
            "and Track 1 returns at frame 110."
        ),
        "verdict": "duplicate_not_avoided",
    },
    ("botsort", "helmet_duplicate"): {
        "observed_behavior": (
            "The same three-frame Track 2 duplicate occurs; primary Track 1 returns."
        ),
        "verdict": "duplicate_not_avoided",
    },
    ("botsort_reid", "helmet_duplicate"): {
        "observed_behavior": (
            "ReID produces the same duplicate sequence as BoT-SORT without ReID."
        ),
        "verdict": "duplicate_not_avoided",
    },
    ("bytetrack", "walking_crossing"): {
        "observed_behavior": (
            "Track 2 began on the dark-clothed worker but is on the red-vest worker "
            "in this window; Track 4 starts on the dark-clothed worker at frame 269."
        ),
        "verdict": "probable_id_switch",
    },
    ("botsort", "walking_crossing"): {
        "observed_behavior": (
            "Track 1 remains on the red-vest worker. Track 4 starts on the "
            "dark-clothed worker at frame 269, so identity transfer is avoided but "
            "dark-worker fragmentation remains."
        ),
        "verdict": "partial_clear_improvement",
    },
    ("botsort_reid", "walking_crossing"): {
        "observed_behavior": (
            "The identity behavior matches BoT-SORT without ReID; ReID adds no visible "
            "continuity benefit in the predeclared window."
        ),
        "verdict": "no_additional_improvement",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _crop(image, crop: tuple[int, int, int, int]):
    x1, y1, x2, y2 = crop
    return image[y1:y2, x1:x2]


def _read_frame(video: Path, frame_index: int):
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, image = capture.read()
    capture.release()
    if not ok or image is None:
        raise RuntimeError(f"Could not read frame {frame_index} from {video}")
    return image


def _tile(image, label: str, width: int = 340):
    scale = width / image.shape[1]
    resized = cv2.resize(image, (width, max(1, round(image.shape[0] * scale))))
    canvas = cv2.copyMakeBorder(resized, 38, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(
        canvas,
        label,
        (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _write_sheet(
    tracker: str,
    video: Path,
    window_name: str,
    window: dict[str, Any],
) -> Path:
    tiles = []
    for frame_index in window["frames"]:
        image = _crop(_read_frame(video, frame_index), window["crop"])
        timestamp = frame_index / window["fps"]
        tiles.append(_tile(image, f"{tracker} | f{frame_index} | {timestamp:.3f}s"))
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
    output = ARTIFACT_ROOT / "sheets" / f"{window_name}_{tracker}.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), cv2.hconcat(padded)):
        raise RuntimeError(f"Could not write {output}")
    return output


def _write_clip(
    tracker: str,
    video: Path,
    window_name: str,
    window: dict[str, Any],
) -> Path:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, window["start"])
    output = ARTIFACT_ROOT / "clips" / f"{window_name}_{tracker}.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    frame_index = window["start"]
    while frame_index <= window["end"]:
        ok, image = capture.read()
        if not ok or image is None:
            break
        cropped = _crop(image, window["crop"])
        scale = 960 / cropped.shape[1]
        rendered = cv2.resize(
            cropped,
            (960, max(1, round(cropped.shape[0] * scale))),
        )
        cv2.putText(
            rendered,
            f"{tracker} | frame {frame_index} | {frame_index / window['fps']:.3f}s",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if writer is None:
            writer = cv2.VideoWriter(
                str(output),
                cv2.VideoWriter_fourcc(*"mp4v"),
                window["fps"],
                (rendered.shape[1], rendered.shape[0]),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not create {output}")
        writer.write(rendered)
        frame_index += 1
    capture.release()
    if writer is not None:
        writer.release()
    return output


def _tracker_clip_metrics(result_dir: Path) -> dict[str, Any]:
    metadata = json.loads((result_dir / "run_metadata.json").read_text())
    tracks = _jsonl(result_dir / "tracks.jsonl")
    compliance = _jsonl(result_dir / "compliance_states.jsonl")
    incidents = _jsonl(result_dir / "incidents.jsonl")
    lengths = Counter(int(item["track_id"]) for item in tracks)
    states = Counter(
        record[equipment]["temporal_state"]
        for record in compliance
        for equipment in ("helmet", "vest")
    )
    return {
        "configuration": metadata["tracking"]["tracker"]["configuration"],
        "tracking": {
            **metadata["tracking"]["diagnostics"],
            "observations_per_track": {
                str(key): value for key, value in sorted(lengths.items())
            },
        },
        "tracking_timing_ms": metadata["tracking"]["timing"],
        "pipeline": {
            "effective_fps": metadata["timing"]["effective_pipeline_fps"],
            "end_to_end_latency_ms": metadata["timing"]["end_to_end_frame_latency"],
        },
        "association": metadata["ppe_association"]["diagnostics"],
        "temporal_state_counts": dict(sorted(states.items())),
        "incidents": incidents,
    }


def main() -> None:
    """Generate machine-readable comparisons and predeclared review artifacts."""
    metrics: dict[str, Any] = {
        "experiment": "bounded real-video tracking robustness comparison",
        "formal_mot_ground_truth": False,
        "public_demo_eligible": False,
        "video_hashes": {
            f"{clip}.mp4": _sha256(ROOT / f"data/demo_videos/{clip}.mp4")
            for clip in CLIPS
        },
        "tracker_configs": {
            "bytetrack": {
                "path": "configs/bytetrack_person.yaml",
                "sha256": _sha256(ROOT / "configs/bytetrack_person.yaml"),
            },
            "botsort": {
                "path": "configs/botsort_person.yaml",
                "sha256": _sha256(ROOT / "configs/botsort_person.yaml"),
            },
            "botsort_reid": {
                "path": "configs/botsort_reid_person.yaml",
                "sha256": _sha256(ROOT / "configs/botsort_reid_person.yaml"),
                "model": "yolo26n-cls.pt",
                "model_sha256": _sha256(ROOT / "yolo26n-cls.pt"),
            },
        },
        "predeclared_windows": WINDOWS,
        "review_findings": {
            f"{tracker}:{window}": finding
            for (tracker, window), finding in REVIEW_FINDINGS.items()
        },
        "final_selection": {
            "tracker": "botsort",
            "reid": False,
            "production_config": "configs/production_tracker.yaml",
            "production_tracker_changed": True,
            "reason": (
                "Prevents the observed walking identity transfer without stable-clip "
                "regression; ReID adds cost without additional continuity benefit."
            ),
        },
        "trackers": {},
        "review_artifacts": {},
    }
    manifest_rows = []
    for tracker, root in TRACKERS.items():
        metrics["trackers"][tracker] = {}
        for clip in CLIPS:
            metrics["trackers"][tracker][clip] = _tracker_clip_metrics(root / clip)
        for window_name, window in WINDOWS.items():
            video = root / window["clip"] / "annotated.mp4"
            sheet = _write_sheet(tracker, video, window_name, window)
            clip = _write_clip(tracker, video, window_name, window)
            metrics["review_artifacts"].setdefault(window_name, {})[tracker] = {
                "sheet": sheet.relative_to(ROOT).as_posix(),
                "clip": clip.relative_to(ROOT).as_posix(),
            }
            frame_tracks = _jsonl(root / window["clip"] / "tracks.jsonl")
            ids = sorted(
                {
                    int(item["track_id"])
                    for item in frame_tracks
                    if window["start"] <= int(item["frame_index"]) <= window["end"]
                }
            )
            manifest_rows.append(
                {
                    "clip": window["clip"],
                    "time_window": (
                        f"frames {window['start']}-{window['end']} / "
                        f"{window['start'] / window['fps']:.3f}-"
                        f"{window['end'] / window['fps']:.3f}s"
                    ),
                    "tracker": tracker,
                    "track_ids": ";".join(str(value) for value in ids),
                    "review_question": window["question"],
                    "observed_behavior": REVIEW_FINDINGS[(tracker, window_name)][
                        "observed_behavior"
                    ],
                    "verdict": REVIEW_FINDINGS[(tracker, window_name)]["verdict"],
                    "notes": sheet.relative_to(ROOT).as_posix(),
                }
            )
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    (DOC_ROOT / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    with (DOC_ROOT / "review_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(json.dumps(metrics["tracker_configs"], indent=2))


if __name__ == "__main__":
    main()
