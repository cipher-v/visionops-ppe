"""Person-only adapter for supported Ultralytics ByteTrack and BoT-SORT."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import ultralytics
import yaml
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker

from visionops.tracking.schema import PersonTrack, TrackerEvents, TrackerFrameResult
from visionops.video.schema import VideoDetection, normalized_box

PERSON_CLASS_ID = 2
PERSON_CLASS_NAME = "Person"
FROZEN_PERSON_THRESHOLD = 0.41


class TrackerInitializationError(RuntimeError):
    """Raised when a supported Ultralytics tracker cannot initialize."""


@dataclass(frozen=True)
class TrackerSettings:
    """Immutable repository-owned Person tracker configuration."""

    tracker_type: str = "bytetrack"
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.80
    fuse_score: bool = True
    appearance_reid: bool = False
    gmc_method: str | None = None
    proximity_thresh: float = 0.50
    appearance_thresh: float = 0.80
    with_reid: bool = False
    model: str = "auto"

    @classmethod
    def from_yaml(cls, path: Path) -> TrackerSettings:
        """Load and strictly validate the checked-in tracker configuration."""
        if not path.is_file():
            raise FileNotFoundError(f"Tracker configuration does not exist: {path}")
        values = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Tracker configuration must be a YAML mapping")
        try:
            settings = cls(**values)
        except TypeError as error:
            raise ValueError(f"Malformed tracker configuration: {error}") from error
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.tracker_type not in {"bytetrack", "botsort"}:
            raise ValueError("Tracker type must be 'bytetrack' or 'botsort'")
        thresholds = (
            self.track_low_thresh,
            self.track_high_thresh,
            self.new_track_thresh,
            self.match_thresh,
        )
        if any(not 0 <= value <= 1 for value in thresholds):
            raise ValueError("Tracker thresholds must be normalized")
        if self.track_low_thresh > self.track_high_thresh:
            raise ValueError("Low tracker threshold cannot exceed high threshold")
        if self.track_buffer < 0:
            raise ValueError("Track buffer cannot be negative")
        if self.appearance_reid:
            raise ValueError("Use the official 'with_reid' setting")
        if self.tracker_type == "bytetrack" and self.with_reid:
            raise ValueError("ByteTrack does not support appearance ReID")
        if self.tracker_type == "botsort":
            if self.gmc_method not in {
                "sparseOptFlow",
                "orb",
                "sift",
                "ecc",
                "none",
            }:
                raise ValueError("BoT-SORT requires a supported GMC method")
            if not 0 <= self.proximity_thresh <= 1:
                raise ValueError("BoT-SORT proximity threshold must be normalized")
            if not 0 <= self.appearance_thresh <= 1:
                raise ValueError("BoT-SORT appearance threshold must be normalized")
            if self.with_reid and self.model == "auto":
                raise ValueError(
                    "This adapter requires an external model for BoT-SORT ReID"
                )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class PersonByteTracker:
    """Feed only frozen-threshold Person detections into one supported tracker."""

    def __init__(self, settings: TrackerSettings) -> None:
        settings.validate()
        try:
            args = SimpleNamespace(
                track_high_thresh=settings.track_high_thresh,
                track_low_thresh=settings.track_low_thresh,
                new_track_thresh=settings.new_track_thresh,
                track_buffer=settings.track_buffer,
                match_thresh=settings.match_thresh,
                fuse_score=settings.fuse_score,
                gmc_method=settings.gmc_method,
                proximity_thresh=settings.proximity_thresh,
                appearance_thresh=settings.appearance_thresh,
                with_reid=settings.with_reid,
                model=settings.model,
                device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            )
            tracker_class = (
                BYTETracker if settings.tracker_type == "bytetrack" else BOTSORT
            )
            self._tracker = tracker_class(args=args)
        except Exception as error:
            raise TrackerInitializationError(
                f"Unable to initialize Ultralytics {settings.tracker_type}: {error}"
            ) from error
        self.settings = settings
        self._seen_ids: set[int] = set()
        self._previous_lost_ids: set[int] = set()
        self._expired_ids: set[int] = set()
        self._last_frame_index: int | None = None
        self._last_timestamp: float | None = None

    @property
    def name(self) -> str:
        return (
            "Ultralytics ByteTrack"
            if self.settings.tracker_type == "bytetrack"
            else "Ultralytics BoT-SORT"
        )

    @property
    def version(self) -> str:
        return ultralytics.__version__

    @property
    def live_track_ids(self) -> tuple[int, ...]:
        """Return IDs still tracked or buffered as lost at end of input."""
        active = {int(track.track_id) for track in self._tracker.tracked_stracks}
        lost = {int(track.track_id) for track in self._tracker.lost_stracks}
        return tuple(sorted((active | lost) - self._expired_ids))

    def metadata(self) -> dict[str, object]:
        implementation = (
            "ultralytics.trackers.byte_tracker.BYTETracker"
            if self.settings.tracker_type == "bytetrack"
            else "ultralytics.trackers.bot_sort.BOTSORT"
        )
        return {
            "name": self.name,
            "ultralytics_version": self.version,
            "implementation": implementation,
            "appearance_reid": self.settings.with_reid,
            "reid_model": self.settings.model if self.settings.with_reid else None,
            "configuration": self.settings.to_dict(),
            "person_class_id": PERSON_CLASS_ID,
            "person_confidence_threshold": FROZEN_PERSON_THRESHOLD,
            "track_ids_local_to_run": True,
        }

    def _validate_frame(self, frame_index: int, timestamp_seconds: float) -> None:
        if frame_index < 0:
            raise ValueError("Frame index cannot be negative")
        if not math.isfinite(timestamp_seconds) or timestamp_seconds < 0:
            raise ValueError("Timestamp must be finite and non-negative")
        if self._last_frame_index is not None and frame_index <= self._last_frame_index:
            raise ValueError("Frame indices must increase strictly")
        if (
            self._last_timestamp is not None
            and timestamp_seconds < self._last_timestamp
        ):
            raise ValueError("Timestamps cannot move backward")
        self._last_frame_index = frame_index
        self._last_timestamp = timestamp_seconds

    @staticmethod
    def _valid_box(detection: VideoDetection) -> bool:
        if len(detection.bbox_xyxy) != 4:
            return False
        x1, y1, x2, y2 = detection.bbox_xyxy
        return all(math.isfinite(value) for value in detection.bbox_xyxy) and (
            x2 > x1 and y2 > y1
        )

    def update(
        self,
        frame_index: int,
        timestamp_seconds: float,
        detections: list[VideoDetection],
        frame_shape: tuple[int, int],
        source_id: str,
        frame: np.ndarray | None = None,
    ) -> TrackerFrameResult:
        """Update one processed frame without accepting PPE or weak Person boxes."""
        self._validate_frame(frame_index, timestamp_seconds)
        height, width = frame_shape
        if height <= 0 or width <= 0:
            raise ValueError("Frame dimensions must be positive")

        person_detections = [
            detection
            for detection in detections
            if detection.class_id == PERSON_CLASS_ID
            and detection.class_name == PERSON_CLASS_NAME
        ]
        accepted = [
            detection
            for detection in person_detections
            if detection.confidence >= FROZEN_PERSON_THRESHOLD
            and math.isfinite(detection.confidence)
            and self._valid_box(detection)
        ]
        rejected = len(person_detections) - len(accepted)
        data = torch.tensor(
            [
                [*detection.bbox_xyxy, detection.confidence, PERSON_CLASS_ID]
                for detection in accepted
            ],
            dtype=torch.float32,
        )
        if not len(accepted):
            data = torch.empty((0, 6), dtype=torch.float32)
        boxes = Boxes(data, (height, width)).cpu().numpy()
        if self.settings.tracker_type == "botsort" and frame is None:
            raise ValueError("BoT-SORT requires the current image frame")
        rows = self._tracker.update(boxes, img=frame)

        current_lost = {int(track.track_id) for track in self._tracker.lost_stracks}
        current_removed = {
            int(track.track_id) for track in self._tracker.removed_stracks
        }
        active_ids = {int(row[4]) for row in rows}
        recovered_ids = active_ids & self._previous_lost_ids
        created_ids = active_ids - self._seen_ids
        lost_ids = current_lost - self._previous_lost_ids
        expired_ids = current_removed - self._expired_ids

        records = []
        matched_indices: set[int] = set()
        for row in rows:
            x1, y1, x2, y2, track_id_value, score, class_id_value, index_value = row
            track_id = int(track_id_value)
            class_id = int(class_id_value)
            detection_index = int(index_value)
            if class_id != PERSON_CLASS_ID or not 0 <= detection_index < len(accepted):
                raise RuntimeError("Tracker returned an invalid Person association")
            matched_indices.add(detection_index)
            state = (
                "new"
                if track_id in created_ids
                else "recovered"
                if track_id in recovered_ids
                else "tracked"
            )
            bbox = tuple(float(value) for value in (x1, y1, x2, y2))
            records.append(
                PersonTrack(
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    track_id=track_id,
                    class_id=PERSON_CLASS_ID,
                    class_name=PERSON_CLASS_NAME,
                    confidence=float(score),
                    bbox_xyxy=bbox,
                    bbox_normalized=normalized_box(bbox, width, height),
                    source_id=source_id,
                    track_state=state,
                )
            )

        self._seen_ids.update(active_ids)
        self._previous_lost_ids = current_lost
        self._expired_ids.update(expired_ids)
        return TrackerFrameResult(
            tracks=tuple(records),
            matched_person_indices=frozenset(matched_indices),
            rejected_person_detections=rejected,
            events=TrackerEvents(
                created_ids=tuple(sorted(created_ids)),
                lost_ids=tuple(sorted(lost_ids)),
                recovered_ids=tuple(sorted(recovered_ids)),
                expired_ids=tuple(sorted(expired_ids)),
            ),
        )
