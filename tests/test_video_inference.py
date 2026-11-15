import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from visionops.video.config import (
    EXPECTED_CLASSES,
    EXPECTED_THRESHOLDS,
    load_frozen_detector_config,
    verify_weights,
)
from visionops.video.policy import passes_class_threshold
from visionops.video.schema import VideoDetection
from visionops.video.source import (
    AnnotatedVideoWriter,
    VideoReader,
    VideoSourceOpenError,
    frame_timestamp_seconds,
    parse_source,
)

ROOT = Path(__file__).parents[1]


def _write_test_video(path: Path, frame_count: int = 6) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (64, 48))
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.full((48, 64, 3), index * 20, dtype=np.uint8)
        cv2.rectangle(frame, (index + 2, 5), (index + 12, 20), (0, 255, 0), -1)
        writer.write(frame)
    writer.release()


def test_model_hash_verification(tmp_path: Path) -> None:
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"frozen-weights")
    expected = hashlib.sha256(b"frozen-weights").hexdigest().upper()
    assert verify_weights(weights, expected) == expected
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_weights(weights, "0" * 64)
    with pytest.raises(FileNotFoundError, match="do not exist"):
        verify_weights(tmp_path / "missing.pt", expected)


def test_production_detector_config_loading() -> None:
    config = load_frozen_detector_config(
        ROOT / "configs/production_detector.yaml", ROOT
    )
    assert config.architecture == "YOLO26n detect"
    assert config.imgsz == 960
    assert config.classes == EXPECTED_CLASSES
    assert config.confidence_thresholds == EXPECTED_THRESHOLDS
    assert config.inference_confidence_floor == 0.11
    assert config.nms_iou == 0.70


@pytest.mark.parametrize(
    ("class_id", "threshold"),
    [
        (class_id, EXPECTED_THRESHOLDS[name])
        for class_id, name in EXPECTED_CLASSES.items()
    ],
)
def test_each_frozen_class_threshold(class_id: int, threshold: float) -> None:
    assert passes_class_threshold(
        class_id, threshold, EXPECTED_CLASSES, EXPECTED_THRESHOLDS
    )
    assert not passes_class_threshold(
        class_id, threshold - 0.0001, EXPECTED_CLASSES, EXPECTED_THRESHOLDS
    )


def test_detection_serialization_has_no_tracking_fields() -> None:
    detection = VideoDetection(
        frame_index=128,
        timestamp_seconds=4.267,
        class_id=3,
        class_name="no_helmet",
        confidence=0.81,
        bbox_xyxy=(412.0, 118.0, 476.0, 207.0),
        bbox_normalized=(0.412, 0.118, 0.476, 0.207),
        source_id="demo.mp4",
    )
    values = detection.to_dict()
    assert values["bbox_xyxy"] == [412.0, 118.0, 476.0, 207.0]
    assert values["bbox_normalized"] == [0.412, 0.118, 0.476, 0.207]
    assert "track_id" not in values


def test_video_timestamp_calculation() -> None:
    assert frame_timestamp_seconds(128, 30.0) == pytest.approx(128 / 30)
    assert frame_timestamp_seconds(3, None, 0.42) == pytest.approx(0.42)
    with pytest.raises(ValueError, match="negative"):
        frame_timestamp_seconds(-1, 30.0)


def test_malformed_and_unopenable_sources(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        parse_source(" ")
    with pytest.raises(ValueError, match="Unsupported"):
        parse_source("ftp://example.com/video")
    safe = parse_source("rtsp://user:secret@example.com/live")
    assert safe.kind == "stream"
    assert "user" not in safe.source_id
    assert "secret" not in safe.source_id
    with pytest.raises(VideoSourceOpenError, match="Unable to open"):
        VideoReader(str(tmp_path / "missing.mp4")).open()


def test_deterministic_video_read_and_output_write(tmp_path: Path) -> None:
    source = tmp_path / "tiny.avi"
    output = tmp_path / "copy.avi"
    _write_test_video(source)

    packets = []
    with VideoReader(str(source)) as reader:
        assert reader.metadata is not None
        assert reader.metadata.original_width == 64
        assert reader.metadata.original_height == 48
        assert reader.metadata.reported_fps == pytest.approx(12.0)
        assert reader.metadata.total_frame_count == 6
        with AnnotatedVideoWriter(output, 64, 48, 12.0) as writer:
            while (packet := reader.read()) is not None:
                packets.append(packet)
                writer.write(packet.image)
        assert reader.reached_end_of_file
        assert reader.read_failures == 0

    assert len(packets) == 6
    assert packets[0].frame_index == 0
    assert packets[-1].timestamp_seconds == pytest.approx(5 / 12)
    with VideoReader(str(output)) as copied:
        copied_count = 0
        while copied.read() is not None:
            copied_count += 1
    assert copied_count == 6
