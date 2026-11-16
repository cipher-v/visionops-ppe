from pathlib import Path

import cv2
import numpy as np
import pytest

from visionops.api.media import finalize_browser_media, resolve_ffmpeg_tools


def test_ffmpeg_finalizes_opencv_mp4_for_browser(tmp_path: Path) -> None:
    try:
        tools = resolve_ffmpeg_tools()
    except RuntimeError as error:
        pytest.skip(str(error))
    source = tmp_path / "opencv.mp4"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 24)
    )
    assert writer.isOpened()
    for value in (0, 80, 160, 240):
        writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    writer.release()

    destination = tmp_path / "presentation.mp4"
    probe = finalize_browser_media(source, destination, tools)

    assert destination.is_file()
    assert probe.browser_compatible
    assert probe.codec_name == "h264"
    assert probe.pixel_format == "yuv420p"
    assert (probe.width, probe.height) == (32, 24)
    assert probe.duration_seconds > 0
