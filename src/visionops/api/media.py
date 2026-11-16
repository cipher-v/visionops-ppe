"""FFmpeg-backed browser presentation media finalization."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MediaProbe:
    format_name: str
    codec_name: str
    pixel_format: str | None
    width: int
    height: int
    duration_seconds: float

    @property
    def browser_compatible(self) -> bool:
        return (
            "mp4" in self.format_name
            and self.codec_name == "h264"
            and self.pixel_format == "yuv420p"
        )


@dataclass(frozen=True)
class FFmpegTools:
    ffmpeg: Path
    ffprobe: Path


def resolve_ffmpeg_tools() -> FFmpegTools:
    """Resolve FFmpeg from configuration, PATH, or the local WinGet package."""
    configured = os.getenv("VISIONOPS_FFMPEG_PATH")
    ffmpeg = Path(configured).resolve() if configured else None
    if ffmpeg is None:
        discovered = shutil.which("ffmpeg")
        ffmpeg = Path(discovered).resolve() if discovered else None
    if ffmpeg is None and os.name == "nt":
        local_app_data = os.getenv("LOCALAPPDATA")
        if local_app_data:
            package_root = Path(local_app_data) / "Microsoft/WinGet/Packages"
            candidates = sorted(package_root.glob("**/ffmpeg.exe"))
            ffmpeg = candidates[-1].resolve() if candidates else None
    if ffmpeg is None or not ffmpeg.is_file():
        raise RuntimeError(
            "FFmpeg was not found; set VISIONOPS_FFMPEG_PATH or add it to PATH"
        )
    ffprobe = ffmpeg.with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
    if not ffprobe.is_file():
        discovered_probe = shutil.which("ffprobe")
        ffprobe = Path(discovered_probe).resolve() if discovered_probe else ffprobe
    if not ffprobe.is_file():
        raise RuntimeError("ffprobe was not found beside FFmpeg or on PATH")
    return FFmpegTools(ffmpeg, ffprobe)


def probe_media(path: Path, ffprobe: Path) -> MediaProbe:
    completed = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=format_name,duration:stream=codec_name,pix_fmt,width,height,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    media_format = payload["format"]
    duration = stream.get("duration") or media_format.get("duration")
    return MediaProbe(
        format_name=str(media_format["format_name"]),
        codec_name=str(stream["codec_name"]),
        pixel_format=stream.get("pix_fmt"),
        width=int(stream["width"]),
        height=int(stream["height"]),
        duration_seconds=float(duration),
    )


def finalize_browser_media(
    source: Path, destination: Path, tools: FFmpegTools
) -> MediaProbe:
    """Create an H.264/yuv420p faststart MP4, remuxing compatible input."""
    source_probe = probe_media(source, tools.ffprobe)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.stem}.tmp.mp4")
    temporary.unlink(missing_ok=True)
    command = [
        str(tools.ffmpeg),
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
    ]
    if source_probe.browser_compatible:
        command.extend(["-c", "copy"])
    else:
        command.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
            ]
        )
    command.extend(["-movflags", "+faststart", str(temporary)])
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        finalized = probe_media(temporary, tools.ffprobe)
        if not finalized.browser_compatible:
            raise RuntimeError("Finalized presentation media is not browser compatible")
        temporary.replace(destination)
        return finalized
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
