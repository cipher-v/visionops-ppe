"""Run frozen VisionOps PPE detection over a streaming video source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from visionops.video.config import load_frozen_detector_config
from visionops.video.pipeline import PipelineOptions, run_video_pipeline

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/production_detector.yaml"
DEFAULT_TRACKER_CONFIG = ROOT / "configs/bytetrack_person.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        help="Local video, non-negative device index, or RTSP/HTTP stream URI",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--enable-tracking", action="store_true")
    parser.add_argument("--tracker-config", type=Path, default=DEFAULT_TRACKER_CONFIG)
    parser.add_argument("--fallback-fps", type=float, default=30.0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional smoke-test limit; omitted means process until source ends",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_frozen_detector_config(args.config, ROOT)
    metadata = run_video_pipeline(
        config,
        PipelineOptions(
            source=args.source,
            output_dir=args.output_dir,
            frame_stride=args.frame_stride,
            render=args.render,
            fallback_fps=args.fallback_fps,
            device=args.device,
            max_frames=args.max_frames,
            enable_tracking=args.enable_tracking,
            tracker_config=args.tracker_config if args.enable_tracking else None,
        ),
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
