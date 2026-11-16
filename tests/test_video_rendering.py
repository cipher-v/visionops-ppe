from types import SimpleNamespace

import numpy as np
import pytest

from visionops.video.pipeline import PipelineOptions
from visionops.video.rendering import (
    output_dimensions,
    panel_origin_outside_box,
    render_demo_overlay,
    render_style,
    scale_for_output,
    temporal_status,
)


def state(name: str, duration: float = 0.0, incident: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        temporal_state=name,
        continuous_violation_duration=duration,
        incident_duration_seconds=duration if incident else None,
        incident_active=incident,
    )


def test_render_style_scales_for_4k() -> None:
    hd = render_style(1920, 1080)
    uhd = render_style(3840, 2160)
    assert uhd.header_font == pytest.approx(hd.header_font * 2)
    assert uhd.box_thickness >= hd.box_thickness * 2
    assert uhd.padding >= hd.padding * 2


def test_output_scaling_preserves_aspect_and_even_dimensions() -> None:
    assert output_dimensions(3840, 2160, 1920) == (1920, 1080)
    assert output_dimensions(2160, 3840, 1920) == (1920, 3412)
    frame = np.zeros((108, 192, 3), dtype=np.uint8)
    assert scale_for_output(frame, (96, 54)).shape == (54, 96, 3)


def test_temporal_status_labels() -> None:
    assert temporal_status(state("compliant")) == "OK"
    assert temporal_status(state("not_required")) == "N/R"
    assert temporal_status(state("unknown")) == "UNKNOWN"
    assert temporal_status(state("conflict")) == "CONFLICT"
    assert temporal_status(state("suspected_violation", 1.2)) == "SUSPECT 1.2s"
    assert temporal_status(state("confirmed_violation", 3.4, True)) == "VIOLATION 3.4s"


def test_panel_is_placed_above_then_below_person_box() -> None:
    assert panel_origin_outside_box(
        (1920, 1080), (500, 400, 1000, 900), (420, 220), 20
    ) == (500, 160)
    assert panel_origin_outside_box(
        (1920, 1080), (500, 40, 1000, 500), (420, 220), 20
    ) == (500, 520)


def test_panel_uses_side_when_vertical_placement_is_impossible() -> None:
    origin = panel_origin_outside_box(
        (1920, 1080), (700, 20, 1600, 1060), (420, 220), 20
    )
    assert origin == (260, 20)
    assert origin[0] + 420 <= 700


def test_panel_stays_above_but_moves_around_legend() -> None:
    origin = panel_origin_outside_box(
        (1920, 1080),
        (170, 196, 800, 1080),
        (250, 80),
        12,
        ((9, 9, 321, 147),),
    )
    assert origin == (333, 104)


def test_demo_overlay_draws_person_panel_incident_zone_and_legend() -> None:
    frame = np.full((720, 1280, 3), 180, dtype=np.uint8)
    track = SimpleNamespace(track_id=7, bbox_xyxy=(300, 120, 800, 680))
    record = SimpleNamespace(
        track_id=7,
        helmet=state("confirmed_violation", 2.5, True),
        vest=state("not_required"),
    )
    membership = SimpleNamespace(track_id=7, stable_zone_ids=("loading",))
    rendered = render_demo_overlay(
        frame,
        (track,),
        (record,),
        zone_records=(membership,),
        zone_names={"loading": "Loading Bay"},
    )
    assert rendered.shape == frame.shape
    assert not np.array_equal(rendered, frame)
    assert tuple(rendered[120, 300]) != (180, 180, 180)
    assert np.any(rendered[:250, :500] < 50)


def test_pipeline_overlay_options_validate() -> None:
    options = PipelineOptions(
        source="video.mp4", output_dir=__import__("pathlib").Path("results/test")
    )
    assert options.overlay_mode == "demo"
    with pytest.raises(ValueError, match="Overlay mode"):
        PipelineOptions(
            source="video.mp4", output_dir=options.output_dir, overlay_mode="other"
        )
    with pytest.raises(ValueError, match="Render width"):
        PipelineOptions(
            source="video.mp4", output_dir=options.output_dir, render_width=1
        )
