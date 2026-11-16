"""Resolution-aware recruiter-facing video overlay utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class RenderStyle:
    scale: float
    header_font: float
    body_font: float
    legend_font: float
    text_thickness: int
    box_thickness: int
    padding: int
    line_gap: int


def render_style(width: int, height: int) -> RenderStyle:
    """Scale all visual dimensions from the shorter frame edge."""
    if width <= 0 or height <= 0:
        raise ValueError("Render dimensions must be positive")
    scale = max(0.75, min(width, height) / 1080.0)
    return RenderStyle(
        scale=scale,
        header_font=0.82 * scale,
        body_font=0.64 * scale,
        legend_font=0.50 * scale,
        text_thickness=max(1, round(2 * scale)),
        box_thickness=max(3, round(4 * scale)),
        padding=max(6, round(9 * scale)),
        line_gap=max(4, round(6 * scale)),
    )


def output_dimensions(
    source_width: int, source_height: int, render_width: int | None
) -> tuple[int, int]:
    """Return even output dimensions while preserving source aspect ratio."""
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Source dimensions must be positive")
    if render_width is None:
        return source_width, source_height
    if render_width <= 0:
        raise ValueError("Render width must be positive")
    width = render_width - render_width % 2
    height = round(source_height * width / source_width)
    height -= height % 2
    return max(2, width), max(2, height)


def scale_for_output(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize only the completed visualization frame."""
    width, height = size
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    interpolation = cv2.INTER_AREA if width < frame.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(frame, (width, height), interpolation=interpolation)


def temporal_status(state: Any) -> str:
    """Map a temporal equipment snapshot to concise demo wording."""
    if state.temporal_state == "not_required":
        return "N/R"
    if state.temporal_state == "compliant":
        return "OK"
    if state.temporal_state == "suspected_violation":
        return f"SUSPECT {state.continuous_violation_duration:.1f}s"
    if state.temporal_state == "confirmed_violation":
        duration = (
            state.incident_duration_seconds
            if state.incident_duration_seconds is not None
            else state.continuous_violation_duration
        )
        return f"VIOLATION {duration:.1f}s"
    if state.temporal_state == "conflict":
        return "CONFLICT"
    return "UNKNOWN"


def _association_status(evidence: Any) -> str:
    return {
        "present": "OK",
        "conflict": "CONFLICT",
        "violation": "SUSPECT 0.0s",
        "unknown": "UNKNOWN",
    }.get(evidence.state, "UNKNOWN")


def _panel_layout(
    lines: list[tuple[str, float, tuple[int, int, int]]], style: RenderStyle
) -> tuple[list[tuple[int, int]], list[int], int, int]:
    sizes = [
        cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, style.text_thickness)[0]
        for text, font, _ in lines
    ]
    line_heights = [height + style.line_gap for _, height in sizes]
    panel_width = max(width for width, _ in sizes) + style.padding * 2
    panel_height = sum(line_heights) + style.padding * 2
    return sizes, line_heights, panel_width, panel_height


def panel_origin_outside_box(
    frame_size: tuple[int, int],
    box: tuple[int, int, int, int],
    panel_size: tuple[int, int],
    gap: int,
    obstacles: tuple[tuple[int, int, int, int], ...] = (),
) -> tuple[int, int]:
    """Place a panel above a box, below it, or beside it as a last resort."""
    frame_width, frame_height = frame_size
    x1, y1, x2, y2 = box
    panel_width, panel_height = panel_size
    max_x = max(0, frame_width - panel_width)
    max_y = max(0, frame_height - panel_height)
    aligned_x = min(max(0, x1), max_x)

    def clear_origin(y: int) -> tuple[int, int] | None:
        x_candidates = [aligned_x]
        for obstacle_x1, _, obstacle_x2, _ in obstacles:
            x_candidates.extend((obstacle_x2 + gap, obstacle_x1 - gap - panel_width))
        for candidate_x in x_candidates:
            if not 0 <= candidate_x <= max_x:
                continue
            panel_rect = (
                candidate_x,
                y,
                candidate_x + panel_width,
                y + panel_height,
            )
            if all(
                panel_rect[2] <= obstacle[0]
                or panel_rect[0] >= obstacle[2]
                or panel_rect[3] <= obstacle[1]
                or panel_rect[1] >= obstacle[3]
                for obstacle in obstacles
            ):
                return candidate_x, y
        return None

    above_y = y1 - gap - panel_height
    if above_y >= 0:
        origin = clear_origin(above_y)
        if origin is not None:
            return origin

    below_y = y2 + gap
    if below_y + panel_height <= frame_height:
        origin = clear_origin(below_y)
        if origin is not None:
            return origin

    # A near-full-height person can leave no vertical space. Prefer a side panel
    # in that exceptional case so the worker remains visible.
    aligned_y = min(max(0, y1), max_y)
    left_x = x1 - gap - panel_width
    if left_x >= 0:
        return left_x, aligned_y
    right_x = x2 + gap
    if right_x + panel_width <= frame_width:
        return right_x, aligned_y

    # Some overlap is geometrically unavoidable when neither axis has room.
    return aligned_x, min(max(0, below_y), max_y)


def _draw_panel(
    image: np.ndarray,
    origin: tuple[int, int],
    lines: list[tuple[str, float, tuple[int, int, int]]],
    style: RenderStyle,
) -> None:
    sizes, line_heights, panel_width, panel_height = _panel_layout(lines, style)
    x = min(max(0, origin[0]), max(0, image.shape[1] - panel_width))
    y = min(max(0, origin[1]), max(0, image.shape[0] - panel_height))
    cv2.rectangle(image, (x, y), (x + panel_width, y + panel_height), (18, 18, 18), -1)
    cursor = y + style.padding
    for (text, font, color), (_, height), line_height in zip(
        lines, sizes, line_heights
    ):
        cursor += height
        cv2.putText(
            image,
            text,
            (x + style.padding, cursor),
            cv2.FONT_HERSHEY_SIMPLEX,
            font,
            color,
            style.text_thickness,
            cv2.LINE_AA,
        )
        cursor += line_height - height


def _legend(image: np.ndarray, style: RenderStyle) -> None:
    lines = _legend_lines(style)
    _draw_panel(
        image,
        (style.padding, style.padding),
        lines,
        style,
    )


def _legend_lines(
    style: RenderStyle,
) -> list[tuple[str, float, tuple[int, int, int]]]:
    return [
        ("OK = confirmed PPE evidence", style.legend_font, (235, 235, 235)),
        ("SUSPECT = below confirmation time", style.legend_font, (235, 235, 235)),
        (
            "VIOLATION = confirmed temporal violation",
            style.legend_font,
            (235, 235, 235),
        ),
        ("UNKNOWN = insufficient evidence", style.legend_font, (235, 235, 235)),
        ("CONFLICT = contradictory detections", style.legend_font, (235, 235, 235)),
        ("N/R = equipment not required", style.legend_font, (235, 235, 235)),
    ]


def render_demo_overlay(
    frame: np.ndarray,
    tracks: tuple[Any, ...],
    temporal_records: tuple[Any, ...] = (),
    association_records: tuple[Any, ...] = (),
    zone_records: tuple[Any, ...] = (),
    zone_names: dict[str, str] | None = None,
) -> np.ndarray:
    """Draw uncluttered per-Person temporal state without raw PPE boxes."""
    rendered = frame.copy()
    height, width = rendered.shape[:2]
    style = render_style(width, height)
    temporal_by_track = {record.track_id: record for record in temporal_records}
    association_by_track = {record.track_id: record for record in association_records}
    membership_by_track = {record.track_id: record for record in zone_records}
    _, _, legend_width, legend_height = _panel_layout(_legend_lines(style), style)
    occupied = [
        (
            style.padding,
            style.padding,
            style.padding + legend_width,
            style.padding + legend_height,
        )
    ]
    for track in tracks:
        x1, y1, x2, y2 = (round(value) for value in track.bbox_xyxy)
        temporal = temporal_by_track.get(track.track_id)
        association = association_by_track.get(track.track_id)
        if temporal is not None:
            helmet = temporal_status(temporal.helmet)
            vest = temporal_status(temporal.vest)
            incident = temporal.helmet.incident_active or temporal.vest.incident_active
        elif association is not None:
            helmet = _association_status(association.helmet)
            vest = _association_status(association.vest)
            incident = False
        else:
            helmet = vest = "UNKNOWN"
            incident = False
        color = (
            (40, 40, 235)
            if incident
            else (0, 165, 255)
            if any(
                value.startswith("SUSPECT") or value == "CONFLICT"
                for value in (helmet, vest)
            )
            else (60, 210, 70)
        )
        cv2.rectangle(
            rendered, (x1, y1), (x2, y2), color, style.box_thickness, cv2.LINE_AA
        )
        lines = [
            (f"PERSON #{track.track_id}", style.header_font, (255, 255, 255)),
            (f"Helmet: {helmet}", style.body_font, (235, 235, 235)),
            (f"Vest: {vest}", style.body_font, (235, 235, 235)),
        ]
        if incident:
            lines.append(("INCIDENT", style.header_font, (80, 80, 255)))
        if zone_names is not None:
            membership = membership_by_track.get(track.track_id)
            zone_ids = membership.stable_zone_ids if membership is not None else ()
            label = ", ".join(zone_names.get(item, item) for item in zone_ids) or "none"
            lines.append((f"Zone: {label}", style.body_font, (220, 190, 255)))
        _, _, panel_width, panel_height = _panel_layout(lines, style)
        panel_gap = max(style.padding, round(12 * style.scale))
        panel_origin = panel_origin_outside_box(
            (width, height),
            (x1, y1, x2, y2),
            (panel_width, panel_height),
            panel_gap,
            tuple(occupied),
        )
        _draw_panel(rendered, panel_origin, lines, style)
        occupied.append(
            (
                panel_origin[0],
                panel_origin[1],
                panel_origin[0] + panel_width,
                panel_origin[1] + panel_height,
            )
        )
    _legend(rendered, style)
    return rendered
