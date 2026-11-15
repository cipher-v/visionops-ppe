"""Pure box geometry for deterministic PPE association."""

from __future__ import annotations

import math
from dataclasses import dataclass

from visionops.association.config import RegionFractions

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class GeometricSignals:
    center_inside_person: bool
    center_inside_region: bool
    person_ioa: float
    region_ioa: float
    normalized_center_distance: float
    center_proximity: float


def valid_box(box: Box) -> bool:
    return (
        len(box) == 4
        and all(math.isfinite(value) for value in box)
        and (box[2] > box[0] and box[3] > box[1])
    )


def box_area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def anatomical_region(person: Box, fractions: RegionFractions) -> Box:
    """Derive an anatomical proxy region from normalized Person fractions."""
    if not valid_box(person):
        raise ValueError("Person box must be finite and have positive area")
    width, height = person[2] - person[0], person[3] - person[1]
    return (
        person[0] + fractions.x_min * width,
        person[1] + fractions.y_min * height,
        person[0] + fractions.x_max * width,
        person[1] + fractions.y_max * height,
    )


def intersection_area(first: Box, second: Box) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def intersection_over_area(subject: Box, container: Box) -> float:
    """Return the fraction of the subject box inside the container."""
    area = box_area(subject)
    return intersection_area(subject, container) / area if area > 0 else 0.0


def box_center(box: Box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def point_inside(point: tuple[float, float], box: Box) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def geometric_signals(ppe: Box, person: Box, region: Box) -> GeometricSignals:
    """Calculate containment and scale-normalized center-proximity signals."""
    if not valid_box(ppe) or not valid_box(person) or not valid_box(region):
        raise ValueError("Geometry signals require finite positive-area boxes")
    ppe_center = box_center(ppe)
    region_center = box_center(region)
    half_width = (region[2] - region[0]) / 2
    half_height = (region[3] - region[1]) / 2
    half_diagonal = math.hypot(half_width, half_height)
    distance = math.hypot(
        ppe_center[0] - region_center[0], ppe_center[1] - region_center[1]
    )
    normalized_distance = distance / half_diagonal if half_diagonal else math.inf
    return GeometricSignals(
        center_inside_person=point_inside(ppe_center, person),
        center_inside_region=point_inside(ppe_center, region),
        person_ioa=intersection_over_area(ppe, person),
        region_ioa=intersection_over_area(ppe, region),
        normalized_center_distance=normalized_distance,
        center_proximity=max(0.0, 1.0 - normalized_distance),
    )
