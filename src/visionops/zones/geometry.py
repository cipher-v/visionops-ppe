"""Dependency-free normalized polygon geometry with inclusive boundaries."""

from __future__ import annotations

import math

Point = tuple[float, float]


def polygon_area(polygon: tuple[Point, ...]) -> float:
    """Return the absolute shoelace area."""
    return (
        abs(
            sum(
                x1 * y2 - x2 * y1
                for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1])
            )
        )
        / 2
    )


def point_on_segment(point: Point, start: Point, end: Point, eps: float = 1e-9) -> bool:
    """Return whether a point lies on a closed line segment."""
    px, py = point
    x1, y1 = start
    x2, y2 = end
    cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
    if not math.isclose(cross, 0.0, abs_tol=eps):
        return False
    return (
        min(x1, x2) - eps <= px <= max(x1, x2) + eps
        and min(y1, y2) - eps <= py <= max(y1, y2) + eps
    )


def point_in_polygon(point: Point, polygon: tuple[Point, ...]) -> bool:
    """Use ray casting; polygon boundaries count as inside."""
    for start, end in zip(polygon, polygon[1:] + polygon[:1]):
        if point_on_segment(point, start, end):
            return True
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y):
            intersection_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < intersection_x:
                inside = not inside
        previous = current
    return inside


def _orientation(a: Point, b: Point, c: Point) -> int:
    value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if math.isclose(value, 0.0, abs_tol=1e-12):
        return 0
    return 1 if value > 0 else 2


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    """Return whether two closed segments intersect."""
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    return any(
        (
            o1 == 0 and point_on_segment(c, a, b),
            o2 == 0 and point_on_segment(d, a, b),
            o3 == 0 and point_on_segment(a, c, d),
            o4 == 0 and point_on_segment(b, c, d),
        )
    )


def polygon_self_intersects(polygon: tuple[Point, ...]) -> bool:
    """Reject intersections between non-adjacent polygon edges."""
    edges = list(zip(polygon, polygon[1:] + polygon[:1]))
    count = len(edges)
    for first_index, first in enumerate(edges):
        for second_index in range(first_index + 1, count):
            if second_index in {first_index, first_index + 1}:
                continue
            if first_index == 0 and second_index == count - 1:
                continue
            if segments_intersect(*first, *edges[second_index]):
                return True
    return False


def bottom_center(bbox: tuple[float, float, float, float]) -> Point:
    """Return the normalized ground-plane proxy for a Person box."""
    x1, _, x2, y2 = bbox
    return ((x1 + x2) / 2, y2)
