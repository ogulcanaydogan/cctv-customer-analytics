from __future__ import annotations

from typing import Tuple

Point = Tuple[float, float]
Line = Tuple[Point, Point]
BBox = Tuple[int, int, int, int]


def normalized_line_to_absolute(line: Line, width: int, height: int) -> Line:
    """Convert a normalized line definition to absolute pixel coordinates."""

    (x1_norm, y1_norm), (x2_norm, y2_norm) = line
    return (
        (x1_norm * width, y1_norm * height),
        (x2_norm * width, y2_norm * height),
    )


def bbox_center(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_side(point: Point, line: Line) -> float:
    """Return the sign of a point relative to a line segment using cross product."""

    (x1, y1), (x2, y2) = line
    x, y = point
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)


def crossed_line(prev_point: Point, current_point: Point, line: Line) -> bool:
    """Return True if the segment between the two points crosses the line."""

    prev_side = point_side(prev_point, line)
    curr_side = point_side(current_point, line)
    return prev_side * curr_side < 0


__all__ = [
    "Point",
    "Line",
    "BBox",
    "normalized_line_to_absolute",
    "bbox_center",
    "point_side",
    "crossed_line",
]
