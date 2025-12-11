from __future__ import annotations

import logging
from typing import Dict, Tuple

from src.config import LineDefinition
from src.utils.geometry import bbox_center, crossed_line, normalized_line_to_absolute, point_side

logger = logging.getLogger(__name__)

Track = Tuple[int, int, int, int, int]


class EntranceCounter:
    """Counts entries and exits across an entrance line."""

    def __init__(self, camera_id: str, entrance_line: LineDefinition) -> None:
        self.camera_id = camera_id
        self.entrance_line_def = entrance_line
        self.track_last_center: Dict[int, Tuple[float, float]] = {}
        self.entered = 0
        self.exited = 0

    def _outside_side(self, frame_width: int, frame_height: int) -> float:
        abs_line = normalized_line_to_absolute(
            (self.entrance_line_def.p1, self.entrance_line_def.p2), frame_width, frame_height
        )
        (x1, y1), (x2, y2) = abs_line
        mid_x = (x1 + x2) / 2.0
        mid_y = (y1 + y2) / 2.0
        if mid_y < frame_height / 2.0:
            reference_point = (mid_x, 0.0)
        else:
            reference_point = (mid_x, float(frame_height))
        return point_side(reference_point, abs_line)

    def update(
        self, tracks: Tuple[Track, ...] | list[Track], frame_width: int, frame_height: int
    ) -> list[tuple[int, str]]:
        abs_line = normalized_line_to_absolute(
            (self.entrance_line_def.p1, self.entrance_line_def.p2), frame_width, frame_height
        )
        outside_side = self._outside_side(frame_width, frame_height)
        events: list[tuple[int, str]] = []

        for track_id, x1, y1, x2, y2 in tracks:
            current_center = bbox_center((x1, y1, x2, y2))
            previous_center = self.track_last_center.get(track_id)
            if previous_center is not None and crossed_line(previous_center, current_center, abs_line):
                prev_side = point_side(previous_center, abs_line)
                curr_side = point_side(current_center, abs_line)
                if prev_side * outside_side >= 0 and curr_side * outside_side < 0:
                    self.entered += 1
                    events.append((track_id, "in"))
                    logger.info("Camera %s: Track %s entered", self.camera_id, track_id)
                elif prev_side * outside_side < 0 and curr_side * outside_side >= 0:
                    self.exited += 1
                    events.append((track_id, "out"))
                    logger.info("Camera %s: Track %s exited", self.camera_id, track_id)
            self.track_last_center[track_id] = current_center

        return events

    def get_counts(self) -> dict[str, int]:
        occupancy = self.entered - self.exited
        return {"entered": self.entered, "exited": self.exited, "current_occupancy": occupancy}


__all__ = ["EntranceCounter"]
