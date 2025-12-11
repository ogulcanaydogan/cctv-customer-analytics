from __future__ import annotations

import logging
from typing import List, Tuple

from deep_sort_realtime.deepsort_tracker import DeepSort

logger = logging.getLogger(__name__)

Track = Tuple[int, int, int, int, int]
Detection = Tuple[int, int, int, int, float]


class PersonTracker:
    """Deep SORT based tracker for person detections."""

    def __init__(self) -> None:
        self.tracker = DeepSort(max_age=30)
        logger.info("Initialized DeepSort tracker")

    def update(self, detections: List[Detection]) -> List[Track]:
        """Update tracker with detections.

        Args:
            detections: List of detections from detector.

        Returns:
            List of tracks as (track_id, x1, y1, x2, y2).
        """

        tracker_inputs = [
            ((x1, y1, x2 - x1, y2 - y1), score, None)
            for (x1, y1, x2, y2, score) in detections
        ]

        tracked_objects = self.tracker.update_tracks(tracker_inputs, frame=None)
        tracks: List[Track] = []
        for obj in tracked_objects:
            if not obj.is_confirmed():
                continue
            track_id = obj.track_id
            ltrb = obj.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            tracks.append((track_id, x1, y1, x2, y2))
        logger.debug("Active tracks: %d", len(tracks))
        return tracks


__all__ = ["PersonTracker", "Track"]
