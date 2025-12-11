from __future__ import annotations

import threading
from collections import defaultdict
from datetime import date, datetime
from typing import TYPE_CHECKING, Dict, List, Literal

if TYPE_CHECKING:
    from src.pipelines.counter import EntranceCounter

from pydantic import BaseModel


class EntranceEvent(BaseModel):
    camera_id: str
    timestamp: datetime
    direction: Literal["in", "out"]
    track_id: int


class EventStore:
    """In-memory, thread-safe event repository."""

    def __init__(self) -> None:
        self._events: List[EntranceEvent] = []
        self._lock = threading.Lock()

    def add_event(self, event: EntranceEvent) -> None:
        with self._lock:
            self._events.append(event)

    def get_recent_events(self, camera_id: str, limit: int = 50) -> List[EntranceEvent]:
        with self._lock:
            filtered = [e for e in self._events if e.camera_id == camera_id]
        return filtered[-limit:]

    def get_counts(self, camera_id: str) -> dict[str, int]:
        with self._lock:
            filtered = [e for e in self._events if e.camera_id == camera_id]
        entered = sum(1 for e in filtered if e.direction == "in")
        exited = sum(1 for e in filtered if e.direction == "out")
        return {"entered": entered, "exited": exited, "current_occupancy": entered - exited}

    def get_events_for_day(self, day: date) -> List[EntranceEvent]:
        """Return all events that occurred on the given calendar day (server timezone)."""
        with self._lock:
            return [event for event in self._events if event.timestamp.date() == day]

    def summarize_daily_counts(
        self, day: date, counters: Dict[str, "EntranceCounter"] | None = None
    ) -> dict[str, dict[str, int | Dict[int, int]]]:
        """
        Aggregate per-camera statistics for the given day.

        Args:
            day: Calendar day to summarize.
            counters: Optional counters to source current occupancy.

        Returns:
            Mapping from camera_id to metrics including total_in_today, total_out_today,
            current_occupancy, and visits_per_hour.
        """

        events = self.get_events_for_day(day)
        per_camera: Dict[str, dict[str, int | Dict[int, int]]] = defaultdict(
            lambda: {"total_in_today": 0, "total_out_today": 0, "visits_per_hour": defaultdict(int)}
        )
        for event in events:
            camera_stats = per_camera[event.camera_id]
            if event.direction == "in":
                camera_stats["total_in_today"] = int(camera_stats["total_in_today"]) + 1
                visits_per_hour = camera_stats["visits_per_hour"]
                if isinstance(visits_per_hour, dict):
                    visits_per_hour[event.timestamp.hour] = visits_per_hour.get(event.timestamp.hour, 0) + 1
            else:
                camera_stats["total_out_today"] = int(camera_stats["total_out_today"]) + 1

        for camera_id, stats in per_camera.items():
            visits_per_hour = stats["visits_per_hour"]
            if isinstance(visits_per_hour, defaultdict):
                stats["visits_per_hour"] = dict(visits_per_hour)
            occupancy = None
            if counters and camera_id in counters:
                occupancy = counters[camera_id].get_counts().get("current_occupancy", 0)
            else:
                occupancy = int(stats["total_in_today"]) - int(stats["total_out_today"])
            stats["current_occupancy"] = occupancy

        if counters:
            for camera_id in counters:
                per_camera.setdefault(
                    camera_id,
                    {
                        "total_in_today": 0,
                        "total_out_today": 0,
                        "visits_per_hour": {},
                        "current_occupancy": counters[camera_id].get_counts().get("current_occupancy", 0),
                    },
                )

        return dict(per_camera)


__all__ = ["EntranceEvent", "EventStore"]
