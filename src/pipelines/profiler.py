from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

BBox = Tuple[int, int, int, int]


class CustomerProfiler:
    """Stub profiler for future anonymous attributes."""

    def profile(self, track_id: int, frame: np.ndarray, bbox: BBox) -> Dict[str, str]:
        # TODO: Integrate age/gender and clothing color models here.
        return {
            "track_id": track_id,
            "age_group": "unknown",
            "gender": "unknown",
            "top_color": "unknown",
            "bottom_color": "unknown",
        }


__all__ = ["CustomerProfiler"]
