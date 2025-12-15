from __future__ import annotations

import threading
from typing import Optional
import logging

import numpy as np

logger = logging.getLogger(__name__)


class FrameBuffer:
    """Thread-safe container for the latest annotated frame per camera."""

    def __init__(self) -> None:
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    def update(self, frame: np.ndarray) -> None:
        """Store a copy of the latest frame."""
        with self._lock:
            self._frame = frame.copy()
        logger.debug("FrameBuffer.update: stored frame shape=%s", getattr(frame, 'shape', None))

    def read(self) -> Optional[np.ndarray]:
        """Return a copy of the latest frame, if available."""
        with self._lock:
            if self._frame is None:
                logger.debug("FrameBuffer.read: no frame available")
                return None
            return self._frame.copy()


__all__ = ["FrameBuffer"]
