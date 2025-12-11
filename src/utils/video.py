from __future__ import annotations

import logging
import time
from typing import Generator, Tuple

import cv2

logger = logging.getLogger(__name__)

Frame = Tuple[any, float]


class CameraStream:
    """RTSP camera stream with reconnect support."""

    def __init__(self, camera_id: str, rtsp_url: str, reconnect_interval: float = 5.0) -> None:
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.reconnect_interval = reconnect_interval
        self.capture: cv2.VideoCapture | None = None

    def _connect(self) -> None:
        if self.capture is not None:
            self.capture.release()
        self.capture = cv2.VideoCapture(self.rtsp_url)
        if not self.capture.isOpened():
            logger.warning("Camera %s: Unable to open RTSP stream", self.camera_id)
        else:
            logger.info("Camera %s: RTSP stream opened", self.camera_id)

    def frames(self) -> Generator[Frame, None, None]:
        """Yield frames and timestamps, reconnecting on failure."""

        self._connect()
        while True:
            if self.capture is None or not self.capture.isOpened():
                logger.info("Camera %s: Attempting to reconnect", self.camera_id)
                time.sleep(self.reconnect_interval)
                self._connect()
                continue

            success, frame = self.capture.read()
            if not success or frame is None:
                logger.warning("Camera %s: Frame read failed, reconnecting", self.camera_id)
                time.sleep(self.reconnect_interval)
                self._connect()
                continue

            yield frame, time.time()

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            logger.info("Camera %s: Stream released", self.camera_id)


__all__ = ["CameraStream"]
