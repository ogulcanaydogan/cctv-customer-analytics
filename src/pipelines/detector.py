from __future__ import annotations

import logging
import threading
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

Detection = Tuple[int, int, int, int, float]


class PersonDetector:
    """YOLOv8 based person detector.

    Loads the model once and exposes a ``detect`` method that returns
    bounding boxes for people in the frame.
    """

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.5) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self._model = YOLO(model_path)
        self._lock = threading.Lock()
        logger.info("Loaded YOLOv8 model from %s", model_path)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run person detection on a BGR frame.

        Args:
            frame: BGR image as numpy array.

        Returns:
            List of detections formatted as (x1, y1, x2, y2, score).
        """

        with self._lock:
            results = self._model.predict(frame, conf=self.conf_threshold, verbose=False)
        detections: List[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id != 0:  # Only person class
                    continue
                score = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append((int(x1), int(y1), int(x2), int(y2), score))
        logger.debug("Detected %d people", len(detections))
        return detections


__all__ = ["PersonDetector", "Detection"]
