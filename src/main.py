from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Dict

import uvicorn

from src.api.server import app, init_app_state
from src.config import AppConfig, load_config
from src.pipelines.counter import EntranceCounter
from src.pipelines.detector import PersonDetector
from src.pipelines.profiler import CustomerProfiler
from src.pipelines.tracker import PersonTracker
from src.utils.events import EntranceEvent, EventStore
from src.utils.streaming import FrameBuffer
from src.utils.video import CameraStream

try:
    import cv2
except ImportError:  # pragma: no cover - OpenCV not available in some test envs
    cv2 = None  # type: ignore

logger = logging.getLogger(__name__)


def _annotate_frame(
    frame,
    tracks,
    counts: dict[str, int],
) -> "cv2.Mat":
    """Draw bounding boxes, IDs, and aggregate counts on the frame."""

    if cv2 is None:
        return frame

    annotated = frame.copy()
    for track_id, x1, y1, x2, y2 in tracks:
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"ID {track_id}",
            (int(x1), max(int(y1) - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    overlay = f"IN: {counts['entered']} OUT: {counts['exited']} OCC: {counts['current_occupancy']}"
    cv2.rectangle(annotated, (10, 10), (280, 40), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        overlay,
        (15, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def process_camera(
    camera_cfg: CameraConfig,
    detector: PersonDetector,
    tracker: PersonTracker,
    counter: EntranceCounter,
    event_store: EventStore,
    profiler: CustomerProfiler,
    frame_buffer: FrameBuffer,
) -> None:
    stream = CameraStream(camera_id=camera_cfg.id, rtsp_url=camera_cfg.rtsp_url)
    for frame, timestamp in stream.frames():
        try:
            detections = detector.detect(frame)
            tracks = tracker.update(detections)
            events = counter.update(tracks, frame.shape[1], frame.shape[0])
            track_boxes = {track_id: (x1, y1, x2, y2) for track_id, x1, y1, x2, y2 in tracks}
            for track_id, direction in events:
                event = EntranceEvent(
                    camera_id=camera_cfg.id,
                    timestamp=datetime.fromtimestamp(timestamp),
                    direction=direction,
                    track_id=track_id,
                )
                event_store.add_event(event)
                bbox = track_boxes.get(track_id)
                if bbox:
                    profiler.profile(track_id, frame, bbox)
            counts = counter.get_counts()
            logger.info(
                "Camera %s (%s): entered=%d exited=%d occupancy=%d",
                camera_cfg.id,
                camera_cfg.name,
                counts["entered"],
                counts["exited"],
                counts["current_occupancy"],
            )

            annotated = _annotate_frame(frame, tracks, counts)
            frame_buffer.update(annotated)
        except Exception:
            logger.exception("Camera %s: error processing frame", camera_cfg.id)


def start_camera_threads(
    config: AppConfig,
    detector: PersonDetector,
    event_store: EventStore,
    profiler: CustomerProfiler,
    frame_buffers: Dict[str, FrameBuffer],
) -> Dict[str, EntranceCounter]:
    counters: Dict[str, EntranceCounter] = {}
    for camera_cfg in config.cameras:
        counter = EntranceCounter(camera_id=camera_cfg.id, entrance_line=camera_cfg.entrance_line)
        tracker = PersonTracker()
        counters[camera_cfg.id] = counter
        frame_buffer = FrameBuffer()
        frame_buffers[camera_cfg.id] = frame_buffer
        thread = threading.Thread(
            target=process_camera,
            args=(camera_cfg, detector, tracker, counter, event_store, profiler, frame_buffer),
            daemon=True,
            name=f"camera-{camera_cfg.id}",
        )
        thread.start()
        logger.info("Started processing thread for camera %s", camera_cfg.id)
    return counters


def main() -> None:
    config = load_config()
    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))
    detector = PersonDetector(model_path=config.model_path)
    event_store = EventStore()
    profiler = CustomerProfiler()

    frame_buffers: Dict[str, FrameBuffer] = {}
    counters = start_camera_threads(config, detector, event_store, profiler, frame_buffers)

    init_app_state(config=config, counters=counters, event_store=event_store, frame_buffers=frame_buffers)

    logger.info(
        "Starting API server on http://%s:%d", config.api_host, config.api_port
    )
    uvicorn.run(
        app,
        host=config.api_host,
        port=config.api_port,
        log_level=config.log_level.lower(),
    )


if __name__ == "__main__":
    main()
