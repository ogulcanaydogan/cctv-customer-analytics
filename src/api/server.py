from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path
from typing import Dict, Iterator, Optional

from contextlib import asynccontextmanager

try:
    import cv2
except ImportError:  # pragma: no cover - optional in tests
    cv2 = None  # type: ignore
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

from src.config import AppConfig, CameraConfig
from src.pipelines.counter import EntranceCounter
from src.utils.events import EntranceEvent, EventStore
from src.utils.streaming import FrameBuffer

logger = logging.getLogger(__name__)

@asynccontextmanager
async def _lifespan(app: FastAPI):  # pragma: no cover - exercised in integration
    """Ensure application state exists before serving requests."""

    _ensure_state()
    yield


app = FastAPI(title="Retail Vision Analytics", lifespan=_lifespan)


class AppState:
    def __init__(
        self,
        config: AppConfig,
        counters: Dict[str, EntranceCounter],
        event_store: EventStore,
        frame_buffers: Dict[str, FrameBuffer],
    ) -> None:
        self.config = config
        self.counters = counters
        self.event_store = event_store
        self.frame_buffers = frame_buffers


_state: AppState | None = None


def init_app_state(
    config: AppConfig,
    counters: Dict[str, EntranceCounter],
    event_store: EventStore,
    frame_buffers: Dict[str, FrameBuffer],
) -> None:
    """Initialize global app state used by API endpoints."""

    global _state
    _state = AppState(
        config=config, counters=counters, event_store=event_store, frame_buffers=frame_buffers
    )
    logger.info("API state initialized with %d cameras", len(counters))


def _ensure_state() -> AppState:
    """Return current state, initializing an empty demo state if needed."""

    global _state
    if _state is None:
        logger.warning("API state was not initialized; falling back to empty demo state")
        _state = AppState(
            config=AppConfig(cameras=[]),
            counters={},
            event_store=EventStore(),
            frame_buffers={},
        )
    return _state


def get_state() -> AppState:
    return _ensure_state()


def mjpeg_generator(camera_id: str) -> Iterator[bytes]:
    """Yield JPEG frames for the requested camera as an MJPEG stream."""

    state = get_state()
    frame_buffer = state.frame_buffers.get(camera_id)
    if frame_buffer is None:
        logger.warning("Requested stream for unknown camera %s", camera_id)
        return iter(())

    if cv2 is None:
        logger.error("OpenCV not available; MJPEG streaming disabled")
        return iter(())

    logger.info("Client connected to stream for camera %s", camera_id)
    last_placeholder = 0.0

    def _encode_placeholder(message: str) -> Optional[bytes]:
        canvas = np.full((360, 640, 3), 220, dtype=np.uint8)
        cv2.putText(canvas, message, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
        ok, jpeg_buf = cv2.imencode(".jpg", canvas)
        if not ok:
            return None
        return b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + bytearray(jpeg_buf) + b"\r\n"

    while True:
        frame = frame_buffer.read()
        if frame is None:
            if time.time() - last_placeholder > 1.5:
                placeholder = _encode_placeholder("Waiting for camera frames...")
                if placeholder:
                    yield placeholder
                last_placeholder = time.time()
            time.sleep(0.05)
            continue
        ret, jpeg = cv2.imencode(".jpg", frame)
        if not ret:
            logger.error("Failed to encode frame for camera %s", camera_id)
            time.sleep(0.05)
            continue
        chunk = b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + bytearray(jpeg) + b"\r\n"
        yield chunk


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/cameras")
def list_cameras() -> list[CameraConfig]:
    return get_state().config.cameras


@app.get("/cameras/{camera_id}/counts")
def camera_counts(camera_id: str) -> dict[str, int]:
    state = get_state()
    counter = state.counters.get(camera_id)
    if counter is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return counter.get_counts()


@app.get("/cameras/{camera_id}/events")
def camera_events(camera_id: str, limit: int = 50) -> list[EntranceEvent]:
    state = get_state()
    if camera_id not in state.counters:
        raise HTTPException(status_code=404, detail="Camera not found")
    return state.event_store.get_recent_events(camera_id=camera_id, limit=limit)


@app.get("/cameras/{camera_id}/stream")
def stream_camera(camera_id: str) -> StreamingResponse:
    state = get_state()
    if camera_id not in state.counters:
        raise HTTPException(status_code=404, detail="Camera not found")
    return StreamingResponse(
        mjpeg_generator(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/live")
def live_view() -> HTMLResponse:
    html_path = Path(__file__).with_name("live.html")
    content = html_path.read_text(encoding="utf-8")
    return HTMLResponse(content=content)


@app.get("/dashboard")
def dashboard_view() -> HTMLResponse:
    html_path = Path(__file__).with_name("dashboard.html")
    content = html_path.read_text(encoding="utf-8")
    return HTMLResponse(content=content)


@app.get("/stats/summary")
def stats_summary() -> dict:
    state = get_state()
    summary = state.event_store.summarize_daily_counts(date.today(), counters=state.counters)
    total_in_today = sum(v["total_in_today"] for v in summary.values()) if summary else 0
    total_out_today = sum(v["total_out_today"] for v in summary.values()) if summary else 0
    busiest_camera: Optional[str] = None
    if summary:
        busiest_camera = max(summary.items(), key=lambda item: item[1]["total_in_today"])[0]
    return {
        "total_in_today": total_in_today,
        "total_out_today": total_out_today,
        "busiest_camera": busiest_camera,
        "cameras": summary,
    }


__all__ = [
    "app",
    "init_app_state",
    "get_state",
    "mjpeg_generator",
]
