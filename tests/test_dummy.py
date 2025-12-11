from datetime import datetime

from fastapi.testclient import TestClient

from src.api.server import app, init_app_state
from src.config import AppConfig, CameraConfig, LineDefinition
from src.pipelines.counter import EntranceCounter
from src.utils.events import EntranceEvent, EventStore
from src.utils.streaming import FrameBuffer


def _setup_app_state() -> TestClient:
    config = AppConfig(
        log_level="INFO",
        model_path="yolov8n.pt",
        cameras=[
            CameraConfig(
                id="cam1",
                name="Front Door",
                rtsp_url="rtsp://example",
                entrance_line=LineDefinition(p1=(0.0, 0.5), p2=(1.0, 0.5)),
            )
        ],
    )
    counters = {"cam1": EntranceCounter(camera_id="cam1", entrance_line=config.cameras[0].entrance_line)}
    event_store = EventStore()
    now = datetime.now()
    event_store.add_event(EntranceEvent(camera_id="cam1", timestamp=now, direction="in", track_id=1))
    event_store.add_event(EntranceEvent(camera_id="cam1", timestamp=now, direction="out", track_id=1))
    frame_buffers = {"cam1": FrameBuffer()}
    init_app_state(config=config, counters=counters, event_store=event_store, frame_buffers=frame_buffers)
    return TestClient(app)


def test_live_and_dashboard_endpoints():
    client = _setup_app_state()
    assert client.get("/live").status_code == 200
    assert client.get("/dashboard").status_code == 200


def test_stats_summary_structure():
    client = _setup_app_state()
    response = client.get("/stats/summary")
    assert response.status_code == 200
    payload = response.json()
    assert "total_in_today" in payload
    assert "total_out_today" in payload
    assert "cameras" in payload
    assert "cam1" in payload["cameras"]
