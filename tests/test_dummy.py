from datetime import datetime, date

from fastapi.responses import HTMLResponse

from src.api import server
from src.config import AppConfig, CameraConfig, LineDefinition
from src.pipelines.counter import EntranceCounter
from src.utils.events import EntranceEvent, EventStore
from src.utils.streaming import FrameBuffer


def _setup_app_state() -> None:
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
    server.init_app_state(config=config, counters=counters, event_store=event_store, frame_buffers=frame_buffers)


def test_live_and_dashboard_endpoints_return_html():
    _setup_app_state()
    live_resp = server.live_view()
    dash_resp = server.dashboard_view()
    assert isinstance(live_resp, HTMLResponse)
    assert isinstance(dash_resp, HTMLResponse)
    assert "text/html" in live_resp.media_type
    assert "text/html" in dash_resp.media_type


def test_stats_summary_structure():
    _setup_app_state()
    payload = server.stats_summary()
    assert "total_in_today" in payload
    assert "total_out_today" in payload
    assert "cameras" in payload
    assert "cam1" in payload["cameras"]
    cam_summary = payload["cameras"]["cam1"]
    assert cam_summary["total_in_today"] >= 1
    assert cam_summary["total_out_today"] >= 1
    # Ensure hourly breakdown includes the current hour key
    hour_key = datetime.now().hour
    assert hour_key in cam_summary["visits_per_hour"]


if __name__ == "__main__":
    _setup_app_state()
    print(server.stats_summary())
