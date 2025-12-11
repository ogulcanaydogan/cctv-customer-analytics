# retail-vision-analytics

Retail Vision Analytics is a Python 3.11 project for real-time footfall analytics using RTSP CCTV streams. It combines YOLOv8 person detection, Deep SORT tracking, entrance line counting, and a FastAPI service for querying live metrics.

## Features
- Connect to one or more RTSP cameras.
- Real-time person detection with YOLOv8 (Ultralytics).
- Deep SORT tracking for stable track IDs.
- Entrance line counting for IN and OUT movements.
- REST API (FastAPI) exposing health, camera metadata, counts, recent events, and live MJPEG streams.
- Stubs for future anonymous customer profiling (age, gender, clothing colors).
- In-memory storage for events and counts.
- Web UI pages for live camera views and aggregated dashboard analytics.

## Project Structure
```
retail-vision-analytics/
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    ├── main.py                 # entry point
    ├── config.py               # YAML & Pydantic configuration
    ├── pipelines/
    │   ├── __init__.py
    │   ├── detector.py         # YOLOv8 person detector
    │   ├── tracker.py          # Deep SORT tracker
    │   ├── counter.py          # entrance line counting
    │   └── profiler.py         # profiling stub
    ├── utils/
    │   ├── __init__.py
    │   ├── video.py            # RTSP handling
    │   ├── geometry.py         # line/geometry helpers
    │   ├── events.py           # in-memory event storage
    │   └── streaming.py        # frame buffers for MJPEG streaming
    └── api/
        ├── __init__.py
        ├── server.py           # FastAPI routes
        ├── live.html           # live camera UI
        └── dashboard.html      # summary dashboard UI
```

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Configuration
Create a `config.yaml` in the project root:
```yaml
log_level: INFO
model_path: yolov8n.pt
api_host: 0.0.0.0
api_port: 8080
cameras:
  - id: "entrance-1"
    name: "Front Door"
    rtsp_url: "rtsp://username:password@camera-ip/stream"
    entrance_line:
      p1: [0.1, 0.8]
      p2: [0.9, 0.8]
```
- `entrance_line` coordinates are normalized (0..1) relative to frame width/height.

## Running
### Start processing and API together
```bash
python -m src.main
```
This will start one processing thread per configured camera and launch the FastAPI server on `http://0.0.0.0:8080` by default. You can change the host and port in `config.yaml` using `api_host` and `api_port`.

### Running the API separately
If you prefer to run only the API (after initializing state in code), you can run:
```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8080
```

## Web UI
- `GET /live` renders live camera views with MJPEG streams, bounding boxes, and counters.
- `GET /dashboard` renders a summary dashboard using in-memory event analytics.
- For a quick preview without a running backend, open the HTML files directly; they automatically fall back to demo mode when served from `file://` or when you append `?demo=1`, e.g. `file:///path/to/repo/src/api/live.html` or `file:///path/to/repo/src/api/dashboard.html`. The live page will also use demo cards if the API returns zero cameras.

## API Endpoints
- `GET /health` - Service status.
- `GET /cameras` - Configured cameras.
- `GET /cameras/{camera_id}/counts` - Entered/exited/current occupancy.
- `GET /cameras/{camera_id}/events` - Recent entrance events.

## Limitations & Next Steps
- Customer profiling is a stub placeholder for age, gender presentation, and clothing colors.
- Data is stored in memory; add persistent storage for production.
- MJPEG streaming is simple and intended for internal use; consider FPS limiting or resizing for large deployments.
- If you start only the API without running `main.py`, the UI will load with an empty demo state until camera workers are running.
- Model weights and RTSP credentials are user-provided.

## Testing
A placeholder test suite is included:
```bash
pytest
```
