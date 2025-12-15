from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import yaml
from pydantic import BaseModel, Field, ValidationError, root_validator

logger = logging.getLogger(__name__)


class LineDefinition(BaseModel):
    p1: Tuple[float, float] = Field(..., description="Start point as normalized coordinates (x, y)")
    p2: Tuple[float, float] = Field(..., description="End point as normalized coordinates (x, y)")

    @root_validator(skip_on_failure=True)
    def validate_points(cls, values: dict) -> dict:
        for key in ("p1", "p2"):
            point = values.get(key)
            if point is None:
                continue
            if not all(0.0 <= coord <= 1.0 for coord in point):
                raise ValueError(f"{key} coordinates must be in the range [0, 1]")
        return values


class CameraConfig(BaseModel):
    id: str
    name: str
    rtsp_url: str
    entrance_line: LineDefinition


class AppConfig(BaseModel):
    log_level: str = Field("INFO", description="Logging level")
    model_path: str = Field("yolov8n.pt", description="Path to YOLOv8 model")
    api_host: str = Field("0.0.0.0", description="Host interface for the API server")
    api_port: int = Field(8080, description="Port for the API server")
    cameras: List[CameraConfig]


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load application configuration from YAML and return validated model.

    Args:
        path: Optional path to configuration file. Defaults to ``config.yaml``.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the configuration content is invalid.

    Returns:
        AppConfig: Parsed application configuration.
    """

    config_path = Path(path or "config.yaml")
    if not config_path.exists():
        logger.error("Configuration file %s not found", config_path)
        raise FileNotFoundError(f"Configuration file {config_path} not found")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw_config = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - YAML errors are unlikely in dummy test
        logger.exception("Failed to parse YAML configuration: %s", exc)
        raise ValueError("Failed to parse YAML configuration") from exc

    try:
        return AppConfig(**raw_config)
    except ValidationError as exc:
        logger.error("Invalid configuration: %s", exc)
        raise ValueError("Invalid configuration") from exc


__all__ = ["AppConfig", "CameraConfig", "LineDefinition", "load_config"]
