from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CameraConfig:
    id: str
    url: str
    label: str | None = None


@dataclass(frozen=True)
class RecognitionConfig:
    threshold: float
    cooldown_seconds: int
    min_frames: int
    process_every_n_frames: int
    face_min_size: int
    blur_threshold: float
    resize_width: int
    match_strategy: str
    top_k: int
    unknown_save_limit_per_hour: int


@dataclass(frozen=True)
class RuntimeConfig:
    reconnect_delay_seconds: int
    queue_poll_interval_seconds: float
    process_start_method: str
    providers: list[str]
    scrfd_model_path: str
    arcface_model_path: str
    detector_input_width: int
    detector_input_height: int
    detector_score_threshold: float
    detector_nms_threshold: float


@dataclass(frozen=True)
class ApiConfig:
    base_url: str
    attendance_path: str
    api_key: str
    timeout_seconds: int
    retry_attempts: int
    retry_backoff_seconds: float
    employees_sync_path: str = "/api/face-attendance/employees"
    enrollment_sync_path: str = "/api/face-enrollment/sync"
    enrollment_status_report_path: str = "/api/face-enrollment/sync/report"
    unknown_person_detect_path: str = "/api/unknown-persons/detect"
    unknown_persons_sync_path: str = "/api/unknown-persons/sync"
    sync_poll_interval_seconds: int = 120
    enrollment_dataset_dir: str = "synced-enrollments"


@dataclass(frozen=True)
class LoggingConfig:
    log_dir: str
    unknown_faces_dir: str
    level: str


@dataclass(frozen=True)
class AppConfig:
    cameras: list[CameraConfig]
    recognition: RecognitionConfig
    runtime: RuntimeConfig
    api: ApiConfig
    logging: LoggingConfig


def load_config(config_path: str | Path) -> AppConfig:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)

    return AppConfig(
        cameras=[CameraConfig(**camera) for camera in payload["cameras"]],
        recognition=RecognitionConfig(**payload["recognition"]),
        runtime=RuntimeConfig(**payload["runtime"]),
        api=ApiConfig(**payload["api"]),
        logging=LoggingConfig(**payload["logging"]),
    )
