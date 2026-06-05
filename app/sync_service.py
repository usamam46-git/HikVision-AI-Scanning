from __future__ import annotations

import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from app.api_client import AttendanceApiClient
from app.utils.config import AppConfig
from app.utils.enroll_embeddings import generate_embeddings
from app.utils.logging_utils import setup_logger


def _safe_name(value: str) -> str:
    cleaned = "".join(ch for ch in value if ch.isalnum() or ch in {"_", "-", " "}).strip()
    return "_".join(cleaned.split()) or "employee"


class EnrollmentSyncService:
    def __init__(self, config: AppConfig, app_dir: Path) -> None:
        self.config = config
        self.app_dir = app_dir
        self.logger = setup_logger(
            name="enrollment_sync",
            log_dir=str(self.app_dir / self.config.logging.log_dir),
            level=self.config.logging.level,
        )
        self.api_client = AttendanceApiClient(config=self.config.api, logger=self.logger)
        self.dataset_dir = self.app_dir / self.config.api.enrollment_dataset_dir
        self.output_file = self.app_dir / "employees.json"
        self.config_path = self.app_dir / "config.yaml"

    def run_once(self) -> bool:
        payload = self.api_client.fetch_enrollment_sync_data()
        if not payload:
            return False

        employees = payload.get("employees") or []
        self._sync_dataset(employees)
        records = generate_embeddings(
          input_dir=self.dataset_dir,
          output=self.output_file,
          config_path=self.config_path,
        )

        records_by_id = {int(item["employee_id"]): item for item in records}
        reported_results = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for employee in employees:
            employee_id = int(employee["employee_id"])
            images_count = int(employee.get("images_count") or 0)
            if images_count <= 0:
                continue

            generated = records_by_id.get(employee_id)
            if generated:
                reported_results.append(
                    {
                        "employee_id": employee_id,
                        "status": "enrolled",
                        "embeddings_count": len(generated.get("embeddings") or []),
                        "error_message": None,
                        "last_enrolled_at": now_iso,
                    }
                )
            else:
                reported_results.append(
                    {
                        "employee_id": employee_id,
                        "status": "failed",
                        "embeddings_count": 0,
                        "error_message": "No embeddings generated from synced enrollment images",
                        "last_enrolled_at": None,
                    }
                )

        if reported_results:
            self.api_client.report_enrollment_sync_status(reported_results)

        self.logger.info(
            "enrollment_sync_completed employees=%s reported=%s",
            len(employees),
            len(reported_results),
        )
        return True

    def run_forever(self, should_stop: threading.Event) -> None:
        self.logger.info(
            "enrollment_sync_started interval_seconds=%s dataset_dir=%s",
            self.config.api.sync_poll_interval_seconds,
            self.dataset_dir,
        )

        while not should_stop.is_set():
            try:
                self.run_once()
            except Exception:
                self.logger.exception("enrollment_sync_cycle_failed")

            should_stop.wait(self.config.api.sync_poll_interval_seconds)

        self.logger.info("enrollment_sync_stopped")

    def _sync_dataset(self, employees: list[dict]) -> None:
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        expected_dirs: set[str] = set()

        for employee in employees:
            employee_id = int(employee["employee_id"])
            images = employee.get("images") or []
            folder_name = f"{employee_id}_{_safe_name(str(employee.get('name') or 'employee'))}"
            employee_dir = self.dataset_dir / folder_name

            if not images:
                if employee_dir.exists():
                    shutil.rmtree(employee_dir, ignore_errors=True)
                continue

            expected_dirs.add(folder_name)
            employee_dir.mkdir(parents=True, exist_ok=True)
            expected_files: set[str] = set()

            for image in images:
                image_id = int(image["image_id"])
                source_url = image.get("url")
                original_name = image.get("original_name") or f"{image_id}.jpg"
                extension = Path(original_name).suffix or ".jpg"
                local_name = f"{image_id}{extension}"
                expected_files.add(local_name)

                destination = employee_dir / local_name
                if destination.exists():
                    continue

                if source_url:
                    self.api_client.download_file(source_url, destination)

            for existing in employee_dir.iterdir():
                if existing.is_file() and existing.name not in expected_files:
                    existing.unlink(missing_ok=True)

        for existing in self.dataset_dir.iterdir():
            if existing.is_dir() and existing.name not in expected_dirs:
                shutil.rmtree(existing, ignore_errors=True)
