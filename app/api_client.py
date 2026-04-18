from __future__ import annotations

import base64
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request

from app.utils.config import ApiConfig


class AttendanceApiClient:
    def __init__(self, config: ApiConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def send_attendance(self, employee_id: int, camera_id: str) -> bool:
        payload = {
            "employee_id": employee_id,
            "camera_id": camera_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return self._post_json(
            path=self.config.attendance_path,
            payload=payload,
            success_log=(
                "attendance_posted employee_id=%s camera_id=%s status=%s",
                employee_id,
                camera_id,
            ),
            failure_log_prefix=(
                "attendance_post_failed employee_id=%s camera_id=%s",
                employee_id,
                camera_id,
            ),
        )

    def send_unknown_person_detection(
        self,
        camera_id: str,
        embedding: list[float],
        image_path: Path | None = None,
        confidence_score: float | None = None,
    ) -> bool:
        face_image_base64 = None
        if image_path is not None and image_path.exists():
            face_image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

        payload = {
            "camera_id": camera_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "embedding": embedding,
            "face_image_base64": face_image_base64,
            "confidence_score": confidence_score,
        }

        return self._post_json(
            path=self.config.unknown_person_detect_path,
            payload=payload,
            success_log=(
                "unknown_person_posted camera_id=%s status=%s",
                camera_id,
            ),
            failure_log_prefix=(
                "unknown_person_post_failed camera_id=%s",
                camera_id,
            ),
        )

    def fetch_unknown_persons_sync(self) -> dict | None:
        return self._get_json(
            path=self.config.unknown_persons_sync_path,
            error_label="unknown_persons_sync_fetch_failed",
        )

    def fetch_enrollment_sync_data(self) -> dict | None:
        return self._get_json(
            path=self.config.enrollment_sync_path,
            error_label="enrollment_sync_fetch_failed",
        )

    def report_enrollment_sync_status(self, results: list[dict]) -> bool:
        payload = {"results": results}
        return self._post_json(
            path=self.config.enrollment_status_report_path,
            payload=payload,
            success_log=("enrollment_sync_reported status=%s results=%s", len(results)),
            failure_log_prefix=("enrollment_sync_report_failed results=%s", len(results)),
        )

    def download_file(self, url: str, destination: Path) -> bool:
        destination.parent.mkdir(parents=True, exist_ok=True)
        req = request.Request(url=url, method="GET")

        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                data = response.read()
                destination.write_bytes(data)
                return True
        except Exception as exc:
            self.logger.error(
                "enrollment_image_download_failed url=%s destination=%s error=%s",
                url,
                destination,
                exc,
            )
            return False

    def _get_json(self, path: str, error_label: str) -> dict | None:
        url = f"{self.config.base_url.rstrip('/')}{path}"
        req = request.Request(
            url=url,
            method="GET",
            headers={"x-api-key": self.config.api_key},
        )

        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                payload_text = response.read().decode("utf-8")
                return json.loads(payload_text) if payload_text else {}
        except Exception as exc:
            self.logger.error("%s error=%s", error_label, exc)
            return None

    def _post_json(
        self,
        path: str,
        payload: dict,
        success_log: tuple,
        failure_log_prefix: tuple,
    ) -> bool:
        encoded_payload = json.dumps(payload).encode("utf-8")
        url = f"{self.config.base_url.rstrip('/')}{path}"

        last_error: Exception | None = None
        for attempt in range(1, self.config.retry_attempts + 1):
            req = request.Request(
                url=url,
                data=encoded_payload,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.config.api_key,
                },
            )
            try:
                with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                    status = getattr(response, "status", 200)
                    if 200 <= status < 300:
                        self.logger.info(success_log[0], *success_log[1:], status)
                        return True

                    body = response.read().decode("utf-8", errors="replace")
                    self.logger.warning(
                        f"{failure_log_prefix[0]} attempt=%s status=%s body=%s",
                        *failure_log_prefix[1:],
                        attempt,
                        status,
                        body[:300],
                    )
            except error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = exc
                self.logger.warning(
                    f"{failure_log_prefix[0]} attempt=%s status=%s body=%s",
                    *failure_log_prefix[1:],
                    attempt,
                    exc.code,
                    body[:300],
                )
            except error.URLError as exc:
                last_error = exc
                self.logger.warning(
                    f"{failure_log_prefix[0]} attempt=%s error=%s",
                    *failure_log_prefix[1:],
                    attempt,
                    exc,
                )

            if attempt < self.config.retry_attempts:
                time.sleep(self.config.retry_backoff_seconds * attempt)

        if last_error is not None:
            self.logger.error("%s exhausted_error=%s", failure_log_prefix[0], last_error)

        return False
