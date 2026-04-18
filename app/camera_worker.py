from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from app.api_client import AttendanceApiClient
from app.recognition import FaceRecognitionEngine, RecognitionResult
from app.utils.config import AppConfig, CameraConfig
from app.utils.logging_utils import setup_logger


@dataclass
class TrackState:
    track_id: str
    recent_ids: deque[int]
    recent_unknowns: deque[bool]
    last_update: float
    bbox: tuple[int, int, int, int]
    unknown_reported: bool = False


class UnknownFaceLimiter:
    def __init__(self, hourly_limit: int) -> None:
        self.hourly_limit = hourly_limit
        self.hour_key = self._current_hour_key()
        self.count = 0

    def allow(self) -> bool:
        current_key = self._current_hour_key()
        if current_key != self.hour_key:
            self.hour_key = current_key
            self.count = 0

        if self.count >= self.hourly_limit:
            return False

        self.count += 1
        return True

    @staticmethod
    def _current_hour_key() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d%H")


class CameraWorker:
    def __init__(self, camera: CameraConfig, config: AppConfig, app_dir: Path) -> None:
        self.camera = camera
        self.config = config
        self.app_dir = app_dir
        self.logger = setup_logger(
            name=f"camera_{camera.id}",
            log_dir=str(self.app_dir / self.config.logging.log_dir),
            level=self.config.logging.level,
        )
        self.recognition = FaceRecognitionEngine(
            recognition_config=self.config.recognition,
            runtime_config=self.config.runtime,
            employees_path=self.app_dir / "employees.json",
            logger=self.logger,
        )
        self.api_client = AttendanceApiClient(config=self.config.api, logger=self.logger)
        self.last_seen: dict[int, float] = {}
        self.track_histories: dict[str, TrackState] = {}
        self.last_reload_check = 0.0
        self.unknown_limiter = UnknownFaceLimiter(
            self.config.recognition.unknown_save_limit_per_hour
        )
        self.unknown_faces_dir = self.app_dir / self.config.logging.unknown_faces_dir
        self.unknown_faces_dir.mkdir(parents=True, exist_ok=True)
        self.unknown_score_cutoff = max(0.35, self.config.recognition.threshold - 0.10)

    def run_forever(self) -> None:
        self.logger.info("camera_worker_started camera_id=%s", self.camera.id)
        while True:
            capture = self._open_stream()
            if capture is None:
                time.sleep(self.config.runtime.reconnect_delay_seconds)
                continue

            try:
                self._process_stream(capture)
            except Exception:
                self.logger.exception("camera_worker_error camera_id=%s", self.camera.id)
            finally:
                capture.release()
                self.logger.warning(
                    "camera_stream_released camera_id=%s reconnecting_in=%ss",
                    self.camera.id,
                    self.config.runtime.reconnect_delay_seconds,
                )
                time.sleep(self.config.runtime.reconnect_delay_seconds)

    def _open_stream(self) -> cv2.VideoCapture | None:
        self.logger.info("camera_connecting camera_id=%s url=%s", self.camera.id, self.camera.url)
        capture = cv2.VideoCapture(self.camera.url, cv2.CAP_FFMPEG)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            self.logger.error("camera_connect_failed camera_id=%s", self.camera.id)
            capture.release()
            return None
        return capture

    def _process_stream(self, capture: cv2.VideoCapture) -> None:
        frame_index = 0
        while True:
            self._reload_employees_if_needed()
            ok, frame = capture.read()
            if not ok or frame is None:
                self.logger.warning("empty_frame camera_id=%s", self.camera.id)
                break

            frame_index += 1
            if frame_index % self.config.recognition.process_every_n_frames != 0:
                continue

            processed_frame = self._resize_frame(frame)
            faces = self.recognition.detect_faces(processed_frame)
            self.logger.info(
                "faces_detected camera_id=%s count=%s frame_index=%s",
                self.camera.id,
                len(faces),
                frame_index,
            )
            self._cleanup_tracks()

            for face in faces:
                result = self.recognition.evaluate_face(face, processed_frame)
                self._handle_face_result(face, processed_frame, result)

    def _handle_face_result(
        self,
        face: dict,
        frame: np.ndarray,
        result: RecognitionResult,
    ) -> None:
        bbox = self._face_bbox(face)
        state = self._get_or_create_track(bbox)
        state.last_update = time.time()
        state.bbox = bbox

        if result.accepted and result.employee_id is not None:
            state.recent_ids.append(result.employee_id)
            state.recent_unknowns.clear()
            state.unknown_reported = False
            confirmed = (
                len(state.recent_ids) == self.config.recognition.min_frames
                and len(set(state.recent_ids)) == 1
            )

            self.logger.info(
                "recognized_face camera_id=%s employee_id=%s name=%s score=%.4f track=%s confirmed=%s",
                self.camera.id,
                result.employee_id,
                result.employee_name,
                result.score,
                state.track_id,
                confirmed,
            )

            if confirmed and self._can_emit_event(result.employee_id):
                sent = self.api_client.send_attendance(
                    employee_id=result.employee_id,
                    camera_id=self.camera.id,
                )
                if sent:
                    self.last_seen[result.employee_id] = time.time()
        else:
            state.recent_ids.clear()
            unknown_candidate = (
                result.embedding is not None
                and (
                    result.reason == "missing_embedding"
                    or (
                        result.reason == "below_threshold"
                        and float(result.score) <= self.unknown_score_cutoff
                    )
                )
            )
            state.recent_unknowns.append(
                unknown_candidate
            )
            confirmed_unknown = (
                len(state.recent_unknowns) == self.config.recognition.min_frames
                and all(state.recent_unknowns)
            )
            self.logger.info(
                "unrecognized_face camera_id=%s reason=%s score=%.4f track=%s",
                self.camera.id,
                result.reason,
                result.score,
                state.track_id,
            )
            if (
                confirmed_unknown
                and not state.unknown_reported
            ):
                saved_path = self._save_unknown_face(face, frame)
                sent = self.api_client.send_unknown_person_detection(
                    camera_id=self.camera.id,
                    embedding=result.embedding.tolist(),
                    image_path=saved_path,
                    confidence_score=float(result.score) if result.score else None,
                )
                state.unknown_reported = sent

    def _can_emit_event(self, employee_id: int) -> bool:
        last_seen_at = self.last_seen.get(employee_id)
        if last_seen_at is None:
            return True

        elapsed = time.time() - last_seen_at
        if elapsed < self.config.recognition.cooldown_seconds:
            self.logger.info(
                "attendance_suppressed employee_id=%s camera_id=%s elapsed=%.2f cooldown=%s",
                employee_id,
                self.camera.id,
                elapsed,
                self.config.recognition.cooldown_seconds,
            )
            return False
        return True

    def _save_unknown_face(self, face: dict, frame: np.ndarray) -> Path | None:
        if not self.unknown_limiter.allow():
            return None

        bbox = np.asarray(face["bbox"], dtype=np.int32)
        left, top, right, bottom = bbox.tolist()
        crop = frame[max(0, top):max(0, bottom), max(0, left):max(0, right)]
        if crop.size == 0:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = self.unknown_faces_dir / f"{self.camera.id}_{timestamp}.jpg"
        cv2.imwrite(str(filename), crop)
        return filename

    def _get_or_create_track(self, bbox: tuple[int, int, int, int]) -> TrackState:
        best_key: str | None = None
        best_iou = 0.0
        for key, state in self.track_histories.items():
            iou = self._calculate_iou(bbox, state.bbox)
            if iou > best_iou:
                best_iou = iou
                best_key = key

        if best_key is not None and best_iou >= 0.3:
            return self.track_histories[best_key]

        track_id = uuid.uuid4().hex[:8]
        state = TrackState(
            track_id=track_id,
            recent_ids=deque(maxlen=self.config.recognition.min_frames),
            recent_unknowns=deque(maxlen=self.config.recognition.min_frames),
            last_update=time.time(),
            bbox=bbox,
            unknown_reported=False,
        )
        self.track_histories[track_id] = state
        return state

    def _cleanup_tracks(self) -> None:
        cutoff = time.time() - 2.5
        expired = [
            key for key, state in self.track_histories.items() if state.last_update < cutoff
        ]
        for key in expired:
            self.track_histories.pop(key, None)

    def _reload_employees_if_needed(self) -> None:
        now = time.time()
        if now - self.last_reload_check < 3:
            return

        self.last_reload_check = now
        try:
            self.recognition.reload_employees_if_changed()
        except Exception:
            self.logger.exception("employees_reload_failed")

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        width = frame.shape[1]
        target_width = self.config.recognition.resize_width
        if width <= target_width:
            return frame

        ratio = target_width / width
        target_height = int(frame.shape[0] * ratio)
        return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _face_bbox(face: dict) -> tuple[int, int, int, int]:
        bbox = np.asarray(face["bbox"], dtype=np.int32)
        left, top, right, bottom = bbox.tolist()
        return left, top, right, bottom

    @staticmethod
    def _calculate_iou(
        box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]
    ) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union


def run_camera_worker(camera: CameraConfig, config: AppConfig, app_dir: str) -> None:
    worker = CameraWorker(camera=camera, config=config, app_dir=Path(app_dir))
    worker.run_forever()
