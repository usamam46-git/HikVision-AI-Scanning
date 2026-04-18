from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from app.utils.config import RecognitionConfig, RuntimeConfig


ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class EmployeeRecord:
    employee_id: int
    name: str
    embeddings: np.ndarray


@dataclass(frozen=True)
class DetectedFace:
    bbox: np.ndarray
    score: float
    kps: np.ndarray


@dataclass(frozen=True)
class RecognitionResult:
    employee_id: int | None
    employee_name: str | None
    score: float
    accepted: bool
    reason: str
    embedding: np.ndarray | None = None


class OnnxSessionFactory:
    def __init__(self, providers: list[str]) -> None:
        available = set(ort.get_available_providers())
        selected = [provider for provider in providers if provider in available]
        self.providers = selected or ["CPUExecutionProvider"]
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self.session_options = session_options

    def create(self, model_path: str | Path) -> ort.InferenceSession:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX model not found: {path}")
        return ort.InferenceSession(
            str(path),
            sess_options=self.session_options,
            providers=self.providers,
        )


class ScrfdDetector:
    def __init__(
        self,
        model_path: str | Path,
        providers: list[str],
        input_size: tuple[int, int],
        score_threshold: float,
        nms_threshold: float,
    ) -> None:
        self.session = OnnxSessionFactory(providers).create(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.input_size = input_size
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

        output_count = len(self.output_names)
        if output_count == 6:
            self.fmc = 3
            self.strides = [8, 16, 32]
            self.num_anchors = 2
            self.use_kps = False
        elif output_count == 9:
            self.fmc = 3
            self.strides = [8, 16, 32]
            self.num_anchors = 2
            self.use_kps = True
        elif output_count == 10:
            self.fmc = 5
            self.strides = [8, 16, 32, 64, 128]
            self.num_anchors = 1
            self.use_kps = False
        elif output_count == 15:
            self.fmc = 5
            self.strides = [8, 16, 32, 64, 128]
            self.num_anchors = 1
            self.use_kps = True
        else:
            raise ValueError(f"Unsupported SCRFD output layout: {output_count}")

        if not self.use_kps:
            raise ValueError("SCRFD model must include 5-point keypoint outputs")

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        resized, scale = self._resize_and_pad(frame)
        blob = cv2.dnn.blobFromImage(
            resized,
            scalefactor=1.0 / 128.0,
            size=self.input_size,
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        outputs = self.session.run(self.output_names, {self.input_name: blob})

        scores_list: list[np.ndarray] = []
        bboxes_list: list[np.ndarray] = []
        kps_list: list[np.ndarray] = []
        input_height, input_width = self.input_size[1], self.input_size[0]

        for idx, stride in enumerate(self.strides):
            scores = self._flatten_scores(outputs[idx])
            bbox_preds = self._flatten_boxes(outputs[idx + self.fmc]) * stride

            height = input_height // stride
            width = input_width // stride
            anchor_centers = np.stack(
                np.mgrid[:height, :width][::-1],
                axis=-1,
            ).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape(-1, 2)
            if self.num_anchors > 1:
                anchor_centers = np.repeat(anchor_centers, self.num_anchors, axis=0)

            positive = np.where(scores >= self.score_threshold)[0]
            if positive.size == 0:
                continue

            bboxes = self._distance2bbox(anchor_centers, bbox_preds)
            scores_list.append(scores[positive])
            bboxes_list.append(bboxes[positive])

            if self.use_kps:
                kps_preds = self._flatten_kps(outputs[idx + self.fmc * 2]) * stride
                kps = self._distance2kps(anchor_centers, kps_preds)
                kps_list.append(kps[positive])

        if not scores_list:
            return []

        scores = np.concatenate(scores_list)
        bboxes = np.concatenate(bboxes_list)
        order = np.argsort(-scores)
        scores = scores[order]
        bboxes = bboxes[order]
        kps = np.concatenate(kps_list)[order] if self.use_kps else None

        keep = self._nms(bboxes, scores, self.nms_threshold)
        faces: list[DetectedFace] = []
        for index in keep:
            bbox = bboxes[index] / scale
            bbox[0::2] = np.clip(bbox[0::2], 0, frame.shape[1] - 1)
            bbox[1::2] = np.clip(bbox[1::2], 0, frame.shape[0] - 1)

            if kps is None:
                continue

            landmarks = kps[index] / scale
            landmarks[:, 0] = np.clip(landmarks[:, 0], 0, frame.shape[1] - 1)
            landmarks[:, 1] = np.clip(landmarks[:, 1], 0, frame.shape[0] - 1)
            faces.append(
                DetectedFace(
                    bbox=bbox.astype(np.float32),
                    score=float(scores[index]),
                    kps=landmarks.astype(np.float32),
                )
            )
        return faces

    def _resize_and_pad(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        target_width, target_height = self.input_size
        height, width = image.shape[:2]
        scale = min(target_width / width, target_height / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(image, (resized_width, resized_height))
        padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        padded[:resized_height, :resized_width] = resized
        return padded, scale

    @staticmethod
    def _flatten_scores(output: np.ndarray) -> np.ndarray:
        scores = np.squeeze(output)
        return scores.reshape(-1).astype(np.float32)

    @staticmethod
    def _flatten_boxes(output: np.ndarray) -> np.ndarray:
        boxes = np.squeeze(output)
        return boxes.reshape(-1, 4).astype(np.float32)

    @staticmethod
    def _flatten_kps(output: np.ndarray) -> np.ndarray:
        kps = np.squeeze(output)
        return kps.reshape(-1, 10).astype(np.float32)

    @staticmethod
    def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)

    @staticmethod
    def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
        kps = np.zeros((distance.shape[0], 5, 2), dtype=np.float32)
        for index in range(5):
            kps[:, index, 0] = points[:, 0] + distance[:, index * 2]
            kps[:, index, 1] = points[:, 1] + distance[:, index * 2 + 1]
        return kps

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep: list[int] = []
        while order.size > 0:
            current = int(order[0])
            keep.append(current)
            if order.size == 1:
                break

            xx1 = np.maximum(x1[current], x1[order[1:]])
            yy1 = np.maximum(y1[current], y1[order[1:]])
            xx2 = np.minimum(x2[current], x2[order[1:]])
            yy2 = np.minimum(y2[current], y2[order[1:]])

            width = np.maximum(0.0, xx2 - xx1 + 1)
            height = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = width * height
            iou = intersection / (areas[current] + areas[order[1:]] - intersection)
            remaining = np.where(iou <= threshold)[0]
            order = order[remaining + 1]
        return keep


class ArcFaceEmbedder:
    def __init__(
        self,
        model_path: str | Path,
        providers: list[str],
    ) -> None:
        self.session = OnnxSessionFactory(providers).create(model_path)
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.output_name = self.session.get_outputs()[0].name
        shape = input_meta.shape
        self.input_width = int(shape[3]) if isinstance(shape[3], int) else 112
        self.input_height = int(shape[2]) if isinstance(shape[2], int) else 112

    def embed(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        aligned = self._norm_crop(frame, landmarks)
        blob = cv2.dnn.blobFromImage(
            aligned,
            scalefactor=1.0 / 127.5,
            size=(self.input_width, self.input_height),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        embedding = self.session.run([self.output_name], {self.input_name: blob})[0][0]
        return self._normalize_vector(np.asarray(embedding, dtype=np.float32))

    def _norm_crop(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        scaled_template = ARCFACE_TEMPLATE.copy()
        if self.input_width != 112:
            ratio = self.input_width / 112.0
            scaled_template *= ratio

        matrix, _ = cv2.estimateAffinePartial2D(
            landmarks.astype(np.float32),
            scaled_template.astype(np.float32),
            method=cv2.LMEDS,
        )
        if matrix is None:
            raise ValueError("Failed to estimate face alignment transform")

        return cv2.warpAffine(
            image,
            matrix,
            (self.input_width, self.input_height),
            borderValue=0.0,
        )

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm


class FaceRecognitionEngine:
    def __init__(
        self,
        recognition_config: RecognitionConfig,
        runtime_config: RuntimeConfig,
        employees_path: str | Path,
        logger: logging.Logger,
    ) -> None:
        self.recognition_config = recognition_config
        self.runtime_config = runtime_config
        self.logger = logger
        self.employees_path = Path(employees_path)
        self._employees_mtime = 0.0
        base_dir = Path(employees_path).resolve().parent
        self.detector = ScrfdDetector(
            model_path=self._resolve_model_path(
                base_dir, self.runtime_config.scrfd_model_path
            ),
            providers=self.runtime_config.providers,
            input_size=(
                self.runtime_config.detector_input_width,
                self.runtime_config.detector_input_height,
            ),
            score_threshold=self.runtime_config.detector_score_threshold,
            nms_threshold=self.runtime_config.detector_nms_threshold,
        )
        self.embedder = ArcFaceEmbedder(
            model_path=self._resolve_model_path(
                base_dir, self.runtime_config.arcface_model_path
            ),
            providers=self.runtime_config.providers,
        )
        self.employees = self._load_employees(self.employees_path)

    def _load_employees(self, employees_path: str | Path) -> list[EmployeeRecord]:
        path = Path(employees_path)
        if not path.exists():
            self.logger.warning("employees_file_missing path=%s", path)
            self._employees_mtime = 0.0
            return []

        self._employees_mtime = path.stat().st_mtime
        with path.open("r", encoding="utf-8") as handle:
            payload: list[dict[str, Any]] = json.load(handle)

        employees: list[EmployeeRecord] = []
        for item in payload:
            embeddings = np.asarray(item["embeddings"], dtype=np.float32)
            if embeddings.ndim != 2 or embeddings.shape[0] == 0:
                self.logger.warning(
                    "invalid_employee_embeddings employee_id=%s name=%s",
                    item.get("employee_id"),
                    item.get("name"),
                )
                continue

            employees.append(
                EmployeeRecord(
                    employee_id=int(item["employee_id"]),
                    name=str(item["name"]),
                    embeddings=self._normalize_matrix(embeddings),
                )
            )

        self.logger.info("loaded_employees count=%s", len(employees))
        return employees

    def reload_employees_if_changed(self) -> bool:
        if not self.employees_path.exists():
            return False

        current_mtime = self.employees_path.stat().st_mtime
        if current_mtime <= self._employees_mtime:
            return False

        self.employees = self._load_employees(self.employees_path)
        self.logger.info("employees_reloaded count=%s", len(self.employees))
        return True

    def detect_faces(self, frame: np.ndarray) -> list[dict[str, Any]]:
        faces = self.detector.detect(frame)
        return [
            {
                "bbox": face.bbox,
                "kps": face.kps,
                "det_score": face.score,
            }
            for face in faces
        ]

    def extract_embedding(self, frame: np.ndarray, face: dict[str, Any]) -> np.ndarray | None:
        landmarks = np.asarray(face.get("kps"), dtype=np.float32)
        if landmarks.shape != (5, 2):
            return None

        try:
            return self.embedder.embed(frame, landmarks)
        except Exception as exc:
            self.logger.warning("embedding_failed error=%s", exc)
            return None

    def evaluate_face(
        self,
        face: dict[str, Any],
        frame: np.ndarray,
    ) -> RecognitionResult:
        bbox = np.asarray(face["bbox"], dtype=np.int32)
        left, top, right, bottom = bbox.tolist()
        width = max(0, right - left)
        height = max(0, bottom - top)

        if min(width, height) < self.recognition_config.face_min_size:
            return RecognitionResult(None, None, 0.0, False, "face_too_small")

        crop = frame[max(0, top):max(0, bottom), max(0, left):max(0, right)]
        if crop.size == 0:
            return RecognitionResult(None, None, 0.0, False, "empty_crop")

        blur_score = cv2.Laplacian(
            cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F
        ).var()
        if blur_score < self.recognition_config.blur_threshold:
            return RecognitionResult(None, None, 0.0, False, "face_too_blurry")

        embedding = self.extract_embedding(frame, face)
        if embedding is None or embedding.size == 0:
            return RecognitionResult(None, None, 0.0, False, "missing_embedding")

        best_match = self._match_employee(embedding)
        if best_match is None:
            return RecognitionResult(
                None,
                None,
                0.0,
                False,
                "no_employees_loaded",
                embedding=embedding,
            )

        employee, score = best_match
        if score < self.recognition_config.threshold:
            return RecognitionResult(
                None,
                None,
                float(score),
                False,
                "below_threshold",
                embedding=embedding,
            )

        return RecognitionResult(
            employee_id=employee.employee_id,
            employee_name=employee.name,
            score=float(score),
            accepted=True,
            reason="matched",
            embedding=embedding,
        )

    def _match_employee(
        self, embedding: np.ndarray
    ) -> tuple[EmployeeRecord, float] | None:
        if not self.employees:
            return None

        scored: list[tuple[EmployeeRecord, float]] = []
        for employee in self.employees:
            similarities = employee.embeddings @ embedding
            if self.recognition_config.match_strategy == "best_match":
                score = float(np.max(similarities))
            else:
                top_k = min(self.recognition_config.top_k, similarities.shape[0])
                top_values = np.sort(similarities)[-top_k:]
                score = float(np.mean(top_values))
            scored.append((employee, score))
        return max(scored, key=lambda item: item[1])

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    @classmethod
    def _normalize_matrix(cls, matrix: np.ndarray) -> np.ndarray:
        rows = [cls._normalize_vector(row) for row in matrix]
        return np.asarray(rows, dtype=np.float32)

    @staticmethod
    def _resolve_model_path(base_dir: Path, configured_path: str) -> Path:
        path = Path(configured_path)
        if path.is_absolute():
            return path
        return (base_dir / path).resolve()
