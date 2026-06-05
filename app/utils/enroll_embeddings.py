from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.recognition import FaceRecognitionEngine
from app.utils.config import load_config
from app.utils.logging_utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate employee embeddings from local image folders."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory structured as input-dir/<employee_id>_<name>/*.jpg",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "employees.json"),
        help="Output employees.json path",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config.yaml"),
        help="Path to config.yaml",
    )
    return parser.parse_args()


def generate_embeddings(
    input_dir: str | Path,
    output: str | Path,
    config_path: str | Path,
) -> list[dict]:
    app_dir = Path(config_path).resolve().parent
    config = load_config(config_path)
    logger = setup_logger(
        name="enroll_embeddings",
        log_dir=str(app_dir / config.logging.log_dir),
        level=config.logging.level,
    )

    engine = FaceRecognitionEngine(
        recognition_config=config.recognition,
        runtime_config=config.runtime,
        employees_path=app_dir / "employees.json",
        logger=logger,
    )

    input_dir = Path(input_dir)
    records = []

    for employee_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        try:
            employee_id_text, employee_name = employee_dir.name.split("_", 1)
            employee_id = int(employee_id_text)
        except ValueError:
            logger.warning("skipping_invalid_directory name=%s", employee_dir.name)
            continue

        embeddings = []
        for image_path in sorted(employee_dir.glob("*")):
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning("image_read_failed path=%s", image_path)
                continue

            faces = engine.detect_faces(image)
            if not faces:
                logger.warning("no_face_detected path=%s", image_path)
                continue

            largest_face = max(
                faces,
                key=lambda face: (face["bbox"][2] - face["bbox"][0])
                * (face["bbox"][3] - face["bbox"][1]),
            )
            embedding = engine.extract_embedding(image, largest_face)
            if embedding is None:
                logger.warning("missing_embedding path=%s", image_path)
                continue

            embeddings.append(embedding.tolist())

        if len(embeddings) < 1:
            logger.warning(
                "insufficient_embeddings employee_id=%s name=%s count=%s",
                employee_id,
                employee_name,
                len(embeddings),
            )
            continue

        records.append(
            {
                "employee_id": employee_id,
                "name": employee_name,
                "embeddings": embeddings,
            }
        )
        logger.info(
            "employee_enrolled employee_id=%s name=%s embeddings=%s",
            employee_id,
            employee_name,
            len(embeddings),
        )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)

    logger.info("enrollment_complete output=%s employees=%s", output_path, len(records))
    return records


def main() -> int:
    args = parse_args()
    generate_embeddings(
        input_dir=args.input_dir,
        output=args.output,
        config_path=args.config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
