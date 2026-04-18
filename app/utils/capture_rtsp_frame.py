from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture a single enrollment frame from a configured RTSP camera."
    )
    parser.add_argument("--config", required=True, help="Path to app/config.yaml")
    parser.add_argument("--camera-id", required=True, help="Configured camera id")
    parser.add_argument("--output", required=True, help="Output JPEG path")
    parser.add_argument("--timeout-seconds", type=int, default=20)
    parser.add_argument("--warmup-frames", type=int, default=5)
    return parser.parse_args()


def load_camera_url(config_path: Path, camera_id: str) -> str:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    for camera in payload.get("cameras", []):
        if camera.get("id") == camera_id:
            return str(camera["url"])
    raise ValueError(f"Camera '{camera_id}' not found in {config_path}")


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    output_path = Path(args.output).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    camera_url = load_camera_url(config_path, args.camera_id)
    capture = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not capture.isOpened():
        raise RuntimeError(f"Unable to open RTSP stream for camera '{args.camera_id}'")

    deadline = time.time() + max(1, args.timeout_seconds)
    warmup_frames = max(0, args.warmup_frames)
    best_frame = None

    try:
        while time.time() < deadline:
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                time.sleep(0.1)
                continue

            best_frame = frame
            if warmup_frames > 0:
                warmup_frames -= 1
                continue
            break
    finally:
        capture.release()

    if best_frame is None:
        raise RuntimeError(f"No usable frame received from camera '{args.camera_id}'")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = cv2.imwrite(str(output_path), best_frame)
    if not written:
        raise RuntimeError(f"Failed to write captured frame to {output_path}")

    print(str(output_path))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        raise
