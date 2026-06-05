from __future__ import annotations

import multiprocessing as mp
import signal
import sys
import threading
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.camera_worker import run_camera_worker
from app.sync_service import EnrollmentSyncService
from app.utils.config import load_config
from app.utils.logging_utils import setup_logger


def main() -> int:
    app_dir = Path(__file__).resolve().parent
    config = load_config(app_dir / "config.yaml")
    logger = setup_logger(
        name="attendance_main",
        log_dir=str(app_dir / config.logging.log_dir),
        level=config.logging.level,
    )

    try:
        mp.set_start_method(config.runtime.process_start_method, force=True)
    except RuntimeError:
        logger.warning("multiprocessing_start_method_already_set")

    processes: list[mp.Process] = []
    should_stop = mp.Event()
    sync_should_stop = threading.Event()

    def _handle_signal(signum: int, _frame: object) -> None:
        logger.warning("shutdown_signal_received signal=%s", signum)
        should_stop.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    sync_service = EnrollmentSyncService(config=config, app_dir=app_dir)
    try:
        sync_service.run_once()
    except Exception:
        logger.exception("initial_enrollment_sync_failed")

    sync_thread = threading.Thread(
        target=sync_service.run_forever,
        args=(sync_should_stop,),
        name="enrollment-sync-thread",
        daemon=True,
    )
    sync_thread.start()

    for camera in config.cameras:
        process = mp.Process(
            target=run_camera_worker,
            args=(camera, config, str(app_dir)),
            name=f"camera-process-{camera.id}",
        )
        process.start()
        processes.append(process)
        logger.info("camera_process_started camera_id=%s pid=%s", camera.id, process.pid)

    try:
        while not should_stop.is_set():
            for index, process in enumerate(processes):
                if process.is_alive():
                    continue

                camera = config.cameras[index]
                logger.error(
                    "camera_process_stopped camera_id=%s exitcode=%s restarting",
                    camera.id,
                    process.exitcode,
                )
                replacement = mp.Process(
                    target=run_camera_worker,
                    args=(camera, config, str(app_dir)),
                    name=f"camera-process-{camera.id}",
                )
                replacement.start()
                processes[index] = replacement
                logger.info(
                    "camera_process_restarted camera_id=%s pid=%s",
                    camera.id,
                    replacement.pid,
                )
            time.sleep(config.runtime.queue_poll_interval_seconds)
    finally:
        sync_should_stop.set()
        sync_thread.join(timeout=10)
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)
        logger.info("attendance_service_stopped")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
