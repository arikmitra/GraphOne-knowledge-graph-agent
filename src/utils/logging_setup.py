"""Structured, run-scoped logging (Phase: Production Readiness requirement)."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(log_dir: str = "data/logs", run_id: str = "run") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"{run_id}.log"

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    root = logging.getLogger("graphone")
    root.setLevel(logging.INFO)
    root.handlers.clear()

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    root.addHandler(stream_handler)

    return root
