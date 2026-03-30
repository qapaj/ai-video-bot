"""
utils/logger.py
Structured logger used by every module.
Writes to stdout (captured by GitHub Actions) and optionally to a file.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


_LOG_FORMAT = "[%(asctime)s] %(levelname)-8s %(name)-22s %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_root_configured = False


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Return a named logger with consistent formatting.
    Safe to call multiple times with the same name.
    """
    global _root_configured

    if not _root_configured:
        logging.basicConfig(
            level=logging.INFO,
            format=_LOG_FORMAT,
            datefmt=_DATE_FORMAT,
            stream=sys.stdout,
        )
        _root_configured = True

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if log_file and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        logger.addHandler(fh)

    return logger


class StageLogger:
    """
    Context manager that logs stage entry/exit with timing.

    Usage:
        with StageLogger(log, "TTS synthesis"):
            ...
    """

    def __init__(self, logger: logging.Logger, stage: str) -> None:
        self._log = logger
        self._stage = stage
        self._start: float = 0.0

    def __enter__(self) -> "StageLogger":
        import time
        self._start = time.monotonic()
        self._log.info("▶ %s", self._stage)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        import time
        elapsed = time.monotonic() - self._start
        if exc_type:
            self._log.error("✗ %s failed after %.1fs: %s", self._stage, elapsed, exc_val)
            return False  # re-raise
        self._log.info("✓ %s completed in %.1fs", self._stage, elapsed)
        return False
