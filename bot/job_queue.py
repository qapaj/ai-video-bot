"""
bot/job_queue.py
Sequential job queue for video generation requests.

Design:
- Jobs run one at a time (GitHub Actions constraint)
- Each job has a unique ID, status, and cancellation flag
- Progress callbacks update the Telegram user in real-time
- Errors are caught and reported without crashing the queue
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("bot.queue")


class JobStatus(Enum):
    QUEUED     = "queued"
    RUNNING    = "running"
    DONE       = "done"
    FAILED     = "failed"
    CANCELLED  = "cancelled"


@dataclass
class Job:
    job_id:   str
    chat_id:  str
    topic:    str
    prefs:    Dict[str, Any]
    status:   JobStatus      = JobStatus.QUEUED
    progress: int            = 0          # 0-100
    message:  str            = ""
    result:   Optional[str]  = None       # final video path
    error:    Optional[str]  = None
    cancelled: bool          = False
    created_at: float        = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    def elapsed(self) -> float:
        if self.started_at:
            end = self.finished_at or time.time()
            return end - self.started_at
        return 0.0


class JobQueue:
    """
    Thread-safe sequential job queue.
    One worker thread processes jobs one at a time.
    """

    def __init__(self) -> None:
        self._queue:   List[Job] = []
        self._current: Optional[Job] = None
        self._lock     = threading.Lock()
        self._event    = threading.Event()
        self._running  = False
        self._thread:  Optional[threading.Thread] = None

        # Callbacks
        self._on_progress: Optional[Callable[[Job], None]] = None
        self._on_complete: Optional[Callable[[Job], None]] = None
        self._on_error:    Optional[Callable[[Job], None]] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_callbacks(
        self,
        on_progress: Callable[[Job], None],
        on_complete: Callable[[Job], None],
        on_error:    Callable[[Job], None],
    ) -> None:
        self._on_progress = on_progress
        self._on_complete = on_complete
        self._on_error    = on_error

    def start(self) -> None:
        """Start the background worker thread."""
        self._running = True
        self._thread  = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        log.info("Job queue started")

    def stop(self) -> None:
        """Stop the worker thread."""
        self._running = False
        self._event.set()

    def submit(self, chat_id: str, topic: str, prefs: dict) -> Job:
        """Add a new video generation job to the queue."""
        job = Job(
            job_id  = str(uuid.uuid4())[:8],
            chat_id = chat_id,
            topic   = topic,
            prefs   = prefs,
        )
        with self._lock:
            self._queue.append(job)
        self._event.set()
        log.info("Job %s queued: %s (position %d)", job.job_id, topic,
                 self.queue_position(chat_id))
        return job

    def cancel(self, chat_id: str) -> bool:
        """Cancel the active or queued job for a chat_id."""
        with self._lock:
            # Cancel active job
            if self._current and self._current.chat_id == str(chat_id):
                self._current.cancelled = True
                log.info("Cancelled active job %s", self._current.job_id)
                return True
            # Cancel queued job
            for job in self._queue:
                if job.chat_id == str(chat_id) and job.status == JobStatus.QUEUED:
                    job.status = JobStatus.CANCELLED
                    self._queue.remove(job)
                    log.info("Cancelled queued job %s", job.job_id)
                    return True
        return False

    def get_status(self, chat_id: str) -> Optional[Job]:
        """Return the most recent job for a chat_id."""
        with self._lock:
            if self._current and self._current.chat_id == str(chat_id):
                return self._current
            for job in reversed(self._queue):
                if job.chat_id == str(chat_id):
                    return job
        return None

    def queue_position(self, chat_id: str) -> int:
        """Return 1-based position in queue (0 if not queued)."""
        with self._lock:
            for i, job in enumerate(self._queue):
                if job.chat_id == str(chat_id):
                    return i + 1
        return 0

    def queue_length(self) -> int:
        with self._lock:
            return len(self._queue)

    # ── Worker ─────────────────────────────────────────────────────────────────

    def _worker(self) -> None:
        while self._running:
            self._event.wait(timeout=5)
            self._event.clear()

            while True:
                with self._lock:
                    if not self._queue:
                        break
                    job = self._queue.pop(0)

                if job.status == JobStatus.CANCELLED:
                    continue

                self._run_job(job)

    def _run_job(self, job: Job) -> None:
        """Execute a single job and dispatch callbacks."""
        with self._lock:
            self._current = job
        job.status     = JobStatus.RUNNING
        job.started_at = time.time()
        log.info("Starting job %s: %s", job.job_id, job.topic)

        try:
            from bot.job_runner import run_pipeline_job
            run_pipeline_job(job, self._progress_callback)
            job.status      = JobStatus.DONE
            job.progress    = 100
            job.finished_at = time.time()
            log.info("Job %s done in %.1fs", job.job_id, job.elapsed())
            if self._on_complete:
                self._on_complete(job)

        except Exception as e:
            job.status      = JobStatus.FAILED
            job.error       = str(e)
            job.finished_at = time.time()
            log.error("Job %s failed: %s", job.job_id, e)
            if self._on_error:
                self._on_error(job)

        finally:
            with self._lock:
                if self._current is job:
                    self._current = None

    def _progress_callback(self, job: Job, progress: int, message: str) -> None:
        job.progress = progress
        job.message  = message
        if self._on_progress:
            try:
                self._on_progress(job)
            except Exception as e:
                log.debug("Progress callback error: %s", e)
