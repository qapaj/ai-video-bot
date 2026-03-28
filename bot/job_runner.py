"""
bot/job_runner.py
Executes the full video generation pipeline for a single Job.
Now with detailed per-stage error messages sent to Telegram user.
"""

from __future__ import annotations

import os
import shutil
import traceback
from typing import Callable

from utils.config import get_config
from utils.helpers import file_ok, safe_filename
from utils.logger import get_logger

log = get_logger("bot.runner")

# Progress milestones
STAGE_SCRIPT   = 5
STAGE_NARRATE  = 20
STAGE_MEDIA    = 40
STAGE_ASSEMBLE = 60
STAGE_UPLOAD   = 85
STAGE_DONE     = 100


def run_pipeline_job(job, progress_cb: Callable) -> None:
    """
    Execute all pipeline stages for a job.
    Sends informative error messages on failure.
    Raises on unrecoverable failure so job_queue marks job FAILED.
    """
    cfg   = get_config()
    prefs = job.prefs
    topic = job.topic

    language     = prefs.get("language",     cfg.language)
    video_type   = prefs.get("video_type",   cfg.video_type)
    voice_gender = prefs.get("voice_gender", cfg.voice_gender)
    quality      = prefs.get("quality",      cfg.quality)

    safe_t   = safe_filename(topic)
    work_dir = str(cfg.paths.work_dir / f"job_{job.job_id}")
    out_dir  = str(cfg.paths.output_dir)
    final    = os.path.join(out_dir, f"video_{video_type}_{safe_t}.mp4")

    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(out_dir,  exist_ok=True)

    log.info("Job %s: topic=%s lang=%s type=%s voice=%s",
             job.job_id, topic, language, video_type, voice_gender)

    try:
        # ── Stage 1: Script ────────────────────────────────────────────────────
        _check_cancel(job)
        progress_cb(job, STAGE_SCRIPT, "📝 Generating script...")

        from model.script_engine import ScriptEngine
        script = ScriptEngine().generate(topic, video_type, language)
        segs   = script["segments"]
        log.info("Script: %d segments, ~%.0fs", len(segs), script["total_duration"])

        if not segs:
            raise RuntimeError("Script engine returned 0 segments")

        # ── Stage 2: Narration ─────────────────────────────────────────────────
        _check_cancel(job)
        progress_cb(job, STAGE_NARRATE, "🎙 Generating narration audio...")

        from audio.narration_generator import NarrationGenerator
        ng    = NarrationGenerator()
        segs_audio = ng.generate_all(
            segs,
            os.path.join(work_dir, "audio"),
            language=language,
            gender=voice_gender,
        )

        valid = [s for s in segs_audio if file_ok(s.get("audio_path", ""), 100)]
        log.info("Narration: %d/%d segments synthesised", len(valid), len(segs))

        if not valid:
            raise RuntimeError(
                "All TTS backends failed — no audio generated.\n"
                "Checked: edge-tts CLI → HF MMS-TTS → espeak-ng.\n"
                f"Language: {language}, Voice: {voice_gender}.\n"
                "Check Actions logs for per-backend error details."
            )

        if len(valid) < len(segs) * 0.5:
            log.warning("Only %d/%d segments have audio — continuing with partial",
                        len(valid), len(segs))

        # ── Stage 3: Media ─────────────────────────────────────────────────────
        _check_cancel(job)
        progress_cb(job, STAGE_MEDIA, "🖼 Fetching media assets...")

        from media.media_fetcher import MediaFetcher
        media = MediaFetcher().fetch(
            topic,
            os.path.join(work_dir, "media"),
            language=language,
        )
        log.info("Media: %d files", media.get("total", 0))

        # ── Stage 4: Video assembly ────────────────────────────────────────────
        _check_cancel(job)
        progress_cb(job, STAGE_ASSEMBLE, "🎬 Rendering video...")

        from video.video_engine import VideoEngine
        ok = VideoEngine().assemble(
            script_data          = script,
            segments_with_audio  = valid,
            media_data           = media,
            work_dir             = os.path.join(work_dir, "assembly"),
            final_output_path    = final,
        )

        if not ok or not file_ok(final, 10_000):
            raise RuntimeError(
                "Video assembly failed at render stage.\n"
                "Check Actions logs for FFmpeg error details."
            )

        size_mb = os.path.getsize(final) / 1_000_000
        log.info("Video ready: %.1fMB at %s", size_mb, final)

        # ── Stage 5: Upload ────────────────────────────────────────────────────
        _check_cancel(job)
        progress_cb(job, STAGE_UPLOAD, "📤 Uploading to Telegram...")

        total_dur = sum(s.get("actual_duration", 5.0) for s in valid)
        _upload(job, final, topic, video_type, total_dur, language)

        job.result = final
        progress_cb(job, STAGE_DONE, "✅ Done!")
        log.info("Job %s complete", job.job_id)

    except _CancelledException:
        log.info("Job %s cancelled by user", job.job_id)
        raise

    except Exception as e:
        log.error("Job %s FAILED: %s\n%s", job.job_id, e, traceback.format_exc())
        raise

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        log.debug("Work dir cleaned: %s", work_dir)


def _upload(job, video_path: str, topic: str, video_type: str,
            duration: float, language: str) -> None:
    from bot.uploader import TelegramUploader
    cfg     = get_config()
    caption = TelegramUploader(cfg.telegram.token).build_caption(
        topic, video_type, duration, language
    )
    ok = TelegramUploader(cfg.telegram.token).send_video(
        chat_id    = job.chat_id,
        video_path = video_path,
        caption    = caption,
    )
    if not ok:
        raise RuntimeError("Telegram upload failed — video may be too large or token invalid")


def _check_cancel(job) -> None:
    if getattr(job, "cancelled", False):
        raise _CancelledException()


class _CancelledException(Exception):
    pass
