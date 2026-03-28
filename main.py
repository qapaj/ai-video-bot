"""
main.py
Entry points:

  python main.py bot       — run the Telegram bot listener
  python main.py pipeline  — run one video generation job (called by GitHub Actions)
  python main.py test      — smoke-test the system
"""

from __future__ import annotations

import os
import sys

from utils.logger import get_logger

log = get_logger("main")


def run_bot() -> None:
    """Start the Telegram bot polling loop."""
    from utils.config import get_config
    cfg = get_config()

    if not cfg.telegram.token:
        log.error("TELEGRAM_BOT_TOKEN not set")
        sys.exit(1)

    from bot.telegram_bot import TelegramBot
    window = int(os.environ.get("POLL_WINDOW_SECONDS", "265"))
    bot = TelegramBot(cfg.telegram.token)
    bot.run(window_seconds=window)


def run_pipeline() -> None:
    """Execute one video generation job from environment variables."""
    from utils.config import get_config
    cfg = get_config()

    topic       = os.environ.get("VIDEO_TOPIC", "").strip()
    chat_id     = os.environ.get("CHAT_ID", "")
    language    = os.environ.get("VIDEO_LANGUAGE", cfg.language)
    video_type  = os.environ.get("VIDEO_TYPE", cfg.video_type)
    voice_gender= os.environ.get("VOICE_GENDER", cfg.voice_gender)
    quality     = os.environ.get("VIDEO_QUALITY", cfg.quality)

    if not topic:
        log.error("VIDEO_TOPIC not set")
        sys.exit(1)

    log.info("Pipeline: %s | %s | %s | %s", topic, video_type, language, voice_gender)

    # Notify Telegram user: pipeline started
    if cfg.telegram.token and chat_id:
        from bot.uploader import TelegramUploader
        u = TelegramUploader(cfg.telegram.token)
        u.send_status(chat_id,
            f"🎬 <b>Production started!</b>\n\n"
            f"📌 <b>{topic}</b>\n"
            f"📂 {video_type} | 🌍 {language} | 🎤 {voice_gender}\n\n"
            f"⏳ 5–8 minutes...")

    # Build a synthetic job object for the runner
    class _Job:
        job_id       = "pipeline"
        cancelled    = False
        topic        = topic
        chat_id      = chat_id
        result: str  = ""
        prefs = {
            "language":     language,
            "video_type":   video_type,
            "voice_gender": voice_gender,
            "quality":      quality,
        }

    job = _Job()

    def progress(j, pct, msg):
        log.info("[%d%%] %s", pct, msg)
        if cfg.telegram.token and chat_id:
            from bot.uploader import TelegramUploader
            TelegramUploader(cfg.telegram.token).send_status(chat_id, f"{msg} ({pct}%)")

    try:
        from bot.job_runner import run_pipeline_job
        run_pipeline_job(job, progress)
        log.info("✓ Pipeline completed")
    except Exception as e:
        log.error("Pipeline failed: %s", e)
        if cfg.telegram.token and chat_id:
            from bot.uploader import TelegramUploader
            TelegramUploader(cfg.telegram.token).send_status(
                chat_id, f"❌ <b>Failed:</b> <code>{str(e)[:200]}</code>"
            )
        sys.exit(1)


def run_test() -> None:
    """Smoke-test key components without API calls."""
    from utils.config import get_config
    from model.script_engine import ScriptEngine
    from video.text_renderer import ArabicTextShaper

    cfg    = get_config()
    script = ScriptEngine().generate("الذكاء الاصطناعي", "news", "ar")
    log.info("Script test: %d segments", len(script["segments"]))

    shaper = ArabicTextShaper()
    shaped = shaper.shape_for_render("مرحباً بكم في نشرتنا الإخبارية")
    log.info("Shaper test: '%s'", shaped[:40])

    log.info("✓ Smoke test passed")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pipeline"
    {"bot": run_bot, "pipeline": run_pipeline, "test": run_test}.get(
        cmd, run_pipeline
    )()
