"""
bot/uploader.py
Uploads the final video to Telegram.
Handles compression if file exceeds the 50MB Telegram limit.
FIX: subprocess.run in compress() is now wrapped in try/except via run_cmd.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

import requests

from utils.helpers import ffprobe_dims, ffprobe_duration, file_ok, run_cmd
from utils.logger import get_logger

log = get_logger("bot.uploader")

MAX_BYTES = 48 * 1_000_000


class TelegramUploader:
    def __init__(self, token: str) -> None:
        self.token    = token
        self.base_url = f"https://api.telegram.org/bot{token}"

    def send_video(self, chat_id: str, video_path: str,
                   caption: str = "", timeout: int = 300) -> bool:
        if not file_ok(video_path, 1000):
            log.error("Video file missing or too small: %s", video_path)
            return False

        final = video_path
        if os.path.getsize(video_path) > MAX_BYTES:
            compressed = video_path.replace(".mp4", "_c.mp4")
            if self._compress(video_path, compressed):
                final = compressed
            else:
                log.warning("Compression failed — sending original")

        thumb = self._extract_thumb(final)
        w, h  = ffprobe_dims(final)
        dur   = int(ffprobe_duration(final))

        self.send_status(chat_id, "📤 Uploading video to Telegram...")

        try:
            with open(final, "rb") as vf:
                files = {"video": (Path(final).name, vf, "video/mp4")}
                data  = {
                    "chat_id":            chat_id,
                    "caption":            caption,
                    "parse_mode":         "HTML",
                    "duration":           dur,
                    "width":              w,
                    "height":             h,
                    "supports_streaming": "true",
                }
                if thumb and os.path.exists(thumb):
                    with open(thumb, "rb") as tf:
                        files["thumbnail"] = ("thumb.jpg", tf, "image/jpeg")
                        resp = requests.post(
                            f"{self.base_url}/sendVideo",
                            data=data, files=files, timeout=timeout,
                        )
                else:
                    resp = requests.post(
                        f"{self.base_url}/sendVideo",
                        data=data, files=files, timeout=timeout,
                    )

            if resp.status_code == 200 and resp.json().get("ok"):
                log.info("✓ Video sent (%dKB, %ds)", os.path.getsize(final) // 1024, dur)
                return True
            log.error("Upload failed: %s — %s", resp.status_code, resp.text[:200])
            return False

        except Exception as e:
            log.error("Upload exception: %s", e)
            return False

    def send_status(self, chat_id: str, text: str,
                    reply_markup: Optional[dict] = None) -> None:
        try:
            payload: dict = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
            if reply_markup:
                payload["reply_markup"] = reply_markup
            requests.post(f"{self.base_url}/sendMessage",
                          json=payload, timeout=15)
        except Exception as e:
            log.debug("send_status error: %s", e)

    def build_caption(self, topic: str, video_type: str,
                      duration: float, language: str = "ar") -> str:
        emojis = {"news": "📰", "story": "📖", "facts": "💡", "education": "🎓"}
        labels_ar = {"news": "فيديو إخباري", "story": "قصة",
                     "facts": "حقائق مذهلة", "education": "تعليمي"}
        labels_en = {"news": "News", "story": "Story",
                     "facts": "Facts", "education": "Educational"}
        emoji = emojis.get(video_type, "🎬")
        label = labels_ar.get(video_type, video_type) if language == "ar" \
                else labels_en.get(video_type, video_type)
        tag   = f"#{topic.replace(' ', '_')} #AI_Video #{video_type}"
        return (f"{emoji} <b>{label}</b>\n\n"
                f"📌 {topic}\n⏱ {int(duration)}s\n\n"
                f"🤖 <i>AI Generated</i>\n\n{tag}")

    def _compress(self, src: str, dst: str, target_mb: int = 44) -> bool:
        dur = ffprobe_duration(src)
        if dur <= 0:
            return False
        bps = int((target_mb * 1024 * 1024 * 8) / (dur * 1000))
        vbr = max(600, bps - 128)
        lp  = "/tmp/_2pass"

        # FIX: use run_cmd instead of bare subprocess.run — catches FileNotFoundError
        run_cmd(
            ["ffmpeg", "-y", "-i", src, "-c:v", "libx264",
             "-b:v", f"{vbr}k", "-pass", "1", "-passlogfile", lp,
             "-an", "-f", "null", "/dev/null"],
            timeout=300, label="compress pass1",
        )
        ok, err = run_cmd(
            ["ffmpeg", "-y", "-i", src,
             "-c:v", "libx264", "-b:v", f"{vbr}k",
             "-pass", "2", "-passlogfile", lp,
             "-c:a", "aac", "-b:a", "128k",
             "-movflags", "+faststart", dst],
            timeout=600, label="compress pass2",
        )
        return ok and file_ok(dst, 1000)

    def _extract_thumb(self, video: str) -> Optional[str]:
        thumb = "/tmp/_thumb.jpg"
        ok, _ = run_cmd(
            ["ffmpeg", "-y", "-i", video,
             "-vframes", "1", "-vf", "scale=320:-1", "-q:v", "5", thumb],
            timeout=30, label="thumbnail",
        )
        return thumb if ok and file_ok(thumb, 100) else None
