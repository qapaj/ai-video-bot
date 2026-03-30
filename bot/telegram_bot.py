"""
bot/telegram_bot.py
Professional Telegram bot with:
- Inline keyboard settings (language, type, voice, quality)
- 3-step /generate wizard
- Real-time progress updates
- Job queue integration
- Error recovery
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import requests

from bot.job_queue import Job, JobQueue, JobStatus
from utils.config import get_config
from utils.helpers import file_ok
from utils.logger import get_logger

log = get_logger("bot.telegram")

# ── Constants ─────────────────────────────────────────────────────────────────

LANGUAGE_OPTIONS = {
    "ar": "🇸🇦 عربي",
    "en": "🇺🇸 English",
    "fr": "🇫🇷 Français",
    "es": "🇪🇸 Español",
    "de": "🇩🇪 Deutsch",
}
TYPE_OPTIONS = {
    "news":      "📰 أخبار / News",
    "story":     "📖 قصة / Story",
    "facts":     "💡 حقائق / Facts",
    "education": "🎓 تعليمي / Education",
}
VOICE_OPTIONS  = {"female": "👩 Female", "male": "👨 Male"}
QUALITY_OPTIONS= {"standard": "⚡ Standard", "high": "✨ High"}

DEFAULT_PREFS = {
    "language":     "ar",
    "video_type":   "news",
    "voice_gender": "female",
    "quality":      "standard",
    "wizard_step":  None,
    "wizard_type":  None,
    "wizard_lang":  None,
}

HELP_TEXT = """\
🎬 <b>AI Video Bot</b>

<b>Commands:</b>
/start — Welcome message
/generate — Create a new video (guided wizard)
/status — Check your job status
/cancel — Cancel current job
/help — This message

<b>Quick commands:</b>
/news <i>topic</i>
/story <i>topic</i>
/facts <i>topic</i>
/education <i>topic</i>

<b>Example:</b> <code>/news artificial intelligence</code>

⏳ Production time: 5–8 minutes"""


class TelegramBot:
    """
    Long-polling Telegram bot with job queue integration.
    """

    def __init__(self, token: str) -> None:
        self.token    = token
        self.base_url = f"https://api.telegram.org/bot{token}"
        self._prefs   = self._load_prefs()
        self._queue   = JobQueue()
        self._setup_queue()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def run(self, window_seconds: int = 265) -> None:
        """Poll for updates for window_seconds, then exit cleanly."""
        self._queue.start()
        offset   = self._load_offset()
        deadline = time.time() + window_seconds
        log.info("Bot polling for %ds (offset=%d)", window_seconds, offset)

        while time.time() < deadline:
            try:
                updates = self._get_updates(offset)
                for upd in updates:
                    uid = upd["update_id"]
                    try:
                        if "callback_query" in upd:
                            self._handle_callback(upd["callback_query"])
                        elif "message" in upd or "edited_message" in upd:
                            msg = upd.get("message") or upd.get("edited_message")
                            if msg:
                                self._handle_message(msg)
                    except Exception as e:
                        log.error("Update %d error: %s", uid, e)
                    offset = uid + 1
                    self._save_offset(offset)
            except requests.exceptions.Timeout:
                pass
            except Exception as e:
                log.error("Poll error: %s", e)
                time.sleep(3)

        log.info("Polling window expired (offset=%d)", offset)
        self._queue.stop()

    # ── Update handlers ────────────────────────────────────────────────────────

    def _handle_message(self, msg: dict) -> None:
        chat_id = str(msg["chat"]["id"])
        msg_id  = str(msg.get("message_id", ""))
        text    = (msg.get("text") or "").strip()
        if not text:
            return

        prefs = self._get_prefs(chat_id)

        # Wizard: awaiting topic text
        if prefs.get("wizard_step") == "awaiting_topic":
            self._wizard_topic_received(chat_id, msg_id, text, prefs)
            return

        if not text.startswith("/"):
            self._send("sendMessage", chat_id=chat_id,
                       text="💬 Use /generate or /help")
            return

        parts = text.split(maxsplit=1)
        cmd   = parts[0].lower().split("@")[0]
        arg   = parts[1].strip() if len(parts) > 1 else ""

        handlers = {
            "/start":     lambda: self._cmd_start(chat_id),
            "/help":      lambda: self._cmd_help(chat_id),
            "/generate":  lambda: self._cmd_generate(chat_id),
            "/status":    lambda: self._cmd_status(chat_id),
            "/cancel":    lambda: self._cmd_cancel(chat_id),
            "/settings":  lambda: self._cmd_settings(chat_id),
        }
        quick = {"/news": "news", "/story": "story",
                 "/facts": "facts", "/education": "education"}

        if cmd in handlers:
            handlers[cmd]()
        elif cmd in quick:
            if not arg:
                self._send("sendMessage", chat_id=chat_id,
                           text=f"⚠️ Usage: <code>{cmd} your topic</code>",
                           parse_mode="HTML")
            else:
                self._set_pref(chat_id, "video_type", quick[cmd])
                self._launch_job(chat_id, arg, self._get_prefs(chat_id))
        else:
            self._send("sendMessage", chat_id=chat_id,
                       text="❓ Unknown command. Use /help")

    def _handle_callback(self, cb: dict) -> None:
        chat_id = str(cb["message"]["chat"]["id"])
        msg_id  = cb["message"]["message_id"]
        data    = cb.get("data", "")
        cb_id   = cb["id"]
        prefs   = self._get_prefs(chat_id)

        self._answer_callback(cb_id)

        # ── Main menu ──────────────────────────────────────────────────────────
        if data == "back_main":
            self._edit(chat_id, msg_id, "⚙️ <b>Settings</b>",
                       self._kb_main())

        elif data == "show_status":
            self._edit(chat_id, msg_id,
                       self._status_text(chat_id, prefs), self._kb_main())

        elif data.startswith("menu_"):
            menu = data[5:]
            texts = {
                "lang":    ("🌍 Choose language:", self._kb_language(prefs["language"])),
                "type":    ("📂 Choose type:",     self._kb_type(prefs["video_type"])),
                "voice":   ("🎤 Choose voice:",    self._kb_voice(prefs["voice_gender"])),
                "quality": ("✨ Choose quality:",  self._kb_quality(prefs["quality"])),
            }
            if menu in texts:
                t, kb = texts[menu]
                self._edit(chat_id, msg_id, t, kb)

        # ── Setting changes ────────────────────────────────────────────────────
        elif data.startswith("set_lang_"):
            v = data[9:]; self._set_pref(chat_id, "language", v)
            self._edit(chat_id, msg_id,
                       f"✅ Language: <b>{LANGUAGE_OPTIONS.get(v, v)}</b>",
                       self._kb_main())

        elif data.startswith("set_type_"):
            v = data[9:]; self._set_pref(chat_id, "video_type", v)
            self._edit(chat_id, msg_id,
                       f"✅ Type: <b>{TYPE_OPTIONS.get(v, v)}</b>",
                       self._kb_main())

        elif data.startswith("set_voice_"):
            v = data[10:]; self._set_pref(chat_id, "voice_gender", v)
            self._edit(chat_id, msg_id,
                       f"✅ Voice: <b>{VOICE_OPTIONS.get(v, v)}</b>",
                       self._kb_main())

        elif data.startswith("set_quality_"):
            v = data[12:]; self._set_pref(chat_id, "quality", v)
            self._edit(chat_id, msg_id,
                       f"✅ Quality: <b>{QUALITY_OPTIONS.get(v, v)}</b>",
                       self._kb_main())

        # ── Wizard ─────────────────────────────────────────────────────────────
        elif data == "wizard_start":
            self._set_pref(chat_id, "wizard_step", "pick_lang")
            self._edit(chat_id, msg_id,
                       "🌍 <b>Step 1/3: Choose language</b>",
                       self._kb_wizard_lang())

        elif data.startswith("wizard_lang_"):
            lang = data[12:]
            self._set_pref(chat_id, "wizard_lang", lang)
            self._set_pref(chat_id, "wizard_step", "pick_type")
            self._edit(chat_id, msg_id,
                       f"✅ Language: <b>{LANGUAGE_OPTIONS.get(lang)}</b>\n\n"
                       f"📂 <b>Step 2/3: Choose video type</b>",
                       self._kb_wizard_type())

        elif data.startswith("wizard_type_"):
            vtype = data[12:]
            self._set_pref(chat_id, "wizard_type", vtype)
            self._set_pref(chat_id, "wizard_step", "awaiting_topic")
            self._edit(chat_id, msg_id,
                       f"✅ Type: <b>{TYPE_OPTIONS.get(vtype)}</b>\n\n"
                       f"✏️ <b>Step 3/3: Type your topic</b>")

        elif data == "wizard_cancel":
            self._set_pref(chat_id, "wizard_step", None)
            self._edit(chat_id, msg_id, "❌ Cancelled.", self._kb_main())

        elif data.startswith("confirm_video_"):
            topic = data[14:]
            prefs = self._get_prefs(chat_id)
            self._edit(chat_id, msg_id,
                       f"⏳ <b>Starting production...</b>\n📌 {topic}")
            self._launch_job(chat_id, topic, prefs)

    # ── Commands ───────────────────────────────────────────────────────────────

    def _cmd_start(self, chat_id: str) -> None:
        self._send("sendMessage", chat_id=chat_id,
                   text=HELP_TEXT, parse_mode="HTML",
                   reply_markup=self._kb_main())

    def _cmd_help(self, chat_id: str) -> None:
        self._cmd_start(chat_id)

    def _cmd_generate(self, chat_id: str) -> None:
        self._set_pref(chat_id, "wizard_step", "pick_lang")
        self._send("sendMessage", chat_id=chat_id,
                   text="🌍 <b>Step 1/3: Choose language</b>",
                   parse_mode="HTML",
                   reply_markup=self._kb_wizard_lang())

    def _cmd_status(self, chat_id: str) -> None:
        job = self._queue.get_status(chat_id)
        if not job:
            self._send("sendMessage", chat_id=chat_id,
                       text="📊 No active or recent jobs.")
            return
        status_map = {
            JobStatus.QUEUED:    "⏳ Queued",
            JobStatus.RUNNING:   "🔄 Running",
            JobStatus.DONE:      "✅ Done",
            JobStatus.FAILED:    "❌ Failed",
            JobStatus.CANCELLED: "🚫 Cancelled",
        }
        text = (
            f"📊 <b>Job Status</b>\n\n"
            f"Topic: <b>{job.topic}</b>\n"
            f"Status: {status_map.get(job.status, str(job.status))}\n"
            f"Progress: {job.progress}%\n"
        )
        if job.message:
            text += f"Stage: {job.message}\n"
        if job.error:
            text += f"Error: <code>{job.error[:100]}</code>\n"
        self._send("sendMessage", chat_id=chat_id,
                   text=text, parse_mode="HTML")

    def _cmd_cancel(self, chat_id: str) -> None:
        ok = self._queue.cancel(chat_id)
        msg = "🚫 Job cancelled." if ok else "❌ No active job to cancel."
        self._send("sendMessage", chat_id=chat_id, text=msg)

    def _cmd_settings(self, chat_id: str) -> None:
        prefs = self._get_prefs(chat_id)
        self._send("sendMessage", chat_id=chat_id,
                   text=self._status_text(chat_id, prefs),
                   parse_mode="HTML",
                   reply_markup=self._kb_main())

    # ── Wizard helpers ─────────────────────────────────────────────────────────

    def _wizard_topic_received(self, chat_id: str, msg_id: str,
                                topic: str, prefs: dict) -> None:
        if len(topic) < 2:
            self._send("sendMessage", chat_id=chat_id,
                       text="⚠️ Please type a longer topic.")
            return

        if prefs.get("wizard_type"):
            self._set_pref(chat_id, "video_type", prefs["wizard_type"])
        if prefs.get("wizard_lang"):
            self._set_pref(chat_id, "language", prefs["wizard_lang"])
        self._set_pref(chat_id, "wizard_step", None)
        prefs = self._get_prefs(chat_id)

        text = (
            f"🎬 <b>Confirm Video</b>\n\n"
            f"📌 Topic: <b>{topic}</b>\n"
            f"📂 Type: <b>{TYPE_OPTIONS.get(prefs['video_type'], '')}</b>\n"
            f"🌍 Language: <b>{LANGUAGE_OPTIONS.get(prefs['language'], '')}</b>\n"
            f"🎤 Voice: <b>{VOICE_OPTIONS.get(prefs['voice_gender'], '')}</b>\n\n"
            f"Proceed?"
        )
        self._send("sendMessage", chat_id=chat_id, text=text, parse_mode="HTML",
                   reply_markup={"inline_keyboard": [
                       [{"text": "✅ Create video",
                         "callback_data": f"confirm_video_{topic[:50]}"}],
                       [{"text": "✏️ Change topic",  "callback_data": "wizard_start"},
                        {"text": "❌ Cancel",         "callback_data": "wizard_cancel"}],
                   ]})

    # ── Job launch ─────────────────────────────────────────────────────────────

    def _launch_job(self, chat_id: str, topic: str, prefs: dict) -> None:
        # Check if already running
        existing = self._queue.get_status(chat_id)
        if existing and existing.status in (JobStatus.QUEUED, JobStatus.RUNNING):
            pos = self._queue.queue_position(chat_id)
            self._send("sendMessage", chat_id=chat_id,
                       text=f"⏳ You already have a job in progress (position {pos}).\n"
                            f"Use /cancel to cancel it first.",
                       parse_mode="HTML")
            return

        job = self._queue.submit(chat_id, topic, prefs)
        pos = self._queue.queue_length()
        self._send("sendMessage", chat_id=chat_id,
                   text=(
                       f"✅ <b>Job queued!</b>\n\n"
                       f"📌 Topic: <b>{topic}</b>\n"
                       f"📂 Type: <b>{TYPE_OPTIONS.get(prefs.get('video_type','news'), '')}</b>\n"
                       f"🌍 Language: <b>{LANGUAGE_OPTIONS.get(prefs.get('language','ar'), '')}</b>\n\n"
                       f"⏳ Expected: 5–8 minutes"
                   ),
                   parse_mode="HTML",
                   reply_markup={"inline_keyboard": [[
                       {"text": "📊 Check status", "callback_data": "show_status"}
                   ]]})

    # ── Queue callbacks ────────────────────────────────────────────────────────

    def _setup_queue(self) -> None:
        self._queue.set_callbacks(
            on_progress = self._on_job_progress,
            on_complete = self._on_job_complete,
            on_error    = self._on_job_error,
        )

    def _on_job_progress(self, job: Job) -> None:
        STAGES = {
            5:  "📝 Generating script...",
            20: "🎙 Generating narration...",
            40: "🖼 Fetching media...",
            60: "🎬 Rendering video...",
            85: "📤 Finalising...",
        }
        if job.progress in STAGES:
            self._send("sendMessage", chat_id=job.chat_id,
                       text=f"{STAGES[job.progress]}\n"
                            f"Progress: {job.progress}%",
                       parse_mode="HTML")

    def _on_job_complete(self, job: Job) -> None:
        self._send("sendMessage", chat_id=job.chat_id,
                   text=f"✅ <b>Video ready!</b>\n\n"
                        f"📌 {job.topic}\n"
                        f"⏱ Produced in {int(job.elapsed())}s",
                   parse_mode="HTML")

    def _on_job_error(self, job: Job) -> None:
        self._send("sendMessage", chat_id=job.chat_id,
                   text=f"❌ <b>Video generation failed</b>\n\n"
                        f"Error: <code>{(job.error or 'unknown')[:200]}</code>\n\n"
                        f"Please try again with /generate",
                   parse_mode="HTML")

    # ── Keyboard builders ──────────────────────────────────────────────────────

    def _kb_main(self) -> dict:
        return {"inline_keyboard": [
            [{"text": "🎬 New video", "callback_data": "wizard_start"}],
            [{"text": "🌍 Language",  "callback_data": "menu_lang"},
             {"text": "📂 Type",      "callback_data": "menu_type"}],
            [{"text": "🎤 Voice",     "callback_data": "menu_voice"},
             {"text": "✨ Quality",   "callback_data": "menu_quality"}],
            [{"text": "📊 Status",    "callback_data": "show_status"}],
        ]}

    def _kb_language(self, current: str) -> dict:
        rows = [[{"text": ("✅ " if k == current else "") + v,
                  "callback_data": f"set_lang_{k}"}]
                for k, v in LANGUAGE_OPTIONS.items()]
        rows.append([{"text": "◀️ Back", "callback_data": "back_main"}])
        return {"inline_keyboard": rows}

    def _kb_type(self, current: str) -> dict:
        rows = [[{"text": ("✅ " if k == current else "") + v,
                  "callback_data": f"set_type_{k}"}]
                for k, v in TYPE_OPTIONS.items()]
        rows.append([{"text": "◀️ Back", "callback_data": "back_main"}])
        return {"inline_keyboard": rows}

    def _kb_voice(self, current: str) -> dict:
        rows = [[{"text": ("✅ " if k == current else "") + v,
                  "callback_data": f"set_voice_{k}"}]
                for k, v in VOICE_OPTIONS.items()]
        rows.append([{"text": "◀️ Back", "callback_data": "back_main"}])
        return {"inline_keyboard": rows}

    def _kb_quality(self, current: str) -> dict:
        rows = [[{"text": ("✅ " if k == current else "") + v,
                  "callback_data": f"set_quality_{k}"}]
                for k, v in QUALITY_OPTIONS.items()]
        rows.append([{"text": "◀️ Back", "callback_data": "back_main"}])
        return {"inline_keyboard": rows}

    def _kb_wizard_lang(self) -> dict:
        rows = [[{"text": v, "callback_data": f"wizard_lang_{k}"}]
                for k, v in LANGUAGE_OPTIONS.items()]
        rows.append([{"text": "❌ Cancel", "callback_data": "wizard_cancel"}])
        return {"inline_keyboard": rows}

    def _kb_wizard_type(self) -> dict:
        rows = [[{"text": v, "callback_data": f"wizard_type_{k}"}]
                for k, v in TYPE_OPTIONS.items()]
        rows.append([{"text": "❌ Cancel", "callback_data": "wizard_cancel"}])
        return {"inline_keyboard": rows}

    # ── Status text ────────────────────────────────────────────────────────────

    def _status_text(self, chat_id: str, prefs: dict) -> str:
        return (
            f"⚙️ <b>Your Settings</b>\n\n"
            f"🌍 Language: <b>{LANGUAGE_OPTIONS.get(prefs['language'], prefs['language'])}</b>\n"
            f"📂 Type:     <b>{TYPE_OPTIONS.get(prefs['video_type'], prefs['video_type'])}</b>\n"
            f"🎤 Voice:    <b>{VOICE_OPTIONS.get(prefs['voice_gender'], '')}</b>\n"
            f"✨ Quality:  <b>{QUALITY_OPTIONS.get(prefs['quality'], '')}</b>"
        )

    # ── Telegram API helpers ───────────────────────────────────────────────────

    def _send(self, method: str, **kwargs) -> dict:
        try:
            r = requests.post(
                f"{self.base_url}/{method}",
                json=kwargs, timeout=20,
            )
            return r.json()
        except Exception as e:
            log.debug("API %s error: %s", method, e)
            return {}

    def _edit(self, chat_id: str, msg_id: int,
              text: str, reply_markup: Optional[dict] = None) -> None:
        payload: dict = {
            "chat_id":    chat_id,
            "message_id": msg_id,
            "text":       text,
            "parse_mode": "HTML",
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup
        self._send("editMessageText", **payload)

    def _answer_callback(self, cb_id: str) -> None:
        self._send("answerCallbackQuery", callback_query_id=cb_id)

    def _get_updates(self, offset: int) -> list:
        r = requests.get(
            f"{self.base_url}/getUpdates",
            params={
                "offset":          offset,
                "timeout":         20,
                "limit":           10,
                "allowed_updates": ["message", "callback_query"],
            },
            timeout=25,
        )
        if r.status_code == 200:
            return r.json().get("result", [])
        return []

    # ── Preference persistence ─────────────────────────────────────────────────

    _PREFS_FILE  = "/tmp/bot_prefs.json"
    _OFFSET_FILE = "/tmp/bot_offset.txt"

    def _load_prefs(self) -> dict:
        try:
            return json.loads(Path(self._PREFS_FILE).read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_prefs(self) -> None:
        try:
            Path(self._PREFS_FILE).write_text(
                json.dumps(self._prefs, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

    def _get_prefs(self, chat_id: str) -> dict:
        return {**DEFAULT_PREFS, **self._prefs.get(str(chat_id), {})}

    def _set_pref(self, chat_id: str, key: str, value) -> None:
        uid = str(chat_id)
        if uid not in self._prefs:
            self._prefs[uid] = {}
        self._prefs[uid][key] = value
        self._save_prefs()

    def _load_offset(self) -> int:
        try:
            return int(Path(self._OFFSET_FILE).read_text().strip())
        except Exception:
            return 0

    def _save_offset(self, offset: int) -> None:
        try:
            Path(self._OFFSET_FILE).write_text(str(offset))
        except Exception:
            pass
