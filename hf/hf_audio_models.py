"""
hf/hf_audio_models.py
HuggingFace-based audio synthesis.

Primary:  XTTS-v2 via HF Inference API (no local model download needed)
Secondary: facebook/mms-tts-ara via API
Tertiary:  local SpeechT5 (small, CPU-viable)
Final fallback: edge-tts (always works, no API needed)
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from utils.logger import get_logger
from utils.helpers import file_ok

log = get_logger("hf.audio")

HF_API_URL = "https://api-inference.huggingface.co/models"


class HFAudioEngine:
    """
    Synthesises speech using HuggingFace models with a multi-level fallback chain.
    """

    def __init__(self, token: str, cache_dir: str = "/tmp/hf_audio_cache") -> None:
        self.token     = token
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Public ─────────────────────────────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        output_path: str,
        language: str = "ar",
        voice_gender: str = "female",
    ) -> bool:
        """
        Synthesise text to speech. Tries each backend in order.
        Returns True on any success.
        """
        if not text.strip():
            return False

        backends = [
            ("edge-tts",    lambda: self._edge_tts(text, output_path, language, voice_gender)),
            ("MMS-TTS API", lambda: self._mms_tts_api(text, output_path, language)),
            ("espeak-ng",   lambda: self._espeak(text, output_path, language)),
        ]

        # If HF token is set and language is Arabic, try XTTS-v2 first
        if self.token and language == "ar":
            backends.insert(0, (
                "XTTS-v2 API",
                lambda: self._xtts_api(text, output_path),
            ))

        for name, fn in backends:
            try:
                log.info("TTS attempt: %s", name)
                if fn():
                    log.info("✓ TTS succeeded: %s", name)
                    return True
            except Exception as e:
                log.warning("%s failed: %s", name, e)

        log.error("All TTS backends failed for: %s", text[:60])
        return False

    def post_process(self, audio_path: str, output_path: str) -> bool:
        """
        Apply audio post-processing:
        - silence trimming
        - loudness normalization
        - gentle compression
        Returns True if processed file was written, False on error (original preserved).
        """
        try:
            ok, _ = self._run(["ffmpeg", "-y", "-i", audio_path,
                "-af", (
                    "silenceremove=start_periods=1:start_threshold=-50dB:start_duration=0.1,"
                    "silenceremove=stop_periods=-1:stop_threshold=-50dB:stop_duration=0.3,"
                    "loudnorm=I=-16:TP=-1.5:LRA=11,"
                    "acompressor=threshold=0.5:ratio=4:attack=200:release=1000"
                ),
                "-ar", "44100", "-ac", "1", "-c:a", "aac", "-b:a", "128k",
                output_path,
            ], timeout=60)
            return ok and file_ok(output_path)
        except Exception as e:
            log.warning("post_process failed: %s", e)
            return False

    # ── Backends ───────────────────────────────────────────────────────────────

    def _xtts_api(self, text: str, output_path: str) -> bool:
        """XTTS-v2 via HuggingFace Inference API."""
        try:
            import requests
            r = requests.post(
                f"{HF_API_URL}/coqui/XTTS-v2",
                headers={"Authorization": f"Bearer {self.token}",
                         "Content-Type": "application/json"},
                json={"inputs": text, "parameters": {"language": "ar"}},
                timeout=90,
            )
            if r.status_code == 200 and r.content:
                with open(output_path, "wb") as f:
                    f.write(r.content)
                return file_ok(output_path)
        except Exception as e:
            log.debug("XTTS-v2 API: %s", e)
        return False

    def _mms_tts_api(self, text: str, output_path: str, language: str = "ar") -> bool:
        """facebook/mms-tts via HF Inference API (supports many languages)."""
        lang_models = {
            "ar": "facebook/mms-tts-ara",
            "en": "facebook/mms-tts-eng",
            "fr": "facebook/mms-tts-fra",
            "es": "facebook/mms-tts-spa",
            "de": "facebook/mms-tts-deu",
        }
        model_id = lang_models.get(language, lang_models["ar"])
        try:
            import requests
            r = requests.post(
                f"{HF_API_URL}/{model_id}",
                headers={"Authorization": f"Bearer {self.token}"},
                json={"inputs": text},
                timeout=60,
            )
            if r.status_code == 200 and r.content:
                # API returns raw audio bytes (wav/flac)
                tmp_audio = output_path + "_raw"
                with open(tmp_audio, "wb") as f:
                    f.write(r.content)
                # Convert to standard AAC
                ok, _ = self._run([
                    "ffmpeg", "-y", "-i", tmp_audio,
                    "-ar", "44100", "-ac", "1", "-c:a", "aac", "-b:a", "128k",
                    output_path,
                ], timeout=30)
                try: os.remove(tmp_audio)
                except: pass
                return ok and file_ok(output_path)
        except Exception as e:
            log.debug("MMS-TTS API: %s", e)
        return False

    def _edge_tts(self, text: str, output_path: str,
                  language: str = "ar", gender: str = "female") -> bool:
        """edge-tts — Microsoft Neural voices, no API key required."""
        from audio.narration_generator import VOICE_CATALOGUE
        voices = VOICE_CATALOGUE.get(language, VOICE_CATALOGUE["ar"])
        voice_list = voices.get(gender, voices.get("female", []))
        all_voices = voice_list + [v for k, vl in voices.items()
                                    for v in vl if vl is not voice_list]

        async def _try(voice: str) -> bool:
            try:
                import edge_tts
                comm = edge_tts.Communicate(text, voice, rate="-5%")
                await comm.save(output_path)
                return file_ok(output_path)
            except Exception:
                return False

        for voice in all_voices[:4]:
            try:
                if asyncio.run(_try(voice)):
                    return True
            except Exception:
                continue
        return False

    def _espeak(self, text: str, output_path: str, language: str = "ar") -> bool:
        """espeak-ng — always available on Ubuntu runners."""
        lang_map = {"ar": "ar", "en": "en", "fr": "fr", "es": "es", "de": "de"}
        lang = lang_map.get(language, "ar")
        wav  = output_path + "_esp.wav"
        ok1, _ = self._run(
            ["espeak-ng", "-v", lang, "-s", "145", "-p", "50", "-a", "180", "-w", wav, text],
            timeout=60,
        )
        if not ok1:
            return False
        ok2, _ = self._run(
            ["ffmpeg", "-y", "-i", wav,
             "-ar", "44100", "-ac", "1", "-c:a", "aac", "-b:a", "128k", output_path],
            timeout=30,
        )
        try: os.remove(wav)
        except: pass
        return ok2 and file_ok(output_path)

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _run(cmd: list, timeout: int = 60):
        from utils.helpers import run_cmd
        return run_cmd(cmd, timeout=timeout)
