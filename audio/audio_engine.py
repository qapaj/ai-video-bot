"""
audio/audio_engine.py

Main audio engine — orchestrates the complete TTS pipeline.

Fallback chain (priority order):
  1. HF XTTS-v2 API           (highest quality, multilingual)
  2. HF MMS-TTS API           (language-specific, reliable)
  3. HF SpeechT5 API          (lightweight, fast)
  4. edge-tts CLI             (Microsoft Neural, no API key)
  5. edge-tts Python API      (same, different invocation path)
  6. espeak-ng                (always on Ubuntu runners)
  7. Silent WAV               (guaranteed non-crash final fallback)

Security: HF_TOKEN is read from environment only — never hardcoded.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from utils.config import get_config
from utils.helpers import ffprobe_duration, file_ok, run_cmd
from utils.logger import StageLogger, get_logger

log = get_logger("audio.engine")

MIN_AUDIO_BYTES = 200

EDGE_TTS_VOICES = {
    "ar": {
        "female": ["ar-EG-SalmaNeural", "ar-SA-ZariyahNeural", "ar-AE-FatimaNeural"],
        "male":   ["ar-SA-HamedNeural",  "ar-EG-ShakirNeural",  "ar-AE-HamdanNeural"],
    },
    "en": {
        "female": ["en-US-JennyNeural", "en-GB-SoniaNeural"],
        "male":   ["en-US-GuyNeural",   "en-GB-RyanNeural"],
    },
    "fr": {"female": ["fr-FR-DeniseNeural"],  "male": ["fr-FR-HenriNeural"]},
    "es": {"female": ["es-ES-ElviraNeural"],  "male": ["es-ES-AlvaroNeural"]},
    "de": {"female": ["de-DE-KatjaNeural"],   "male": ["de-DE-ConradNeural"]},
}

ESPEAK_VOICES = {
    "ar": ["ar", "ar+f3", "ar+m3", "ar+f1"],
    "en": ["en-us", "en", "en+f3"],
    "fr": ["fr",  "fr+f3"],
    "es": ["es",  "es+f3"],
    "de": ["de",  "de+f3"],
}


class AudioEngine:
    """Produces narration audio for script segments with a 7-level fallback chain."""

    def __init__(self) -> None:
        self._cfg       = get_config()
        self._hf        = self._init_hf()
        self._processor = self._init_processor()
        self._cache     = self._init_cache()
        self._edge_cli  = self._find_edge_cli()
        self._espeak_ok = self._check_espeak()

        log.info("AudioEngine | HF=%s | edge-tts=%s | espeak=%s",
                 "YES" if self._hf else "NO (set HF_TOKEN secret)",
                 "YES" if self._edge_cli else "NO",
                 "YES" if self._espeak_ok else "NO")

    # ── Public ─────────────────────────────────────────────────────────────────

    def generate_all(self, segments: list[dict], output_dir: str,
                     language: str = "ar", gender: str = "female") -> list[dict]:
        """Synthesise audio for all segments. Returns segments with audio_path added."""
        os.makedirs(output_dir, exist_ok=True)
        results: list[dict] = []

        with StageLogger(log, f"TTS ({len(segments)} segments, lang={language})"):
            for i, seg in enumerate(segments):
                text = (seg.get("text") or "").strip()
                if not text:
                    continue

                out = os.path.join(output_dir, f"seg_{i:03d}.aac")
                log.info("Seg %d/%d: %.55s...", i + 1, len(segments), text)

                if self._synthesise(text, out, language, gender):
                    dur = ffprobe_duration(out)
                    if dur < 0.1:
                        dur = max(2.0, len(text) / 18.0)
                    results.append({**seg, "audio_path": out, "actual_duration": dur})
                    log.info("  → %.2fs", dur)
                else:
                    log.error("  ALL backends failed — segment %d skipped", i)

        log.info("TTS complete: %d/%d segments", len(results), len(segments))
        return results

    def concat(self, audio_paths: list[str], output_path: str) -> bool:
        """Merge segment audio files into a single track."""
        return self._processor.concat(audio_paths, output_path)

    def mix_with_music(self, video_path: str, voice_path: str,
                       output_path: str, music_path: Optional[str] = None) -> bool:
        """Attach voice + optional background music to a silent video."""
        music = music_path or self._find_music()
        if music and os.path.exists(music):
            ok = self._mix_voice_music(video_path, voice_path, music, output_path)
            if ok:
                return True
            log.warning("Music mix failed, trying voice-only")
        return self._mix_voice_only(video_path, voice_path, output_path)

    # ── Synthesis dispatcher ───────────────────────────────────────────────────

    def _synthesise(self, text: str, out: str, lang: str, gender: str) -> bool:
        # 1. Cache lookup
        if self._cache:
            ck = self._cache.key(text, lang, "multi", gender)
            if self._cache.serve(ck, out):
                log.debug("  Cache HIT")
                return True

        # 2. Try backends in priority order
        backends = [
            ("HF TTS API",       lambda: self._b_hf(text, out, lang)),
            ("edge-tts CLI",     lambda: self._b_edge_cli(text, out, lang, gender)),
            ("edge-tts Python",  lambda: self._b_edge_python(text, out, lang, gender)),
            ("espeak-ng",        lambda: self._b_espeak(text, out, lang)),
            ("silent fallback",  lambda: self._b_silence(out)),
        ]

        for name, fn in backends:
            log.info("  [%s]", name)
            try:
                if fn():
                    # Post-process (non-fatal)
                    pp = out + "_pp.aac"
                    if self._processor.process(out, pp):
                        shutil.move(pp, out)
                    # Cache
                    if self._cache:
                        try:
                            ck = self._cache.key(text, lang, "multi", gender)
                            self._cache.put(ck, out, ".aac")
                        except Exception:
                            pass
                    return True
                log.warning("  ✗ %s → False", name)
            except Exception as e:
                log.warning("  ✗ %s → %s: %s", name, type(e).__name__, e)

        return False  # never reached (silence always succeeds)

    # ── Backend 1: HuggingFace API ─────────────────────────────────────────────

    def _b_hf(self, text: str, out: str, lang: str) -> bool:
        if not self._hf:
            return False
        try:
            return self._hf.synthesize(text, out, lang)
        except Exception as e:
            log.warning("  HF API error: %s", e)
            return False

    # ── Backend 2: edge-tts CLI ────────────────────────────────────────────────

    def _b_edge_cli(self, text: str, out: str, lang: str, gender: str) -> bool:
        if not self._edge_cli:
            return False
        voices = EDGE_TTS_VOICES.get(lang, EDGE_TTS_VOICES["ar"])
        vlist  = voices.get(gender, voices.get("female", []))
        mp3    = out.replace(".aac", "_edge.mp3")

        for voice in vlist:
            try:
                r = subprocess.run(
                    [self._edge_cli, "--voice", voice, "--rate", "-5%",
                     "--text", text, "--write-media", mp3],
                    capture_output=True, text=True, timeout=45,
                )
                if r.returncode == 0 and file_ok(mp3, MIN_AUDIO_BYTES):
                    ok, _ = run_cmd(["ffmpeg", "-y", "-i", mp3,
                                     "-ar", "44100", "-ac", "1",
                                     "-c:a", "aac", "-b:a", "128k", out],
                                    timeout=30)
                    _rm(mp3)
                    if ok and file_ok(out, MIN_AUDIO_BYTES):
                        return True
                log.debug("  edge CLI %s rc=%d", voice, r.returncode)
            except subprocess.TimeoutExpired:
                log.debug("  edge CLI timeout: %s", voice)
            except Exception as e:
                log.debug("  edge CLI error: %s", e)
        _rm(mp3)
        return False

    # ── Backend 3: edge-tts Python API ────────────────────────────────────────

    def _b_edge_python(self, text: str, out: str, lang: str, gender: str) -> bool:
        try:
            import edge_tts
        except ImportError:
            return False

        voices = EDGE_TTS_VOICES.get(lang, EDGE_TTS_VOICES["ar"])
        vlist  = voices.get(gender, voices.get("female", []))
        mp3    = out.replace(".aac", "_edgepy.mp3")

        for voice in vlist:
            loop = None
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                async def _go(v: str) -> None:
                    await edge_tts.Communicate(text, v, rate="-5%").save(mp3)

                loop.run_until_complete(_go(voice))
                if file_ok(mp3, MIN_AUDIO_BYTES):
                    ok, _ = run_cmd(["ffmpeg", "-y", "-i", mp3,
                                     "-ar", "44100", "-ac", "1",
                                     "-c:a", "aac", "-b:a", "128k", out],
                                    timeout=30)
                    _rm(mp3)
                    if ok and file_ok(out, MIN_AUDIO_BYTES):
                        return True
            except Exception as e:
                log.debug("  edge Python %s: %s", voice, e)
            finally:
                if loop:
                    try: loop.close()
                    except: pass
        _rm(mp3)
        return False

    # ── Backend 4: espeak-ng ───────────────────────────────────────────────────

    def _b_espeak(self, text: str, out: str, lang: str) -> bool:
        if not self._espeak_ok:
            return False
        wav = out + "_esp.wav"
        for vid in ESPEAK_VOICES.get(lang, ["ar"]):
            try:
                r = subprocess.run(
                    ["espeak-ng", "-v", vid, "-s", "145",
                     "-p", "50", "-a", "180", "-g", "5", "-w", wav, text],
                    capture_output=True, text=True, timeout=60,
                )
                if r.returncode == 0 and file_ok(wav, MIN_AUDIO_BYTES):
                    ok, _ = run_cmd(["ffmpeg", "-y", "-i", wav,
                                     "-ar", "44100", "-ac", "1",
                                     "-c:a", "aac", "-b:a", "128k", out],
                                    timeout=30)
                    _rm(wav)
                    if ok and file_ok(out, MIN_AUDIO_BYTES):
                        return True
                log.debug("  espeak %s rc=%d", vid, r.returncode)
            except Exception as e:
                log.debug("  espeak %s: %s", vid, e)
        _rm(wav)
        return False

    # ── Backend 5: silent fallback ─────────────────────────────────────────────

    @staticmethod
    def _b_silence(out: str, duration: float = 3.0) -> bool:
        """Generate silent audio — always works, never crashes the pipeline."""
        log.warning("  Using SILENT audio — video will render without voice")
        ok, _ = run_cmd([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"anullsrc=r=44100:cl=mono:d={duration}",
            "-c:a", "aac", "-b:a", "64k", out,
        ], timeout=15, label="silence")
        return ok and file_ok(out, 50)

    # ── Audio mixing ───────────────────────────────────────────────────────────

    def _mix_voice_music(self, video: str, voice: str, music: str, out: str) -> bool:
        mv = self._cfg.audio.music_volume
        ok, err = run_cmd([
            "ffmpeg", "-y", "-i", video, "-i", voice,
            "-stream_loop", "-1", "-i", music,
            "-filter_complex", (
                f"[1:a]volume=1.0[v];[2:a]volume={mv}[m];"
                "[v][m]amix=inputs=2:duration=first:dropout_transition=2[aout]"
            ),
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", out,
        ], timeout=600, label="mix+music")
        return ok and file_ok(out, 1000)

    def _mix_voice_only(self, video: str, voice: str, out: str) -> bool:
        ok, err = run_cmd([
            "ffmpeg", "-y", "-i", video, "-i", voice,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", out,
        ], timeout=600, label="mix voice-only")
        if not ok:
            log.error("Voice-only mix failed: %s", err[:200])
        return ok and file_ok(out, 1000)

    # ── Initialisers ───────────────────────────────────────────────────────────

    def _init_hf(self):
        token = self._cfg.hf.token
        if not token:
            return None
        try:
            from audio.hf_tts_client import HFTTSClient
            return HFTTSClient(token=token)
        except Exception as e:
            log.warning("HFTTSClient init error: %s", e)
            return None

    @staticmethod
    def _init_processor():
        from audio.audio_processor import AudioProcessor
        return AudioProcessor()

    def _init_cache(self):
        try:
            from audio.audio_cache import AudioCache
            d = str(self._cfg.paths.cache_dir / "audio")
            c = AudioCache(d)
            c.prune()
            return c
        except Exception as e:
            log.warning("AudioCache init error (disabled): %s", e)
            return None

    @staticmethod
    def _find_edge_cli() -> Optional[str]:
        r = subprocess.run(["which", "edge-tts"],
                           capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
        r2 = subprocess.run(["edge-tts", "--list-voices"],
                            capture_output=True, text=True, timeout=10)
        return "edge-tts" if r2.returncode == 0 else None

    @staticmethod
    def _check_espeak() -> bool:
        r = subprocess.run(["espeak-ng", "--version"], capture_output=True)
        return r.returncode == 0

    def _find_music(self) -> Optional[str]:
        d = self._cfg.paths.music_dir
        if d.exists():
            for ext in ("*.mp3", "*.wav", "*.ogg"):
                f = list(d.glob(ext))
                if f:
                    return str(f[0])
        return None


def _rm(path: str) -> None:
    try: os.remove(path)
    except: pass
