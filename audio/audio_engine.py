"""
audio/audio_engine.py

Fixes all FileNotFoundError crashes from missing system binaries.

CRASH SITES FIXED:
  1. _check_espeak(): called subprocess.run(["espeak-ng"]) without try/except
     → now uses shutil.which() first, subprocess wrapped in try/except
  2. _b_espeak(): bare subprocess.run inside inner loop
     → already had try/except, hardened further
  3. All binary calls now use shutil.which() guard before subprocess.run
  4. _ensure_espeak() added: auto-installs via apt-get if missing

FALLBACK CHAIN (all crash-proof):
  1. edge-tts async Python API  → no binary, no PATH dependency
  2. edge-tts python -m module  → PATH-independent (uses sys.executable)
  3. HuggingFace Inference API  → remote, no binary
  4. espeak-ng                  → only called if shutil.which() confirms it exists
  5. FFmpeg anullsrc silence    → ffmpeg always present, generates real audio file

The pipeline NEVER raises FileNotFoundError. Worst case: silent audio.
"""

from __future__ import annotations

import asyncio
import importlib
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

EDGE_VOICES: dict[str, dict[str, list[str]]] = {
    "ar": {
        "female": ["ar-EG-SalmaNeural", "ar-SA-ZariyahNeural", "ar-AE-FatimaNeural"],
        "male":   ["ar-SA-HamedNeural",  "ar-EG-ShakirNeural",  "ar-AE-HamdanNeural"],
    },
    "en": {
        "female": ["en-US-JennyNeural",  "en-GB-SoniaNeural"],
        "male":   ["en-US-GuyNeural",    "en-GB-RyanNeural"],
    },
    "fr": {"female": ["fr-FR-DeniseNeural"], "male": ["fr-FR-HenriNeural"]},
    "es": {"female": ["es-ES-ElviraNeural"], "male": ["es-ES-AlvaroNeural"]},
    "de": {"female": ["de-DE-KatjaNeural"],  "male": ["de-DE-ConradNeural"]},
}

ESPEAK_VOICES: dict[str, list[str]] = {
    "ar": ["ar", "ar+f3", "ar+m3"],
    "en": ["en-us", "en", "en+f3"],
    "fr": ["fr", "fr+f3"],
    "es": ["es", "es+f3"],
    "de": ["de", "de+f3"],
}


# ── Safe binary locator ────────────────────────────────────────────────────────

def _find_binary(name: str) -> Optional[str]:
    """
    Return the full path to a binary, or None if not found.
    Uses shutil.which() which is safe — never raises FileNotFoundError.
    """
    return shutil.which(name)


# ── edge-tts self-installer ────────────────────────────────────────────────────

def _ensure_edge_tts() -> bool:
    """
    Guarantee edge_tts Python package is importable.
    If missing: installs via pip using sys.executable (correct environment).
    Never raises.
    """
    try:
        importlib.import_module("edge_tts")
        return True
    except ImportError:
        pass

    log.info("edge_tts missing — auto-installing...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "edge-tts==6.1.9",
             "-q", "--disable-pip-version-check"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            importlib.invalidate_caches()
            try:
                importlib.import_module("edge_tts")
                log.info("✓ edge-tts installed")
                return True
            except ImportError as e:
                log.error("edge-tts installed but import still failed: %s", e)
        else:
            log.error("pip install edge-tts failed: %s", result.stderr[:200])
    except Exception as e:
        log.error("edge-tts auto-install error: %s", e)

    return False


# ── espeak-ng checker (CRASH FIX: was bare subprocess.run) ────────────────────

def _check_espeak() -> Optional[str]:
    """
    Return the path to espeak-ng binary, or None if not available.
    Uses shutil.which() — NEVER raises FileNotFoundError.
    """
    path = _find_binary("espeak-ng")
    if path:
        log.debug("espeak-ng found: %s", path)
        return path

    # Try to auto-install via apt-get (works on GitHub Actions Ubuntu runners)
    log.info("espeak-ng not found — attempting apt-get install...")
    try:
        r = subprocess.run(
            ["sudo", "apt-get", "install", "-y", "-qq",
             "espeak-ng", "espeak-ng-data"],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode == 0:
            path = _find_binary("espeak-ng")
            if path:
                log.info("✓ espeak-ng installed: %s", path)
                return path
            log.warning("apt-get succeeded but espeak-ng still not on PATH")
        else:
            log.warning("apt-get install espeak-ng failed: %s", r.stderr[:150])
    except FileNotFoundError:
        log.warning("sudo/apt-get not available (not on Ubuntu runner?)")
    except Exception as e:
        log.warning("espeak-ng auto-install error: %s", e)

    log.warning("espeak-ng unavailable — this backend will be skipped")
    return None


class AudioEngine:
    """
    Produces narration audio for script segments.
    Every backend is crash-proof: FileNotFoundError is NEVER propagated.
    """

    def __init__(self) -> None:
        self._cfg        = get_config()
        self._edge_ok    = _ensure_edge_tts()
        self._espeak_bin = _check_espeak()         # full path or None
        self._hf         = self._init_hf()
        self._processor  = self._init_processor()
        self._cache      = self._init_cache()

        log.info(
            "AudioEngine | edge-tts=%s | espeak=%s | HF=%s",
            "✓" if self._edge_ok        else "✗",
            f"✓ ({self._espeak_bin})" if self._espeak_bin else "✗",
            "✓" if self._hf             else "✗ (set HF_TOKEN secret)",
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate_all(
        self,
        segments:   list[dict],
        output_dir: str,
        language:   str = "ar",
        gender:     str = "female",
    ) -> list[dict]:
        """
        Synthesise audio for all segments.
        Every segment gets a result — worst case is a silent clip.
        Never raises, never crashes.
        """
        os.makedirs(output_dir, exist_ok=True)
        results: list[dict] = []

        with StageLogger(log, f"TTS ({len(segments)} segs, {language}/{gender})"):
            for i, seg in enumerate(segments):
                text = (seg.get("text") or "").strip()
                if not text:
                    continue

                out = os.path.join(output_dir, f"seg_{i:03d}.aac")
                log.info("Seg %d/%d: %.55s...", i + 1, len(segments), text)

                ok = self._synthesise(text, out, language, gender)

                dur = ffprobe_duration(out) if ok and file_ok(out, 50) else 0.0
                if dur < 0.1:
                    dur = max(2.5, len(text) / 18.0)

                results.append({
                    **seg,
                    "audio_path":      out,
                    "actual_duration": dur,
                })
                log.info("  %s %.2fs", "✓" if ok else "⚠ silent", dur)

        log.info("TTS done: %d/%d segments", len(results), len(segments))
        return results

    def concat(self, audio_paths: list[str], output_path: str) -> bool:
        return self._processor.concat(audio_paths, output_path)

    def mix_with_music(
        self,
        video_path:  str,
        voice_path:  str,
        output_path: str,
        music_path:  Optional[str] = None,
    ) -> bool:
        music = music_path or self._find_music()
        if music and os.path.exists(music):
            if self._mix_with_music(video_path, voice_path, music, output_path):
                return True
            log.warning("Music mix failed — voice-only fallback")
        return self._mix_voice_only(video_path, voice_path, output_path)

    # ── Synthesis dispatcher ───────────────────────────────────────────────────

    def _synthesise(self, text: str, out: str, lang: str, gender: str) -> bool:
        # Cache lookup
        if self._cache:
            ck = self._cache.key(text, lang, "v8", gender)
            if self._cache.serve(ck, out):
                log.debug("  Cache HIT")
                return True

        backends: list[tuple[str, object]] = [
            ("edge-tts async",   lambda: self._b_edge_async(text, out, lang, gender)),
            ("edge-tts -m",      lambda: self._b_edge_module(text, out, lang, gender)),
            ("HuggingFace API",  lambda: self._b_hf(text, out, lang)),
            ("espeak-ng",        lambda: self._b_espeak(text, out, lang)),
            ("silent fallback",  lambda: self._b_silence(out)),
        ]

        for name, fn in backends:
            log.info("  [%s]", name)
            try:
                if fn():
                    log.info("  ✓ %s", name)
                    self._post_process(out)
                    if self._cache:
                        try:
                            ck = self._cache.key(text, lang, "v8", gender)
                            self._cache.put(ck, out, ".aac")
                        except Exception:
                            pass
                    return True
                log.warning("  ✗ %s → False", name)

            except FileNotFoundError as e:
                # Binary missing — log clearly, continue to next backend
                log.error(
                    "  ✗ %s → FileNotFoundError: %s  "
                    "[binary not on PATH, skipping this backend]",
                    name, e,
                )
            except Exception as e:
                log.warning("  ✗ %s → %s: %s", name, type(e).__name__, e)

        log.error("ALL backends failed including silence — critical bug")
        return False

    # ── Backend 1: edge-tts async Python API (no binary, no PATH) ─────────────

    def _b_edge_async(self, text: str, out: str, lang: str, gender: str) -> bool:
        if not self._edge_ok:
            return False

        voices = EDGE_VOICES.get(lang, EDGE_VOICES["ar"])
        vlist  = voices.get(gender, voices.get("female", []))
        mp3    = out.replace(".aac", "_eas.mp3")

        for voice in vlist:
            loop: Optional[asyncio.AbstractEventLoop] = None
            try:
                import edge_tts  # type: ignore

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                async def _run(v: str) -> None:
                    comm = edge_tts.Communicate(text, v, rate="-5%", volume="+0%")
                    await comm.save(mp3)

                loop.run_until_complete(_run(voice))

                if file_ok(mp3, MIN_AUDIO_BYTES):
                    ok, _ = run_cmd(
                        ["ffmpeg", "-y", "-i", mp3,
                         "-ar", "44100", "-ac", "1",
                         "-c:a", "aac", "-b:a", "128k", out],
                        timeout=30, label="edge-async → aac",
                    )
                    _rm(mp3)
                    if ok and file_ok(out, MIN_AUDIO_BYTES):
                        return True
                else:
                    log.debug("  edge-async %s: mp3 missing/empty", voice)
                    _rm(mp3)

            except ImportError:
                self._edge_ok = False
                _rm(mp3)
                return False
            except Exception as e:
                log.debug("  edge-async %s: %s: %s", voice, type(e).__name__, e)
                _rm(mp3)
            finally:
                if loop:
                    try: loop.close()
                    except Exception: pass

        return False

    # ── Backend 2: edge-tts via sys.executable -m edge_tts (PATH-independent) ──

    def _b_edge_module(self, text: str, out: str, lang: str, gender: str) -> bool:
        if not self._edge_ok:
            return False

        voices = EDGE_VOICES.get(lang, EDGE_VOICES["ar"])
        vlist  = voices.get(gender, voices.get("female", []))
        mp3    = out.replace(".aac", "_em.mp3")

        for voice in vlist:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "edge_tts",
                     "--voice", voice, "--rate", "-5%",
                     "--text", text, "--write-media", mp3],
                    capture_output=True, text=True, timeout=60,
                )
                if result.returncode == 0 and file_ok(mp3, MIN_AUDIO_BYTES):
                    ok, _ = run_cmd(
                        ["ffmpeg", "-y", "-i", mp3,
                         "-ar", "44100", "-ac", "1",
                         "-c:a", "aac", "-b:a", "128k", out],
                        timeout=30, label="edge-module → aac",
                    )
                    _rm(mp3)
                    if ok and file_ok(out, MIN_AUDIO_BYTES):
                        return True
                else:
                    log.debug("  edge -m %s rc=%d: %s",
                              voice, result.returncode,
                              (result.stderr or "")[:100])
                    _rm(mp3)

            except subprocess.TimeoutExpired:
                log.debug("  edge -m %s: timeout", voice)
                _rm(mp3)
            except Exception as e:
                log.debug("  edge -m %s: %s", voice, e)
                _rm(mp3)

        return False

    # ── Backend 3: HuggingFace Inference API ──────────────────────────────────

    def _b_hf(self, text: str, out: str, lang: str) -> bool:
        if not self._hf:
            return False
        try:
            return self._hf.synthesize(text, out, lang)
        except Exception as e:
            log.warning("  HF API: %s", e)
            return False

    # ── Backend 4: espeak-ng (CRASH FIX: uses stored path, never raw "espeak-ng") ─

    def _b_espeak(self, text: str, out: str, lang: str) -> bool:
        """
        CRASH FIX: Use self._espeak_bin (full path from shutil.which).
        Never call subprocess.run(["espeak-ng", ...]) directly — that raises
        FileNotFoundError if the binary isn't on PATH.
        """
        if not self._espeak_bin:
            log.debug("  espeak-ng path not known — skipping")
            return False

        wav = out + "_esp.wav"
        for voice_id in ESPEAK_VOICES.get(lang, ["ar"]):
            try:
                # Use self._espeak_bin (full verified path), not bare "espeak-ng"
                result = subprocess.run(
                    [self._espeak_bin,
                     "-v", voice_id,
                     "-s", "145", "-p", "50", "-a", "180", "-g", "5",
                     "-w", wav, text],
                    capture_output=True, text=True, timeout=60,
                )
                if result.returncode == 0 and file_ok(wav, MIN_AUDIO_BYTES):
                    ok, _ = run_cmd(
                        ["ffmpeg", "-y", "-i", wav,
                         "-ar", "44100", "-ac", "1",
                         "-c:a", "aac", "-b:a", "128k", out],
                        timeout=30, label="espeak → aac",
                    )
                    _rm(wav)
                    if ok and file_ok(out, MIN_AUDIO_BYTES):
                        return True
                else:
                    log.debug("  espeak %s rc=%d", voice_id, result.returncode)

            except FileNotFoundError:
                # Binary disappeared from disk after we checked — handle gracefully
                log.warning("  espeak-ng binary vanished: %s", self._espeak_bin)
                self._espeak_bin = None
                _rm(wav)
                return False
            except subprocess.TimeoutExpired:
                log.debug("  espeak %s: timeout", voice_id)
            except Exception as e:
                log.debug("  espeak %s: %s: %s", voice_id, type(e).__name__, e)

        _rm(wav)
        return False

    # ── Backend 5: FFmpeg silence (GUARANTEED — never raises) ─────────────────

    @staticmethod
    def _b_silence(out: str, duration: float = 3.0) -> bool:
        """
        Generate a real AAC audio file containing silence.
        Uses ffmpeg anullsrc — requires only ffmpeg (always present on Ubuntu).
        This is the final guarantee: the pipeline completes, video renders.
        """
        log.warning("  Using SILENT audio for this segment")
        ok, err = run_cmd(
            ["ffmpeg", "-y",
             "-f", "lavfi",
             "-i", f"anullsrc=r=44100:cl=mono:d={duration}",
             "-c:a", "aac", "-b:a", "64k",
             out],
            timeout=15, label="silence",
        )
        if ok and file_ok(out, 50):
            return True
        log.error("  Even silence generation failed: %s — ffmpeg may be missing", err)
        return False

    # ── Post-processing ────────────────────────────────────────────────────────

    def _post_process(self, audio_path: str) -> None:
        """Loudness normalise + silence trim in-place. Non-fatal on any error."""
        tmp = audio_path + "_pp.aac"
        try:
            ok, _ = run_cmd(
                ["ffmpeg", "-y", "-i", audio_path,
                 "-af", (
                     "silenceremove="
                     "start_periods=1:start_threshold=-50dB:start_duration=0.1,"
                     "silenceremove="
                     "stop_periods=-1:stop_threshold=-50dB:stop_duration=0.2,"
                     "loudnorm=I=-16:TP=-1.5:LRA=11"
                 ),
                 "-ar", "44100", "-ac", "1",
                 "-c:a", "aac", "-b:a", "128k",
                 tmp],
                timeout=60,
            )
            if ok and file_ok(tmp, MIN_AUDIO_BYTES):
                shutil.move(tmp, audio_path)
        except Exception as e:
            log.debug("post-process error (non-fatal): %s", e)
        finally:
            _rm(tmp)

    # ── Audio mixing ───────────────────────────────────────────────────────────

    def _mix_with_music(self, video: str, voice: str, music: str, out: str) -> bool:
        mv = self._cfg.audio.music_volume
        ok, err = run_cmd(
            ["ffmpeg", "-y",
             "-i", video, "-i", voice,
             "-stream_loop", "-1", "-i", music,
             "-filter_complex", (
                 f"[1:a]volume=1.0[v];"
                 f"[2:a]volume={mv}[m];"
                 "[v][m]amix=inputs=2:duration=first:dropout_transition=2[aout]"
             ),
             "-map", "0:v", "-map", "[aout]",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest",
             out],
            timeout=600, label="mix+music",
        )
        return ok and file_ok(out, 1000)

    def _mix_voice_only(self, video: str, voice: str, out: str) -> bool:
        ok, err = run_cmd(
            ["ffmpeg", "-y",
             "-i", video, "-i", voice,
             "-map", "0:v", "-map", "1:a",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest",
             out],
            timeout=600, label="mix voice-only",
        )
        if not ok:
            log.error("voice-only mix failed: %s", err[:200])
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
            log.warning("HFTTSClient init: %s", e)
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
            log.warning("AudioCache init (disabled): %s", e)
            return None

    def _find_music(self) -> Optional[str]:
        d = self._cfg.paths.music_dir
        if d.exists():
            for ext in ("*.mp3", "*.wav", "*.ogg"):
                files = list(d.glob(ext))
                if files:
                    return str(files[0])
        return None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _rm(path: str) -> None:
    try: os.remove(path)
    except OSError: pass
