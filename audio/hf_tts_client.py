"""
audio/hf_tts_client.py

HuggingFace Inference API client for text-to-speech.

Responsibilities:
  - Send authenticated POST requests to HF Inference API
  - Handle retries with exponential backoff
  - Handle 503 "model loading" responses (model cold-start)
  - Validate that the response is real audio data
  - Never download or load a model locally
  - Never block the pipeline — all failures are returned as False

Security: the HF token is ALWAYS read from the environment variable HF_TOKEN.
It is NEVER hardcoded in source code.

Supported models (all API-based, no local download):
  - coqui/XTTS-v2              (multilingual, highest quality)
  - facebook/mms-tts-ara       (Arabic optimised)
  - facebook/mms-tts-eng       (English optimised)
  - microsoft/speecht5_tts     (lightweight, fast)
  - espnet/kan-bayashi_ljspeech_vits  (English VITS)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import requests

from utils.logger import get_logger

log = get_logger("audio.hf_client")

# ── HuggingFace Inference API base URL ────────────────────────────────────────
HF_API_BASE = "https://api-inference.huggingface.co/models"

# ── Magic bytes that identify common audio formats ────────────────────────────
# Used to validate that the API returned real audio, not an error JSON.
AUDIO_MAGIC = {
    b"RIFF":     "wav",    # WAV
    b"ID3\x03":  "mp3",    # MP3 with ID3v2.3 tag
    b"ID3\x04":  "mp3",    # MP3 with ID3v2.4 tag
    b"\xff\xfb": "mp3",    # MP3 frame sync
    b"\xff\xf3": "mp3",    # MP3 frame sync variant
    b"\xff\xf2": "mp3",    # MP3 frame sync variant
    b"fLaC":     "flac",   # FLAC
    b"OggS":     "ogg",    # OGG
    b"\x1aE\xdf\xa3": "webm",  # WebM / Matroska
}

# ── Model registry ────────────────────────────────────────────────────────────
# Maps language codes to ordered list of (model_id, request_payload_builder).
# Models are tried in order; first successful response wins.

def _xtts_payload(text: str, language: str) -> dict:
    return {
        "inputs": text,
        "parameters": {
            "language": language,
            "speaker_embedding": None,  # use model default voice
        },
    }

def _mms_payload(text: str, language: str) -> dict:
    return {"inputs": text}

def _speecht5_payload(text: str, language: str) -> dict:
    return {
        "inputs": text,
        "parameters": {"vocoder_tag": "hifigan_unit"},
    }

def _generic_payload(text: str, language: str) -> dict:
    return {"inputs": text}


# (model_id, languages_supported, payload_builder, expected_content_types)
MODEL_REGISTRY = [
    # ── Tier 1: XTTS-v2 — best quality, multilingual ─────────────────────────
    {
        "id":        "coqui/XTTS-v2",
        "name":      "XTTS-v2",
        "languages": ["ar", "en", "fr", "es", "de", "pt", "it", "nl", "ru", "zh"],
        "payload":   _xtts_payload,
        "timeout":   120,   # XTTS is slower
    },
    # ── Tier 2: MMS-TTS per language ─────────────────────────────────────────
    {
        "id":        "facebook/mms-tts-ara",
        "name":      "MMS-Arabic",
        "languages": ["ar"],
        "payload":   _mms_payload,
        "timeout":   60,
    },
    {
        "id":        "facebook/mms-tts-eng",
        "name":      "MMS-English",
        "languages": ["en"],
        "payload":   _mms_payload,
        "timeout":   60,
    },
    {
        "id":        "facebook/mms-tts-fra",
        "name":      "MMS-French",
        "languages": ["fr"],
        "payload":   _mms_payload,
        "timeout":   60,
    },
    {
        "id":        "facebook/mms-tts-spa",
        "name":      "MMS-Spanish",
        "languages": ["es"],
        "payload":   _mms_payload,
        "timeout":   60,
    },
    {
        "id":        "facebook/mms-tts-deu",
        "name":      "MMS-German",
        "languages": ["de"],
        "payload":   _mms_payload,
        "timeout":   60,
    },
    # ── Tier 3: SpeechT5 — fast, lightweight, English only ───────────────────
    {
        "id":        "microsoft/speecht5_tts",
        "name":      "SpeechT5",
        "languages": ["en", "ar", "fr", "es", "de"],  # works for all, en best
        "payload":   _speecht5_payload,
        "timeout":   45,
    },
]


class HFTTSClient:
    """
    HuggingFace Inference API client for text-to-speech.

    Usage:
        client = HFTTSClient(token=os.environ["HF_TOKEN"])
        ok = client.synthesize("Hello world", "/tmp/out.wav", language="en")
    """

    def __init__(
        self,
        token: str,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        model_loading_wait: float = 20.0,
    ) -> None:
        if not token:
            raise ValueError(
                "HF_TOKEN is empty. Add it as a GitHub Secret: "
                "Settings → Secrets → Actions → New secret → HF_TOKEN"
            )
        self._token              = token
        self._max_retries        = max_retries
        self._retry_delay        = retry_delay
        self._model_loading_wait = model_loading_wait
        self._session            = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self._token}",
            "Content-Type":  "application/json",
            "Accept":        "audio/*, application/octet-stream",
        })

    # ── Public ─────────────────────────────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        output_path: str,
        language: str = "ar",
    ) -> bool:
        """
        Synthesise text to speech using the best available HF model.

        Tries each model in MODEL_REGISTRY order, skipping those that
        don't support the requested language.

        Returns True and writes output_path on success.
        Returns False if all models fail — never raises.
        """
        if not text.strip():
            log.warning("synthesize: empty text, skipping")
            return False

        # Filter models that support this language
        candidates = [m for m in MODEL_REGISTRY if language in m["languages"]]
        if not candidates:
            log.warning("No HF models registered for language '%s', trying all", language)
            candidates = MODEL_REGISTRY[:]

        for model in candidates:
            log.info("Trying HF model: %s (%s)", model["name"], model["id"])
            ok = self._call_model(
                model_id    = model["id"],
                payload     = model["payload"](text, language),
                output_path = output_path,
                timeout     = model["timeout"],
            )
            if ok:
                log.info("✓ HF TTS success: %s → %s (%d bytes)",
                         model["name"], Path(output_path).name,
                         os.path.getsize(output_path))
                return True
            log.warning("✗ %s failed, trying next model...", model["name"])

        log.error("All HF TTS models failed for language='%s' text='%.40s...'",
                  language, text)
        return False

    def check_model_available(self, model_id: str) -> bool:
        """
        Ping a model to check if the API can reach it.
        Returns True if the model responds (even if loading).
        """
        try:
            r = self._session.head(
                f"{HF_API_BASE}/{model_id}",
                timeout=10,
            )
            # 200 = ready, 503 = loading (still reachable), 401/403 = auth issue
            return r.status_code in (200, 503)
        except Exception as e:
            log.debug("check_model_available(%s): %s", model_id, e)
            return False

    def list_available_models(self, language: str = "ar") -> list[str]:
        """Return names of models that are reachable for the given language."""
        available = []
        for model in MODEL_REGISTRY:
            if language not in model["languages"]:
                continue
            if self.check_model_available(model["id"]):
                available.append(model["name"])
        return available

    # ── Private ────────────────────────────────────────────────────────────────

    def _call_model(
        self,
        model_id:    str,
        payload:     dict,
        output_path: str,
        timeout:     int,
    ) -> bool:
        """
        Call one HF model with retry + model-loading wait logic.
        Returns True if audio was written successfully.
        """
        url = f"{HF_API_BASE}/{model_id}"

        for attempt in range(1, self._max_retries + 1):
            try:
                log.debug("  Attempt %d/%d: POST %s",
                          attempt, self._max_retries, url)
                response = self._session.post(
                    url,
                    json=payload,
                    timeout=timeout,
                    stream=True,   # stream to handle large audio files
                )

                # ── 503: model is loading (cold start) ────────────────────────
                if response.status_code == 503:
                    wait = self._model_loading_wait * attempt
                    log.info("  Model loading, waiting %.0fs...", wait)
                    time.sleep(wait)
                    continue

                # ── 429: rate limited ──────────────────────────────────────────
                if response.status_code == 429:
                    retry_after = float(
                        response.headers.get("Retry-After", self._retry_delay * attempt)
                    )
                    log.warning("  Rate limited, waiting %.0fs...", retry_after)
                    time.sleep(retry_after)
                    continue

                # ── 401/403: auth failure — no point retrying ─────────────────
                if response.status_code in (401, 403):
                    log.error(
                        "  Auth failed (HTTP %d). Check your HF_TOKEN secret.",
                        response.status_code,
                    )
                    return False

                # ── Non-200 other errors ───────────────────────────────────────
                if response.status_code != 200:
                    log.warning("  HTTP %d from %s",
                                response.status_code, model_id)
                    time.sleep(self._retry_delay * attempt)
                    continue

                # ── 200: read response body ────────────────────────────────────
                audio_bytes = response.content
                if not audio_bytes:
                    log.warning("  Empty response body from %s", model_id)
                    time.sleep(self._retry_delay)
                    continue

                # Validate that it's actually audio
                fmt = self._detect_audio_format(audio_bytes)
                if fmt is None:
                    # Could be a JSON error wrapped in a 200
                    try:
                        err = response.json()
                        log.warning("  API returned JSON instead of audio: %s",
                                    str(err)[:200])
                    except Exception:
                        log.warning("  Response is not audio and not JSON "
                                    "(first 20 bytes: %s)", audio_bytes[:20])
                    time.sleep(self._retry_delay * attempt)
                    continue

                # Write to output path (use detected format extension for temp)
                tmp_path = output_path + f"_hf.{fmt}"
                with open(tmp_path, "wb") as f:
                    f.write(audio_bytes)

                # Verify written file
                written_size = os.path.getsize(tmp_path)
                if written_size < 200:
                    log.warning("  Written file too small (%d bytes)", written_size)
                    os.remove(tmp_path)
                    continue

                # If target needs a specific format, convert; otherwise rename
                if output_path.endswith(f".{fmt}"):
                    os.rename(tmp_path, output_path)
                else:
                    from utils.helpers import run_cmd
                    ok, err = run_cmd([
                        "ffmpeg", "-y", "-i", tmp_path,
                        "-ar", "44100", "-ac", "1",
                        "-c:a", "aac", "-b:a", "128k",
                        output_path,
                    ], timeout=30, label=f"convert {fmt}→aac")
                    try: os.remove(tmp_path)
                    except: pass
                    if not ok:
                        log.warning("  Format conversion failed: %s", err)
                        continue

                return True

            except requests.exceptions.Timeout:
                log.warning("  Timeout on attempt %d/%d (model=%s)",
                            attempt, self._max_retries, model_id)
                time.sleep(self._retry_delay * attempt)

            except requests.exceptions.ConnectionError as e:
                log.warning("  Connection error attempt %d/%d: %s",
                            attempt, self._max_retries, e)
                time.sleep(self._retry_delay * attempt)

            except Exception as e:
                log.error("  Unexpected error in _call_model(%s): %s: %s",
                          model_id, type(e).__name__, e)
                time.sleep(self._retry_delay)

        return False

    @staticmethod
    def _detect_audio_format(data: bytes) -> Optional[str]:
        """
        Detect audio format from magic bytes.
        Returns format string ('wav', 'mp3', 'flac', etc.) or None if not audio.
        """
        if len(data) < 4:
            return None
        header = data[:4]
        for magic, fmt in AUDIO_MAGIC.items():
            if header.startswith(magic):
                return fmt
        # Additional check: WAV files start with RIFF but need 12-byte check
        if data[:4] == b"RIFF" and len(data) > 12 and data[8:12] == b"WAVE":
            return "wav"
        return None
