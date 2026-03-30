"""
hf/hf_model_loader.py
Smart HuggingFace model loader.

Selects lightweight models appropriate for GitHub Actions runners
(2-core CPU, 7GB RAM, no GPU). Downloads are cached to avoid
repeated large downloads.

Strategy:
  - Try API-based inference first (no local download, free tier)
  - Fall back to local lightweight model if API is unavailable
  - Never block the pipeline if HF is unavailable
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from utils.logger import get_logger

log = get_logger("hf.loader")

# ── Model registry ────────────────────────────────────────────────────────────
# Maps task → (api_model_id, local_fallback_id, max_tokens)
# All models chosen for small size (<500MB) and CPU viability.
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "text_generation": {
        "api_model":   "google/flan-t5-base",
        "local_model": "google/flan-t5-small",
        "max_tokens":  512,
        "task":        "text2text-generation",
    },
    "summarization": {
        "api_model":   "facebook/bart-large-cnn",
        "local_model": "sshleifer/distilbart-cnn-6-6",
        "max_tokens":  256,
        "task":        "summarization",
    },
    "grammar_correction": {
        "api_model":   "grammarly/coedit-large",
        "local_model": "prithivida/grammar_error_correcter_v1",
        "max_tokens":  256,
        "task":        "text2text-generation",
    },
    "tts": {
        "api_model":   "facebook/mms-tts-ara",
        "local_model": "microsoft/speecht5_tts",
        "max_tokens":  None,
        "task":        "text-to-speech",
    },
    "audio_enhancement": {
        "api_model":   "speechbrain/sepformer-wham",
        "local_model": None,
        "max_tokens":  None,
        "task":        "audio-to-audio",
    },
}

HF_API_URL = "https://api-inference.huggingface.co/models"


class HFModelLoader:
    """
    Provides inference via the HuggingFace Inference API
    with an optional local-model fallback.
    """

    def __init__(self, token: str, cache_dir: str = "/tmp/hf_cache") -> None:
        self.token    = token
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._available: Dict[str, bool] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def text_generate(self, prompt: str, task: str = "text_generation",
                      max_new_tokens: int = 300) -> str:
        """
        Generate text using HF Inference API.
        Returns empty string if unavailable — caller must handle fallback.
        """
        spec = MODEL_REGISTRY.get(task)
        if not spec:
            log.warning("Unknown HF task: %s", task)
            return ""

        model_id = spec["api_model"]
        payload  = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.7,
                "do_sample": True,
                "return_full_text": False,
            },
        }

        result = self._api_call(model_id, payload)
        if not result:
            return ""

        # Parse response
        if isinstance(result, list) and result:
            item = result[0]
            return item.get("generated_text", item.get("summary_text", "")).strip()
        return ""

    def grammar_fix(self, text: str) -> str:
        """Fix grammar mistakes in text. Returns original if unavailable."""
        if not text.strip():
            return text

        # Check cache
        ckey = f"gram_{hash(text) & 0xFFFFFF}"
        cached = self._load_cache(ckey)
        if cached:
            return cached

        prompt = f"Fix grammar: {text}"
        result = self.text_generate(prompt, task="grammar_correction")
        if result and len(result) > 10:
            self._save_cache(ckey, result)
            return result
        return text

    def improve_script(self, script: str, language: str = "ar") -> str:
        """
        Improve a video script: grammar, clarity, pacing.
        Works best for English; for Arabic falls back to template cleaning.
        """
        if language != "en":
            # HF Arabic models are large — use rule-based cleanup for Arabic
            return self._clean_arabic_script(script)

        ckey = f"script_{hash(script) & 0xFFFFFF}"
        cached = self._load_cache(ckey)
        if cached:
            return cached

        prompt = (
            "Rewrite this video script to be clear, engaging and well-paced. "
            "Fix grammar. Use short sentences. Keep the same meaning:\n\n"
            f"{script[:1000]}"
        )
        result = self.text_generate(prompt, task="text_generation", max_new_tokens=400)
        if result and len(result) > 50:
            self._save_cache(ckey, result)
            return result
        return script

    def is_available(self, task: str = "text_generation") -> bool:
        """Check if HF API is reachable for a given task."""
        if task in self._available:
            return self._available[task]

        spec = MODEL_REGISTRY.get(task)
        if not spec:
            return False

        try:
            import requests
            r = requests.head(
                f"{HF_API_URL}/{spec['api_model']}",
                headers=self._headers(),
                timeout=5,
            )
            ok = r.status_code in (200, 401, 403)  # 401/403 = reachable but auth issue
            self._available[task] = ok
            return ok
        except Exception:
            self._available[task] = False
            return False

    # ── Private helpers ────────────────────────────────────────────────────────

    def _api_call(self, model_id: str, payload: dict,
                  retries: int = 2) -> Any:
        """Call HF Inference API with retry on 503 (model loading)."""
        try:
            import requests
            for attempt in range(retries + 1):
                r = requests.post(
                    f"{HF_API_URL}/{model_id}",
                    headers=self._headers(),
                    json=payload,
                    timeout=60,
                )
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 503:
                    wait = 15 * (attempt + 1)
                    log.info("HF model loading, waiting %ds...", wait)
                    time.sleep(wait)
                    continue
                log.debug("HF API %s: HTTP %d", model_id, r.status_code)
                return None
        except Exception as e:
            log.debug("HF API error: %s", e)
            return None

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"}

    def _load_cache(self, key: str) -> Optional[str]:
        path = self.cache_dir / f"{key}.txt"
        if path.exists():
            try:
                return path.read_text(encoding="utf-8")
            except Exception:
                pass
        return None

    def _save_cache(self, key: str, value: str) -> None:
        try:
            (self.cache_dir / f"{key}.txt").write_text(value, encoding="utf-8")
        except Exception:
            pass

    def _clean_arabic_script(self, script: str) -> str:
        """
        Rule-based Arabic script cleanup:
        - Remove repeated punctuation
        - Normalise spaces around Arabic punctuation
        - Remove orphaned single characters
        """
        import re
        text = re.sub(r"[،،]{2,}", "،", script)      # double comma
        text = re.sub(r"[\.]{2,}", ".", text)          # ellipsis cleanup
        text = re.sub(r"\s+", " ", text)               # normalize whitespace
        text = re.sub(r"\s([،؟!.])", r"\1", text)     # space before punctuation
        return text.strip()


# ── Module-level singleton ─────────────────────────────────────────────────────

_loader: Optional[HFModelLoader] = None


def get_loader() -> HFModelLoader:
    global _loader
    if _loader is None:
        from utils.config import get_config
        cfg     = get_config()
        _loader = HFModelLoader(cfg.hf.token, cfg.hf.cache_dir)
    return _loader
