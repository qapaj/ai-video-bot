"""
utils/helpers.py
Shared utility functions: subprocess, file ops, text helpers, network.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import unicodedata
from pathlib import Path
from typing import Any, List, Optional, Tuple


# ── Subprocess ────────────────────────────────────────────────────────────────

def run_cmd(cmd: List[str], timeout: int = 300, label: str = "",
            capture: bool = True) -> Tuple[bool, str]:
    """Run a subprocess. Returns (success, stderr_tail). Never raises."""
    try:
        result = subprocess.run(cmd, capture_output=capture,
                                timeout=timeout, text=True)
        if result.returncode != 0:
            tail = (result.stderr or "")[-400:]
            if label:
                from utils.logger import get_logger
                get_logger("helpers").warning("✗ %s: %s", label, tail)
            return False, tail
        return True, ""
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    except FileNotFoundError as e:
        return False, f"not found: {e}"
    except Exception as e:
        return False, str(e)


def ffprobe_duration(path: str) -> float:
    """Return media duration in seconds using ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", path],
            capture_output=True, text=True, timeout=15,
        )
        return float(json.loads(r.stdout).get("format", {}).get("duration", 0))
    except Exception:
        return 0.0


def ffprobe_dims(path: str) -> Tuple[int, int]:
    """Return (width, height) of video using ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-select_streams", "v:0", path],
            capture_output=True, text=True, timeout=15,
        )
        for s in json.loads(r.stdout).get("streams", []):
            if s.get("width"):
                return s["width"], s["height"]
    except Exception:
        pass
    return 1080, 1920


# ── File helpers ──────────────────────────────────────────────────────────────

def safe_remove(path: str) -> None:
    try: os.remove(path)
    except Exception: pass


def ensure_dir(path: "str | Path") -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def file_ok(path: "str | Path", min_bytes: int = 500) -> bool:
    try:
        return Path(path).exists() and Path(path).stat().st_size >= min_bytes
    except Exception:
        return False


def cache_key(text: str, *extras: Any) -> str:
    payload = text + "".join(str(e) for e in extras)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ── Text helpers ──────────────────────────────────────────────────────────────

def is_arabic(text: str) -> bool:
    return bool(re.search(
        r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]", text))


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    return text.strip()


def word_wrap(text: str, max_chars: int = 25) -> List[str]:
    words = text.split()
    lines: List[str] = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_chars:
            cur = (cur + " " + w).strip()
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    return lines or [text]


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.،؟!۔\n])\s*", text)
    return [p.strip() for p in parts if p.strip()]


def estimate_duration(text: str, wpm: int = 130) -> float:
    words = len(text) / 4.5
    return max(3.0, (words / wpm) * 60)


def fmt_timestamp(t: float) -> str:
    h, rem = divmod(int(t), 3600)
    m, s   = divmod(rem, 60)
    ms     = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def safe_filename(text: str, max_len: int = 30) -> str:
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s]+", "_", text.strip())
    return text[:max_len] or "video"


# ── Network ────────────────────────────────────────────────────────────────────

def download_file(url: str, output_path: str, timeout: int = 30,
                  headers: Optional[dict] = None) -> bool:
    try:
        import requests
        resp = requests.get(url, headers=headers or {}, timeout=timeout, stream=True)
        if resp.status_code != 200:
            return False
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        return file_ok(output_path, 1000)
    except Exception:
        return False
