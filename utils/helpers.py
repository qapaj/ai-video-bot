"""
utils/helpers.py
Shared utility functions: subprocess, file ops, text helpers, network.

KEY RULE: Every subprocess call in this codebase MUST go through run_cmd().
run_cmd() catches FileNotFoundError, TimeoutExpired, and all other exceptions.
It NEVER raises — callers always get (bool, str).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path
from typing import Any, List, Optional, Tuple


# ── Subprocess ────────────────────────────────────────────────────────────────

def run_cmd(
    cmd: List[str],
    timeout: int = 300,
    label: str = "",
    capture: bool = True,
) -> Tuple[bool, str]:
    """
    Run a subprocess command safely.
    Returns (success: bool, error_message: str).
    NEVER raises — catches FileNotFoundError, TimeoutExpired, and all exceptions.

    FileNotFoundError is the most common silent killer:
      subprocess.run(["edge-tts", ...])  →  [Errno 2] No such file or directory
    This function converts that into (False, "not found: edge-tts").
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            timeout=timeout,
            text=True,
        )
        if result.returncode != 0:
            tail = (result.stderr or "")[-400:]
            if label:
                from utils.logger import get_logger
                get_logger("helpers").warning("✗ %s: %s", label, tail)
            return False, tail
        return True, ""

    except FileNotFoundError as e:
        msg = f"binary not found: {e.filename or cmd[0]}"
        if label:
            from utils.logger import get_logger
            get_logger("helpers").warning("✗ %s: %s", label, msg)
        return False, msg

    except subprocess.TimeoutExpired:
        msg = f"timeout after {timeout}s"
        if label:
            from utils.logger import get_logger
            get_logger("helpers").warning("✗ %s: %s", label, msg)
        return False, msg

    except PermissionError as e:
        msg = f"permission denied: {e}"
        if label:
            from utils.logger import get_logger
            get_logger("helpers").warning("✗ %s: %s", label, msg)
        return False, msg

    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if label:
            from utils.logger import get_logger
            get_logger("helpers").warning("✗ %s: %s", label, msg)
        return False, msg


def binary_exists(name: str) -> bool:
    """
    Check if a binary exists on PATH or is accessible.
    Uses shutil.which() — safe, never raises.
    """
    return shutil.which(name) is not None


def probe_binary(name: str) -> bool:
    """
    Check if a binary exists AND responds to --version or --help.
    Uses run_cmd() so it never raises.
    Returns True if binary is callable, False otherwise.
    """
    ok, _ = run_cmd([name, "--version"], timeout=10)
    return ok


def ffprobe_duration(path: str) -> float:
    """Return media duration in seconds. Returns 0.0 on any error."""
    ok, _ = run_cmd(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
        timeout=15,
        label="",
    )
    if not ok:
        return 0.0
    try:
        import subprocess as sp
        r = sp.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
            capture_output=True, text=True, timeout=15,
        )
        return float(json.loads(r.stdout).get("format", {}).get("duration", 0))
    except Exception:
        return 0.0


def ffprobe_dims(path: str) -> Tuple[int, int]:
    """Return (width, height) of video. Returns (1080, 1920) on error."""
    try:
        import subprocess as sp
        r = sp.run(
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
    """Remove a file silently. Never raises."""
    try:
        os.remove(path)
    except OSError:
        pass


def ensure_dir(path: "str | Path") -> Path:
    """Create directory tree and return Path. Never raises."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def file_ok(path: "str | Path", min_bytes: int = 500) -> bool:
    """Return True if file exists and meets minimum size."""
    try:
        return Path(path).exists() and Path(path).stat().st_size >= min_bytes
    except Exception:
        return False


def cache_key(text: str, *extras: Any) -> str:
    """SHA-256 hash of text + extras — used as cache filename stem."""
    payload = text + "".join(str(e) for e in extras)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ── Text helpers ──────────────────────────────────────────────────────────────

def is_arabic(text: str) -> bool:
    """Return True if text contains Arabic characters."""
    return bool(re.search(
        r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]",
        text,
    ))


def normalize_unicode(text: str) -> str:
    """NFC normalize and strip zero-width characters."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    return text.strip()


def word_wrap(text: str, max_chars: int = 25) -> List[str]:
    """Word-wrap text at word boundaries."""
    words = text.split()
    lines: List[str] = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_chars:
            cur = (cur + " " + w).strip()
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [text]


def split_sentences(text: str) -> List[str]:
    """Split text at sentence boundaries (Arabic + Latin)."""
    parts = re.split(r"(?<=[.،؟!۔\n])\s*", text)
    return [p.strip() for p in parts if p.strip()]


def estimate_duration(text: str, wpm: int = 130) -> float:
    """Estimate speech duration from character count."""
    words = len(text) / 4.5
    return max(3.0, (words / wpm) * 60)


def fmt_timestamp(t: float) -> str:
    """Format seconds as SRT timestamp HH:MM:SS,mmm."""
    h, rem = divmod(int(t), 3600)
    m, s   = divmod(rem, 60)
    ms     = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def safe_filename(text: str, max_len: int = 30) -> str:
    """Convert arbitrary text to a safe filename stem."""
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s]+", "_", text.strip())
    return text[:max_len] or "video"


# ── Network helpers ────────────────────────────────────────────────────────────

def download_file(
    url: str,
    output_path: str,
    timeout: int = 30,
    headers: Optional[dict] = None,
) -> bool:
    """Download a URL to a file. Returns True on success. Never raises."""
    try:
        import requests
        resp = requests.get(
            url, headers=headers or {}, timeout=timeout, stream=True,
        )
        if resp.status_code != 200:
            return False
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        return file_ok(output_path, 1000)
    except Exception:
        return False
