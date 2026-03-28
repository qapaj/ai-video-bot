"""
audio/audio_cache.py

Content-addressed audio cache using SHA-256 hashes.

Before any API call:
  1. Hash the input text + voice parameters
  2. Check if a valid cached file exists
  3. Return the cached path if found

After a successful synthesis:
  4. Store the result in the cache
  5. Return the cached path for future calls

This eliminates redundant API calls for repeated text
(e.g. generating the same script twice, or retrying after a video error).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

log = get_logger("audio.cache")

# Minimum valid audio file size (bytes). Smaller = likely corrupt/empty.
MIN_AUDIO_BYTES = 200

# Maximum cache age in seconds (7 days). Older entries are auto-pruned.
MAX_CACHE_AGE_SEC = 7 * 24 * 3600

# Maximum total cache entries to keep (prevents unbounded disk growth).
MAX_CACHE_ENTRIES = 500


class AudioCache:
    """
    Persistent audio cache stored in a single directory.
    Files are named by hash; no metadata file needed.
    Thread-safe for read; writes use atomic copy.
    """

    def __init__(self, cache_dir: str = "cache/audio") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        log.debug("Audio cache: %s", self.cache_dir)

    # ── Public API ─────────────────────────────────────────────────────────────

    def key(self, text: str, language: str, voice: str, gender: str) -> str:
        """
        Generate a deterministic cache key from synthesis parameters.
        The hash covers text content AND voice settings so that the same
        text synthesised with different voices produces different cache entries.
        """
        payload = f"{text.strip()}|{language}|{voice}|{gender}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def get(self, cache_key: str) -> Optional[str]:
        """
        Return path to a cached audio file if it exists and is valid.
        Returns None if not cached or cache entry is corrupt/expired.
        """
        for ext in (".wav", ".mp3", ".aac", ".ogg"):
            path = self.cache_dir / f"{cache_key}{ext}"
            if path.exists():
                stat = path.stat()
                # Check size
                if stat.st_size < MIN_AUDIO_BYTES:
                    log.debug("Cache entry too small, removing: %s", path.name)
                    path.unlink(missing_ok=True)
                    continue
                # Check age
                age = time.time() - stat.st_mtime
                if age > MAX_CACHE_AGE_SEC:
                    log.debug("Cache entry expired (%dd old): %s",
                              int(age / 86400), path.name)
                    path.unlink(missing_ok=True)
                    continue
                log.info("Cache HIT: %s (%d bytes, %.0fh old)",
                         path.name, stat.st_size, age / 3600)
                return str(path)
        return None

    def put(self, cache_key: str, source_path: str, ext: str = ".wav") -> str:
        """
        Store an audio file in the cache.
        Uses atomic copy: writes to a temp path, then renames,
        so a partially-written file is never served as a cache hit.

        Returns the path of the cached file.
        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source audio not found: {source_path}")
        if os.path.getsize(source_path) < MIN_AUDIO_BYTES:
            raise ValueError(f"Source audio too small to cache: {source_path}")

        ext = ext if ext.startswith(".") else f".{ext}"
        dest = self.cache_dir / f"{cache_key}{ext}"
        tmp  = self.cache_dir / f"{cache_key}.tmp"

        shutil.copy2(source_path, str(tmp))
        tmp.rename(dest)

        log.info("Cache PUT: %s (%d bytes)", dest.name,
                 os.path.getsize(str(dest)))
        return str(dest)

    def serve(self, cache_key: str, target_path: str) -> bool:
        """
        Copy a cached file to target_path.
        Returns True on success, False if not in cache.
        """
        cached = self.get(cache_key)
        if not cached:
            return False
        shutil.copy2(cached, target_path)
        log.debug("Cache served %s → %s", Path(cached).name,
                  Path(target_path).name)
        return True

    def prune(self) -> int:
        """
        Remove expired and excess cache entries.
        Returns the number of files removed.
        """
        entries = sorted(
            self.cache_dir.glob("*.*"),
            key=lambda p: p.stat().st_mtime,
        )
        removed = 0

        for path in entries:
            if path.suffix == ".tmp":
                path.unlink(missing_ok=True)
                removed += 1
                continue
            age = time.time() - path.stat().st_mtime
            if age > MAX_CACHE_AGE_SEC:
                path.unlink(missing_ok=True)
                removed += 1

        # Enforce max entries (remove oldest first)
        entries = sorted(
            self.cache_dir.glob("*.*"),
            key=lambda p: p.stat().st_mtime,
        )
        if len(entries) > MAX_CACHE_ENTRIES:
            for path in entries[: len(entries) - MAX_CACHE_ENTRIES]:
                path.unlink(missing_ok=True)
                removed += 1

        if removed:
            log.info("Cache pruned: %d files removed", removed)
        return removed

    def stats(self) -> dict:
        """Return cache statistics."""
        files  = list(self.cache_dir.glob("*.*"))
        total  = sum(f.stat().st_size for f in files)
        return {
            "entries": len(files),
            "total_bytes": total,
            "total_mb": round(total / 1_000_000, 2),
            "cache_dir": str(self.cache_dir),
        }
