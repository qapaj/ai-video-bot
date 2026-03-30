"""
config/arabic_utils.py
Arabic text shaping utilities.

Root cause of disconnected letters:
  Arabic Unicode stores letters as isolated codepoints.
  Rendering engines that lack Arabic shaping support draw each letter
  in its isolated form → م ر ح ب ا instead of مرحبا.

Fix:
  1. arabic_reshaper  — converts isolated codepoints to contextual forms
  2. python-bidi      — applies Unicode Bidirectional Algorithm (RTL ordering)

Both steps are required for correct visual rendering.
For TTS: use prepare_for_tts() which only normalises (no bidi reordering).
"""

from __future__ import annotations

import re
import unicodedata
from typing import List

_RESHAPER_OK: bool | None = None


def _check_reshaper() -> bool:
    global _RESHAPER_OK
    if _RESHAPER_OK is None:
        try:
            import arabic_reshaper       # noqa: F401
            from bidi.algorithm import get_display  # noqa: F401
            _RESHAPER_OK = True
        except ImportError:
            _RESHAPER_OK = False
    return _RESHAPER_OK


def shape_for_render(text: str) -> str:
    """
    Shape Arabic text for visual rendering (ImageMagick, PIL, etc.).
    Applies reshaping + RTL bidi reordering.
    Safe to call on mixed Arabic/Latin text.
    """
    if not text or not _contains_arabic(text):
        return text
    if not _check_reshaper():
        return text  # degrade gracefully

    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        return text


def prepare_for_tts(text: str) -> str:
    """
    Prepare text for TTS synthesis.
    Only normalises Unicode — does NOT apply bidi reordering
    because TTS engines read Unicode sequentially.
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    return text.strip()


def wrap_and_shape(text: str, max_chars: int = 22) -> List[str]:
    """
    Word-wrap Arabic text and shape each line for rendering.
    Returns a list of shaped lines ready for ImageMagick.
    """
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

    return [shape_for_render(line) for line in lines] if lines else [shape_for_render(text)]


def _contains_arabic(text: str) -> bool:
    return bool(re.search(
        r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]",
        text,
    ))
