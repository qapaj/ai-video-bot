"""
video/text_renderer.py
Renders subtitle frames as transparent PNGs using ImageMagick.
Arabic text is pre-shaped via arabic_reshaper + python-bidi before rendering.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import List, Optional

from utils.config import get_config
from utils.helpers import file_ok, is_arabic, normalize_unicode, word_wrap, fmt_timestamp
from utils.logger import get_logger

log = get_logger("video.text")


def shape_for_render(text: str) -> str:
    """Shape Arabic text for visual rendering."""
    if not text or not is_arabic(text):
        return text
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(text))
    except ImportError:
        log.warning("arabic-reshaper not installed — text may render incorrectly")
        return text
    except Exception as e:
        log.debug("shape_for_render: %s", e)
        return text


def shape_for_tts(text: str) -> str:
    """Normalise only — no bidi reordering (TTS reads Unicode directly)."""
    return normalize_unicode(text)


class TextRenderer:
    """Renders subtitle text as transparent PNG frames using ImageMagick."""

    def __init__(self) -> None:
        self.cfg   = get_config()
        self.sc    = self.cfg.subtitle
        self.vc    = self.cfg.video
        self._font = self._discover_font()

    def render_subtitle_png(self, text: str, output_png: str,
                            max_width: Optional[int] = None) -> bool:
        """Render shaped text as a transparent PNG. Returns True on success."""
        width = max_width or (self.vc.width - 80)

        if is_arabic(text):
            raw_lines = word_wrap(text, self.sc.max_chars_per_line)
            lines     = [shape_for_render(l) for l in raw_lines]
        else:
            lines = word_wrap(text, 30)

        label    = "\n".join(lines)
        n_lines  = len(lines) or 1
        height   = n_lines * (self.sc.font_size + 14) + 24

        cmd = ["convert", "-size", f"{width}x{height}", "xc:none",
               "-gravity", "Center"]
        if self._font:
            cmd += ["-font", self._font]
        cmd += [
            "-pointsize", str(self.sc.font_size),
            "-fill", f"rgba(0,0,0,{self.sc.shadow_opacity})",
            "-stroke", self.sc.outline_color,
            "-strokewidth", str(self.sc.outline_width),
            "-annotate", "0", label,
            "-fill", self.sc.text_color,
            "-stroke", "none",
            "-annotate", "0", label,
            output_png,
        ]

        env = os.environ.copy()
        env.update({"LANG": "en_US.UTF-8", "LC_ALL": "en_US.UTF-8"})
        r = subprocess.run(cmd, capture_output=True, timeout=20, env=env)
        if r.returncode != 0:
            log.warning("subtitle PNG: %s", r.stderr.decode(errors="replace")[-150:])
        return file_ok(output_png)

    def build_all_subtitle_frames(self, segments: List[dict], work_dir: str) -> List[dict]:
        """Render a PNG for every segment. Returns list of frame dicts."""
        sub_dir = Path(work_dir) / "subs"
        sub_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        t = 0.0
        for i, seg in enumerate(segments):
            text = seg["text"].strip()
            dur  = seg.get("actual_duration", seg.get("duration", 5.0))
            png  = str(sub_dir / f"sub_{i:03d}.png")
            if self.render_subtitle_png(text, png):
                frames.append({"image": png, "start": t, "end": t + dur, "text": text})
            t += dur
        log.info("%d/%d subtitle frames rendered", len(frames), len(segments))
        return frames

    def build_srt(self, segments: List[dict], srt_path: str) -> bool:
        """Write SRT file from segments (fallback subtitle method)."""
        try:
            with open(srt_path, "w", encoding="utf-8") as f:
                t = 0.0
                for i, seg in enumerate(segments):
                    dur  = seg.get("actual_duration", seg.get("duration", 5.0))
                    text = seg.get("text", "")
                    # Shape for rendering in SRT as well
                    shaped = shape_for_render(text) if is_arabic(text) else text
                    f.write(f"{i+1}\n{fmt_timestamp(t)} --> {fmt_timestamp(t+dur)}\n{shaped}\n\n")
                    t += dur
            return True
        except Exception as e:
            log.error("SRT build failed: %s", e)
            return False

    def _discover_font(self) -> Optional[str]:
        """Find the best Arabic-capable font available on this system."""
        search_paths = [
            "/usr/local/share/fonts/arabic/NotoNaskhArabic-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
            "/usr/share/fonts/opentype/noto/NotoNaskhArabic-Regular.ttf",
        ]
        # Check configured fonts dir
        fonts_dir = self.cfg.paths.fonts_dir
        if fonts_dir.exists():
            for f in fonts_dir.glob("*.ttf"):
                search_paths.insert(0, str(f))

        for path in search_paths:
            if os.path.exists(path):
                log.info("Subtitle font: %s", path)
                return path

        # fc-list fallback
        try:
            r = subprocess.run(
                ["fc-list", ":lang=ar", "--format=%{file}\n"],
                capture_output=True, text=True, timeout=10,
            )
            fonts = [l.strip() for l in r.stdout.splitlines()
                     if l.strip().endswith(".ttf") and "Naskh" in l]
            if not fonts:
                fonts = [l.strip() for l in r.stdout.splitlines()
                         if l.strip().endswith(".ttf")]
            if fonts:
                log.info("fc-list font: %s", fonts[0])
                return fonts[0]
        except Exception:
            pass

        for fallback in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        ):
            if os.path.exists(fallback):
                return fallback

        log.warning("No suitable font found — subtitles may be unstyled")
        return None
