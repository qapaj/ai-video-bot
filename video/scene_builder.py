"""
video/scene_builder.py
Converts individual media assets (images, videos) into timed video clips.
Each clip matches the duration of its corresponding audio segment.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Optional

from utils.config import get_config
from utils.helpers import file_ok, run_cmd
from utils.logger import get_logger
from video.transitions import ken_burns_filter

log = get_logger("video.scene")

EFFECTS = ["zoom_in", "zoom_out", "pan_right", "pan_left"]


class SceneBuilder:
    """
    Builds individual video clips from images or source videos.
    Handles:
    - Ken Burns effects on images
    - Source video reformatting and trimming
    - Gradient card generation (when no media is available)
    """

    def __init__(self) -> None:
        self.cfg = get_config()
        self.vc  = self.cfg.video

    # ── Public ─────────────────────────────────────────────────────────────────

    def build_clip(
        self,
        media_item: Optional[dict],
        duration: float,
        output_path: str,
        effect_index: int = 0,
        topic: str = "",
        video_type: str = "news",
        color_index: int = 0,
    ) -> bool:
        """
        Build a single video clip.
        If media_item is None or fails, generates a gradient card.
        """
        duration = max(self.cfg.video.min_segment_sec, duration)
        effect   = EFFECTS[effect_index % len(EFFECTS)]

        if media_item:
            local_path = media_item.get("local_path", "")
            if local_path and os.path.exists(local_path):
                if media_item.get("type") == "video":
                    ok = self._from_video(local_path, duration, output_path)
                else:
                    ok = self._from_image(local_path, duration, output_path, effect)
                if ok:
                    return True
                log.warning("Media clip failed, using gradient card")

        return self._gradient_card(topic, video_type, duration, output_path, color_index)

    def build_intro_card(
        self,
        topic: str,
        video_type: str,
        duration: float,
        output_path: str,
        font_path: Optional[str] = None,
    ) -> bool:
        """Build a branded intro title card."""
        return self._gradient_card(topic, video_type, duration, output_path,
                                   color_index=0, font_path=font_path)

    # ── Image → clip ───────────────────────────────────────────────────────────

    def _from_image(self, path: str, duration: float,
                    output_path: str, effect: str = "zoom_in") -> bool:
        """Apply Ken Burns effect to an image and render as a video clip."""
        w, h = self.vc.width, self.vc.height

        # Step 1: Normalise image to exact output resolution
        norm = output_path + "_n.jpg"
        ok, _ = run_cmd([
            "ffmpeg", "-y", "-i", path,
            "-vf", (
                f"scale={w*2}:{h*2}:force_original_aspect_ratio=increase,"
                f"crop={w*2}:{h*2},scale={w}:{h}"
            ),
            "-frames:v", "1", "-q:v", "2", norm,
        ], timeout=30, label="image normalise")
        src = norm if (ok and file_ok(norm)) else path

        # Step 2: Ken Burns zoompan
        vf = ken_burns_filter(duration, self.vc.fps, w, h, effect)
        ok2, _ = run_cmd([
            "ffmpeg", "-y", "-loop", "1", "-i", src,
            "-vf", vf,
            "-t", str(duration),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", self.vc.preset, "-crf", str(self.vc.crf),
            output_path,
        ], timeout=180, label=f"ken_burns {effect}")

        if os.path.exists(norm):
            try: os.remove(norm)
            except: pass

        if not ok2:
            # Fallback: simple loop without Ken Burns
            ok2, _ = run_cmd([
                "ffmpeg", "-y", "-loop", "1", "-i", src,
                "-vf", f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}",
                "-t", str(duration),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-preset", self.vc.preset, "-crf", str(self.vc.crf),
                output_path,
            ], timeout=120, label="simple loop")

        return ok2 and file_ok(output_path)

    # ── Video → clip ───────────────────────────────────────────────────────────

    def _from_video(self, path: str, duration: float, output_path: str) -> bool:
        """Trim and reformat a source video to vertical output dimensions."""
        w, h = self.vc.width, self.vc.height
        ok, _ = run_cmd([
            "ffmpeg", "-y", "-i", path,
            "-vf", (
                f"scale={w*2}:{h*2}:force_original_aspect_ratio=increase,"
                f"crop={w}:{h},scale={w}:{h}"
            ),
            "-t", str(duration),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", self.vc.preset, "-crf", str(self.vc.crf),
            "-an",
            output_path,
        ], timeout=180, label="video clip")
        return ok and file_ok(output_path)

    # ── Gradient card ──────────────────────────────────────────────────────────

    def _gradient_card(
        self,
        topic: str,
        video_type: str,
        duration: float,
        output_path: str,
        color_index: int = 0,
        font_path: Optional[str] = None,
    ) -> bool:
        """Generate a branded gradient card as a video clip."""
        w, h = self.vc.width, self.vc.height
        palettes = [
            ("#0a1628", "#1e3a5f"), ("#1a0a2e", "#3d1a6e"),
            ("#0a2818", "#1a5c38"), ("#1a1a00", "#4a4a00"),
            ("#1a0a0a", "#5c1a1a"),
        ]
        accents = {
            "news":      "#3b82f6",
            "story":     "#8b5cf6",
            "facts":     "#10b981",
            "education": "#f59e0b",
        }
        labels = {
            "news": "أخبار", "story": "قصة",
            "facts": "حقائق", "education": "تعليم",
        }
        bg_dark, bg_light = palettes[color_index % len(palettes)]
        accent = accents.get(video_type, "#ffffff")

        # Shape Arabic text for rendering
        from video.text_renderer import shape_text
        label = shape_text(labels.get(video_type, "فيديو"))
        topic_text = shape_text(topic[:35])

        img = output_path.replace(".mp4", "_card.jpg")
        cmd = ["convert", "-size", f"{w}x{h}",
               f"gradient:{bg_light}-{bg_dark}"]

        fp = font_path or self._find_font()
        if fp:
            cmd += [
                "-gravity", "Center",
                "-fill", accent, "-font", fp,
                "-pointsize", "72", "-annotate", "0+0-300", label,
                "-fill", "white", "-font", fp,
                "-pointsize", "44", "-annotate", "0+0-120", topic_text,
            ]
        cmd.append(img)

        r_ok, _ = run_cmd(cmd, timeout=30, label="gradient card img")
        if not r_ok or not file_ok(img):
            return False

        result = self._from_image(img, duration, output_path, "zoom_in")
        try: os.remove(img)
        except: pass
        return result

    def _find_font(self) -> Optional[str]:
        """Quick font lookup — delegates to TextRenderer."""
        from video.text_renderer import TextRenderer
        return TextRenderer()._font
