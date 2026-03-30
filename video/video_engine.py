"""
video/video_engine.py
Main video assembly engine.

Pipeline:
  1. Build one visual clip per audio segment (SceneBuilder)
  2. Concatenate clips with xfade transitions
  3. Loop/pad visual track if shorter than audio
  4. Render subtitle PNG frames (TextRenderer)
  5. Overlay subtitles on silent video
  6. Mix audio (AudioEngine)
  7. Return final MP4 path
"""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path
from typing import List, Optional

from audio.audio_engine import AudioEngine
from audio.narration_generator import NarrationGenerator
from utils.config import get_config
from utils.helpers import ffprobe_duration, file_ok, run_cmd
from utils.logger import StageLogger, get_logger
from video.scene_builder import SceneBuilder
from video.text_renderer import TextRenderer
from video.transitions import build_xfade_filter, simple_concat_filter

log = get_logger("video.engine")


class VideoEngine:
    """
    Orchestrates all video assembly stages.
    """

    def __init__(self) -> None:
        self.cfg     = get_config()
        self.vc      = self.cfg.video
        self.scene   = SceneBuilder()
        self.text    = TextRenderer()
        self.audio_e = AudioEngine()
        self.narrate = NarrationGenerator()

    # ── Public ─────────────────────────────────────────────────────────────────

    def assemble(
        self,
        script_data: dict,
        segments_with_audio: List[dict],
        media_data: dict,
        work_dir: str,
        final_output_path: str,
    ) -> bool:
        """
        Full assembly pipeline.
        Returns True and writes final_output_path on success.
        """
        os.makedirs(work_dir, exist_ok=True)
        clips_dir = os.path.join(work_dir, "clips")
        os.makedirs(clips_dir, exist_ok=True)

        topic      = script_data.get("topic", "")
        video_type = script_data.get("video_type", "news")

        valid_segs = [
            s for s in segments_with_audio
            if s.get("audio_path") and file_ok(s.get("audio_path", ""))
        ]
        if not valid_segs:
            log.error("No valid audio segments to assemble")
            return False

        total_audio = sum(
            s.get("actual_duration", s.get("duration", 5.0)) for s in valid_segs
        )
        log.info("Assembly: %d segments, %.1fs total audio", len(valid_segs), total_audio)

        all_media = media_data.get("videos", []) + media_data.get("images", [])
        import random; random.shuffle(all_media)

        # ── Stage 1: Visual clips ──────────────────────────────────────────────
        with StageLogger(log, "Building visual clips"):
            clips = self._build_clips(
                valid_segs, all_media, clips_dir, topic, video_type
            )
        if not clips:
            log.error("No clips generated")
            return False

        # ── Stage 2: Visual track (with loop to match audio) ──────────────────
        silent_video = os.path.join(work_dir, "silent.mp4")
        with StageLogger(log, "Building visual track"):
            if not self._build_visual_track(clips, total_audio, work_dir, silent_video):
                log.error("Visual track failed")
                return False

        # ── Stage 3: Concat narration ──────────────────────────────────────────
        voice_track = os.path.join(work_dir, "voice.aac")
        with StageLogger(log, "Concatenating narration"):
            audio_paths = [s["audio_path"] for s in valid_segs]
            if not self.narrate.concat_segments(audio_paths, voice_track):
                log.error("Audio concat failed")
                return False

        # ── Stage 4: Subtitles ─────────────────────────────────────────────────
        with StageLogger(log, "Rendering subtitles"):
            frames = self.text.build_all_subtitle_frames(valid_segs, work_dir)

        subbed_video = os.path.join(work_dir, "subbed.mp4")
        with StageLogger(log, "Overlaying subtitles"):
            self._overlay_subtitles(silent_video, frames, subbed_video)

        # ── Stage 5: Mix audio ─────────────────────────────────────────────────
        with StageLogger(log, "Mixing audio"):
            if not self.audio_e.mix_final(subbed_video, voice_track, final_output_path):
                log.error("Audio mix failed")
                return False

        size = os.path.getsize(final_output_path) / 1_000_000
        dur  = ffprobe_duration(final_output_path)
        log.info("Final video: %.1fMB, %.1fs → %s", size, dur, final_output_path)
        return True

    # ── Stage helpers ──────────────────────────────────────────────────────────

    def _build_clips(
        self,
        segments: List[dict],
        media: List[dict],
        clips_dir: str,
        topic: str,
        video_type: str,
    ) -> List[str]:
        clips: List[str] = []

        # Intro card (2.5s)
        intro = os.path.join(clips_dir, "c000_intro.mp4")
        if self.scene.build_intro_card(topic, video_type, 2.5, intro,
                                        self.text._font):
            clips.append(intro)
            log.info("✓ Intro card")

        media_idx = 0
        for i, seg in enumerate(segments):
            dur  = max(self.vc.min_segment_sec,
                       seg.get("actual_duration", seg.get("duration", 5.0)))
            clip = os.path.join(clips_dir, f"c{i+1:03d}.mp4")

            # Pick media item (cycle through all available)
            item = None
            for _ in range(max(len(media), 1)):
                if not media:
                    break
                candidate = media[media_idx % len(media)]
                media_idx += 1
                if candidate.get("local_path") and os.path.exists(
                    candidate["local_path"]
                ):
                    item = candidate
                    break

            ok = self.scene.build_clip(
                media_item=item,
                duration=dur,
                output_path=clip,
                effect_index=i,
                topic=topic,
                video_type=video_type,
                color_index=(i + 1) % 5,
            )
            if ok:
                clips.append(clip)
                log.info("✓ Clip %d: %.1fs", i + 1, dur)
            else:
                log.warning("✗ Clip %d failed", i + 1)

        return clips

    def _build_visual_track(
        self,
        clips: List[str],
        total_audio: float,
        work_dir: str,
        output_path: str,
    ) -> bool:
        """Concatenate clips and loop if shorter than audio."""
        first = os.path.join(work_dir, "vt_pass1.mp4")
        if not self._concat_clips(clips, first):
            return False

        dur = ffprobe_duration(first)
        log.info("Visual track: %.1fs vs audio %.1fs", dur, total_audio)

        if dur >= total_audio:
            shutil.copy(first, output_path)
            return True

        # Loop
        repeats = min(math.ceil(total_audio / max(dur, 0.1)), 10)
        log.info("Looping clips ×%d to cover audio", repeats)

        lst = "/tmp/_vt_loop.txt"
        with open(lst, "w", encoding="utf-8") as f:
            for _ in range(repeats):
                f.write(f"file '{first}'\n")

        looped = os.path.join(work_dir, "vt_looped.mp4")
        ok, _ = run_cmd([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", self.vc.preset, "-crf", str(self.vc.crf),
            looped,
        ], timeout=600, label="loop concat")

        if not ok:
            shutil.copy(first, output_path)
            return True

        ok2, _ = run_cmd([
            "ffmpeg", "-y", "-i", looped, "-t", str(total_audio),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", self.vc.preset, "-crf", str(self.vc.crf),
            output_path,
        ], timeout=300, label="trim to audio")
        return ok2 and file_ok(output_path)

    def _concat_clips(self, clips: List[str], output: str) -> bool:
        """Concatenate clips with xfade transitions, fall back to simple concat."""
        valid = [c for c in clips if file_ok(c)]
        if not valid:
            return False
        if len(valid) == 1:
            shutil.copy(valid[0], output)
            return True

        durations  = [ffprobe_duration(c) for c in valid]
        filt, out_label = build_xfade_filter(
            len(valid), durations, self.vc.transition_sec
        )

        cmd = ["ffmpeg", "-y"]
        for c in valid:
            cmd += ["-i", c]
        cmd += [
            "-filter_complex", filt,
            "-map", out_label,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", self.vc.preset, "-crf", str(self.vc.crf),
            output,
        ]
        ok, _ = run_cmd(cmd, timeout=600, label="xfade concat")

        if not ok or not file_ok(output):
            log.warning("xfade failed — simple concat fallback")
            return self._simple_concat(valid, output)
        return True

    def _simple_concat(self, clips: List[str], output: str) -> bool:
        lst = "/tmp/_clips_list.txt"
        with open(lst, "w", encoding="utf-8") as f:
            for c in clips:
                f.write(f"file '{c}'\n")
        ok, _ = run_cmd([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", self.vc.preset, "-crf", str(self.vc.crf),
            output,
        ], timeout=600, label="simple concat")
        return ok and file_ok(output)

    def _overlay_subtitles(
        self,
        silent_video: str,
        frames: List[dict],
        output_path: str,
    ) -> bool:
        """Overlay subtitle PNGs on the video using FFmpeg overlay filter."""
        if not frames:
            shutil.copy(silent_video, output_path)
            return True

        cmd = ["ffmpeg", "-y", "-i", silent_video]
        for sf in frames:
            cmd += ["-i", sf["image"]]

        parts: List[str] = []
        prev = "[0:v]"
        for idx, sf in enumerate(frames):
            out  = f"[ov{idx}]" if idx < len(frames) - 1 else "[vout]"
            en   = f"between(t,{sf['start']:.3f},{sf['end']:.3f})"
            sub  = self.cfg.subtitle
            x    = "(W-w)/2"
            y    = f"H-h-{sub.margin_bottom}"
            parts.append(
                f"{prev}[{idx+1}:v]overlay={x}:{y}:enable='{en}'{out}"
            )
            prev = out

        cmd += [
            "-filter_complex", ";".join(parts),
            "-map", "[vout]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", self.vc.preset, "-crf", str(self.vc.crf),
            output_path,
        ]
        ok, _ = run_cmd(cmd, timeout=600, label="subtitle overlay")

        if not ok or not file_ok(output_path):
            log.warning("Overlay failed — SRT fallback")
            return self._srt_fallback(silent_video, frames, output_path)
        return True

    def _srt_fallback(
        self, video: str, frames: List[dict], output: str
    ) -> bool:
        srt = "/tmp/_sub.srt"
        self.text.build_srt(frames, srt)  # frames have 'text' key

        font      = self.text._font or ""
        sub_cfg   = self.cfg.subtitle
        font_name = Path(font).stem if font else "DejaVuSans"
        font_dir  = str(Path(font).parent) if font else ""
        style = (
            f"FontSize={sub_cfg.font_size},"
            f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
            f"Outline=3,Shadow=1,Alignment=2,MarginV={sub_cfg.margin_bottom}"
        )
        if font:
            style = f"FontName={font_name}," + style

        srt_e = srt.replace(":", "\\:")
        vf = (
            f"subtitles='{srt_e}':fontsdir='{font_dir}':force_style='{style}'"
            if font_dir else
            f"subtitles='{srt_e}':force_style='{style}'"
        )
        ok, _ = run_cmd([
            "ffmpeg", "-y", "-i", video, "-vf", vf,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", self.vc.preset, "-crf", str(self.vc.crf),
            "-c:a", "copy", output,
        ], timeout=600, label="srt subtitle")

        if not ok:
            shutil.copy(video, output)
        return True
