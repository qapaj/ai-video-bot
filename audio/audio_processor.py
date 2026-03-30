"""
audio/audio_processor.py

Audio post-processing pipeline.

Operations applied to every synthesised audio file:
  1. Silence trimming  — removes leading/trailing silence
  2. Loudness normalisation  — targets -16 LUFS (broadcast standard)
  3. Sample rate standardisation  — 44100 Hz mono
  4. Light dynamic compression  — evens out volume peaks

All operations use FFmpeg (always available on GitHub Actions runners).
No heavy Python audio libraries required.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from utils.helpers import ffprobe_duration, file_ok, run_cmd
from utils.logger import get_logger

log = get_logger("audio.processor")

# Target loudness in LUFS (EBU R128 broadcast standard)
TARGET_LUFS   = -16
TARGET_TP     = -1.5   # true peak dBFS
TARGET_LRA    = 11     # loudness range

# Silence detection thresholds
SILENCE_DB    = -50    # dB below which is considered silence
SILENCE_DUR   = 0.15   # seconds of silence before trimming

# Output format
OUT_SAMPLE_RATE = 44100
OUT_CHANNELS    = 1
OUT_CODEC       = "aac"
OUT_BITRATE     = "128k"

# Minimum valid output duration in seconds
MIN_DURATION    = 0.5


class AudioProcessor:
    """
    Applies a fixed post-processing chain to raw TTS audio.
    Designed to be lightweight: no imports beyond subprocess/FFmpeg.
    """

    def process(self, input_path: str, output_path: str) -> bool:
        """
        Run the full post-processing chain.

        Args:
            input_path:  Path to raw audio (any FFmpeg-supported format)
            output_path: Path to write processed AAC audio

        Returns:
            True on success, False on any failure.
            On failure, output_path is NOT written — the caller
            can decide to use the original unprocessed file.
        """
        if not file_ok(input_path, 100):
            log.error("process: input missing or empty: %s", input_path)
            return False

        tmp = output_path + "_proc_tmp.aac"

        # Build the FFmpeg audio filter chain
        af = self._build_filter_chain()

        ok, err = run_cmd([
            "ffmpeg", "-y",
            "-i",    input_path,
            "-af",   af,
            "-ar",   str(OUT_SAMPLE_RATE),
            "-ac",   str(OUT_CHANNELS),
            "-c:a",  OUT_CODEC,
            "-b:a",  OUT_BITRATE,
            tmp,
        ], timeout=60, label="audio post-process")

        if not ok or not file_ok(tmp, 100):
            log.warning("Post-process failed (using raw audio): %s", err[:150])
            try: os.remove(tmp)
            except: pass
            return False

        # Sanity check: processed file should have valid duration
        dur = ffprobe_duration(tmp)
        if dur < MIN_DURATION:
            log.warning("Processed audio too short (%.2fs) — using raw", dur)
            try: os.remove(tmp)
            except: pass
            return False

        os.rename(tmp, output_path)
        log.debug("Post-processed: %.2fs → %s", dur, Path(output_path).name)
        return True

    def process_or_copy(self, input_path: str, output_path: str) -> str:
        """
        Process audio if possible, otherwise copy the raw file.
        Always returns a valid output path.
        """
        if self.process(input_path, output_path):
            return output_path
        # Fallback: convert to standard format without processing
        ok, _ = run_cmd([
            "ffmpeg", "-y", "-i", input_path,
            "-ar", str(OUT_SAMPLE_RATE),
            "-ac", str(OUT_CHANNELS),
            "-c:a", OUT_CODEC, "-b:a", OUT_BITRATE,
            output_path,
        ], timeout=30)
        if ok and file_ok(output_path, 100):
            return output_path
        # Last resort: raw copy
        shutil.copy2(input_path, output_path)
        return output_path

    def get_duration(self, path: str) -> float:
        """Return audio duration in seconds. Returns 0 on error."""
        return ffprobe_duration(path)

    def concat(self, input_paths: list[str], output_path: str) -> bool:
        """
        Concatenate multiple audio files into one.
        Re-encodes all inputs to a uniform format before merging.
        Returns True on success.
        """
        valid = [p for p in input_paths if file_ok(p, 100)]
        if not valid:
            log.error("concat: no valid input files")
            return False

        if len(valid) == 1:
            return self._reencode(valid[0], output_path)

        # Step 1: Convert all inputs to uniform WAV
        wavs: list[str] = []
        for i, src in enumerate(valid):
            wav = f"/tmp/_proc_concat_{i:04d}.wav"
            ok, err = run_cmd([
                "ffmpeg", "-y", "-i", src,
                "-ar", str(OUT_SAMPLE_RATE),
                "-ac", str(OUT_CHANNELS),
                wav,
            ], timeout=60)
            if ok and file_ok(wav, 100):
                wavs.append(wav)
            else:
                log.warning("concat: segment %d failed to convert: %s", i, err[:100])

        if not wavs:
            log.error("concat: all segments failed WAV conversion")
            return False

        # Step 2: Create concat list
        lst = "/tmp/_proc_concat_list.txt"
        with open(lst, "w", encoding="utf-8") as f:
            for w in wavs:
                f.write(f"file '{w}'\n")

        # Step 3: Concat WAVs (lossless copy, same format)
        combined = "/tmp/_proc_concat_combined.wav"
        ok, err = run_cmd([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", lst,
            "-c", "copy",
            combined,
        ], timeout=120, label="concat WAVs")

        if not ok or not file_ok(combined, 100):
            log.error("concat: WAV concat failed: %s", err[:150])
            _cleanup(wavs + [combined])
            return False

        # Step 4: Encode to AAC
        ok, err = run_cmd([
            "ffmpeg", "-y", "-i", combined,
            "-ar", str(OUT_SAMPLE_RATE),
            "-ac", str(OUT_CHANNELS),
            "-c:a", OUT_CODEC, "-b:a", OUT_BITRATE,
            output_path,
        ], timeout=60, label="concat → AAC")

        _cleanup(wavs + [combined])

        if not ok:
            log.error("concat: AAC encoding failed: %s", err[:150])
        return ok and file_ok(output_path, 100)

    # ── Private ────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_filter_chain() -> str:
        """
        Build the FFmpeg -af filter string.

        Chain:
          silenceremove → loudnorm → acompressor

        silenceremove: strip leading/trailing silence
        loudnorm: EBU R128 loudness normalisation
        acompressor: light compression to even out volume
        """
        return (
            # Trim leading silence
            f"silenceremove="
            f"start_periods=1:"
            f"start_threshold={SILENCE_DB}dB:"
            f"start_duration={SILENCE_DUR},"
            # Trim trailing silence
            f"silenceremove="
            f"stop_periods=-1:"
            f"stop_threshold={SILENCE_DB}dB:"
            f"stop_duration={SILENCE_DUR},"
            # Loudness normalisation (two-pass in one filter)
            f"loudnorm="
            f"I={TARGET_LUFS}:"
            f"TP={TARGET_TP}:"
            f"LRA={TARGET_LRA},"
            # Light dynamic compression
            f"acompressor="
            f"threshold=0.5:"
            f"ratio=3:"
            f"attack=200:"
            f"release=1000:"
            f"makeup=1"
        )

    @staticmethod
    def _reencode(src: str, dst: str) -> bool:
        ok, _ = run_cmd([
            "ffmpeg", "-y", "-i", src,
            "-ar", str(OUT_SAMPLE_RATE),
            "-ac", str(OUT_CHANNELS),
            "-c:a", OUT_CODEC, "-b:a", OUT_BITRATE,
            dst,
        ], timeout=30)
        return ok and file_ok(dst, 100)


def _cleanup(paths: list[str]) -> None:
    for p in paths:
        try: os.remove(p)
        except: pass
