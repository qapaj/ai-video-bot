"""
audio/narration_generator.py

Thin adapter that routes calls from the rest of the pipeline into AudioEngine.
Preserves the existing interface so no other module needs to change.
"""

from __future__ import annotations

from typing import List

from audio.audio_engine import AudioEngine
from utils.logger import get_logger

log = get_logger("audio.narration")


class NarrationGenerator:
    """Wraps AudioEngine with the legacy generate_all / concat_segments interface."""

    def __init__(self) -> None:
        self._engine = AudioEngine()

    def generate_all(
        self,
        segments:   List[dict],
        output_dir: str,
        language:   str = "ar",
        gender:     str = "female",
    ) -> List[dict]:
        """Synthesise audio for every segment. Returns segments with audio_path added."""
        return self._engine.generate_all(segments, output_dir, language, gender)

    def concat_segments(self, audio_paths: List[str], output_path: str) -> bool:
        """Merge segment audio into a single track."""
        return self._engine.concat(audio_paths, output_path)
