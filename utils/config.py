"""
utils/config.py
Single source of truth for all configuration.
Values are read from environment variables with sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class VideoConfig:
    width: int = 1080
    height: int = 1920
    fps: int = 30
    crf: int = 23
    preset: str = "fast"
    format: str = "mp4"
    max_duration_sec: int = 60
    min_segment_sec: float = 3.0
    transition_sec: float = 0.4
    ken_burns_zoom: float = 1.25
    ken_burns_speed: float = 0.001


@dataclass
class AudioConfig:
    sample_rate: int = 44100
    channels: int = 1
    bitrate: str = "128k"
    music_volume: float = 0.07
    voice_volume: float = 1.0
    primary_voice: str = "ar-EG-SalmaNeural"
    fallback_voices: List[str] = field(default_factory=lambda: [
        "ar-SA-ZariyahNeural",
        "ar-SA-HamedNeural",
        "ar-AE-FatimaNeural",
    ])
    tts_rate: str = "-5%"


@dataclass
class SubtitleConfig:
    font_size: int = 42
    margin_bottom: int = 130
    max_chars_per_line: int = 22
    outline_width: int = 5
    text_color: str = "white"
    outline_color: str = "black"
    shadow_opacity: float = 0.85


@dataclass
class HFConfig:
    token: str = ""
    text_model: str = "google/flan-t5-base"
    tts_model: str = "microsoft/speecht5_tts"
    use_hf_tts: bool = False
    use_hf_text: bool = False
    cache_dir: str = "/tmp/hf_cache"
    timeout_sec: int = 90


@dataclass
class TelegramConfig:
    token: str = ""
    max_video_mb: int = 48
    upload_timeout_sec: int = 300
    poll_window_sec: int = 265


@dataclass
class PathConfig:
    work_dir: Path = Path("/tmp/pipeline_work")
    output_dir: Path = Path("output")
    assets_dir: Path = Path("assets")
    fonts_dir: Path = Path("assets/fonts")
    music_dir: Path = Path("assets/music")
    cache_dir: Path = Path("/tmp/pipeline_cache")


@dataclass
class AppConfig:
    video: VideoConfig = field(default_factory=VideoConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    subtitle: SubtitleConfig = field(default_factory=SubtitleConfig)
    hf: HFConfig = field(default_factory=HFConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    language: str = "ar"
    voice_gender: str = "female"
    video_type: str = "news"
    quality: str = "standard"
    debug: bool = False


def load_config() -> AppConfig:
    """Build AppConfig from environment variables."""
    cfg = AppConfig()

    # Telegram
    cfg.telegram.token = os.environ.get("TELEGRAM_BOT_TOKEN", "")

    # HuggingFace
    cfg.hf.token = os.environ.get("HF_TOKEN", "")
    cfg.hf.use_hf_tts = os.environ.get("USE_HF_TTS", "0") == "1"
    cfg.hf.use_hf_text = os.environ.get("USE_HF_TEXT", "0") == "1"

    # Pipeline inputs
    cfg.language    = os.environ.get("VIDEO_LANGUAGE", "ar")
    cfg.voice_gender= os.environ.get("VOICE_GENDER", "female")
    cfg.video_type  = os.environ.get("VIDEO_TYPE", "news")
    cfg.quality     = os.environ.get("VIDEO_QUALITY", "standard")
    cfg.debug       = os.environ.get("DEBUG", "0") == "1"

    # Quality adjustments
    if cfg.quality == "high":
        cfg.video.crf     = 20
        cfg.video.preset  = "medium"
        cfg.audio.bitrate = "192k"

    # Paths
    cfg.paths.work_dir  = Path(os.environ.get("WORK_DIR", "/tmp/pipeline_work"))
    cfg.paths.output_dir = Path(os.environ.get("OUTPUT_DIR", "output"))

    # Ensure dirs exist
    for p in (cfg.paths.work_dir, cfg.paths.output_dir, cfg.paths.cache_dir):
        p.mkdir(parents=True, exist_ok=True)

    return cfg


# Module-level singleton
_config: AppConfig | None = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config
