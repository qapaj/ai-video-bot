# 🎬 AI Video Bot — Production v5

> Fully automated professional short-video factory.  
> Runs 100% free on GitHub Actions. Delivered via Telegram.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Telegram User                            │
│   /generate  /news  /story  /facts  /education  + settings     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│              bot/telegram_bot.py  (bot_listener.yml)            │
│  • Inline keyboard settings (language/type/voice/quality)       │
│  • 3-step /generate wizard                                      │
│  • Progress notifications                                       │
│  • Job queue management                                         │
└──────┬──────────────────────────────────────┬───────────────────┘
       │ workflow_dispatch                     │ status updates
┌──────▼──────────────────────────────────────▼───────────────────┐
│                   video_pipeline.yml                            │
│              (GitHub Actions — free compute)                    │
└──────┬──────────────────────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────────────────┐
│                    main.py pipeline                             │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │model/       │  │audio/        │  │video/                 │  │
│  │script_engine│  │narration_    │  │video_engine.py        │  │
│  │             │  │generator.py  │  │scene_builder.py       │  │
│  │Templates +  │  │              │  │text_renderer.py       │  │
│  │HF NLP opt.  │  │edge-tts ───► │  │transitions.py        │  │
│  └─────────────┘  │MMS-TTS API   │  └───────────────────────┘  │
│                   │espeak-ng     │                              │
│  ┌─────────────┐  └──────────────┘  ┌───────────────────────┐  │
│  │media/       │                    │hf/                    │  │
│  │media_fetcher│                    │hf_model_loader.py     │  │
│  │             │                    │hf_audio_models.py     │  │
│  │Pexels API   │                    │                       │  │
│  │Pixabay API  │                    │XTTS-v2 / MMS-TTS API  │  │
│  │Wikimedia    │                    │HF text improvement    │  │
│  └─────────────┘                    └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                          │
                   Final MP4 video
                   1080×1920 vertical
                   sent to Telegram
```

---

## Quick Setup (5 minutes)

### 1. Fork this repository

### 2. Add GitHub Secrets

`Settings → Secrets and variables → Actions → New repository secret`

| Secret | Value |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Your bot token from @BotFather |
| `PEXELS_API_KEY` | Free at pexels.com/api |
| `PIXABAY_API_KEY` | Free at pixabay.com/api |
| `HF_TOKEN` | Optional — enables XTTS-v2 and MMS-TTS |

### 3. Set Actions permissions

`Settings → Actions → General → Workflow permissions → Read and write`

### 4. Enable workflows

`Actions tab → Enable workflows`

### 5. Test the bot

Send to your bot: `/news artificial intelligence`

---

## Bot Commands

| Command | Description |
|---|---|
| `/generate` | Launch 3-step video wizard |
| `/news topic` | Quick news video |
| `/story topic` | Quick story video |
| `/facts topic` | Quick facts video |
| `/education topic` | Quick educational video |
| `/status` | Check job status |
| `/cancel` | Cancel current job |
| `/settings` | Change language, type, voice, quality |
| `/help` | Full help |

---

## Module Reference

```
main.py                     Entry points (bot / pipeline / test)
├── model/
│   └── script_engine.py    Multilingual script generation + HF improvement
├── audio/
│   ├── narration_generator.py  TTS with caching (edge-tts → MMS → espeak)
│   └── audio_engine.py     Final audio mixing + background music
├── video/
│   ├── video_engine.py     Main assembly orchestrator
│   ├── scene_builder.py    Image/video → clip (Ken Burns, gradient cards)
│   ├── text_renderer.py    Arabic subtitle rendering (ImageMagick + bidi)
│   └── transitions.py      FFmpeg xfade filter builders
├── bot/
│   ├── telegram_bot.py     Full bot with inline keyboards
│   ├── job_queue.py        Sequential job queue
│   ├── job_runner.py       Pipeline executor per job
│   └── uploader.py         Telegram video uploader + compression
├── hf/
│   ├── hf_model_loader.py  HuggingFace Inference API client
│   └── hf_audio_models.py  XTTS-v2 + MMS-TTS backends
├── media/
│   └── media_fetcher.py    Pexels + Pixabay + Wikimedia downloader
├── config/
│   └── arabic_utils.py     Arabic reshaping + bidi utilities
└── utils/
    ├── config.py           Central configuration (env vars)
    ├── helpers.py          Subprocess, file, text, network utilities
    └── logger.py           Structured logging with StageLogger
```

---

## Output Specs

| Property | Value |
|---|---|
| Resolution | 1080 × 1920 (vertical) |
| Format | MP4 (H.264 + AAC) |
| Duration | 30–60 seconds |
| Subtitles | Burned-in, Arabic-shaped, 42px |
| Platforms | TikTok, Instagram Reels, YouTube Shorts |

---

## Languages Supported

Arabic 🇸🇦 · English 🇺🇸 · French 🇫🇷 · Spanish 🇪🇸 · German 🇩🇪

---

## Free Tier Budget

| Resource | Free allowance | Usage |
|---|---|---|
| GitHub Actions | 2,000 min/month | ~8 min/video → **250 videos/month** |
| Pexels API | Unlimited | Media |
| Pixabay API | Unlimited | Media |
| edge-tts | Unlimited | TTS |
| HF Inference API | Free tier | Optional enhancement |
