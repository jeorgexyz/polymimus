<P align="center">
<img width="350px" src="./assets/mockingbird.png"
alt="mockingbird">
</p>

# PolyMimus

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white)](#installation)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Runs locally](https://img.shields.io/badge/runs-local%20only-2ea44f)](#overview)
[![Powered by faster-whisper](https://img.shields.io/badge/powered%20by-faster--whisper-orange)](https://github.com/SYSTRAN/faster-whisper)

Local multilingual speech transcription powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Runs entirely on your machine — no API key, no internet required after the first model download.

## Overview

PolyMimus transcribes audio files in 99 languages and can optionally translate them to English. It uses OpenAI's Whisper model via the faster-whisper backend (a CTranslate2 port that is ~4x faster and uses less memory than the original).

## Installation

Requires Python 3.9+ and [ffmpeg](https://ffmpeg.org/download.html) on your PATH.

```bash
pip install -e .
```

## Usage

**Transcribe a file:**
```bash
# Auto-detects language
polymimus transcribe audio.mp3

# Use a larger model for better accuracy
polymimus transcribe audio.mp3 --model medium

# Translate to English
polymimus transcribe audio.mp3 --translate

# Save as plain text or SRT subtitles
polymimus transcribe audio.mp3 --output transcript.txt
polymimus transcribe audio.mp3 --output subtitles.srt
```

**Live microphone transcription:**
```bash
# Listen and transcribe in real-time
polymimus listen

# With translation and a more accurate model
polymimus listen --model small --translate

# Tune VAD sensitivity (0=permissive, 3=strict) and silence cutoff
polymimus listen --aggressiveness 3 --silence 0.8
```

**Model sizes** (accuracy vs. speed tradeoff):

| Model  | Size  | Notes                        |
|--------|-------|------------------------------|
| tiny   | 75 MB | Fastest, least accurate      |
| base   | 145 MB| Good default                 |
| small  | 465 MB| Better accuracy              |
| medium | 1.5 GB| High accuracy                |
| large  | 3 GB  | Best accuracy, slowest       |

Models are downloaded automatically on first use and cached locally.

## License

MIT. See [LICENSE](./LICENSE).

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) — the underlying speech recognition model
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — efficient CTranslate2 inference
