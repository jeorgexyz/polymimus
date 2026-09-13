<p align="center">
<img width="350px" src="./assets/mockingbird.png"
alt="mockingbird">
</p>

# polymimus

[![CI](https://github.com/jeorgexyz/polymimus/actions/workflows/ci.yml/badge.svg)](https://github.com/jeorgexyz/polymimus/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white)](#installation)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Runs locally](https://img.shields.io/badge/runs-local%20only-2ea44f)](#overview)
[![Powered by faster-whisper](https://img.shields.io/badge/powered%20by-faster--whisper-orange)](https://github.com/SYSTRAN/faster-whisper)

Offline multilingual speech transcription CLI powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Runs entirely on your machine, with no API key required and no internet needed after the first model download.

## Overview

polymimus transcribes audio files in 99 languages and can optionally translate them to English. It uses OpenAI's Whisper model via the faster-whisper backend, a CTranslate2 port that is faster and more memory-efficient than the original implementation.

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

# Force a language instead of auto-detecting
polymimus transcribe audio.mp3 --language es

# Translate to English
polymimus transcribe audio.mp3 --translate

# Skip long stretches of non-speech (music, silence)
polymimus transcribe audio.mp3 --vad-filter

# Save as plain text, SRT, or WebVTT subtitles
polymimus transcribe audio.mp3 --output transcript.txt
polymimus transcribe audio.mp3 --output subtitles.srt
polymimus transcribe audio.mp3 --output subtitles.vtt
```

**Live microphone transcription:**
```bash
# Listen and transcribe in real-time
polymimus listen

# With translation and a more accurate model
polymimus listen --model small --translate

# Save the session transcript as it goes
polymimus listen --output session.txt

# Tune VAD sensitivity (0=permissive, 3=strict) and silence cutoff
polymimus listen --aggressiveness 3 --silence 0.8
```

**GPU / precision:** both commands accept `--device` (`auto`, `cpu`, `cuda`) and `--compute-type` (`int8`, `int8_float16`, `float16`, `float32`). The default `int8` on CPU is a good balance; on an NVIDIA GPU try `--device cuda --compute-type float16`.

**Model sizes** (accuracy vs. speed tradeoff):

| Model  | Size  | Notes                        |
|--------|-------|------------------------------|
| tiny   | 75 MB | Fastest, least accurate      |
| base   | 145 MB| Good default                 |
| small  | 465 MB| Better accuracy              |
| medium | 1.5 GB| High accuracy                |
| large  | 3 GB  | Best accuracy, slowest       |

Models are downloaded automatically on first use and cached locally.

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

## License

MIT. See [LICENSE](./LICENSE).

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - the underlying speech recognition model
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - efficient CTranslate2 inference
