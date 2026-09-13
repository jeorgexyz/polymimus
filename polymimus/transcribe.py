from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    language: str
    language_probability: float
    segments: list[Segment]
    duration: float


def load_model(
    model_size: str = "base",
    device: str = "auto",
    compute_type: str = "int8",
) -> WhisperModel:
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe(
    model: WhisperModel,
    audio_path: Path,
    translate: bool = False,
    language: Optional[str] = None,
    vad_filter: bool = False,
) -> TranscriptionResult:
    task = "translate" if translate else "transcribe"
    segments_iter, info = model.transcribe(
        str(audio_path),
        task=task,
        language=language,
        vad_filter=vad_filter,
    )

    segments = [
        Segment(start=s.start, end=s.end, text=s.text.strip())
        for s in segments_iter
    ]

    return TranscriptionResult(
        language=info.language,
        language_probability=info.language_probability,
        segments=segments,
        duration=info.duration,
    )
