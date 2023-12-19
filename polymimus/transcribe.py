from dataclasses import dataclass
from pathlib import Path

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


def load_model(model_size: str = "base") -> WhisperModel:
    return WhisperModel(model_size, device="auto", compute_type="int8")


def transcribe(
    model: WhisperModel,
    audio_path: Path,
    translate: bool = False,
) -> TranscriptionResult:
    task = "translate" if translate else "transcribe"
    segments_iter, info = model.transcribe(str(audio_path), task=task)

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
