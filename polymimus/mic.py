import queue
import threading
from collections import deque
from typing import Callable, Optional

import numpy as np
import webrtcvad
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480 samples per frame
MIN_SEGMENT_BYTES = int(SAMPLE_RATE * 0.3 * 2)  # discard segments shorter than 0.3s


class SegmentAssembler:
    """Turns a stream of (frame, is_speech) pairs into complete speech segments.

    A segment opens when more than TRIGGER_RATIO of the last `silence_frames`
    frames are speech, and closes when more than TRIGGER_RATIO are silence.
    The ring buffer doubles as pre-speech padding, so the start of an
    utterance is not clipped.
    """

    TRIGGER_RATIO = 0.75

    def __init__(self, silence_frames: int):
        self._ring: deque[tuple[bytes, bool]] = deque(maxlen=max(1, silence_frames))
        self._speech: list[bytes] = []
        self._triggered = False

    def feed(self, frame: bytes, is_speech: bool) -> Optional[bytes]:
        """Feed one frame; returns a completed segment or None."""
        ring = self._ring
        ring.append((frame, is_speech))

        if not self._triggered:
            if sum(1 for _, v in ring if v) / ring.maxlen > self.TRIGGER_RATIO:
                self._triggered = True
                self._speech.extend(f for f, _ in ring)
                ring.clear()
            return None

        self._speech.append(frame)
        if sum(1 for _, v in ring if not v) / ring.maxlen > self.TRIGGER_RATIO:
            self._triggered = False
            segment = b"".join(self._speech)
            self._speech.clear()
            ring.clear()
            return segment
        return None


class MicTranscriber:
    def __init__(
        self,
        model: WhisperModel,
        aggressiveness: int = 2,
        silence_duration: float = 0.6,
        translate: bool = False,
        language: Optional[str] = None,
    ):
        self.model = model
        self.vad = webrtcvad.Vad(aggressiveness)
        # how many consecutive silence frames before we cut a segment
        self.silence_frames = max(1, int(silence_duration * 1000 / FRAME_MS))
        self.translate = translate
        self.language = language
        self._audio_q: queue.Queue[bytes] = queue.Queue()
        self._segment_q: queue.Queue[Optional[bytes]] = queue.Queue()
        self._running = False

    def _audio_callback(self, indata, frames, time_info, status):
        self._audio_q.put(bytes(indata))

    def _transcribe_segment(self, pcm: bytes) -> str:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        task = "translate" if self.translate else "transcribe"
        segments, _ = self.model.transcribe(audio, task=task, language=self.language)
        return " ".join(s.text.strip() for s in segments)

    def _worker(self, on_transcript: Callable[[str], None]) -> None:
        while True:
            pcm = self._segment_q.get()
            if pcm is None:
                return
            text = self._transcribe_segment(pcm)
            if text.strip():
                on_transcript(text)

    def run(self, on_transcript: Callable[[str], None]) -> None:
        # imported here so the rest of the CLI works on machines without PortAudio
        import sounddevice as sd

        self._running = True
        assembler = SegmentAssembler(self.silence_frames)

        # transcription happens off-thread so a slow model never stalls
        # the audio loop and piles up latency
        worker = threading.Thread(target=self._worker, args=(on_transcript,), daemon=True)
        worker.start()

        try:
            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                blocksize=FRAME_SAMPLES,
                dtype="int16",
                channels=1,
                callback=self._audio_callback,
            ):
                while self._running:
                    try:
                        frame = self._audio_q.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    segment = assembler.feed(frame, self.vad.is_speech(frame, SAMPLE_RATE))
                    if segment is not None and len(segment) >= MIN_SEGMENT_BYTES:
                        self._segment_q.put(segment)
        finally:
            # flush pending segments, then let the worker exit
            self._segment_q.put(None)
            worker.join()

    def stop(self) -> None:
        self._running = False
