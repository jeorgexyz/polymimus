import queue
import tempfile
import wave
from collections import deque
from pathlib import Path

import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)  # 480 samples per frame
MIN_SEGMENT_BYTES = int(SAMPLE_RATE * 0.3 * 2)  # discard segments shorter than 0.3s


class MicTranscriber:
    def __init__(
        self,
        model: WhisperModel,
        aggressiveness: int = 2,
        silence_duration: float = 0.6,
        translate: bool = False,
    ):
        self.model = model
        self.vad = webrtcvad.Vad(aggressiveness)
        # how many consecutive silence frames before we cut a segment
        self.silence_frames = max(1, int(silence_duration * 1000 / FRAME_MS))
        self.translate = translate
        self._audio_q: queue.Queue[bytes] = queue.Queue()
        self._running = False

    def _audio_callback(self, indata, frames, time_info, status):
        self._audio_q.put(bytes(indata))

    def _transcribe_segment(self, pcm: bytes) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = Path(f.name)

        with wave.open(str(tmp), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm)

        task = "translate" if self.translate else "transcribe"
        segments, _ = self.model.transcribe(str(tmp), task=task)
        text = " ".join(s.text.strip() for s in segments)
        tmp.unlink(missing_ok=True)
        return text

    def run(self, on_transcript) -> None:
        self._running = True

        # ring buffer doubles as pre-speech padding and end-of-speech detector
        ring: deque[tuple[bytes, bool]] = deque(maxlen=self.silence_frames)
        speech: list[bytes] = []
        triggered = False

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

                is_speech = self.vad.is_speech(frame, SAMPLE_RATE)

                if not triggered:
                    ring.append((frame, is_speech))
                    if sum(1 for _, v in ring if v) / ring.maxlen > 0.75:
                        triggered = True
                        speech.extend(f for f, _ in ring)
                        ring.clear()
                else:
                    speech.append(frame)
                    ring.append((frame, is_speech))
                    if sum(1 for _, v in ring if not v) / ring.maxlen > 0.75:
                        triggered = False
                        audio = b"".join(speech)
                        if len(audio) >= MIN_SEGMENT_BYTES:
                            text = self._transcribe_segment(audio)
                            if text.strip():
                                on_transcript(text)
                        speech.clear()
                        ring.clear()

    def stop(self) -> None:
        self._running = False
