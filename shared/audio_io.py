"""Local audio capture and transcription. Fully local, no API cost, no
rate limit, since this runs on-device via faster-whisper."""

import logging
import os
import tempfile
import wave

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

logger = logging.getLogger("resonance.audio")

SAMPLE_RATE = 16000
MIN_RECORDING_SECONDS = 0.3
MAX_RECORDING_SECONDS = 30  # a stuck/held key shouldn't record indefinitely


class VoiceCapture:
    def __init__(self, model_size: str = "base.en", max_seconds: float = MAX_RECORDING_SECONDS):
        logger.info("Loading local Whisper model (%s)...", model_size)
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.max_seconds = max_seconds
        self._chunks: list[np.ndarray] = []
        self._recording = False
        self._stream = None
        self._frame_count = 0
        self._truncated = False

    def _callback(self, indata, frames, time_info, status):
        if not self._recording:
            return
        self._chunks.append(indata.copy())
        self._frame_count += frames
        if self._frame_count >= SAMPLE_RATE * self.max_seconds:
            self._recording = False
            self._truncated = True
            raise sd.CallbackStop()

    def start(self) -> None:
        self._chunks = []
        self._frame_count = 0
        self._truncated = False
        self._recording = True
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=self._callback
        )
        self._stream.start()
        logger.info("Listening...")

    def stop_and_transcribe(self) -> str:
        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()

        if self._truncated:
            logger.warning("Recording hit the %ss cap, stopped automatically.", self.max_seconds)

        if not self._chunks:
            return ""

        audio = np.concatenate(self._chunks, axis=0)
        if len(audio) < SAMPLE_RATE * MIN_RECORDING_SECONDS:
            logger.info("Recording too short, ignored.")
            return ""

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            self._write_wav(f.name, audio)
            path = f.name

        try:
            segments, _ = self.model.transcribe(path, language="en")
            return " ".join(seg.text for seg in segments).strip()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass  # best-effort cleanup, not worth failing the turn over

    @staticmethod
    def _write_wav(path: str, audio: np.ndarray) -> None:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio.tobytes())
