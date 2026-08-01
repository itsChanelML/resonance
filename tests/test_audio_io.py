from unittest.mock import MagicMock

import numpy as np
import pytest
import sounddevice as sd

from shared.audio_io import SAMPLE_RATE, VoiceCapture


@pytest.fixture
def capture(monkeypatch):
    monkeypatch.setattr("shared.audio_io.WhisperModel", MagicMock())
    vc = VoiceCapture(max_seconds=0.1)  # 1600 frames at 16kHz
    vc._recording = True
    vc._chunks = []
    vc._frame_count = 0
    return vc


def test_callback_ignores_frames_when_not_recording(capture):
    capture._recording = False
    capture._callback(np.zeros((100, 1), dtype="int16"), 100, None, None)
    assert capture._chunks == []


def test_callback_accumulates_under_the_cap(capture):
    chunk = np.zeros((100, 1), dtype="int16")
    capture._callback(chunk, 100, None, None)
    assert capture._frame_count == 100
    assert capture._truncated is False
    assert capture._recording is True


def test_callback_stops_stream_at_the_cap(capture):
    chunk = np.zeros((1600, 1), dtype="int16")  # exactly the 0.1s cap at 16kHz
    with pytest.raises(sd.CallbackStop):
        capture._callback(chunk, 1600, None, None)
    assert capture._truncated is True
    assert capture._recording is False


def test_stop_and_transcribe_returns_empty_string_with_no_audio(capture):
    capture._recording = True
    assert capture.stop_and_transcribe() == ""
