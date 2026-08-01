from unittest.mock import MagicMock, patch

import pytest
import requests

from shared.voice_out import ElevenLabsVoice


def _stream_response(ok=True, chunks=(b"abc", b"def"), status_code=200, detail_message=None):
    resp = MagicMock()
    resp.ok = ok
    resp.status_code = status_code
    resp.iter_content.return_value = iter(chunks)
    resp.text = detail_message or ""
    resp.json.return_value = {"detail": {"message": detail_message}} if detail_message else {}
    return resp


def _fake_player(exit_code=0):
    player = MagicMock()
    player.stdin = MagicMock()
    player.wait.return_value = exit_code
    player.poll.return_value = None
    return player


def test_speak_streams_chunks_into_player():
    voice = ElevenLabsVoice(api_key="key")
    player = _fake_player(exit_code=0)
    with patch("shared.voice_out.requests.post", return_value=_stream_response()):
        with patch("shared.voice_out.subprocess.Popen", return_value=player) as mock_popen:
            voice.speak("hello there")
    mock_popen.assert_called_once()
    assert player.stdin.write.call_count == 2
    player.stdin.close.assert_called_once()


def test_speak_raises_with_vendor_message_on_error():
    voice = ElevenLabsVoice(api_key="key")
    resp = _stream_response(ok=False, status_code=402, detail_message="quota exceeded")
    with patch("shared.voice_out.requests.post", return_value=resp):
        with patch("shared.voice_out.subprocess.Popen") as mock_popen:
            with pytest.raises(requests.exceptions.HTTPError, match="quota exceeded"):
                voice.speak("hello")
    mock_popen.assert_not_called()


def test_speak_warns_on_nonzero_exit(capsys):
    voice = ElevenLabsVoice(api_key="key")
    player = _fake_player(exit_code=1)
    with patch("shared.voice_out.requests.post", return_value=_stream_response()):
        with patch("shared.voice_out.subprocess.Popen", return_value=player):
            voice.speak("hello")
    assert "Playback may have failed" in capsys.readouterr().out


def test_speak_records_usage():
    usage = MagicMock()
    usage.record_elevenlabs_characters.return_value = None
    voice = ElevenLabsVoice(api_key="key", usage=usage)
    with patch("shared.voice_out.requests.post", return_value=_stream_response()):
        with patch("shared.voice_out.subprocess.Popen", return_value=_fake_player()):
            voice.speak("hello there")
    usage.record_elevenlabs_characters.assert_called_once_with(len("hello there"))


def test_stop_terminates_active_process_and_suppresses_warning(capsys):
    voice = ElevenLabsVoice(api_key="key")
    fake_process = MagicMock()
    fake_process.poll.return_value = None
    voice._process = fake_process

    voice.stop()

    fake_process.terminate.assert_called_once()
    assert voice._interrupted is True


def test_stop_is_noop_when_nothing_is_playing():
    voice = ElevenLabsVoice(api_key="key")
    voice.stop()  # should not raise
    assert voice._interrupted is True
