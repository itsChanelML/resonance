from unittest.mock import MagicMock, patch

import pytest
import requests

from shared.llm_client import NimClient, NimConfig


def _response(status_code=200, content="pong"):
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(f"{status_code} error")
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    # tenacity's wait_exponential otherwise makes retry tests slow for real.
    monkeypatch.setattr("time.sleep", lambda seconds: None)


def test_chat_returns_reply_content():
    client = NimClient(NimConfig(api_key="key"))
    with patch("shared.llm_client.requests.post", return_value=_response(content="hi there")):
        reply = client.chat([{"role": "user", "content": "hello"}])
    assert reply == "hi there"


def test_chat_retries_then_succeeds():
    client = NimClient(NimConfig(api_key="key"))
    calls = [requests.exceptions.ConnectionError("boom"), _response(content="ok")]

    def side_effect(*args, **kwargs):
        result = calls.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    with patch("shared.llm_client.requests.post", side_effect=side_effect) as mock_post:
        reply = client.chat([{"role": "user", "content": "hello"}])
    assert reply == "ok"
    assert mock_post.call_count == 2


def test_chat_reraises_after_exhausting_retries():
    client = NimClient(NimConfig(api_key="key"))
    with patch("shared.llm_client.requests.post", side_effect=requests.exceptions.ConnectionError("down")) as mock_post:
        with pytest.raises(requests.exceptions.ConnectionError):
            client.chat([{"role": "user", "content": "hello"}])
    assert mock_post.call_count == 5  # stop_after_attempt(5)


def test_chat_records_usage_on_success():
    usage = MagicMock()
    usage.record_nim_request.return_value = None
    client = NimClient(NimConfig(api_key="key"), usage=usage)
    with patch("shared.llm_client.requests.post", return_value=_response()):
        client.chat([{"role": "user", "content": "hello"}])
    usage.record_nim_request.assert_called_once()


def test_chat_logs_usage_warning(caplog):
    usage = MagicMock()
    usage.record_nim_request.return_value = "NVIDIA NIM requests: 800/1000 used this month (~80% of the free-tier estimate)."
    client = NimClient(NimConfig(api_key="key"), usage=usage)
    with patch("shared.llm_client.requests.post", return_value=_response()):
        with caplog.at_level("WARNING"):
            client.chat([{"role": "user", "content": "hello"}])
    assert "80%" in caplog.text
