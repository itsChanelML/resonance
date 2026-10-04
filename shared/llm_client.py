"""Thin client for NVIDIA NIM chat completions. Retries on transient
network failures and supports structured JSON output for mode routing."""

import logging
from dataclasses import dataclass

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger("resonance.llm")

NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


@dataclass
class NimConfig:
    api_key: str
    model: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"  # nemotron-3-nano-30b-a3b hit EOL 2026-09-01; this is its live successor on NIM
    max_tokens: int = 300
    temperature: float = 0.4


class NimClient:
    def __init__(self, config: NimConfig, usage=None):
        self.config = config
        self.usage = usage  # optional shared.usage.UsageTracker

    def _payload(self, messages, max_tokens, thinking, extra=None, model=None) -> dict:
        payload = {
            "model": model or self.config.model,
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        if thinking is not None:
            # Nemotron reasoning model: thinking off cuts project-prompt latency
            # from ~30-85s to ~4s (measured live, see docs/BASELINE.md).
            payload["chat_template_kwargs"] = {"enable_thinking": thinking}
        payload.update(extra or {})
        return payload

    @retry(
        stop=stop_after_attempt(5),  # free tier returns frequent 503s under load
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        reraise=True,
    )
    def _post(self, payload: dict) -> dict:
        response = requests.post(
            NIM_URL,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        if response.status_code == 429:
            logger.warning("NIM rate limit hit (40 req/min free tier), backing off")
        response.raise_for_status()

        if self.usage is not None:
            warning = self.usage.record_nim_request()
            if warning:
                logger.warning(warning)
        return response.json()

    def chat(self, messages: list[dict], json_mode: bool = False,
             max_tokens: int | None = None, thinking: bool | None = None) -> str:
        extra = {"response_format": {"type": "json_object"}} if json_mode else None
        data = self._post(self._payload(messages, max_tokens, thinking, extra))
        content = data["choices"][0]["message"].get("content")
        if not content:
            raise ValueError("model returned no content (reasoning may have used the token budget)")
        return content

    def chat_with_tools(self, messages: list[dict], tools: list[dict] | None,
                        max_tokens: int | None = None, thinking: bool | None = None,
                        tool_choice: str = "auto", model: str | None = None) -> dict:
        """One model hop that may return tool_calls instead of content.
        Returns the raw assistant message dict."""
        extra = {"tools": tools, "tool_choice": tool_choice} if tools else None
        data = self._post(self._payload(messages, max_tokens, thinking, extra, model))
        return data["choices"][0]["message"]
