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
    model: str = "nvidia/nemotron-3-nano-30b-a3b"  # NVIDIA's own model, tuned for exactly this: low-latency agentic and voice-assistant workloads
    max_tokens: int = 300
    temperature: float = 0.4


class NimClient:
    def __init__(self, config: NimConfig, usage=None):
        self.config = config
        self.usage = usage  # optional shared.usage.UsageTracker

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        reraise=True,
    )
    def chat(self, messages: list[dict], json_mode: bool = False) -> str:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

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

        return response.json()["choices"][0]["message"]["content"]
