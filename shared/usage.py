"""Local usage tracking against the free-tier caps mentioned in the README.

Persisted to a small JSON file next to the repo so counts survive between
runs, and reset automatically each calendar month since that's how
ElevenLabs' free tier resets. Purely advisory: this estimates usage from
request counts and character counts, it doesn't call either vendor's
billing API, so treat the warnings as a heads-up, not an exact balance.
"""

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger("resonance.usage")

DEFAULT_PATH = Path(__file__).resolve().parent.parent / ".usage.json"

NIM_REQUEST_CAP = int(os.environ.get("NIM_REQUEST_WARNING", 1000))
ELEVENLABS_CHARACTER_CAP = int(os.environ.get("ELEVENLABS_CHARACTER_WARNING", 10000))

# fire a warning once per threshold crossing, not on every call after
WARNING_THRESHOLDS = (0.8, 1.0)


class UsageTracker:
    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = path
        self._data = self._load()

    def _load(self) -> dict:
        month = _current_month()
        data = {}
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                logger.warning("Could not read usage file at %s, starting fresh.", self.path)
        if data.get("month") != month:
            data = {"month": month, "nim_requests": 0, "elevenlabs_characters": 0}
        return data

    def _save(self) -> None:
        try:
            self.path.write_text(json.dumps(self._data))
        except OSError:
            logger.warning("Could not persist usage stats to %s", self.path)

    def record_nim_request(self) -> str | None:
        self._data["nim_requests"] = self._data.get("nim_requests", 0) + 1
        return self._check_threshold("nim_requests", NIM_REQUEST_CAP, "NVIDIA NIM requests")

    def record_elevenlabs_characters(self, count: int) -> str | None:
        self._data["elevenlabs_characters"] = self._data.get("elevenlabs_characters", 0) + count
        return self._check_threshold("elevenlabs_characters", ELEVENLABS_CHARACTER_CAP, "ElevenLabs characters")

    def _check_threshold(self, key: str, cap: int, label: str) -> str | None:
        used = self._data.get(key, 0)
        warned_key = f"{key}_warned_at"
        already_warned = self._data.get(warned_key, 0.0)
        ratio = used / cap if cap else 0.0

        warning = None
        for threshold in WARNING_THRESHOLDS:
            if ratio >= threshold and already_warned < threshold:
                self._data[warned_key] = threshold
                pct = int(threshold * 100)
                warning = f"{label}: {used}/{cap} used this month (~{pct}% of the free-tier estimate)."

        self._save()
        return warning


def _current_month() -> str:
    return time.strftime("%Y-%m")
