"""ElevenLabs Flash text-to-speech playback, with retry/backoff."""

import logging
import subprocess
import threading

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger("mentor.voice")


class ElevenLabsVoice:
    def __init__(self, api_key: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM", usage=None):
        self.api_key = api_key
        self.voice_id = voice_id
        self.usage = usage  # optional shared.usage.UsageTracker
        self._lock = threading.Lock()
        self._process = None
        self._interrupted = False

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        reraise=True,
    )
    def speak(self, text: str) -> None:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream"
        response = requests.post(
            url,
            headers={"xi-api-key": self.api_key, "Content-Type": "application/json"},
            json={
                "text": text,
                "model_id": "eleven_flash_v2_5",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            },
            stream=True,
            timeout=30,
        )
        if not response.ok:
            # ElevenLabs puts the real reason in the JSON body; raise_for_status()
            # alone would discard it and just say "402 Client Error".
            try:
                detail = response.json().get("detail", {})
                message = detail.get("message", response.text)
            except Exception:
                message = response.text
            response.close()
            raise requests.exceptions.HTTPError(
                f"{response.status_code} from ElevenLabs: {message}"
            )

        if self.usage is not None:
            warning = self.usage.record_elevenlabs_characters(len(text))
            if warning:
                logger.warning(warning)

        # Pipe MP3 chunks straight into ffplay's stdin as they arrive rather
        # than buffering the whole reply to disk first: playback starts on
        # the first chunk instead of after the full download.
        player = subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-i", "pipe:0"],
            stdin=subprocess.PIPE,
        )
        with self._lock:
            self._interrupted = False
            self._process = player

        try:
            for chunk in response.iter_content(chunk_size=4096):
                if not chunk:
                    continue
                try:
                    player.stdin.write(chunk)
                except (BrokenPipeError, OSError):
                    break  # player exited early (e.g. interrupted via stop())
        finally:
            try:
                player.stdin.close()
            except OSError:
                pass

        exit_code = player.wait()

        with self._lock:
            interrupted = self._interrupted
            self._process = None

        if not interrupted and exit_code != 0:
            print(f"  Playback may have failed (ffplay exit code {exit_code}). "
                  f"Check System Settings > Sound > Output.")

    def stop(self) -> None:
        """Cut off in-progress playback, e.g. when the user barges in with
        a new question while a reply is still being spoken."""
        with self._lock:
            process = self._process
            self._interrupted = True
        if process and process.poll() is None:
            process.terminate()
