"""
Resonance — voice as daily development infrastructure

The premise: reading is expensive. Every time you break flow to parse a
wall of terminal output, your working memory is holding two things at
once, the problem you were solving and the text you're now decoding. That
split is where cognitive load piles up, and it's the same split that
happens dozens of times a day across an entire engineering org.

Voice runs on a different channel. Hearing an answer while your eyes stay
on the code doesn't compete with the visual work you're already doing,
it runs alongside it. Less load spent parsing, more bandwidth left for
the actual problem. That's the bet this is built on: not "voice is a
novelty," but "voice is lower-friction infrastructure for how developers
already think."

This app takes input either way, talked or typed, because forcing one
input mode onto every environment isn't infrastructure, it's a demo.
Some people are heads-down and typing is faster. Some are away from the
keyboard and talking is faster. Either path lands in the same reasoning
engine (NVIDIA NIM) and comes back as audio (ElevenLabs Flash), so the
answer costs you a listen, not a re-read.

Same free-tier stack throughout: NVIDIA NIM for reasoning, ElevenLabs
Flash for voice, local Whisper for transcription when you talk.

Voice input:  hold the hotkey (default right-Control), talk, release
Text input:   type at the prompt, press enter
Both funnel into the same handler, same system prompt, same response.

Voice replies run on a background thread so the hotkey listener stays
responsive: holding the hotkey again while Resonance is still speaking
interrupts playback and starts a new recording immediately (barge-in),
instead of waiting for the current reply to finish.

VOICE_OUTPUT=false drops to a fully silent, text-only mode.
"""

import logging
import os
import shutil
import subprocess
import sys
import threading

from pynput import keyboard
from dotenv import load_dotenv

load_dotenv()  # reads .env in the current directory, if present

from shared.audio_io import VoiceCapture
from shared.llm_client import NimClient, NimConfig
from shared.usage import UsageTracker
from shared.voice_out import ElevenLabsVoice

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("resonance")

VOICE_OUTPUT = os.environ.get("VOICE_OUTPUT", "true").lower() != "false"


def _resolve_hotkey(name: str):
    """Accepts a pynput Key name (f9, ctrl_r, alt, ...) or a single
    character. Defaults to right-Control, which is rarely claimed by OS
    shortcuts."""
    name = name.strip()
    if hasattr(keyboard.Key, name):
        return getattr(keyboard.Key, name)
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    raise ValueError(
        f"Unrecognized RESONANCE_HOTKEY {name!r}. Use a pynput Key name "
        f"(e.g. f9, ctrl_r, alt) or a single character."
    )


HOTKEY = _resolve_hotkey(os.environ.get("RESONANCE_HOTKEY", "ctrl_r"))

SYSTEM_PROMPT = (
    "You are Resonance, an ambient engineering assistant designed to be "
    "heard, not read. You get questions either typed or spoken, doesn't "
    "matter which, and you answer in 1-3 short sentences built for "
    "listening while someone keeps working. No markdown, no code blocks, "
    "no bullet points, no filler. Plain, direct, technically precise "
    "language, every word earning its place in the sentence."
)

MAX_HISTORY_TURNS = 6  # keep recent context only, so NIM calls stay cheap


def _notify(title: str, message: str) -> None:
    """Best-effort OS notification so a reply is visible even if you've
    looked away from the terminal. macOS only today; silently skipped
    elsewhere since there's no cross-platform equivalent worth the
    dependency."""
    if sys.platform != "darwin" or not shutil.which("osascript"):
        return
    escaped = message.replace("\\", "\\\\").replace('"', '\\"')
    escaped_title = title.replace("\\", "\\\\").replace('"', '\\"')
    script = f'display notification "{escaped}" with title "{escaped_title}"'
    try:
        subprocess.run(["osascript", "-e", script], check=False, timeout=5)
    except Exception:
        pass  # notification is a nicety, never worth crashing the app over


class Resonance:
    def __init__(self):
        self.capture = VoiceCapture()
        self.usage = UsageTracker()
        self.nim = NimClient(NimConfig(api_key=os.environ["NVIDIA_API_KEY"]), usage=self.usage)
        self.voice = ElevenLabsVoice(
            api_key=os.environ["ELEVENLABS_API_KEY"],
            voice_id=os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
            usage=self.usage,
        )
        self.history: list[dict] = []
        self._history_lock = threading.Lock()
        self._speaking = threading.Event()

    def handle(self, text: str, source: str) -> None:
        if not text.strip():
            return

        logger.info("[%s input] %s", source, text)
        print("  Thinking...")

        with self._history_lock:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages += self.history[-MAX_HISTORY_TURNS:]
        messages.append({"role": "user", "content": text})

        reply = self.nim.chat(messages)

        with self._history_lock:
            self.history.append({"role": "user", "content": text})
            self.history.append({"role": "assistant", "content": reply})

        print(f"> {reply}")
        _notify("Resonance", reply)

        if VOICE_OUTPUT:
            print("  Speaking...")
            self._speaking.set()
            try:
                self.voice.speak(reply)
            finally:
                self._speaking.clear()

    def _start_hotkey_listener(self) -> None:
        def on_press(key):
            if key == HOTKEY:
                if self._speaking.is_set():
                    self.voice.stop()  # barge-in: cut off the current reply
                self.capture.start()

        def on_release(key):
            if key == HOTKEY:
                text = self.capture.stop_and_transcribe()
                if text:
                    # Runs off the listener thread so holding the hotkey
                    # again (barge-in) is caught immediately instead of
                    # waiting for this reply to finish speaking.
                    threading.Thread(target=self.handle, args=(text, "voice"), daemon=True).start()

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()  # background thread, non-blocking

    def run(self) -> None:
        self._start_hotkey_listener()

        mode = "voice + text output" if VOICE_OUTPUT else "text output only"
        logger.info(
            "Resonance ready (%s). Hold %s to talk, or type below and press enter.",
            mode, HOTKEY,
        )

        while True:
            try:
                typed = input("> ")
            except (EOFError, KeyboardInterrupt):
                break
            self.handle(typed, source="typed")


if __name__ == "__main__":
    Resonance().run()
