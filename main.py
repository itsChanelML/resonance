"""
Terminal Whisperer - Phase 3: hotkey-triggered voice assistant

Hold a key to record a question, release to send it through NVIDIA NIM
for reasoning and ElevenLabs Flash for a spoken reply.

This is event-triggered, not continuous. One NIM call and one ElevenLabs
call per press-and-release cycle. Nothing runs in the background between
presses, so you stay well inside both free tiers regardless of how long
you leave this running.

Setup:
    pip install -r requirements.txt
    export NVIDIA_API_KEY="nvapi-..."
    export ELEVENLABS_API_KEY="sk_..."
    python main.py

Requires ffmpeg installed and on PATH (for playback via ffplay).
"""

import os
import wave
import tempfile

import numpy as np
import requests
import sounddevice as sd
from pynput import keyboard
from dotenv import load_dotenv
from faster_whisper import WhisperModel

load_dotenv()  # reads .env in the current directory, if present

# ---- Config ----
NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]
ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # "Rachel", default

HOTKEY = keyboard.Key.f9  # hold this to talk, release to send
SAMPLE_RATE = 16000
MIN_RECORDING_SECONDS = 0.3  # ignore accidental taps
MAX_RECORDING_SECONDS = 30  # a stuck/held key shouldn't record indefinitely

NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
# NVIDIA's own Nemotron 3 Nano: a mixture-of-experts model built for exactly
# this kind of low-latency, agentic, voice-assistant workload.
NIM_MODEL = "nvidia/nemotron-3-nano-30b-a3b"

SYSTEM_PROMPT = (
    "You are Terminal Whisperer, an ambient voice pair programmer. "
    "A developer describes terminal output, git diffs, or code context to "
    "you out loud; you don't see any of it directly, only what they say. "
    "Answer in 1-3 short spoken sentences. No markdown, no code blocks, no "
    "bullet points. Plain spoken language a developer can listen to "
    "without looking at a screen."
)

# ---- Load the local transcription model once, at startup ----
print("Loading local Whisper model (first run downloads it, ~150MB)...")
whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")
print(f"Ready. Hold {HOTKEY} to talk, release to send. Ctrl+C to quit.")

# ---- Recording state ----
_recording_chunks = []
_is_recording = False
_stream = None
_frame_count = 0
_truncated = False


def _audio_callback(indata, frames, time_info, status):
    global _frame_count, _is_recording, _truncated
    if not _is_recording:
        return
    _recording_chunks.append(indata.copy())
    _frame_count += frames
    if _frame_count >= SAMPLE_RATE * MAX_RECORDING_SECONDS:
        _is_recording = False
        _truncated = True
        raise sd.CallbackStop()


def start_recording():
    global _recording_chunks, _is_recording, _stream, _frame_count, _truncated
    _recording_chunks = []
    _frame_count = 0
    _truncated = False
    _is_recording = True
    _stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=_audio_callback
    )
    _stream.start()
    print("Listening...")


def stop_recording_and_process():
    global _is_recording, _stream
    _is_recording = False
    if _stream:
        _stream.stop()
        _stream.close()

    if _truncated:
        print(f"Recording hit the {MAX_RECORDING_SECONDS}s cap, stopped automatically.")

    if not _recording_chunks:
        return

    audio = np.concatenate(_recording_chunks, axis=0)
    if len(audio) < SAMPLE_RATE * MIN_RECORDING_SECONDS:
        print("Too short, ignored.")
        return

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        _write_wav(f.name, audio, SAMPLE_RATE)
        wav_path = f.name

    print("Transcribing locally...")
    text = _transcribe(wav_path)
    os.unlink(wav_path)

    if not text.strip():
        print("Didn't catch that.")
        return

    print(f"You said: {text}")
    print("Thinking (NIM)...")
    reply = _ask_nim(text)
    print(f"Reply: {reply}")

    print("Speaking (ElevenLabs Flash)...")
    _speak(reply)


def _write_wav(path, audio, sample_rate):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())


def _transcribe(wav_path):
    segments, _ = whisper_model.transcribe(wav_path, language="en")
    return " ".join(seg.text for seg in segments).strip()


def _ask_nim(user_text):
    response = requests.post(
        NIM_URL,
        headers={
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": NIM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            "max_tokens": 150,
            "temperature": 0.4,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def _speak(text):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"
    response = requests.post(
        url,
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "text": text,
            "model_id": "eleven_flash_v2_5",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        },
        stream=True,
        timeout=30,
    )
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
        mp3_path = f.name

    _play_mp3(mp3_path)
    os.unlink(mp3_path)


def _play_mp3(path):
    # ffplay (bundled with ffmpeg) avoids extra Python audio-backend
    # headaches across macOS/Windows/Linux. Swap for simpleaudio or
    # a proper streaming player once this is working end to end.
    os.system(f'ffplay -nodisp -autoexit -loglevel quiet "{path}"')


# ---- Hotkey wiring ----
def on_press(key):
    if key == HOTKEY and not _is_recording:
        start_recording()


def on_release(key):
    if key == HOTKEY and _is_recording:
        stop_recording_and_process()


if __name__ == "__main__":
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
