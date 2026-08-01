"""
Pre-flight check for Resonance / Terminal Whisperer.

Tests each piece of the stack on its own, in order, so if something's
broken you know exactly which layer failed instead of debugging the
whole hotkey loop at once.

Run this before you ever touch resonance.py or main.py:

    python test_setup.py
"""

import os
import shutil
import sys

from dotenv import load_dotenv

load_dotenv()


def check(label: str):
    print(f"\n--- {label} ---")


def ok(msg: str):
    print(f"  PASS  {msg}")


def fail(msg: str):
    print(f"  FAIL  {msg}")


def main():
    failures = []

    # 1. ffmpeg on PATH
    check("ffmpeg")
    if shutil.which("ffplay"):
        ok("ffplay found on PATH")
    else:
        fail("ffplay not found. Install ffmpeg (see README setup steps).")
        failures.append("ffmpeg")

    # 2. env vars present
    check("Environment variables")
    nvidia_key = os.environ.get("NVIDIA_API_KEY")
    eleven_key = os.environ.get("ELEVENLABS_API_KEY")

    if nvidia_key and nvidia_key != "nvapi-your-key-here":
        ok("NVIDIA_API_KEY is set")
    else:
        fail("NVIDIA_API_KEY missing or still the placeholder. Check your .env file.")
        failures.append("NVIDIA_API_KEY")

    if eleven_key and eleven_key != "sk_your-key-here":
        ok("ELEVENLABS_API_KEY is set")
    else:
        fail("ELEVENLABS_API_KEY missing or still the placeholder. Check your .env file.")
        failures.append("ELEVENLABS_API_KEY")

    if failures:
        print("\nStopping here, fix the above before testing live API calls.")
        sys.exit(1)

    # 3. NIM reasoning call
    check("NVIDIA NIM (reasoning)")
    try:
        from shared.llm_client import NimClient, NimConfig

        client = NimClient(NimConfig(api_key=nvidia_key))
        reply = client.chat([
            {"role": "user", "content": "Reply with exactly the word: pong"}
        ])
        ok(f"NIM responded: {reply.strip()!r}")
    except Exception as e:
        fail(f"NIM call failed: {e}")
        failures.append("NIM")

    # 4. ElevenLabs voice call (this one actually plays audio)
    check("ElevenLabs Flash (voice output)")
    try:
        from shared.voice_out import ElevenLabsVoice

        voice = ElevenLabsVoice(
            api_key=eleven_key,
            voice_id=os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
        )
        print("  Playing a short test phrase, you should hear it now...")
        voice.speak("Resonance is online.")
        ok("Audio call completed without error (confirm you actually heard it)")
    except Exception as e:
        fail(f"ElevenLabs call failed: {e}")
        failures.append("ElevenLabs")

    # 5. Microphone availability (does not record, just checks a device exists)
    check("Microphone")
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        input_devices = [d for d in devices if d["max_input_channels"] > 0]
        if input_devices:
            default = sd.query_devices(kind="input")
            ok(f"Input device available: {default['name']}")
        else:
            fail("No input device found. Check mic permissions for your terminal app.")
            failures.append("microphone")
    except Exception as e:
        fail(f"Could not query audio devices: {e}")
        failures.append("microphone")

    print("\n" + "=" * 40)
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        sys.exit(1)
    else:
        print("All checks passed. Run resonance.py or main.py.")


if __name__ == "__main__":
    main()