"""Design a custom ElevenLabs voice from a text description (no cloning).

Usage:
    python scripts/design_voice.py              # generate previews, save MP3s
    python scripts/design_voice.py --pick 2     # save preview #2 as a permanent voice

Previews land in ./voice_previews/. Listen, then re-run with --pick N. The
script prints the voice ID to put in .env as ELEVENLABS_VOICE_ID.
"""
import argparse
import base64
import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API = "https://api.elevenlabs.io/v1"
PREVIEW_DIR = Path("voice_previews")
STATE_FILE = PREVIEW_DIR / "previews.json"

VOICE_NAME = "Resonance"
DESCRIPTION = (
    "A composed, refined British male AI butler voice in his 40s. Warm but precise "
    "baritone, unhurried and quietly confident, with dry understated wit. "
    "Crisp enunciation, calm even pacing, subtle polish like a highly capable "
    "digital assistant. Clean studio recording, no background noise."
)
PREVIEW_TEXT = (
    "Good evening. All systems are running at full capacity, the build passed on the "
    "first attempt, and I took the liberty of flagging two warnings you may want to "
    "look at. Shall I walk you through them, or would you prefer to ignore them as usual?"
)


def check(r: requests.Response) -> None:
    if not r.ok:
        raise SystemExit(f"{r.status_code} from ElevenLabs: {r.text}")


def headers() -> dict:
    return {"xi-api-key": os.environ["ELEVENLABS_API_KEY"]}


def generate() -> None:
    r = requests.post(
        f"{API}/text-to-voice/design",
        headers=headers(),
        json={"voice_description": DESCRIPTION, "text": PREVIEW_TEXT},
        timeout=120,
    )
    check(r)
    previews = r.json()["previews"]
    PREVIEW_DIR.mkdir(exist_ok=True)
    ids = []
    for i, p in enumerate(previews, start=1):
        (PREVIEW_DIR / f"preview_{i}.mp3").write_bytes(base64.b64decode(p["audio_base_64"]))
        ids.append(p["generated_voice_id"])
    STATE_FILE.write_text(json.dumps(ids))
    print(f"Saved {len(ids)} previews to {PREVIEW_DIR}/. Listen, then run with --pick N.")


def pick(n: int) -> None:
    ids = json.loads(STATE_FILE.read_text())
    r = requests.post(
        f"{API}/text-to-voice",
        headers=headers(),
        json={
            "voice_name": VOICE_NAME,
            "voice_description": DESCRIPTION,
            "generated_voice_id": ids[n - 1],
        },
        timeout=60,
    )
    check(r)
    print(f"Created voice. Set in .env:\n  ELEVENLABS_VOICE_ID={r.json()['voice_id']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", type=int, help="save preview N as a permanent voice")
    args = ap.parse_args()
    pick(args.pick) if args.pick else generate()
