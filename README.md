<div align="center">

```
 ⌇⌇⌇  R E S O N A N C E  ⌇⌇⌇
```

# Eliminate developer fatigue.

**A universal voice companion for how developers actually think.**

Talk or type, it reasons, then answers out loud, before you've even
looked away from the code.

![status](https://img.shields.io/badge/status-active-1D9E75)
![stack](https://img.shields.io/badge/stack-NVIDIA%20NIM%20%2B%20ElevenLabs%20Flash-534AB7)
![cost](https://img.shields.io/badge/cost-%240%20to%20build-D85A30)

| 🗣️ Talk or type | 🧠 NVIDIA reasoning | ⚡ ~75ms voice | 🆓 Free to build |
|:---:|:---:|:---:|:---:|
| whichever fits the room you're in | Nemotron 3 Nano, NVIDIA's own model | never breaks your flow | no paid API, ever |

</div>

---

## 🎙️ The fatigue is real

Every context switch has a cost. Reading a terminal error, a git diff, a
stack trace, pulls a developer's attention off the problem they were
solving and onto decoding text, and working memory has to hold both at
once until the switch resolves. That's the cognitive load tax on a
developer's day, paid dozens of times, by every engineer, on every team,
every single day, and it's the thing this was built to eliminate.

This isn't a new observation. Cognitive load theory and the modality
effect in multimedia learning research both point at the same mechanism:
offloading information to a second channel, hearing instead of reading,
frees up working memory rather than competing for it. Voice output
doesn't just feel different from a wall of text, it runs on a different
channel entirely. Less load spent parsing, more bandwidth left for the
actual engineering problem, and at the scale of an entire org, that
recovered bandwidth is where creativity and innovation actually come
from.

That's the thesis: voice isn't a novelty layered onto developer tools.
**It's lower-friction infrastructure for how developers already think**,
and it belongs in the daily productivity environment, not a demo reel.

---

## 📊 The proof

GenAI adoption among developers is already high and still climbing, but
a growing body of 2026 research is catching up to what that adoption
costs:

- Using GenAI for programming tends to shift work from active
  problem-solving into passive review of AI-generated output, a shift
  linked to *increased* cognitive load and reduced performance,
  particularly for experienced developers.
  ([meta-analysis, arXiv 2605.04779](https://arxiv.org/pdf/2605.04779))
- In field studies of professional developers, perceived cognitive load
  during development-heavy tasks tracks the AI interaction itself, not
  just the output quality.
  ([arXiv 2607.02337](https://arxiv.org/abs/2607.02337v1))
- It takes an average of 23 minutes to fully recover focus after a task
  interruption.
  ([UC Irvine research, via MindStudio](https://www.mindstudio.ai/blog/ai-brain-fry-cognitive-fatigue-ai-tools))

And there's early, direct evidence that voice specifically helps with
exactly this problem:

- A controlled study on voice-assisted Python debugging measured a 37%
  reduction in cognitive load and 78% faster error identification versus
  reading a traditional stack trace, with response latency under 1.2
  seconds. One study, not yet independently replicated at scale, but a
  close match for the problem this repo is built around.
  ([arXiv 2507.15007](https://arxiv.org/pdf/2507.15007))
- In industrial human-machine interaction research, voice control showed
  up to a 67% time-efficiency advantage over touchscreen interaction for
  complex, multi-step commands.
  ([HMI cognitive load study](https://www.researchgate.net/publication/301258336_Measurement_of_efficiency_of_auditory_vs_visual_communication_in_HMI_A_cognitive_load_approach))

**Worth being honest about:** voice isn't automatically free of cognitive
cost either. Continuous, unrequested audio layered on top of a visual
task has been shown to *increase* load without improving outcomes. That's
not an argument against voice, it's an argument for *how* it's used, and
it's why Resonance only speaks when explicitly triggered (a held key, a
typed line), never as constant ambient narration.

---

## 🔊 What's inside

| App | What it does |
|---|---|
| `resonance.py` | Talk or type, either way, get reasoning back as speech. Built so voice fits into real work environments, not just a quiet home office. |
| `main.py` | The earlier, voice-only version of the same idea: hold a key, ask, get a spoken answer. |

Both run on the same free-tier stack, no paid API required to build on
it, run it, or read every layer of it end to end:

```
  🧠  reasoning   →  NVIDIA NIM · nvidia/nemotron-3-nano-30b-a3b
  🗣️  voice       →  ElevenLabs Flash · eleven_flash_v2_5, ~75ms
  👂  listening   →  faster-whisper, fully local, zero API cost
```

**Why this model:** `nvidia/nemotron-3-nano-30b-a3b` is NVIDIA's own
mixture-of-experts model, purpose-built for exactly this kind of
low-latency, agentic, tool-calling workload, including local voice
assistants specifically. It's not a generic instruct model pressed into
service, it's NVIDIA's stack running NVIDIA's model, tuned for the job
it's actually doing here.

---

## 🏗️ How it's built

```
resonance/
├── resonance.py           # Talk-or-type input, always-on voice output
├── main.py                # Earlier voice-only version of the same idea
├── test_setup.py          # catch a broken API key or missing ffmpeg
├── conftest.py             # so `pytest` resolves the shared/ package
├── requirements.txt
├── requirements-dev.txt      # requirements.txt + pytest
├── .env.example             # template for your API keys, committed
├── .env                       # your real keys, gitignored, never committed
├── .gitignore
├── README.md                 # you are here
├── shared/
│   ├── __init__.py
│   ├── audio_io.py          # mic capture + local Whisper transcription
│   ├── llm_client.py        # NVIDIA NIM client, retry/backoff, JSON mode
│   ├── voice_out.py         # ElevenLabs Flash TTS client, streaming playback, barge-in
│   └── usage.py               # local free-tier usage tracking, persisted to .usage.json
└── tests/
    ├── test_resonance.py     # handle(), hotkey/barge-in wiring, notify, run loop
    ├── test_llm_client.py
    ├── test_voice_out.py
    ├── test_audio_io.py
    └── test_usage.py
```

**Why it's split this way:** `resonance.py` and `main.py` are the
product. `shared/` is the infrastructure underneath both of them. Neither
app talks to `requests` or `sounddevice` directly, they talk to
`shared/`. That means swapping Whisper for a hosted STT, or ElevenLabs
for another voice vendor, is a change in one file, not a scavenger hunt
through the whole codebase.

### Who owns what

- **`shared/audio_io.py`** — `VoiceCapture` class. Owns the microphone
  stream and the local Whisper model. Input: raw audio. Output: a
  transcript string. Knows nothing about NIM, ElevenLabs, or hotkeys.

- **`shared/llm_client.py`** — `NimClient` class. Owns the NVIDIA NIM
  connection: auth, retry/backoff on network failures, optional
  structured JSON output. Knows nothing about what the reasoning is for.

- **`shared/voice_out.py`** — `ElevenLabsVoice` class. Owns
  text-to-speech and playback, same retry pattern as the NIM client.
  Streams MP3 chunks straight into `ffplay`'s stdin as they arrive
  instead of buffering the whole reply to disk first, and exposes
  `stop()` so a new hotkey press can cut off playback mid-sentence
  (barge-in). Knows nothing about what the text means.

- **`shared/usage.py`** — `UsageTracker` class. Counts NIM requests and
  ElevenLabs characters against the free-tier caps in the math below,
  persisted to `.usage.json` (gitignored) so the count survives between
  runs and resets itself each calendar month. Purely advisory: it's
  estimating from request/character counts, not reading either vendor's
  real billing API.

- **`resonance.py`** — wires the shared modules together with two entry
  points into one handler: a hotkey for talking, a prompt for typing.
  Same reasoning, same voice, same output, regardless of which one you
  used. Voice replies run on a background thread so the hotkey listener
  stays free to catch a new press immediately, which is what makes
  barge-in possible.

- **`main.py`** — the single-entry-point version: hotkey only, voice
  only. Kept in the repo as the earlier iteration of the same idea.

### ⚡ One turn, start to finish

```
  you talk or type
        │
        ▼
  🎧  local Whisper transcribes        (no API call)
        │
        ▼
  🧠  Nemotron 3 Nano reasons          (1 NIM call)
        │
        ▼
  🗣️  Eleven Flash speaks it back      (1 ElevenLabs call, ~75ms)
```

1. Either input path fires: hold F9 and talk (routed through
   `VoiceCapture` and local Whisper), or type at the prompt and hit
   enter.
2. `Resonance.handle()` sends the text to `NimClient.chat()` with a
   short, direct system prompt tuned for something spoken, not read. One
   NIM call.
3. The reply prints to the terminal always, so it works with sound off.
   If `VOICE_OUTPUT` is enabled (the default), it also plays through
   `ElevenLabsVoice.speak()`.

One input event, one NIM call, one optional ElevenLabs call. No hidden
polling, no continuous listening, no background cost between turns.

---

## ⚙️ Get it running

1. **Install ffmpeg** (handles audio playback):
   - macOS: `brew install ffmpeg`
   - Windows: `winget install ffmpeg`
   - Linux: `sudo apt install ffmpeg`

2. **Install Python deps:**
   ```
   pip install -r requirements.txt
   ```

3. **Get your free API keys:**
   - NVIDIA NIM: sign up at build.nvidia.com, generate a key (`nvapi-...`)
   - ElevenLabs: sign up at elevenlabs.io, grab your key from Settings > API Keys

4. **Set your API keys.** Copy the template and fill in your real keys:
   ```
   cp .env.example .env
   ```
   Then edit `.env`:
   ```
   NVIDIA_API_KEY=nvapi-...
   ELEVENLABS_API_KEY=sk_...
   ```
   `.env` is gitignored, it never gets committed. If you'd rather not use
   a file at all, exporting the same variables in your shell works too:
   ```
   export NVIDIA_API_KEY="nvapi-..."
   export ELEVENLABS_API_KEY="sk_..."
   ```

Model and voice are already set (`nvidia/nemotron-3-nano-30b-a3b` and
Eleven Flash), no extra config needed to get running. Swap either in
`shared/llm_client.py` or `shared/voice_out.py` if you want to try
something else.

## ▶️ Talk to it

```
python resonance.py
```
Hold **right-Control** and talk, or type at the `>` prompt and hit
enter. Set `RESONANCE_HOTKEY` in `.env` to use a different key. Set
`VOICE_OUTPUT=false` for a fully silent, text-only mode. While a reply
is speaking, holding the hotkey again interrupts it and starts a new
recording immediately (barge-in) instead of waiting for it to finish.

```
python main.py
```
The voice-only version: hold **F9**, talk, let go.

First run of either app downloads the local Whisper model (~150MB),
one-time cost.

### 🧪 Running the tests

```
pip install -r requirements-dev.txt
pytest
```
Everything's mocked, no API keys or microphone required. Covers the
retry/backoff paths on both clients, the recording-length cap, the
usage tracker's threshold/reset logic, and `resonance.py`'s own
orchestration: `handle()`, hotkey/barge-in wiring, OS notifications,
and the typed-input run loop.

---

## 💸 The free-tier math

- **NVIDIA NIM**: ~1,000 signup credits, ~40 requests/minute. One
  input event equals one NIM call.
- **ElevenLabs Flash**: 10,000 credits/month (~10-20 min of audio).
  Replies are capped at a few spoken sentences by design.
- **Whisper transcription**: fully local. No API call, no rate limit,
  no cost.

`resonance.py` tracks requests and characters against these caps
locally (`shared/usage.py`, see `.usage.json`) and logs a warning at
~80% and 100% of whichever cap you set via `NIM_REQUEST_WARNING` /
`ELEVENLABS_CHARACTER_WARNING` in `.env`.

---

## 🗺️ Where this is headed

Update this as milestones land. A stale roadmap is worse than none.

- [x] Voice-only ambient assistant (`main.py`)
- [x] Retry/backoff, structured logging, modular client architecture
- [x] Dual voice/text input, built for real work environments
      (`resonance.py`)
- [x] Running on NVIDIA's own Nemotron 3 Nano, not a generic model
- [x] Recording-length cap so a stuck/held key can't record forever
- [x] Streaming TTS playback straight into `ffplay`, instead of
      downloading the full reply before it starts playing
- [x] Barge-in: hold the hotkey again mid-reply to interrupt playback
      and start a new recording immediately
- [x] Configurable hotkey (`RESONANCE_HOTKEY`), OS notification on
      reply, local usage tracking against each vendor's free tier
- [x] Unit tests for both clients, the recording cap, and usage tracking
      (mocked, no live API calls)
- [x] Unit tests for `resonance.py`'s own orchestration: `handle()`,
      hotkey/barge-in wiring, notifications, and the run loop
- [ ] Feed real terminal output and git diffs into the prompt directly,
      not just what's said or typed
- [ ] Stream the NIM reply itself and start speaking the first sentence
      before the rest has finished generating, rather than waiting for
      the full reply
- [ ] Confirm-before-acting step for anything that would run a command
      on the user's behalf
- [ ] Live web search for grounded, current answers rather than reasoning
      from training data alone
- [ ] Concurrency: multiple people sharing a NIM key at once needs real
      rate-limit handling, not just retry/backoff

---

<div align="center">

### Fatigue eliminated, one interruption at a time.

One developer using this saves a few seconds per interruption. A whole
org running it is a different number entirely: less time lost to context
switching, more attention left over for the actual hard problems, which
is where innovation comes from in the first place.

`⌇⌇⌇⌇⌇⌇⌇⌇⌇⌇⌇⌇⌇⌇⌇⌇⌇⌇⌇⌇⌇⌇⌇⌇⌇⌇⌇⌇⌇⌇⌇⌇`

</div>