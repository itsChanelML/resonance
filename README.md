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
│   ├── project_context.py   # read-only project root, exclusions, attachments
│   ├── project_tools.py     # list_files, search_code, read_file, git_diff (validated, bounded)
│   ├── inspector.py         # bounded tool-calling loop (6 rounds, 30s, output budget)
│   ├── investigation_state.py # session memory: constraints, hypotheses, experiments, recap
│   ├── context_builder.py   # budgeted context, spoken/detail split, citation check
│   ├── turn_controller.py   # turn IDs, stale-reply suppression
│   ├── trace.py             # per-turn timing records
│   ├── voice_out.py         # ElevenLabs Flash TTS client, streaming playback, barge-in
│   └── usage.py               # local free-tier usage tracking, persisted to .usage.json
├── demo_projects/
│   └── rag_regression/      # synthetic RAG project with a seeded evidence bug
├── scripts/
│   ├── design_voice.py
│   └── build_demo_repo.py   # builds a git repo of the demo with a real uncommitted diff
├── docs/
│   └── BASELINE.md          # audit of current behavior and live NIM measurements
└── tests/
    ├── test_resonance.py     # handle(), hotkey/barge-in wiring, notify, run loop
    ├── test_project_context.py  # boundaries, exclusions, redaction, citations
    ├── test_project_tools.py    # tool validation, git, inspection loop
    ├── test_investigation_state.py  # state, rule-outs, recap
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

- **`shared/project_context.py`**: `ProjectSession`. Owns the project
  root, the exclusion policy (defaults, `.gitignore`, `.resonanceignore`),
  secret redaction, and attachments. Every file read goes through it, so
  the boundary lives in one place. Read-only.

- **`shared/project_tools.py`**: `ProjectTools`. The four model-callable
  tools (`list_files`, `search_code`, `read_file`, `git_diff`): argument
  validation, output bounds, structured statuses. Never raises into the
  caller.

- **`shared/inspector.py`**: `run_inspection`. The bounded tool-calling
  loop: rounds, deadline, output budget, cancellation. Knows nothing about
  voice or the terminal.

- **`shared/investigation_state.py`**: `InvestigationState`. Session memory for one
  investigation: the problem, the engineer's constraints and corrections,
  observations (only with validated citations), hypotheses, and experiments.
  Deterministic on purpose: only explicit engineer actions change a
  hypothesis, and an experiment stays "proposed" until you supply a result.
  Never saved to disk.

- **`shared/context_builder.py`**: prompts, the context budget, the
  spoken/detail split, citation checks, and the turn hints.

- **`shared/turn_controller.py`, `shared/trace.py`**: turn IDs with
  stale-reply suppression, and per-turn timing records.

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
  🧠  Nemotron 3 Nano reasons          (1 NIM call; in project mode,
        │                               several, see below)
        ▼
  🗣️  Eleven Flash speaks it back      (1 ElevenLabs call, ~75ms)
```

1. Either input path fires: hold the hotkey (right-Control by default;
   `main.py` uses F9) and talk, routed through `VoiceCapture` and local
   Whisper, or type at the prompt and hit enter.
2. `Resonance.handle()` gives the turn an ID and sends the text to
   `NimClient.chat()` with a short, direct system prompt tuned for
   something spoken, not read. One NIM call.
3. The reply prints to the terminal always, so it works with sound off.
   If `VOICE_OUTPUT` is enabled (the default), it also plays through
   `ElevenLabsVoice.speak()`. If you press the hotkey or start another
   turn first, the old reply is dropped and the old audio stops.

**Project mode** (`--project`) changes step 2: the model may call the
read-only tools, each a separate NIM call, before it answers. A typical
turn is 2 to 7 NIM calls and takes 5 to 20 seconds; the loop is capped at
6 tool rounds. The terminal shows the full evidence and file references,
and only a short summary is spoken.

No hidden polling, no continuous listening, no background cost between
turns.

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

### 📂 Project mode

```
python resonance.py --project ./my_project --context src/retriever.py --context logs/eval.json
```
Point Resonance at a project and attach the files you want it to reason
about. Answers are grounded in those excerpts: the full detail (observed
evidence with `path:line` references, hypotheses, one next experiment,
what's missing) prints in the terminal, and only a short summary is
spoken. Citations that don't match an attached excerpt are flagged.

With a project selected, the model can also inspect it on its own through
four read-only tools: `list_files`, `search_code` (literal text),
`read_file` (max 200 lines per call), and `git_diff` (unstaged, staged,
or against a commit). Each call is validated and shown as it happens, e.g.
`[tool] git_diff(target='unstaged') -> OK`. An investigation stops after 6
tool rounds, 30 seconds, or a fixed output budget, and then answers with
what it has and says what is missing. Citations are checked against
everything inspected this session; evidence from a file that has since
changed is forgotten.

Commands at the `>` prompt: `/project <dir>`, `/attach <file>`,
`/context` (what's attached), `/preview` (the exact attached text that
will be sent), `/trace` (last turn's timing and tool calls), `/state`
(investigation notes), `/recap`, `/verify <id> <what you observed>`,
`/ruleout <id>`, `/mute`, `/unmute`, `/clear`. Run
`python resonance.py --preflight` to check keys, microphone, project root,
and the NIM connection before a session.

`git_diff` needs `git` installed; without it, or in a non-git folder, the
tool reports that and the rest still works.

To try it on a ready-made scenario (a RAG evaluation that regressed after
a change), run `python scripts/build_demo_repo.py /tmp/rag_demo`, then
`python resonance.py --project /tmp/rag_demo --context artifacts/eval_after.json`.
The demo project has a deliberately failing test
(`demo_projects/rag_regression/tests/test_prompt_builder.py`); it proves the
seeded bug and is excluded from the main `pytest` run.

**Investigation memory.** Within a session Resonance keeps notes: the
problem, your constraints ("we only have time for one experiment", "assume
retrieval is fine"), what it has observed in the files (only claims with
valid citations), hypotheses, and proposed experiments. These go back to the
model on every turn, so a correction sticks beyond the last few messages.
A claim you challenge ("you suggested X, what evidence?") that it cannot back
up is recorded as unsupported and is not offered again as the cause. An
experiment stays **proposed, not run** until you record a result with
`/verify`. Say "I'm back, give me a recap" for a short spoken summary
(what is ruled out, what is still uncertain, the next step) with the full
notes in the terminal. "Say that again" and "tell me more" also work. These
three are answered locally, so they are instant and use no NIM request.
Notes are session memory only: nothing is saved, and switching project or
`/clear` wipes them. Notes about a file are marked stale if it changes.

Rules it enforces (for attachments and every tool):
- **Read-only.** Resonance never writes to your project. Git runs with
  fixed arguments, never a model-written command.
- **Stays inside the root.** Paths are resolved first, so `../` and
  symlinks pointing outside the project are blocked.
- **Excluded by default:** `.env`, keys and credentials, `.git`,
  `venv`/`node_modules`, build output, binaries, plus anything in your
  `.gitignore` or a `.resonanceignore` file. Secret-like values in
  attached files are redacted; that filter is a safety net, not a
  guarantee.
- **Fresh.** An attached file that changed on disk is re-read on the next
  turn. Switching projects clears the conversation.

**What leaves your machine:** your question, recent conversation, the
attached file excerpts, and any file lines or diffs the model reads
through its tools go to NVIDIA NIM; only the spoken summary goes
to ElevenLabs. Transcription runs locally. Use `/preview` to see the
file text before sending.

Questions that challenge a claim or ask for a single experiment get extra
guidance (`RESONANCE_TURN_HINTS=false` turns it off). Two experimental
switches are off by default: `RESONANCE_FINAL_THINKING=true` rewrites the
final answer with thinking on, and `RESONANCE_FINAL_MODEL=<nim model id>`
rewrites it with a different model. Neither beat the hints alone in testing
and both add latency (see `docs/BASELINE.md`).

Project mode turns the model's thinking off to keep replies around 4-5
seconds. With it on, the same prompt took 27-86 seconds in testing (see
`docs/BASELINE.md`).

**Status:** attaching files, searching, reading, and git diffs work
today. The model is small and sometimes cites a wider line range than it
read or gets a detail wrong; the citation check flags those, so treat
answers as leads to verify. Measured results, including that the
brief's release gate is not yet met, are in `docs/BASELINE.md`. Running tests or evaluations on your behalf is
planned, not built: an experiment Resonance proposes is something you run.

### 🧪 Running the tests

```
pip install -r requirements-dev.txt
pytest
```
Everything's mocked, no API keys or microphone required. Covers the
retry/backoff paths on both clients, the recording-length cap, the
usage tracker's threshold/reset logic, and `resonance.py`'s own
orchestration: `handle()`, hotkey/barge-in wiring, OS notifications,
and the typed-input run loop. Also covers project-mode boundaries,
exclusions, redaction, citation checks, and stale-reply handling.

---

## 💸 The free-tier math

- **NVIDIA NIM**: ~1,000 signup credits, ~40 requests/minute. Without a
  project, one input event equals one NIM call. In project mode each tool
  round is another call, so a turn costs roughly 2 to 7. Expect the
  request counter to climb faster, and occasional `503` errors when the
  shared free tier is busy (the client retries 5 times). A recap, replay, or "tell me more" costs
  no NIM request.
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
- [x] Project mode: attach files, read-only boundary, exclusions,
      source references, short spoken summary with detail in the terminal
- [x] Turn IDs so an interrupted or superseded reply is never spoken
- [x] Project search, file listing, line-range reads, and git diffs the
      model can request, with a bounded inspection loop
- [x] Citation check against everything inspected this session
- [x] Session investigation memory, corrections that stick, and a local
      recap that never reports a proposal as verified
- [x] Mute, microphone-failure fallback, and per-turn timing from the
      hotkey press
- [ ] Feed real terminal output into the prompt directly, not just
      what's said or typed
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