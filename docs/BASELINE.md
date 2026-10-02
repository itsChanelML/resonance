# Resonance baseline audit (brief B01)

Measured 2026-10-02 against `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` on NIM's free tier.

## Turn flow
hotkey press -> `TurnController.cancel()` + `voice.stop()` + `capture.start()` ->
release -> Whisper transcribe -> `handle()` on a thread -> `NimClient.chat` ->
history append -> print -> `voice.speak(spoken)`. Typed input enters `handle()` directly.

## Context and history
- History: last 6 messages (`MAX_HISTORY_TURNS`), in memory only, cleared on `/project`.
- Context sent: system prompt + history + question; with attachments, also line-numbered
  file excerpts (6000-token budget). `/preview` shows the exact text.
- Services that receive data: NVIDIA NIM (question, history, file excerpts), ElevenLabs
  (spoken text only). Whisper transcription is local.

## Live findings
| Question | Result |
|---|---|
| Does the model follow the SPOKEN/OBSERVATIONS/... format? | Yes, with thinking off. Citations sometimes use odd hyphens or invented path prefixes; `check_citations` flags both. |
| Latency, thinking ON, project prompt | 27-86 s, and can spend the whole token budget before answering |
| Latency, thinking OFF | 3.8-5.3 s, so project mode sets `enable_thinking=False` |
| Structured tool calls? | **Yes.** `tools` + `tool_choice=auto` returned a valid `tool_calls` entry, so Milestone 2 can use model-driven tools |
| Reliability | One 503 (`Worker local total request limit reached`) seen under load; `NimClient` retries 3x |

## Not yet verified by hand
Hotkey recording, Whisper transcription, and audible barge-in need a person at the
machine. Run `python resonance.py --preflight` first, then try one voice turn and one
interruption.
