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

## Milestone 2 rehearsal (live NIM, RAG regression demo)

Five questions per session (diagnose, one experiment, challenge, find caller,
ties), five sessions per prompt variant, run through the real `Resonance.handle`
with a mocked speaker. Scoring is keyword-based and approximate. Turns within a
session are correlated, so treat differences of a few points as noise.

| Question | Before fixes (1 run, informal) | Same prompt, run A | Same prompt, run B |
|---|---|---|---|
| diagnose from diff | found cause 3-5 of 5 | 4/5 | 3/5 |
| one experiment (fixed-input comparison) | not proposed | 4/5 | 2/5 |
| challenge a false premise | accepted it once | 0/5 | 1/5 |
| find the caller | right file, wrong spoken claim | 4/5 | 4/5 |
| retriever ties (unread code) | spoke a wrong, confident claim | 5/5 | 4/5 |

Totals: 17/25 and 14/25. That is below the brief's release gate (adequate
diagnosis and correct next experiment in 4 of 5 repeated rehearsals), so the
gate is **not met**. Weakest: the challenge turn (the model often restates its
diagnosis instead of answering "what evidence connects this to X") and the
fixed-input experiment.

Prompt variants tried and dropped: a rule that SPOKEN must answer the user's
latest question first scored 13/25 and 12/25 in two runs, with experiment and
caller turns getting worse. Storing the proposed next step in chat history made
later answers echo earlier ones, so only the spoken summary is kept.

Ideas not yet tried: thinking on for the final answer hop only, a larger model
for the final hop, and deterministic handling for challenge questions.
