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

## Challenge and experiment turns: variants measured

Six sessions of diagnose -> experiment -> challenge per variant (live NIM).
Scores are keyword-based; "challenge" counts a reply whose spoken line says the
claim was not made or no evidence was found.

| Variant | diagnose | experiment | challenge |
|---|---|---|---|
| V0 baseline | 5/6 | 3/6 | 0/6 |
| V1 turn hints (kept) | 5/6 | 6/6 | 6/6 |
| V2 thinking on for the final answer hop | 6/6 | 4/6 | 0/6 |
| V3 hints + thinking | 5/6 | 6/6 | 6/6 |
| V4 hints + `nvidia/nemotron-3-super-120b-a12b` for the final hop | 4/6 | 6/6 | 3/6 |

Turn hints (`RESONANCE_TURN_HINTS`, default on) append guidance to the model
request only when the question looks like a challenge ("what evidence...",
"you suggested...") or a request for one experiment. They are not stored in
history. Thinking and the larger model added latency (up to 43 s on one turn)
without beating hints alone, so both stay off by default
(`RESONANCE_FINAL_THINKING`, `RESONANCE_FINAL_MODEL`).

Full five-question set with hints on (6 sessions): diagnose 4/6, experiment 6/6,
challenge 6/6, caller 3/6, ties 5/6 = 24/30 (earlier: 14-17 of 25).

Reading the answers by hand tempers the experiment score. All six proposed a
single-factor comparison on identical saved inputs, but only three (varying
prompt assembly or crossing old/new retriever and prompt) would actually
separate the causes. The other three varied only `top_k` while `MAX_EVIDENCE`
stayed at 1, which cannot change what reaches the prompt. The challenge
answers read as intended ("I did not claim a model change caused the
regression; I found no evidence supporting that claim"). The release gate (4 of
5 correct next experiments) is still not met for the experiment turn.

### Downstream-of-retrieval rule added to the experiment hint

The experiment hint now also says: if retrieval metrics such as recall are fine
but answer quality dropped, hold the retrieved documents fixed and vary how
they are assembled into the prompt; a change to a parameter that cannot alter
what reaches the next stage tells you nothing. Eight sessions, graded by hand:
7 of 8 proposed the informative comparison (same retrieved documents, old vs.
new prompt assembly). The eighth compared the current configuration with
itself. Before the rule, 3 of 6 were informative. Diagnose scored 6/8 in the
same sessions.

**Disclosure for the talk:** this rule is a general RAG-debugging heuristic, but
it also steers the assistant toward the demo scenario's answer. Present the
experiment as the assistant following debugging guidance, not as it deducing
the cause unaided.

## Milestone 3: investigation state, recap, continuity

**Finding that shaped the design.** I first tried asking the model to maintain
the notes (JSON mode works on this model). It was unreliable: given "You
suggested changing the model, what evidence?" it marked both unrelated
hypotheses ruled out, with reasons like "User did not reject the hypothesis",
copied the instructions into `constraints`, and mangled the JSON on a plain
question. So state updates are deterministic: observations need validated
citations, a hypothesis changes status only on an explicit engineer action
(`/ruleout`, "rule out X", "assume X is fine", or a challenged claim the
answer found no evidence for), constraints and corrections are the engineer's
own sentences, and an experiment is "proposed" until `/verify` supplies a
result. Recap, replay and "tell me more" are built locally with no model call.

**Bugs found in live testing and fixed:** matching a one-word phrase such as
"retrieval" ruled out any hypothesis that mentioned it (now only 3+ meaningful
words match an existing hypothesis, with stopwords removed); the recap ran over
70 words and ended in fragments (now a 55-word budget that drops the least
important parts and cuts at clause boundaries); constraints stored the whole
message, questions included (now only the matching sentence).

**Acceptance run** (6 sessions, live NIM, RAG demo: diagnose, one experiment
with a constraint, challenge a claim never made, "assume retrieval is fine",
then "I'm back, give me a recap"). Checks are mine and keyword-based, on one
scenario, so they show the mechanism works, not that it generalizes.

| Check | Result |
|---|---|
| Both constraints present in notes | 6/6 |
| Challenged claim recorded as unsupported | 6/6 (5/6 in an earlier run: the model ignored the challenge once, and nothing is recorded when it does) |
| Recap lists the challenged claim as ruled out | 6/6 |
| Recap says the next step is "not yet run" | 6/6 |
| Recap says nothing has been verified | 6/6 |
| Recap at most 70 words | 6/6 |
| Recap in under 1.5 s (local, no model call) | 6/6 |
| No ruled-out idea re-raised by the model | 6/6 |

Known limits: the model sometimes proposes a tool action ("run git_diff") as
its next step, and still invents baseline values (it said `MAX_EVIDENCE` was 5
when the diff shows 3); the notes record what it said, they do not fix it.
Timing marks cover activation, transcription, model reply, speech start and
interruption. First audible output is approximated by speech start, and an
in-flight NIM request cannot be cancelled, so a superseded reply is discarded
when it returns.
