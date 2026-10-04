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

import argparse
import logging
import os
import shutil
import subprocess
import sys
import threading
import time

from pynput import keyboard
from dotenv import load_dotenv

load_dotenv()  # reads .env in the current directory, if present

from shared.audio_io import VoiceCapture
from shared.context_builder import (
    build_context_block, check_citations, compact_reply, hedge_spoken,
    inspection_prompt, project_prompt, split_reply, turn_hint,
)
from shared.llm_client import NimClient, NimConfig
from shared.investigation_state import (
    EXPAND_INTENT, RECAP_INTENT, REPLAY_INTENT, InvestigationState, short,
)
from shared.inspector import INCOMPLETE, run_inspection
from shared.project_context import ProjectSession
from shared.project_tools import ProjectTools
from shared.trace import TurnTrace
from shared.turn_controller import TurnController
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

# With project context attached the spoken part is a summary; the detail
# stays in the terminal (see shared/context_builder.PROJECT_PROMPT).
SYSTEM_PROMPT_PROJECT = (
    "You are Resonance, a voice-first engineering companion. Be precise and "
    "separate observed evidence from hypotheses."
)

# Answer-quality switches, measured in docs/BASELINE.md.
TURN_HINTS = os.environ.get("RESONANCE_TURN_HINTS", "true").lower() != "false"
FINAL_THINKING = os.environ.get("RESONANCE_FINAL_THINKING", "false").lower() == "true"
FINAL_MODEL = os.environ.get("RESONANCE_FINAL_MODEL") or None

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
    def __init__(self, project: str | None = None, context: list[str] | None = None):
        self.project: ProjectSession | None = None
        self.tools: ProjectTools | None = None
        self.last_trace: list[dict] = []
        self.ledger: list = []  # everything inspected this session, for citation checks
        self.state = InvestigationState()  # session memory for the current investigation
        self.last_spoken = ""
        self.last_reply = ""
        self.muted = False
        self._pressed_at: float | None = None
        self._mic_failed = False
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
        self.turns = TurnController()
        self.last_turn: TurnTrace | None = None
        if project:
            self.set_project(project)
        for path in context or []:
            self.attach(path)

    def set_project(self, root: str) -> None:
        self.project = ProjectSession(root)
        self.tools = ProjectTools(self.project)
        self.ledger.clear()
        self.state.clear()  # another project's investigation must not carry over
        with self._history_lock:
            self.history.clear()  # don't carry another project's context over
        print(f"  Project: {self.project.root}")

    def attach(self, path: str) -> None:
        if self.project is None:
            print("  No project selected. Start with --project <dir> or /project <dir>.")
            return
        result = self.project.attach(path)
        if result.status != "OK":
            print(f"  [{result.status}] {result.message}")
            return
        src = result.sources[0]
        notes = []
        if src.truncated:
            notes.append("truncated")
        if src.redactions:
            notes.append(f"{src.redactions} secret-like value(s) redacted")
        suffix = f" ({', '.join(notes)})" if notes else ""
        print(f"  Attached {src.ref}, {len(src.content)} chars{suffix}")

    def _prune_ledger(self) -> set[str]:
        """Forget inspected evidence whose file changed since it was read, so
        old observations cannot be cited as if still true."""
        root = self.project.root
        fresh, changed = [], set()
        for src in self.ledger:
            try:
                if (root / src.path).stat().st_mtime <= src.captured_at:
                    fresh.append(src)
                else:
                    changed.add(src.path)
            except OSError:
                changed.add(src.path)
        self.ledger[:] = fresh
        return changed

    @staticmethod
    def _print_call(call: dict) -> None:
        args = ", ".join(f"{k}={v!r}" for k, v in (call["args"] or {}).items()) if isinstance(call["args"], dict) else "?"
        flag = " (truncated)" if call["truncated"] else ""
        print(f"  [tool] {call['tool']}({args}) -> {call['status']}{flag}")

    def _command(self, text: str) -> bool:
        """Handles /project, /attach, /context. Returns True if consumed."""
        if not text.startswith("/"):
            return False
        cmd, _, arg = text.partition(" ")
        arg = arg.strip()
        if cmd == "/project" and arg:
            try:
                self.set_project(arg)
            except NotADirectoryError as exc:
                print(f"  [NOT_FOUND] {exc}")
        elif cmd == "/attach" and arg:
            self.attach(arg)
        elif cmd == "/context":
            if not self.project or not self.project.attachments:
                print("  No context attached.")
            else:
                _, trace = build_context_block(list(self.project.attachments.values()))
                for t in trace:
                    flag = " (omitted for budget)" if t["omitted_for_budget"] else ""
                    print(f"  {t['path']}  lines {t['lines']}  {t['chars_sent']} chars{flag}")
        elif cmd == "/trace":
            t = self.last_turn
            if not t:
                print("  No turns yet.")
            else:
                print(f"  {t.summary()}")
                for c in getattr(t, "tool_calls", []):
                    print(f"  [tool] {c['tool']} {c['args']} -> {c['status']} ({c['chars']} chars)")
        elif cmd in ("/state", "/notes"):
            print(self.state.notes() if not self.state.is_empty() else "  No investigation notes yet.")
        elif cmd == "/recap":
            self._say_local(*self.state.recap())
        elif cmd == "/verify":
            num, _, result = arg.partition(" ")
            ok = num.lstrip("eE").isdigit() and self.state.verify(int(num.lstrip("eE")), result)
            print("  Experiment marked verified." if ok else "  Usage: /verify <experiment id> <what you observed>")
        elif cmd == "/ruleout":
            num, _, reason = arg.partition(" ")
            ok = num.lstrip("hH").isdigit() and self.state.rule_out(int(num.lstrip("hH")), reason.strip() or "engineer ruled it out")
            print("  Hypothesis ruled out." if ok else "  Usage: /ruleout <hypothesis id> [reason]")
        elif cmd == "/mute":
            self.muted = True
            self.voice.stop()
            print("  Muted. Replies print only; /unmute to hear them again.")
        elif cmd == "/unmute":
            self.muted = False
            print("  Unmuted.")
        elif cmd == "/preview":
            if not self.project or not self.project.attachments:
                print("  No context attached.")
            else:
                block, _ = build_context_block(list(self.project.attachments.values()))
                print(block)
                print("  ^ exactly this text, plus your question, is sent to NVIDIA NIM.")
        elif cmd == "/clear":
            if self.project:
                self.project.clear()
            self.ledger.clear()
            self.state.clear()
            print("  Context and investigation notes cleared.")
        else:
            print("  Commands: /project <dir>, /attach <file>, /context, /preview, /trace, /state, /recap,\n  /verify <id> <result>, /ruleout <id>, /mute, /unmute, /clear")
        return True

    def _speak(self, spoken: str, trace: TurnTrace) -> None:
        if not VOICE_OUTPUT or self.muted:
            return
        print("  Speaking...")
        self._speaking.set()
        trace.mark("speech_start")
        try:
            self.voice.speak(spoken)
        except Exception as exc:
            trace.status = "SPEECH_ERROR"
            print(f"  [ERROR] speech failed ({exc}); reply is shown above as text.")
        finally:
            self._speaking.clear()

    def _say_local(self, spoken: str, notes: str | None = None, trace: TurnTrace | None = None) -> None:
        """Answer without calling the model: print the full notes, speak the short line."""
        if notes:
            print(notes)
        print(f"  Spoken: {spoken}")
        self.last_spoken = spoken
        self._speak(spoken, trace or TurnTrace(0, "local"))

    def _expansion(self) -> str:
        """Spoken detail for 'tell me more', built from the last answer's
        observations, with citations stripped."""
        from shared.investigation_state import _bullets
        from shared.context_builder import _section
        items = [short(b, 14) for b in _bullets(_section(self.last_reply, "OBSERVATIONS") or "")[:3]]
        return ("More detail: " + "; ".join(items) + ".") if items else "There is no more detail on that."

    def _local_intent(self, text: str):
        """Recap, replay, and expand are answered locally: instant, free, and
        a recap can never describe a proposal as a verified result."""
        if RECAP_INTENT.search(text):
            spoken, notes = self.state.recap()
            return spoken, notes
        if REPLAY_INTENT.search(text) and self.last_spoken:
            return self.last_spoken, None
        if EXPAND_INTENT.search(text) and self.last_reply:
            return self._expansion(), None
        return None

    def handle(self, text: str, source: str, activated_at: float | None = None,
               transcribed_at: float | None = None) -> None:
        if not text.strip():
            return
        if source == "typed" and self._command(text.strip()):
            return

        logger.info("[%s input] %s", source, text)
        self._interrupt_speech()  # a new turn always silences the previous answer
        turn_id = self.turns.begin()
        number = self.turns.number(turn_id)
        trace = self.last_turn = TurnTrace(number, source, start=activated_at)
        if transcribed_at is not None:
            trace.mark_at("transcribed", transcribed_at)

        local = self._local_intent(text)
        if local:
            trace.status = "LOCAL"
            self._say_local(local[0], local[1], trace)
            logger.info(trace.summary())
            return
        print("  Thinking...")

        with self._history_lock:
            history = self.history[-MAX_HISTORY_TURNS:]
        system = SYSTEM_PROMPT
        sources = []
        inspecting = self.tools is not None
        if inspecting:
            changed = set(self.project.refresh_stale())
            for rel in changed:
                print(f"  (re-read {rel}: file changed)")
            changed |= self._prune_ledger()
            if changed and self.state.mark_stale(changed):
                print("  (some earlier observations are now stale; notes updated)")
            self.state.apply_user_turn(text, number)  # the engineer's constraints take effect this turn
            sources = list(self.project.attachments.values())
            block, self.last_trace = build_context_block(sources)
            trace.context = self.last_trace
            system = SYSTEM_PROMPT_PROJECT + inspection_prompt(sources)
            memory = self.state.to_prompt()
            if memory:
                system += "\n\n" + memory
            if block:
                system += "\n\nAttached files:\n\n" + block
        messages = [{"role": "system", "content": system}] + history
        hint = turn_hint(text) if (inspecting and TURN_HINTS) else ""
        messages.append({"role": "user", "content": f"{text}\n\n[Guidance: {hint}]" if hint else text})

        evidence = list(sources) + self.ledger
        try:
            if inspecting:
                result = run_inspection(
                    self.nim, messages, self.tools,
                    is_current=lambda: self.turns.is_current(turn_id),
                    on_call=self._print_call,
                    require_tool=len(text.split()) >= 4,  # skip for 'thanks'-style turns
                    final_thinking=FINAL_THINKING, final_model=FINAL_MODEL,
                )
                trace.tool_calls = result.calls
                evidence += result.evidence
                self.ledger.extend(result.evidence)
                reply = result.reply
                if result.status == INCOMPLETE:
                    print(f"  ! Investigation incomplete: {result.reason}")
                if reply is None:
                    if not self.turns.is_current(turn_id):
                        trace.status = "STALE"
                        return
                    raise ValueError("no answer produced")
            else:
                reply = self.nim.chat(messages)
        except Exception as exc:
            trace.status = "ERROR"
            if self.turns.is_current(turn_id):
                print(f"  [ERROR] model request failed: {exc}. Try again, or type your question.")
            return
        trace.mark("model_reply")

        if not self.turns.is_current(turn_id):
            trace.status = "STALE"  # user moved on; drop this answer entirely
            logger.info("dropped stale reply for turn %s", turn_id)
            return

        spoken, detail = split_reply(reply) if inspecting else (reply, reply)
        if inspecting:
            spoken = hedge_spoken(spoken, reply)
            self.state.apply_reply(reply, evidence, number)
            self.state.apply_challenge(text, f"{spoken} {detail}", number)
        self.last_spoken, self.last_reply = spoken, reply
        with self._history_lock:
            self.history.append({"role": "user", "content": text})
            self.history.append({"role": "assistant",
                                 "content": compact_reply(reply) if inspecting else reply})

        if inspecting:
            print(f"> {detail}")
            print(f"  Spoken: {spoken}")
            bad = check_citations(detail, evidence)
            if bad:
                print(f"  ! Unsupported citations (not in anything inspected this session): {', '.join(bad)}")
            if self.state.reasserted:
                print(f"  ! Re-raised a ruled-out idea: {self.state.reasserted[-1]}")
        else:
            print(f"> {reply}")
        _notify("Resonance", spoken)

        self._speak(spoken, trace)
        logger.info(trace.summary())

    def _interrupt_speech(self) -> None:
        """Stop playback if speaking and record it on the turn being cut off."""
        if self._speaking.is_set():
            if self.last_turn:
                self.last_turn.mark("interrupted")
                self.last_turn.status = "INTERRUPTED"
            self.voice.stop()

    def _start_hotkey_listener(self) -> None:
        def on_press(key):
            if key == HOTKEY:
                self.turns.cancel()  # barge-in: any in-flight reply is now stale
                self._interrupt_speech()  # cut off the current reply
                self._pressed_at = time.monotonic()
                try:
                    self.capture.start()
                    self._mic_failed = False
                except Exception as exc:
                    self._mic_failed = True
                    print(f"  [ERROR] microphone unavailable ({exc}). Type your question instead.")

        def on_release(key):
            if key == HOTKEY:
                if self._mic_failed:
                    return
                try:
                    text = self.capture.stop_and_transcribe()
                except Exception as exc:
                    print(f"  [ERROR] transcription failed ({exc}). Type your question instead.")
                    return
                transcribed_at = time.monotonic()
                if text:
                    # Runs off the listener thread so holding the hotkey
                    # again (barge-in) is caught immediately instead of
                    # waiting for this reply to finish speaking.
                    threading.Thread(
                        target=self.handle, args=(text, "voice"),
                        kwargs={"activated_at": self._pressed_at, "transcribed_at": transcribed_at},
                        daemon=True,
                    ).start()

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


def preflight(project: str | None = None, check_provider: bool = True) -> bool:
    """Startup checks (brief B12). Prints one line per check, never a key."""
    ok = True

    def report(name, passed, detail=""):
        nonlocal ok
        ok = ok and passed
        print(f"  [{'OK' if passed else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

    for var in ("NVIDIA_API_KEY", "ELEVENLABS_API_KEY"):
        report(var, bool(os.environ.get(var)), "" if os.environ.get(var) else "not set")
    if project:
        report("project root", os.path.isdir(os.path.expanduser(project)), project)
    try:
        import sounddevice as sd
        sd.check_input_settings()
        report("microphone", True)
    except Exception as exc:
        report("microphone", False, f"{exc} (typed input still works)")
    if check_provider and os.environ.get("NVIDIA_API_KEY"):
        try:
            NimClient(NimConfig(api_key=os.environ["NVIDIA_API_KEY"], max_tokens=5)).chat(
                [{"role": "user", "content": "ping"}], thinking=False)
            report("NVIDIA NIM", True)
        except Exception as exc:
            report("NVIDIA NIM", False, str(exc)[:120])
    return ok


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Resonance voice companion")
    parser.add_argument("--project", help="project root Resonance may inspect (read-only)")
    parser.add_argument("--context", action="append", default=[],
                        help="file to attach, relative to the project root (repeatable)")
    parser.add_argument("--preflight", action="store_true", help="check setup and exit")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    if args.preflight:
        sys.exit(0 if preflight(args.project) else 1)
    if args.context and not args.project:
        sys.exit("--context requires --project")
    Resonance(project=args.project, context=args.context).run()
