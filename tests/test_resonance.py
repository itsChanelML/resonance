from unittest.mock import MagicMock, patch

import pytest

import resonance


class _SyncThread:
    """Stand-in for threading.Thread that runs its target immediately,
    so dispatched work (e.g. handle() from the hotkey listener) can be
    asserted on synchronously instead of racing a real background thread."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-test")
    with patch("resonance.VoiceCapture"), \
         patch("resonance.UsageTracker"), \
         patch("resonance.NimClient"), \
         patch("resonance.ElevenLabsVoice"):
        instance = resonance.Resonance()
    instance.nim.chat.return_value = "a spoken reply"
    return instance


def _wire_hotkey_listener(app):
    with patch("resonance.keyboard.Listener") as mock_listener_cls:
        mock_listener = MagicMock()
        mock_listener_cls.return_value = mock_listener
        app._start_hotkey_listener()
    mock_listener.start.assert_called_once()
    _, kwargs = mock_listener_cls.call_args
    return kwargs["on_press"], kwargs["on_release"]


class TestHandle:
    def test_blank_text_is_a_noop(self, app):
        app.handle("   ", source="typed")
        app.nim.chat.assert_not_called()

    def test_updates_history_and_speaks_when_voice_output_on(self, app, monkeypatch):
        monkeypatch.setattr(resonance, "VOICE_OUTPUT", True)
        app.handle("hello", source="typed")
        assert app.history[-2] == {"role": "user", "content": "hello"}
        assert app.history[-1] == {"role": "assistant", "content": "a spoken reply"}
        app.voice.speak.assert_called_once_with("a spoken reply")

    def test_skips_speak_when_voice_output_off(self, app, monkeypatch):
        monkeypatch.setattr(resonance, "VOICE_OUTPUT", False)
        app.handle("hello", source="typed")
        app.voice.speak.assert_not_called()

    def test_speaking_flag_clears_even_if_speak_raises(self, app, monkeypatch):
        monkeypatch.setattr(resonance, "VOICE_OUTPUT", True)
        app.voice.speak.side_effect = RuntimeError("playback died")
        app.handle("hello", source="typed")  # speech failure falls back to text
        assert not app._speaking.is_set()
        assert app.last_turn.status == "SPEECH_ERROR"

    def test_only_recent_history_is_sent_to_nim(self, app, monkeypatch):
        monkeypatch.setattr(resonance, "VOICE_OUTPUT", False)
        for i in range(10):
            app.handle(f"message {i}", source="typed")
        sent_messages = app.nim.chat.call_args.args[0]
        # system prompt + last MAX_HISTORY_TURNS history entries + this turn's user message
        assert len(sent_messages) == 1 + resonance.MAX_HISTORY_TURNS + 1
        assert sent_messages[0]["role"] == "system"
        assert sent_messages[-1] == {"role": "user", "content": "message 9"}


class TestHotkeyListener:
    def test_press_starts_capture(self, app):
        on_press, _ = _wire_hotkey_listener(app)
        on_press(resonance.HOTKEY)
        app.capture.start.assert_called_once()
        app.voice.stop.assert_not_called()

    def test_press_while_speaking_interrupts_first(self, app):
        on_press, _ = _wire_hotkey_listener(app)
        app._speaking.set()
        on_press(resonance.HOTKEY)
        app.voice.stop.assert_called_once()
        app.capture.start.assert_called_once()

    def test_press_ignores_other_keys(self, app):
        on_press, _ = _wire_hotkey_listener(app)
        on_press(resonance.keyboard.Key.esc)
        app.capture.start.assert_not_called()

    def test_release_with_transcript_dispatches_handle(self, app, monkeypatch):
        monkeypatch.setattr(resonance, "VOICE_OUTPUT", False)
        _, on_release = _wire_hotkey_listener(app)
        app.capture.stop_and_transcribe.return_value = "what does this error mean"

        with patch("resonance.threading.Thread", _SyncThread):
            on_release(resonance.HOTKEY)

        app.nim.chat.assert_called_once()
        assert app.history[-2] == {"role": "user", "content": "what does this error mean"}

    def test_release_with_no_transcript_does_not_dispatch(self, app):
        _, on_release = _wire_hotkey_listener(app)
        app.capture.stop_and_transcribe.return_value = ""
        with patch("resonance.threading.Thread") as mock_thread:
            on_release(resonance.HOTKEY)
        mock_thread.assert_not_called()

    def test_release_ignores_other_keys(self, app):
        _, on_release = _wire_hotkey_listener(app)
        with patch("resonance.threading.Thread") as mock_thread:
            on_release(resonance.keyboard.Key.esc)
        app.capture.stop_and_transcribe.assert_not_called()
        mock_thread.assert_not_called()


class TestResolveHotkey:
    def test_named_key(self):
        assert resonance._resolve_hotkey("f9") == resonance.keyboard.Key.f9

    def test_single_character(self):
        assert resonance._resolve_hotkey("a") == resonance.keyboard.KeyCode.from_char("a")

    def test_unrecognized_name_raises(self):
        with pytest.raises(ValueError, match="Unrecognized RESONANCE_HOTKEY"):
            resonance._resolve_hotkey("not_a_real_key")


class TestNotify:
    def test_noop_off_darwin(self, monkeypatch):
        monkeypatch.setattr(resonance.sys, "platform", "linux")
        with patch("resonance.subprocess.run") as mock_run:
            resonance._notify("title", "message")
        mock_run.assert_not_called()

    def test_noop_without_osascript(self, monkeypatch):
        monkeypatch.setattr(resonance.sys, "platform", "darwin")
        monkeypatch.setattr(resonance.shutil, "which", lambda name: None)
        with patch("resonance.subprocess.run") as mock_run:
            resonance._notify("title", "message")
        mock_run.assert_not_called()

    def test_escapes_quotes_in_the_applescript(self, monkeypatch):
        monkeypatch.setattr(resonance.sys, "platform", "darwin")
        monkeypatch.setattr(resonance.shutil, "which", lambda name: "/usr/bin/osascript")
        with patch("resonance.subprocess.run") as mock_run:
            resonance._notify('a "quoted" title', 'a "quoted" reply')
        script = mock_run.call_args.args[0][2]
        assert '\\"quoted\\"' in script

    def test_swallows_subprocess_errors(self, monkeypatch):
        monkeypatch.setattr(resonance.sys, "platform", "darwin")
        monkeypatch.setattr(resonance.shutil, "which", lambda name: "/usr/bin/osascript")
        with patch("resonance.subprocess.run", side_effect=OSError("no such tool")):
            resonance._notify("title", "message")  # must not raise


class TestRun:
    def test_typed_input_dispatches_handle_until_eof(self, app, monkeypatch):
        monkeypatch.setattr(resonance, "VOICE_OUTPUT", False)
        with patch.object(resonance.Resonance, "_start_hotkey_listener") as mock_listener, \
             patch("builtins.input", side_effect=["hello", EOFError]):
            app.run()
        mock_listener.assert_called_once()
        app.nim.chat.assert_called_once()

    def test_keyboard_interrupt_breaks_the_loop(self, app):
        with patch.object(resonance.Resonance, "_start_hotkey_listener"), \
             patch("builtins.input", side_effect=KeyboardInterrupt):
            app.run()  # must return instead of raising


class TestTurnsAndErrors:
    def test_model_error_leaves_app_usable(self, app, capsys):
        app.nim.chat.side_effect = RuntimeError("503")
        app.handle("hello", source="typed")
        assert "[ERROR]" in capsys.readouterr().out
        assert app.history == [] and app.last_turn.status == "ERROR"

    def test_stale_reply_is_dropped(self, app, monkeypatch):
        monkeypatch.setattr(resonance, "VOICE_OUTPUT", True)

        def barge_in(*a, **k):
            app.turns.cancel()  # user pressed the hotkey while the model was working
            return "obsolete answer"

        app.nim.chat.side_effect = barge_in
        app.handle("hello", source="typed")
        app.voice.speak.assert_not_called()
        assert app.history == [] and app.last_turn.status == "STALE"

    def test_hotkey_press_cancels_in_flight_turn(self, app):
        on_press, _ = _wire_hotkey_listener(app)
        tid = app.turns.begin()
        on_press(resonance.HOTKEY)
        assert not app.turns.is_current(tid)

    def test_project_mode_disables_thinking_and_splits_speech(self, app, tmp_path, monkeypatch):
        monkeypatch.setattr(resonance, "VOICE_OUTPUT", True)
        (tmp_path / "a.py").write_text("x = 1\n")
        app.set_project(str(tmp_path))
        app.attach("a.py")
        app.nim.chat_with_tools.return_value = {"content": "SPOKEN: Short.\nOBSERVATIONS:\n- a.py:1-1 x"}
        app.handle("why", source="typed")
        assert app.nim.chat_with_tools.call_args.kwargs["thinking"] is False
        app.voice.speak.assert_called_once_with("Short.")

    def test_slash_commands_are_not_sent_to_model(self, app):
        app.handle("/context", source="typed")
        app.nim.chat.assert_not_called()


class TestPreflight:
    def test_reports_missing_keys(self, monkeypatch, capsys):
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        assert resonance.preflight(check_provider=False) is False
        assert "[FAIL] NVIDIA_API_KEY" in capsys.readouterr().out


class TestReviewFixes:
    def test_interruption_is_recorded(self, app):
        app.last_turn = resonance.TurnTrace(1, "voice")
        app._speaking.set()
        app._interrupt_speech()
        app.voice.stop.assert_called_once()
        assert app.last_turn.status == "INTERRUPTED" and "interrupted" in app.last_turn.marks

    def test_turn_numbers_skip_cancels(self, app):
        app.turns.cancel()
        tid = app.turns.begin()
        assert app.turns.number(tid) == 1

    def test_spoken_line_printed_and_prompt_uses_real_path(self, app, tmp_path, capsys):
        (tmp_path / "a.py").write_text("x = 1\n")
        app.set_project(str(tmp_path))
        app.attach("a.py")
        app.nim.chat_with_tools.return_value = {"content": "SPOKEN: Short.\nOBSERVATIONS:\n- a.py:1-1 x"}
        app.handle("why", source="typed")
        assert "Spoken: Short." in capsys.readouterr().out
        system = app.nim.chat_with_tools.call_args.args[0][0]["content"]
        assert "a.py:1-1" in system and "src/a.py" not in system


class TestSessionEvidence:
    def _project(self, app, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\ny = 2\n")
        app.set_project(str(tmp_path))

    def test_citation_from_earlier_turn_is_not_flagged(self, app, tmp_path, capsys):
        self._project(app, tmp_path)
        read = {"content": None, "tool_calls": [{"id": "1", "type": "function", "function": {
            "name": "read_file", "arguments": '{"path": "a.py"}'}}]}
        app.nim.chat_with_tools.side_effect = [read, {"content": "SPOKEN: ok\nOBSERVATIONS:\n- a.py:1-2 x"},
                                               {"content": "SPOKEN: again\nOBSERVATIONS:\n- a.py:2-2 y"}]
        app.handle("first", source="typed")
        app.handle("second", source="typed")  # no tool call this turn
        assert "Unsupported citations" not in capsys.readouterr().out

    def test_changed_file_evidence_is_forgotten(self, app, tmp_path):
        import os, time
        self._project(app, tmp_path)
        app.nim.chat_with_tools.return_value = {"content": "SPOKEN: ok"}
        app.ledger.append(app.tools.read_file({"path": "a.py"}).sources[0])
        future = time.time() + 5
        os.utime(tmp_path / "a.py", (future, future))
        app._prune_ledger()
        assert app.ledger == []

    def test_switching_project_clears_ledger(self, app, tmp_path):
        self._project(app, tmp_path)
        app.ledger.append(app.tools.read_file({"path": "a.py"}).sources[0])
        app.set_project(str(tmp_path))
        assert app.ledger == []


class TestTurnHintsAndFinalHop:
    def _project(self, app, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        app.set_project(str(tmp_path))

    def test_hint_goes_to_model_but_not_history(self, app, tmp_path, monkeypatch):
        self._project(app, tmp_path)
        monkeypatch.setattr(resonance, "TURN_HINTS", True)
        app.nim.chat_with_tools.return_value = {"content": "SPOKEN: ok"}
        app.handle("What evidence connects this regression to the model?", source="typed")
        sent = app.nim.chat_with_tools.call_args.args[0][-1]["content"]
        assert "[Guidance:" in sent
        assert app.history[0]["content"] == "What evidence connects this regression to the model?"

    def test_hints_can_be_switched_off(self, app, tmp_path, monkeypatch):
        self._project(app, tmp_path)
        monkeypatch.setattr(resonance, "TURN_HINTS", False)
        app.nim.chat_with_tools.return_value = {"content": "SPOKEN: ok"}
        app.handle("What evidence connects this regression to the model?", source="typed")
        assert "[Guidance:" not in app.nim.chat_with_tools.call_args.args[0][-1]["content"]


class TestMilestone3:
    def _project(self, app, tmp_path):
        (tmp_path / "a.py").write_text("MAX = 1\nx = 2\n")
        app.set_project(str(tmp_path))

    REPLY = ("OBSERVATIONS:\n- `MAX` is 1 at a.py:1-1\nHYPOTHESES:\n- Capping evidence lowers accuracy\n"
             "NEXT: Replay identical inputs with MAX raised\nMISSING: none\nSPOKEN: Evidence is capped.")

    def _read_then(self, reply):
        call = {"content": None, "tool_calls": [{"id": "1", "type": "function", "function": {
            "name": "read_file", "arguments": '{"path": "a.py"}'}}]}
        return [call, {"content": reply}]

    def test_state_is_built_from_a_turn_and_recap_is_local(self, app, tmp_path, monkeypatch):
        monkeypatch.setattr(resonance, "VOICE_OUTPUT", True)
        self._project(app, tmp_path)
        app.nim.chat_with_tools.side_effect = self._read_then(self.REPLY)
        app.handle("Accuracy dropped after the change, what should I look at first?", source="typed")
        assert app.state.experiments[0].status == "proposed" and app.state.observations
        calls_before = app.nim.chat_with_tools.call_count
        app.voice.speak.reset_mock()
        app.handle("I'm back. Give me a 20 second recap.", source="typed")
        assert app.nim.chat_with_tools.call_count == calls_before  # no model call
        spoken = app.voice.speak.call_args.args[0]
        assert "not yet run" in spoken and "Nothing has been verified" in spoken
        assert app.last_turn.status == "LOCAL"

    def test_constraint_is_in_the_prompt_the_same_turn_and_persists(self, app, tmp_path):
        self._project(app, tmp_path)
        app.nim.chat_with_tools.return_value = {"content": "SPOKEN: ok"}
        app.handle("We only have time for one experiment, which one?", source="typed")
        first = app.nim.chat_with_tools.call_args.args[0][0]["content"]
        assert "only have time for one experiment" in first
        app.handle("What about the second file please?", source="typed")
        assert "only have time for one experiment" in app.nim.chat_with_tools.call_args.args[0][0]["content"]

    def test_challenge_rules_out_claim_and_prompt_forbids_repeating_it(self, app, tmp_path):
        self._project(app, tmp_path)
        app.nim.chat_with_tools.return_value = {"content": "SPOKEN: I did not claim that; I found no evidence."}
        app.handle("You suggested changing the model. What evidence connects this?", source="typed")
        assert app.state.hypotheses[0].status == "unsupported"
        app.nim.chat_with_tools.return_value = {"content": "SPOKEN: ok"}
        app.handle("Okay so what is the next step then?", source="typed")
        assert "do NOT present these as the cause" in app.nim.chat_with_tools.call_args.args[0][0]["content"]

    def test_replay_and_expand_are_local(self, app, tmp_path, monkeypatch):
        monkeypatch.setattr(resonance, "VOICE_OUTPUT", True)
        self._project(app, tmp_path)
        app.nim.chat_with_tools.side_effect = self._read_then(self.REPLY)
        app.handle("Why did accuracy drop after this change?", source="typed")
        n = app.nim.chat_with_tools.call_count
        app.voice.speak.reset_mock()
        app.handle("Wait, can you say that again?", source="typed")
        app.voice.speak.assert_called_once_with("Evidence is capped.")
        app.handle("Tell me more about that.", source="typed")
        assert app.voice.speak.call_args.args[0].startswith("More detail:")
        assert app.nim.chat_with_tools.call_count == n

    def test_changed_file_marks_observations_stale(self, app, tmp_path):
        import os, time
        self._project(app, tmp_path)
        app.nim.chat_with_tools.side_effect = self._read_then(self.REPLY)
        app.handle("Why did accuracy drop after this change?", source="typed")
        future = time.time() + 5
        os.utime(tmp_path / "a.py", (future, future))
        app.nim.chat_with_tools.side_effect = None
        app.nim.chat_with_tools.return_value = {"content": "SPOKEN: ok"}
        app.handle("And what about the other part now?", source="typed")
        assert app.state.observations[0].stale

    def test_switching_project_clears_investigation(self, app, tmp_path):
        self._project(app, tmp_path)
        app.state.apply_user_turn("We only have time for one experiment.", 1)
        app.set_project(str(tmp_path))
        assert app.state.is_empty() and not app.state.constraints

    def test_verify_and_ruleout_commands(self, app, tmp_path, capsys):
        self._project(app, tmp_path)
        app.state.apply_reply(self.REPLY, [], 1)
        app.handle("/verify 1 not a number of an experiment", source="typed")
        assert "Usage" in capsys.readouterr().out
        exp = app.state.experiments[0]
        app.handle(f"/verify {exp.id} accuracy recovered", source="typed")
        assert exp.status == "verified"
        hyp = app.state.hypotheses[0]
        app.handle(f"/ruleout {hyp.id} checked", source="typed")
        assert hyp.status == "ruled_out"

    def test_mute_suppresses_speech_until_unmute(self, app, monkeypatch):
        monkeypatch.setattr(resonance, "VOICE_OUTPUT", True)
        app.handle("/mute", source="typed")
        app.handle("hello there", source="typed")
        app.voice.speak.assert_not_called()
        app.handle("/unmute", source="typed")
        app.handle("hello again", source="typed")
        app.voice.speak.assert_called_once()

    def test_microphone_failure_leaves_typed_input_usable(self, app, capsys):
        on_press, on_release = _wire_hotkey_listener(app)
        app.capture.start.side_effect = RuntimeError("no input device")
        on_press(resonance.HOTKEY)
        assert "microphone unavailable" in capsys.readouterr().out
        with patch("resonance.threading.Thread") as mock_thread:
            on_release(resonance.HOTKEY)
        mock_thread.assert_not_called()
        app.handle("typed fallback works", source="typed")
        app.nim.chat.assert_called()

    def test_voice_turn_records_activation_and_transcription_marks(self, app, monkeypatch):
        monkeypatch.setattr(resonance, "VOICE_OUTPUT", False)
        _, on_release = _wire_hotkey_listener(app)
        app._pressed_at = 100.0
        app.capture.stop_and_transcribe.return_value = "what does this error mean"
        with patch("resonance.threading.Thread", _SyncThread):
            on_release(resonance.HOTKEY)
        marks = app.last_turn.elapsed()
        assert "transcribed" in marks and marks["transcribed"] > 0 and "model_reply" in marks
