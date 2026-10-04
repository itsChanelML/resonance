import json
import os
import subprocess

import pytest

from shared.inspector import CANCELLED, COMPLETE, INCOMPLETE, run_inspection
from shared.project_context import ProjectSession
from shared.project_tools import ProjectTools


@pytest.fixture
def tools(tmp_path):
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "src" / "pb.py").write_text("def build(docs):\n    evidence = docs[:1]\n    return evidence\n")
    (root / "src" / "caller.py").write_text("from pb import build\nbuild([1, 2])\n")
    (root / ".env").write_text("SECRET=hunter2hunter2hunter2\n")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "x.js").write_text("evidence")
    (tmp_path / "outside.py").write_text("evidence = 1\n")
    os.symlink(tmp_path / "outside.py", root / "src" / "link.py")
    return ProjectTools(ProjectSession(root))


def test_list_hides_excluded_and_outside_symlinks(tools):
    out = tools.call("list_files", {}).output
    assert "src/pb.py" in out and ".env" not in out and "node_modules" not in out and "link.py" not in out


def test_list_blocks_traversal_and_excluded_dir(tools):
    assert tools.call("list_files", {"relative_path": ".."}).status == "BLOCKED"
    assert tools.call("list_files", {"relative_path": "node_modules"}).status == "BLOCKED"


def test_search_finds_literal_and_skips_excluded(tools):
    r = tools.call("search_code", {"query": "docs[:1]"})
    assert r.status == "OK" and "src/pb.py:2:" in r.output
    assert r.sources[0].ref == "src/pb.py:2-2"
    out = tools.call("search_code", {"query": "evidence"}).output
    assert "node_modules" not in out and "outside" not in out and "link.py" not in out
    assert "Searched, no match" not in out or ".env" not in out


def test_search_zero_results_is_bounded_and_clear(tools):
    out = tools.call("search_code", {"query": "nonexistent_symbol"}).output
    assert out.startswith("0 matches") and "files searched" in out


def test_search_reports_files_without_match_and_glob(tools):
    out = tools.call("search_code", {"query": "docs[:1]", "glob": "*.py"}).output
    assert "Searched, no match: src/caller.py" in out


def test_search_caps_matches(tmp_path):
    (tmp_path / "big.py").write_text("hit\n" * 100)
    r = ProjectTools(ProjectSession(tmp_path)).call("search_code", {"query": "hit", "max_matches": 500})
    assert len(r.sources) == 20 and r.truncated


def test_read_file_bounds_and_policy(tools):
    r = tools.call("read_file", {"path": "src/pb.py", "start_line": 2, "end_line": 3})
    assert r.status == "OK" and r.output.splitlines()[1].startswith("2: ")
    assert tools.call("read_file", {"path": ".env"}).status == "BLOCKED"
    assert tools.call("read_file", {"path": "src/link.py"}).status == "BLOCKED"
    assert tools.call("read_file", {"path": "../outside.py"}).status == "BLOCKED"
    assert tools.call("read_file", {"path": "missing.py"}).status == "NOT_FOUND"


def test_read_file_caps_at_200_lines(tmp_path):
    (tmp_path / "long.py").write_text("\n".join(f"x{i}" for i in range(500)))
    src = ProjectTools(ProjectSession(tmp_path)).call("read_file", {"path": "long.py"}).sources[0]
    assert (src.start_line, src.end_line) == (1, 200)


def test_malformed_arguments_become_errors(tools):
    assert tools.call("read_file", {}).status == "ERROR"
    assert tools.call("read_file", {"path": "a.py", "start_line": "x"}).status == "ERROR"
    assert tools.call("nope", {}).status == "ERROR"
    assert tools.call("search_code", None).status == "ERROR"


def _git(root, *a):
    subprocess.run(["git", "-C", str(root), "-c", "user.email=t@t", "-c", "user.name=t", *a],
                   check=True, capture_output=True)


def test_git_diff_non_git_and_clean_and_changes(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    t = ProjectTools(ProjectSession(tmp_path))
    assert t.call("git_diff", {"target": "unstaged"}).status == "NOT_GIT"
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "init")
    assert "clean" in t.call("git_diff", {"target": "unstaged"}).output
    (tmp_path / "a.py").write_text("x = 2\n")
    (tmp_path / "new.py").write_text("y = 1\n")
    r = t.call("git_diff", {"target": "unstaged"})
    assert "+x = 2" in r.output and "Untracked files" in r.output and "new.py" in r.output
    assert r.sources[0].path == "a.py"
    _git(tmp_path, "add", "a.py")
    assert "+x = 2" in t.call("git_diff", {"target": "staged"}).output
    assert "+x = 2" in t.call("git_diff", {"target": "baseline", "baseline": "HEAD"}).output


def test_git_diff_rejects_option_injection_and_hides_excluded(tmp_path):
    t = ProjectTools(ProjectSession(tmp_path))
    assert t.call("git_diff", {"target": "baseline", "baseline": "--output=/tmp/x"}).status == "ERROR"
    assert t.call("git_diff", {"target": "baseline", "baseline": "a..b"}).status == "ERROR"
    (tmp_path / ".env").write_text("A=1\n")
    (tmp_path / "ok.py").write_text("a = 1\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-f", ".")
    _git(tmp_path, "commit", "-qm", "init")
    (tmp_path / ".env").write_text("A=2\n")
    (tmp_path / "ok.py").write_text("a = 2\n")
    out = t.call("git_diff", {"target": "unstaged"}).output
    assert "+a = 2" in out and "A=2" not in out and "hidden by exclusion" in out


# ---- inspector ----------------------------------------------------------
class FakeNim:
    def __init__(self, *messages):
        self.messages = list(messages)
        self.calls = []

    def chat_with_tools(self, messages, tools, max_tokens=None, thinking=None, tool_choice="auto"):
        self.calls.append((len(messages), tools))
        self.choices = getattr(self, "choices", []) + [tool_choice]
        return self.messages.pop(0)


def _call(name, args, id="c1"):
    return {"content": None, "tool_calls": [{"id": id, "type": "function",
            "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}]}


def test_loop_runs_tool_then_answers(tools):
    nim = FakeNim(_call("search_code", {"query": "docs[:1]"}), {"content": "SPOKEN: found it"})
    r = run_inspection(nim, [{"role": "user", "content": "q"}], tools)
    assert r.status == COMPLETE and r.reply == "SPOKEN: found it"
    assert r.calls[0]["tool"] == "search_code" and r.evidence[0].path == "src/pb.py"


def test_loop_survives_malformed_tool_arguments(tools):
    nim = FakeNim(_call("read_file", "{not json"), {"content": "done"})
    r = run_inspection(nim, [], tools)
    assert r.status == COMPLETE and r.calls[0]["status"] == "ERROR"


def test_loop_stops_after_max_rounds_and_reports_incomplete(tools):
    nim = FakeNim(*[_call("list_files", {}, f"c{i}") for i in range(3)], {"content": "partial answer"})
    r = run_inspection(nim, [], tools, max_rounds=3)
    assert r.status == INCOMPLETE and "rounds" in r.reason and r.reply == "partial answer"
    assert nim.calls[-1][1] is None  # final forced answer offers no tools


def test_loop_deadline(tools):
    nim = FakeNim({"content": "forced"})
    r = run_inspection(nim, [], tools, deadline_s=-1)
    assert r.status == INCOMPLETE and "deadline" in r.reason


def test_loop_cancellation_discards_answer(tools):
    nim = FakeNim({"content": "obsolete"})
    r = run_inspection(nim, [], tools, is_current=lambda: False)
    assert r.status == CANCELLED and r.reply is None


def test_first_hop_requires_a_tool_only_when_asked(tools):
    nim = FakeNim(_call("list_files", {}), {"content": "done"})
    run_inspection(nim, [], tools, require_tool=True)
    assert nim.choices == ["required", "auto"]
    nim = FakeNim({"content": "done"})
    run_inspection(nim, [], tools)
    assert nim.choices == ["auto"]
