import os

import pytest

from shared.context_builder import check_citations, split_reply, build_context_block
from shared.project_context import ProjectSession


@pytest.fixture
def proj(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("def f():\n    return 1\n")
    (root / ".env").write_text("KEY=secret\n")
    (root / "blob.py").write_bytes(b"\x00\x01\x02")
    (tmp_path / "outside.txt").write_text("nope")
    os.symlink(tmp_path / "outside.txt", root / "link.txt")
    return ProjectSession(root)


def test_reads_relative_file(proj):
    r = proj.attach("a.py")
    assert r.status == "OK" and r.sources[0].ref == "a.py:1-2"


def test_traversal_and_symlink_blocked(proj):
    assert proj.read_file("../outside.txt").status == "BLOCKED"
    assert proj.read_file("link.txt").status == "BLOCKED"


def test_excluded_missing_binary(proj):
    assert proj.read_file(".env").status == "BLOCKED"
    assert proj.read_file("nope.py").status == "NOT_FOUND"
    assert proj.read_file("blob.py").status == "UNSUPPORTED"


def test_resonanceignore(tmp_path):
    (tmp_path / "x.txt").write_text("hi")
    (tmp_path / ".resonanceignore").write_text("x.txt\n")
    assert ProjectSession(tmp_path).read_file("x.txt").status == "BLOCKED"


def test_secret_redaction(tmp_path):
    (tmp_path / "c.py").write_text('api_key = "abcdefghijklmnopqrstuvwxyz"\n')
    src = ProjectSession(tmp_path).read_file("c.py").sources[0]
    assert "abcdefghij" not in src.content and src.redactions == 1


def test_stale_refresh(proj):
    proj.attach("a.py")
    (proj.root / "a.py").write_text("def f():\n    return 2\n")
    assert proj.refresh_stale() == ["a.py"]


def test_budget_trace(proj):
    proj.attach("a.py")
    text, trace = build_context_block(list(proj.attachments.values()), budget_tokens=2)
    assert trace[0]["omitted_for_budget"] and len(text) < 40


def test_split_and_citations(proj):
    proj.attach("a.py")
    reply = "SPOKEN: Look at f.\nOBSERVATIONS:\n- a.py:1-2 returns 1\n- b.py:9 bogus\nNEXT: run it"
    spoken, detail = split_reply(reply)
    assert spoken == "Look at f." and "OBSERVATIONS" in detail
    assert check_citations(detail, list(proj.attachments.values())) == ["b.py:9"]
    assert split_reply("plain answer") == ("plain answer", "plain answer")


def test_gitignore_honored(tmp_path):
    (tmp_path / "x.txt").write_text("hi")
    (tmp_path / ".gitignore").write_text("# c\nx.txt\n")
    assert ProjectSession(tmp_path).read_file("x.txt").status == "BLOCKED"


def test_unicode_hyphen_and_lines_citation(proj):
    proj.attach("a.py")
    srcs = list(proj.attachments.values())
    assert check_citations("a.py lines 1‑2", srcs) == []
    assert check_citations("a.py lines 1‑9", srcs) == ["a.py:1-9"]


def test_missing_spoken_falls_back_to_next_line():
    spoken, detail = split_reply("OBSERVATIONS:\n- a.py:1-2 x\nNEXT: Print docs length.\nMISSING: none")
    assert spoken == "Print docs length." and "OBSERVATIONS" in detail
