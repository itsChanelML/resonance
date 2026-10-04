import os

import pytest

from shared.context_builder import (
    build_context_block, check_citations, compact_reply, hedge_spoken, missing_items, split_reply,
)
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


def test_fallback_never_reads_labels_or_citations_aloud():
    spoken, _ = split_reply("OBSERVATIONS: - a.py:1-2 shows x is one HYPOTHESES: maybe")
    assert spoken == "shows x is one" or "OBSERVATIONS" not in spoken and "a.py" not in spoken
    long_reply = "OBSERVATIONS: " + "word " * 100
    assert len(split_reply(long_reply)[0].split()) <= 46


LATE = "OBSERVATIONS:\n- a.py:1-2 f\nNEXT: Print docs.\nMISSING: the retriever code\nSPOKEN: The retriever ties are broken in order."


def test_spoken_can_come_last_and_is_removed_from_detail():
    spoken, detail = split_reply(LATE)
    assert spoken == "The retriever ties are broken in order." and "SPOKEN" not in detail
    assert detail.endswith("MISSING: the retriever code")


def test_spoken_is_hedged_when_something_is_missing():
    spoken, _ = split_reply(LATE)
    assert missing_items(LATE) == "the retriever code"
    assert hedge_spoken(spoken, LATE).startswith("Not fully confirmed.")
    assert hedge_spoken("This is likely the cause.", LATE) == "This is likely the cause."
    assert hedge_spoken(spoken, LATE.replace("the retriever code", "none")) == spoken


def test_compact_reply_keeps_spoken_only():
    out = compact_reply(LATE)
    assert out == "The retriever ties are broken in order."


def test_citation_content_check(tmp_path):
    (tmp_path / "pb.py").write_text("\n".join(["# c", "MAX = 1", "", "x = 2"]))
    sess = ProjectSession(tmp_path)
    sess.attach("pb.py")
    srcs = list(sess.attachments.values())
    assert check_citations("- `MAX` is 1 at pb.py:2-2", srcs) == []
    flagged = check_citations("- `MAX` is 1 at pb.py:3-3", srcs)
    assert flagged and "do not contain MAX" in flagged[0]
    assert check_citations("- it is plain at pb.py:3-3", srcs) == []  # no identifier to check


def test_inline_sections_and_bullet_prefix():
    reply = "OBSERVATIONS: a.py:1-2 f HYPOTHESES: maybe NEXT: Print docs. MISSING: none SPOKEN: - It keeps order."
    spoken, detail = split_reply(reply)
    assert spoken == "It keeps order." and "SPOKEN" not in detail and "NEXT: Print docs." in detail
    assert missing_items(reply) == ""
    assert compact_reply(reply) == "It keeps order."


def test_section_header_is_not_treated_as_an_identifier(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    sess = ProjectSession(tmp_path)
    sess.attach("a.py")
    assert check_citations("OBSERVATIONS: x is set at a.py:1-1", list(sess.attachments.values())) == []


def test_turn_hint_routes_challenges_and_experiments_only():
    from shared.context_builder import CHALLENGE_HINT, EXPERIMENT_HINT, turn_hint
    assert turn_hint("You suggested changing the model. What evidence connects this?") == CHALLENGE_HINT
    assert turn_hint("We only have time for one experiment. Which comparison separates those causes?") == EXPERIMENT_HINT
    assert turn_hint("Find where build evidence is called.") == ""


def test_experiment_hint_prefers_holding_retrieval_fixed():
    from shared.context_builder import EXPERIMENT_HINT
    assert "downstream of retrieval" in EXPERIMENT_HINT and "retrieved documents" in EXPERIMENT_HINT
