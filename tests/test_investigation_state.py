from shared.investigation_state import (
    EXPAND_INTENT, OPEN, RECAP_INTENT, REPLAY_INTENT, RULED_OUT, UNSUPPORTED, InvestigationState,
)
from shared.project_context import ContextSource


def src(path="src/pb.py", lo=1, hi=20):
    return ContextSource(path, lo, hi, "\n".join(f"line {i} MAX_EVIDENCE" for i in range(lo, hi + 1)),
                         "h", 0.0, "tool")


REPLY = (
    "OBSERVATIONS:\n- `MAX_EVIDENCE` limits evidence at src/pb.py:4-4\n- invented at nowhere.py:9-9\n"
    "HYPOTHESES:\n- Limiting evidence to one passage lowers grounded accuracy\n"
    "NEXT: Replay identical inputs through old and new prompt assembly\n"
    "MISSING:\n- Which case regressed first\n"
    "SPOKEN: Evidence is capped at one passage."
)


def test_reply_populates_state_and_drops_unsupported_observations():
    s = InvestigationState()
    s.apply_reply(REPLY, [src()], turn=1)
    assert [o.refs for o in s.observations] == [["src/pb.py:4-4"]]  # uncited/invalid claim not remembered
    assert s.hypotheses[0].status == OPEN
    assert s.experiments[0].status == "proposed" and s.unresolved == ["Which case regressed first"]


def test_experiment_stays_proposed_until_result_supplied():
    s = InvestigationState()
    s.apply_reply(REPLY, [src()], 1)
    assert not s.verify(s.experiments[0].id, "   ")
    assert s.experiments[0].status == "proposed"
    assert s.verify(s.experiments[0].id, "accuracy recovered to 0.75")
    assert s.experiments[0].status == "verified"


def test_constraints_and_corrections_are_captured_verbatim():
    s = InvestigationState()
    s.apply_user_turn("We only have time for one experiment.", 1)
    s.apply_user_turn("Actually these are validation errors, not network failures.", 2)
    assert s.constraints == ["We only have time for one experiment."]
    assert "validation errors" in s.corrections[0]


def test_rule_out_phrase_marks_matching_hypothesis():
    s = InvestigationState()
    s.apply_reply(REPLY, [src()], 1)
    s.apply_user_turn("Rule out the limiting evidence to one passage idea.", 2)
    assert s.hypotheses[0].status == RULED_OUT


def test_assume_fine_records_ruled_out_idea_even_with_no_match():
    s = InvestigationState()
    s.apply_user_turn("Assume retrieval is fine, only look at prompt assembly.", 1)
    assert any(h.status == RULED_OUT and "retrieval" in h.text for h in s.hypotheses)
    assert s.constraints


def test_challenge_without_evidence_records_unsupported_claim():
    s = InvestigationState()
    s.apply_challenge("You suggested changing the model. What evidence connects this?",
                      "I did not claim that; I found no evidence for a model change.", 3)
    assert s.hypotheses[0].status == UNSUPPORTED and "changing the model" in s.hypotheses[0].text
    s2 = InvestigationState()
    s2.apply_challenge("You suggested changing the model.", "Yes, the model is the likely cause.", 3)
    assert s2.hypotheses == []


def test_ruled_out_idea_reraised_is_not_added_as_open():
    s = InvestigationState()
    s.rule_out_text("changing the model fixes accuracy", "engineer ruled it out")
    s.apply_reply("HYPOTHESES:\n- Changing the model fixes accuracy\nNEXT: x", [], 4)
    assert all(h.status != OPEN for h in s.hypotheses) and s.reasserted


def test_stale_observation_when_file_changes():
    s = InvestigationState()
    s.apply_reply(REPLY, [src()], 1)
    assert s.mark_stale({"src/pb.py"}) == 1 and s.observations[0].stale
    assert "STALE" in s.to_prompt()


def test_prompt_separates_ruled_out_proposed_and_verified():
    s = InvestigationState()
    s.apply_user_turn("Retrieval recall improved but accuracy dropped, what first?", 1)
    s.apply_reply(REPLY, [src()], 1)
    s.rule_out_text("changing the model", "engineer ruled it out")
    p = s.to_prompt()
    assert "do NOT present these as the cause" in p and "PROPOSED but NOT RUN" in p
    assert "Verified results" not in p


def test_recap_preserves_correction_and_never_reports_proposal_as_verified():
    s = InvestigationState()
    s.apply_user_turn("Retrieval recall improved but grounded accuracy dropped after the change.", 1)
    s.apply_reply(REPLY, [src()], 1)
    s.apply_challenge("You suggested changing the model. What evidence connects it?",
                      "I found no evidence for that.", 2)
    spoken, notes = s.recap()
    assert "Ruled out" in spoken and "changing the model" in spoken
    assert "not yet run" in spoken and "Nothing has been verified" in spoken
    assert "PROPOSED, NOT RUN" in notes and "UNSUPPORTED" in notes
    assert len(spoken.split()) <= 70
    s.verify(s.experiments[0].id, "recovered")
    assert "Verified" in s.recap()[0] and "Nothing has been verified" not in s.recap()[0]


def test_empty_recap_and_clear():
    s = InvestigationState()
    assert "haven't investigated" in s.recap()[0]
    s.apply_user_turn("We only have time for one experiment.", 1)
    s.clear()
    assert s.is_empty() and not s.constraints


def test_intents():
    assert RECAP_INTENT.search("I'm back. Give me a 20 second recap.")
    assert RECAP_INTENT.search("catch me up")
    assert REPLAY_INTENT.search("Wait, can you say that again?")
    assert EXPAND_INTENT.search("tell me more about that")
    assert not RECAP_INTENT.search("Does the retriever handle ties?")


def test_assume_fine_does_not_rule_out_unrelated_hypotheses_mentioning_the_word():
    s = InvestigationState()
    s.apply_reply("HYPOTHESES:\n- Evidence is truncated in prompt assembly while retrieval is unchanged\nNEXT: x", [], 1)
    s.apply_user_turn("Assume retrieval is fine.", 2)
    assert s.hypotheses[0].status == OPEN  # the assistant's own idea is untouched
    assert s.hypotheses[1].status == RULED_OUT and s.hypotheses[1].text == "retrieval"


def test_recap_stays_short_and_has_no_ellipsis():
    s = InvestigationState()
    s.apply_user_turn("Retrieval recall improved after this change but grounded accuracy dropped. Read the eval "
                      "results and the diff and tell me what to investigate first please.", 1)
    s.apply_reply(REPLY.replace("Which case regressed first", "Which case regressed first " + "word " * 40), [src()], 1)
    s.rule_out_text("changing the model", "engineer ruled it out")
    s.rule_out_text("the retrieval breadth change " + "alpha beta " * 10, "engineer ruled it out")
    spoken, _ = s.recap()
    assert len(spoken.split()) <= 60 and "..." not in spoken
    assert "Nothing has been verified" in spoken and "not yet run" in spoken and "Ruled out" in spoken


def test_stopwords_do_not_make_short_phrases_match_other_hypotheses():
    s = InvestigationState()
    s.apply_reply("HYPOTHESES:\n- The reduction in MAX_EVIDENCE may cause the model to miss the answer\nNEXT: x", [], 1)
    s.apply_challenge("You suggested changing the model. What evidence connects this?", "I found no evidence.", 2)
    assert s.hypotheses[0].status == OPEN  # the assistant's own hypothesis is untouched
    assert s.hypotheses[1].status == UNSUPPORTED and s.hypotheses[1].text == "changing the model"


def test_more_ways_of_saying_there_is_no_evidence():
    for reply in ("The evidence does not show a link to the model.", "There is no connection to the model.",
                  "Nothing in the files points to a model problem.", "That claim is not supported by anything I read."):
        s = InvestigationState()
        s.apply_challenge("You suggested changing the model.", reply, 1)
        assert s.hypotheses and s.hypotheses[0].status == UNSUPPORTED, reply


def test_short_does_not_leave_trailing_punctuation():
    from shared.investigation_state import short
    assert short("alpha beta gamma, delta epsilon", 3, ellipsis=False) == "alpha beta gamma"


def test_only_the_constraint_sentence_is_stored_not_the_whole_message():
    s = InvestigationState()
    s.apply_user_turn("We changed two things together. We only have time for one experiment. Which one first?", 1)
    assert s.constraints == ["We only have time for one experiment."]


def test_short_cuts_at_a_clause_boundary():
    from shared.investigation_state import short
    t = "The reduction in MAX_EVIDENCE to 1 may cause fewer passages, so grounding drops when the first one lacks the answer"
    assert short(t, 12, ellipsis=False) == "The reduction in MAX_EVIDENCE to 1 may cause fewer passages"
    assert short("alpha beta gamma delta epsilon zeta", 5, ellipsis=False) == "alpha beta gamma delta epsilon"
