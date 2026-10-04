"""Session memory for one investigation (brief B08): the problem, evidence,
the engineer's constraints and corrections, hypotheses, experiments, and open
questions.

Everything here is deterministic. A small model with thinking off proved
unreliable at maintaining these notes (it marked hypotheses ruled out that
nobody rejected), so only explicit engineer actions change a hypothesis's
status, observations need validated citations, and an experiment stays
"proposed" until a result is supplied. Session memory only: nothing is saved."""

import re
from dataclasses import dataclass, field

from shared.context_builder import _CITE, _section, check_citations

OPEN, RULED_OUT, UNSUPPORTED = "open", "ruled_out", "unsupported"
PROPOSED, VERIFIED = "proposed", "verified"

_CONSTRAINT = re.compile(
    r"only (have )?time for|we (can|could) only|\bassume\b|\btreat (it|that|this)\b|"
    r"don'?t (look at|touch|consider|change)|do not (look at|touch|consider|change)|"
    r"only look at|\bignore\b|must not|\bconstraint\b|stick to|have to keep", re.I)
_CORRECTION = re.compile(
    r"\bactually\b|that'?s not|that isn'?t|that is not|\bthese are\b.{0,60}\bnot\b|"
    r"\bnot (a|an|the)\b.{0,40}\b(error|failure|issue|cause|problem)\b|\bwrong\b|\bincorrect\b", re.I)
_CLAIM = re.compile(r"you (?:suggested|said|claimed|mentioned)\s+(?:that\s+)?(.+?)(?:[.?!]|$)", re.I)
_NO_EVIDENCE = re.compile(
    r"no (direct |supporting |clear )?(evidence|link|connection|relationship)|did not (claim|suggest|say|mention)|"
    r"didn'?t (claim|suggest|say|mention)|haven'?t|have not|never (said|suggested|claimed|mentioned)|"
    r"not (supported|linked|connected|related)|nothing (in|that|i)|(does|do|did) not (show|support|indicate|point)|"
    r"(doesn'?t|don'?t|didn'?t) (show|support|indicate|point)|found none|unsupported", re.I)
_RULE_OUT = re.compile(r"\brule[ds]? out\s+(?:the\s+)?(.+?)(?:[.?!]|$)", re.I)
_ASSUME_FINE = re.compile(
    r"\b(?:assume|treat)\s+(?:that\s+)?(.+?)\s+(?:is|are)\s+(?:fine|ok|okay|correct|good|not\s+(?:the\s+)?(?:problem|cause|issue))", re.I)

RECAP_INTENT = re.compile(
    r"\brecap\b|catch me up|where (are|were) we|what did we (rule out|decide|find)|i'?m back|summari[sz]e (the )?(investigation|where)", re.I)
REPLAY_INTENT = re.compile(r"(say|repeat) that again|say it again|repeat that|what did you (just )?say", re.I)
EXPAND_INTENT = re.compile(r"tell me more|more detail|go into (more )?detail|expand on that|elaborate", re.I)


_STOP = {"the", "and", "for", "that", "this", "with", "are", "was", "were", "not", "but", "can", "may",
         "might", "from", "into", "than", "then", "when", "what", "which", "their", "there", "its", "has",
         "have", "had", "does", "did", "too", "any", "all", "our", "your", "you", "she", "his", "her"}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_]+", text.lower()) if len(w) > 2 and w not in _STOP}


def _similar(a: str, b: str, threshold: float = 0.6) -> bool:
    wa, wb = _words(a), _words(b)
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= threshold


def short(text: str, max_words: int = 14, ellipsis: bool = True) -> str:
    """Trim to about max_words, preferring to stop at a clause boundary so the
    spoken fragment does not end mid-thought."""
    text = " ".join(_CITE.sub(" ", text).split()).strip(" -*•.;,")
    words = text.split()
    if len(words) <= max_words:
        return text
    head = words[:max_words]
    for i in range(len(head) - 1, 3, -1):  # keep at least 4 words
        if head[i].endswith((",", ";", ":", ".")):
            head = head[:i + 1]
            break
    cut = " ".join(head).rstrip(" ,;:.-")
    return cut + ("..." if ellipsis else "")


def _first_sentence(text: str) -> str:
    return re.split(r"(?<=[.?!])\s", text.strip(), maxsplit=1)[0]


def _bullets(text: str) -> list[str]:
    parts = re.split(r"(?:^|\s)[-*•]\s+", text.strip())
    parts = [" ".join(p.split()) for p in parts if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _add_unique(items: list[str], text: str, cap: int = 8) -> None:
    text = text.strip()[:200]
    if text and not any(_similar(text, existing, 0.8) for existing in items):
        items.append(text)
        del items[:-cap]


@dataclass
class Observation:
    text: str
    refs: list[str]
    turn: int
    stale: bool = False


@dataclass
class Hypothesis:
    id: int
    text: str
    status: str = OPEN
    reason: str = ""
    turn: int = 0


@dataclass
class Experiment:
    id: int
    text: str
    status: str = PROPOSED
    result: str = ""
    turn: int = 0


@dataclass
class InvestigationState:
    problem: str = ""
    constraints: list[str] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    experiments: list[Experiment] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    reasserted: list[str] = field(default_factory=list)  # ruled-out ideas the model raised again
    _next_id: int = 1

    def _id(self) -> int:
        self._next_id += 1
        return self._next_id - 1

    def is_empty(self) -> bool:
        return not (self.problem or self.observations or self.hypotheses or self.experiments)

    # -- the engineer's turn ------------------------------------------------
    def apply_user_turn(self, text: str, turn: int) -> None:
        if not self.problem and len(text.split()) >= 6 and not RECAP_INTENT.search(text):
            self.problem = text.strip()[:240]
        for sentence in re.split(r"(?<=[.?!])\s+", text.strip()):
            if _CONSTRAINT.search(sentence):
                _add_unique(self.constraints, sentence)
            if _CORRECTION.search(sentence):
                _add_unique(self.corrections, sentence)
        for m in _RULE_OUT.finditer(text):
            self.rule_out_text(m.group(1), "engineer ruled it out", turn)
        for m in _ASSUME_FINE.finditer(text):
            self.rule_out_text(m.group(1), "engineer said to assume it is fine", turn)

    def rule_out_text(self, text: str, reason: str, turn: int = 0, status: str = RULED_OUT) -> Hypothesis:
        """Mark the best-matching open hypothesis ruled out, or record the idea
        as ruled out if nothing matches, so it is not re-raised as a cause."""
        # A short phrase like "retrieval" appears in many hypotheses; matching on
        # it would rule out ideas the engineer never mentioned. Only phrases
        # of 3+ meaningful words match an existing hypothesis by coverage, and
        # shorter ones need near-identical wording. Otherwise record a new,
        # separate ruled-out idea in the engineer's own words.
        key = _words(text)
        best, score = None, 0.0
        for h in self.hypotheses:
            if h.status != OPEN:
                continue
            overlap = len(key & _words(h.text)) / len(key) if len(key) >= 3 else 0.0
            if overlap > score:
                best, score = h, overlap
        if best and score >= 0.5:
            best.status, best.reason = status, reason
            return best
        for h in self.hypotheses:
            if _similar(text, h.text, 0.7):
                h.status, h.reason = status, reason
                return h
        h = Hypothesis(self._id(), text.strip()[:200], status, reason, turn)
        self.hypotheses.append(h)
        return h

    def apply_challenge(self, user_text: str, reply_text: str, turn: int) -> None:
        """'You suggested X. What evidence...?' answered with no evidence: record
        X as unsupported so it is not presented as a cause later."""
        m = _CLAIM.search(user_text)
        if m and _NO_EVIDENCE.search(reply_text):
            self.rule_out_text(m.group(1), "engineer challenged it and no supporting evidence was found",
                               turn, UNSUPPORTED)

    # -- the assistant's turn -----------------------------------------------
    def apply_reply(self, reply: str, evidence: list, turn: int) -> None:
        for bullet in _bullets(_section(reply, "OBSERVATIONS") or ""):
            refs = [f"{p}:{a}" + (f"-{b}" if b else "") for p, a, b in _CITE.findall(bullet)]
            if refs and not check_citations(bullet, evidence):  # only validated facts are remembered
                if not any(_similar(bullet, o.text, 0.8) for o in self.observations):
                    self.observations.append(Observation(" ".join(bullet.split())[:240], refs, turn))
        for bullet in _bullets(_section(reply, "HYPOTHESES") or ""):
            if any(h.status != OPEN and _similar(bullet, h.text, 0.5) for h in self.hypotheses):
                _add_unique(self.reasserted, bullet)
            elif not any(_similar(bullet, h.text, 0.7) for h in self.hypotheses):
                self.hypotheses.append(Hypothesis(self._id(), " ".join(bullet.split())[:200], OPEN, "", turn))
        nxt = _section(reply, "NEXT")
        if nxt:
            text = " ".join(nxt.split()).lstrip("-*• ")[:300]
            if text and not any(_similar(text, e.text, 0.7) for e in self.experiments):
                self.experiments.append(Experiment(self._id(), text, PROPOSED, "", turn))
        missing = _section(reply, "MISSING") or ""
        if not re.fullmatch(r"\W*(none|n/a|nothing)\W*", missing, re.I):
            for item in _bullets(missing):
                _add_unique(self.unresolved, item, cap=5)

    # -- freshness and explicit actions -------------------------------------
    def mark_stale(self, changed_paths: set[str]) -> int:
        n = 0
        for o in self.observations:
            if not o.stale and any(r.rsplit(":", 1)[0] in changed_paths for r in o.refs):
                o.stale, n = True, n + 1
        return n

    def verify(self, exp_id: int, result: str) -> bool:
        for e in self.experiments:
            if e.id == exp_id and result.strip():
                e.status, e.result = VERIFIED, result.strip()[:300]
                return True
        return False

    def rule_out(self, hyp_id: int, reason: str = "engineer ruled it out") -> bool:
        for h in self.hypotheses:
            if h.id == hyp_id:
                h.status, h.reason = RULED_OUT, reason
                return True
        return False

    def clear(self) -> None:
        self.__init__()

    # -- views ---------------------------------------------------------------
    def to_prompt(self) -> str:
        """Compact session memory for the model. The engineer's constraints and
        corrections override anything the assistant said earlier."""
        if self.is_empty() and not (self.constraints or self.corrections):
            return ""
        out = ["Investigation so far (session memory; the engineer's constraints and "
               "corrections override your earlier statements):"]
        if self.problem:
            out.append(f"Problem: {short(self.problem, 40)}")
        for title, items in (("Engineer's constraints", self.constraints),
                             ("Engineer's corrections", self.corrections)):
            if items:
                out.append(f"{title}:\n" + "\n".join(f"- {short(i, 30)}" for i in items))
        obs = [f"- {short(o.text, 30)} [{', '.join(o.refs[:2])}]"
               + (" (STALE: file changed, re-read before relying on it)" if o.stale else "")
               for o in self.observations[-6:]]
        if obs:
            out.append("Observed in inspected files:\n" + "\n".join(obs))
        ruled = [h for h in self.hypotheses if h.status != OPEN]
        if ruled:
            out.append("Ruled out or unsupported (do NOT present these as the cause):\n" + "\n".join(
                f"- {short(h.text, 20)} ({h.reason})" for h in ruled))
        open_h = [h for h in self.hypotheses if h.status == OPEN][-4:]
        if open_h:
            out.append("Open hypotheses (unverified):\n" + "\n".join(f"- {short(h.text, 25)}" for h in open_h))
        proposed = [e for e in self.experiments if e.status == PROPOSED][-2:]
        if proposed:
            out.append("Experiments PROPOSED but NOT RUN (never describe them as done):\n"
                       + "\n".join(f"- {short(e.text, 30)}" for e in proposed))
        done = [e for e in self.experiments if e.status == VERIFIED]
        if done:
            out.append("Verified results:\n" + "\n".join(f"- {short(e.text, 15)}: {short(e.result, 25)}" for e in done))
        if self.unresolved:
            out.append("Unresolved:\n" + "\n".join(f"- {short(u, 20)}" for u in self.unresolved))
        return "\n".join(out)

    def recap(self) -> tuple[str, str]:
        """(spoken, notes). Built locally, never by the model, so a proposal
        cannot be reported as a verified result. Spoken is about 20 seconds."""
        if self.is_empty():
            return "We haven't investigated anything yet this session.", "No investigation notes yet."
        ruled = [h for h in self.hypotheses if h.status != OPEN]
        open_h = [h for h in self.hypotheses if h.status == OPEN]
        proposed = [e for e in self.experiments if e.status == PROPOSED]
        verified = [e for e in self.experiments if e.status == VERIFIED]
        # Parts in priority order; the lowest-priority ones are dropped to stay
        # near 20 seconds of speech (about 55 words). "Nothing verified" is
        # always kept: the recap must never blur proposed and verified.
        s = lambda t, n: short(t, n, ellipsis=False)
        if verified:
            status = f"Verified: {s(verified[-1].text, 7)}, {s(verified[-1].result, 7)}."
        else:
            status = "Nothing has been verified by running anything yet."
        parts = [
            ("ruled", f"Ruled out: {'; '.join(s(h.text, 6) for h in ruled[:2])}." if ruled else ""),
            ("status", status),
            ("next", f"Next step, not yet run: {s(proposed[-1].text, 14)}." if proposed else
                     (f"Open question: {s(self.unresolved[-1], 10)}." if self.unresolved else "")),
            ("open", f"Still uncertain: {s(open_h[-1].text, 10)}." if open_h else ""),
            ("problem", f"Problem: {s(_first_sentence(self.problem), 12)}." if self.problem else ""),
        ]
        keep = {k for k, v in parts if v}
        for drop in ("problem", "open"):  # least important first
            if sum(len(v.split()) for k, v in parts if k in keep) <= 55:
                break
            keep.discard(drop)
        order = ["problem", "ruled", "status", "open", "next"]
        by_key = dict(parts)
        spoken = [by_key[k] for k in order if k in keep and by_key[k]]
        return " ".join(spoken), self.notes()

    def notes(self) -> str:
        lines = []
        if self.problem:
            lines.append(f"Problem: {self.problem}")
        for title, items in (("Constraints", self.constraints), ("Corrections", self.corrections)):
            if items:
                lines.append(f"{title}:")
                lines += [f"  - {i}" for i in items]
        if self.observations:
            lines.append("Observed (validated citations):")
            lines += [f"  - {o.text}" + ("  [STALE: file changed]" if o.stale else "") for o in self.observations]
        for h in self.hypotheses:
            tag = {"open": "OPEN", "ruled_out": "RULED OUT", "unsupported": "UNSUPPORTED"}[h.status]
            lines.append(f"  H{h.id} [{tag}] {h.text}" + (f"  ({h.reason})" if h.reason else ""))
        for e in self.experiments:
            tag = "VERIFIED" if e.status == VERIFIED else "PROPOSED, NOT RUN"
            lines.append(f"  E{e.id} [{tag}] {e.text}" + (f"\n        result: {e.result}" if e.result else ""))
        if self.unresolved:
            lines.append("Unresolved:")
            lines += [f"  - {u}" for u in self.unresolved]
        if self.reasserted:
            lines.append("Note: the assistant re-raised these ruled-out ideas (treat with suspicion):")
            lines += [f"  - {r}" for r in self.reasserted]
        return "\n".join(lines)
