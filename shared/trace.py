"""Per-turn timing and context trace (brief B10/B12). Stores stage
timestamps and which sources were sent, never raw source text or keys."""

import time


class TurnTrace:
    def __init__(self, turn_id: int, source: str, start: float | None = None):
        self.turn_id = turn_id
        self.source = source
        # start is the hotkey activation time for voice turns, so the offsets
        # below include recording and transcription, not just the model call.
        self.marks: dict[str, float] = {"start": start if start is not None else time.monotonic()}
        self.context: list[dict] = []
        self.tool_calls: list[dict] = []
        self.revisions: list[str] = []
        self.status = "OK"

    def mark(self, stage: str) -> None:
        self.marks.setdefault(stage, time.monotonic())

    def mark_at(self, stage: str, when: float) -> None:
        self.marks.setdefault(stage, when)

    def elapsed(self) -> dict[str, float]:
        t0 = self.marks["start"]
        return {k: round(v - t0, 2) for k, v in self.marks.items() if k != "start"}

    def summary(self) -> str:
        stages = ", ".join(f"{k} {v}s" for k, v in self.elapsed().items())
        return f"turn {self.turn_id} [{self.status}] {stages}"
