"""Per-turn timing and context trace (brief B10/B12). Stores stage
timestamps and which sources were sent, never raw source text or keys."""

import time


class TurnTrace:
    def __init__(self, turn_id: int, source: str):
        self.turn_id = turn_id
        self.source = source
        self.marks: dict[str, float] = {"start": time.monotonic()}
        self.context: list[dict] = []
        self.status = "OK"

    def mark(self, stage: str) -> None:
        self.marks.setdefault(stage, time.monotonic())

    def elapsed(self) -> dict[str, float]:
        t0 = self.marks["start"]
        return {k: round(v - t0, 2) for k, v in self.marks.items() if k != "start"}

    def summary(self) -> str:
        stages = ", ".join(f"{k} {v}s" for k, v in self.elapsed().items())
        return f"turn {self.turn_id} [{self.status}] {stages}"
