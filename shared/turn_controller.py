"""Turn IDs and stale-result suppression (brief B10). Each user turn gets a
monotonically increasing ID; a turn is stale once a newer one begins or the
user barges in. Work for a stale turn must not touch history, the terminal,
or the speaker."""

import threading


class TurnController:
    def __init__(self):
        self._lock = threading.Lock()
        self._current = 0  # generation; bumped by begin() and cancel()
        self._numbers: dict[int, int] = {}
        self._count = 0  # user-visible turn number, bumped only by begin()

    def begin(self) -> int:
        with self._lock:
            self._current += 1
            self._count += 1
            self._numbers[self._current] = self._count
            return self._current

    def number(self, turn_id: int) -> int:
        """Display number for a turn (cancel() does not consume one)."""
        with self._lock:
            return self._numbers.get(turn_id, turn_id)

    def cancel(self) -> None:
        """Invalidate whatever turn is in flight (e.g. hotkey barge-in)."""
        with self._lock:
            self._current += 1

    def is_current(self, turn_id: int) -> bool:
        with self._lock:
            return turn_id == self._current
