"""The Query <-> Answer bar under the memory field.

A marker bounces along a dotted track so the panel reads as a live round trip
between the question and the recalled answer.
"""

from __future__ import annotations

from rich.text import Text
from textual.timer import Timer
from textual.widgets import Static

from everos.entrypoints.tui.demo.theme import (
    EVEROS_AMBER,
    EVEROS_CYAN,
    EVEROS_GREEN,
    EVEROS_YELLOW,
)


class QueryAnswerBar(Static):
    """Query <-> Answer bar with a marker that propagates back and forth."""

    TRACK_WIDTH = 11

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._pos = 0
        self._dir = 1
        self._timer: Timer | None = None

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.1, self._advance)

    def _advance(self) -> None:
        self._pos += self._dir
        if self._pos >= self.TRACK_WIDTH - 1:
            self._pos = self.TRACK_WIDTH - 1
            self._dir = -1
        elif self._pos <= 0:
            self._pos = 0
            self._dir = 1
        self.refresh()

    def render(self) -> Text:
        glyph = "▶" if self._dir > 0 else "◀"
        left = "·" * self._pos
        right = "·" * (self.TRACK_WIDTH - 1 - self._pos)
        return Text.assemble(
            ("Query ", f"bold {EVEROS_CYAN}"),
            (f" {left}", EVEROS_AMBER),
            (glyph, f"bold {EVEROS_YELLOW}"),
            (f"{right} ", EVEROS_AMBER),
            ("Answer", f"bold {EVEROS_GREEN}"),
        )
