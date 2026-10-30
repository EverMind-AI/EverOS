"""The animated dot-sphere widget and its animation clock.

The widget owns time only: it advances a tick at ``SPHERE_FPS``, decides which
sphere state to show (an idle loop, or a state pinned by the app), eases
between states and posts ``StageChanged`` when the trace stage moves. Every
pixel comes from the pure renderer in :mod:`.sphere`.
"""

from __future__ import annotations

from textual.message import Message
from textual.timer import Timer
from textual.widgets import Static

from everos.entrypoints.tui.demo.theme import EVEROS_SURFACE
from everos.entrypoints.tui.demo.widgets.sphere import (
    blend_dot_sphere_frames,
    build_dot_sphere,
    render_dot_sphere_text,
)

SPHERE_FRAME_WIDTH = 37
SPHERE_FRAME_HEIGHT = 17
TERMINAL_CELL_HEIGHT_RATIO = 2.0

# Sphere animation cadence. Each named state (and its highlighted trace word)
# dwells for SPHERE_STAGE_SECONDS so a viewer can read the stage it represents.
SPHERE_FPS = 24
SPHERE_STAGE_SECONDS = 3.0
SPHERE_STAGE_TICKS = round(SPHERE_FPS * SPHERE_STAGE_SECONDS)
SPHERE_TRANSITION_SECONDS = 0.35
SPHERE_TRANSITION_TICKS = round(SPHERE_FPS * SPHERE_TRANSITION_SECONDS)
SPHERE_SUPERNOVA_CYCLE_SECONDS = 8.0
SPHERE_SUPERNOVA_CYCLE_TICKS = round(SPHERE_FPS * SPHERE_SUPERNOVA_CYCLE_SECONDS)

SPHERE_IDLE_STATES = (
    "booting",
    "ingesting",
    "extracting",
    "indexing",
    "recalling",
    "celebrating",
)

_STATE_TO_STAGE = {
    "ingesting": 0,
    "extracting": 1,
    "indexing": 2,
    "recalling": 3,
    "remembered": 3,
    "source": 3,
}


def _state_to_stage(state_key: str) -> int:
    """Map a sphere state to its trace-stage index (-1 = no stage highlighted)."""

    return _STATE_TO_STAGE.get(state_key, -1)


def _sphere_state_phase(state: str, state_tick: int) -> float:
    """Return a one-shot phase for stages and a seamless loop for celebration."""

    if state == "celebrating":
        return (state_tick % SPHERE_SUPERNOVA_CYCLE_TICKS) / (
            SPHERE_SUPERNOVA_CYCLE_TICKS - 1
        )
    return min(1.0, state_tick / SPHERE_STAGE_TICKS)


def _idle_sphere_state(tick: int) -> str:
    """Start with Working, then loop ingest through the full celebration."""

    if tick < SPHERE_STAGE_TICKS:
        return SPHERE_IDLE_STATES[0]
    pipeline_state_count = len(SPHERE_IDLE_STATES) - 2
    pipeline_ticks = pipeline_state_count * SPHERE_STAGE_TICKS
    cycle_ticks = pipeline_ticks + SPHERE_SUPERNOVA_CYCLE_TICKS
    cycle_tick = (tick - SPHERE_STAGE_TICKS) % cycle_ticks
    if cycle_tick < pipeline_ticks:
        return SPHERE_IDLE_STATES[1 + cycle_tick // SPHERE_STAGE_TICKS]
    return SPHERE_IDLE_STATES[-1]


class DotSphereWidget(Static):
    """Animated dot sphere that represents EverOS memory activity."""

    DEFAULT_CSS = """
    DotSphereWidget {
        height: 1fr;
        content-align: center middle;
    }
    """

    STATES = SPHERE_IDLE_STATES

    class StageChanged(Message):
        """Posted when the sphere enters a different trace stage."""

        def __init__(self, stage: int) -> None:
            self.stage = stage
            super().__init__()

    def __init__(self) -> None:
        super().__init__()
        self._phase = 0.0
        self._tick = 0
        self._last_stage = -2
        self._rendered_state: str | None = None
        self._transition_from_state: str | None = None
        self._transition_tick = 0
        self._state_tick = 0
        self._celebration_source_phase = 0.0
        self._animation_timer: Timer | None = None
        # When set, the sphere is pinned to a pipeline state (synced to the
        # signal rail during a round). When None it free-runs the idle loop.
        self._driven_state: str | None = None

    def drive_state(self, state: str | None) -> None:
        """Pin the sphere to a pipeline state, or None to resume the idle loop."""

        self._driven_state = state
        self._advance()  # reflect the change without waiting for the next tick

    def on_mount(self) -> None:
        self._animation_timer = self.set_interval(1 / SPHERE_FPS, self._advance)
        self._advance()

    def pause_animation(self) -> None:
        if self._animation_timer is not None:
            self._animation_timer.pause()

    def _frame_size(self) -> tuple[int, int]:
        """Size the sphere to its actual box so it never clips and stays round.

        Terminal cells are ~2x taller than wide, so a round sphere needs
        ``width ≈ 2 * height``. We fill whichever dimension is the constraint.
        Before the first layout the widget reports a 0x0 size, so fall back to
        the default frame.
        """

        width, height = self.size.width, self.size.height
        if width <= 0 or height <= 0:
            return SPHERE_FRAME_WIDTH, SPHERE_FRAME_HEIGHT
        frame_height = height
        frame_width = round(height * TERMINAL_CELL_HEIGHT_RATIO) + 3
        if frame_width > width:
            frame_width = width
            frame_height = round((width - 3) / TERMINAL_CELL_HEIGHT_RATIO)
        # The builder needs at least 13x7; clamp up even on a tiny box (it just
        # clips a little) rather than crash.
        return max(13, frame_width), max(7, frame_height)

    def _advance(self) -> None:
        # Keep time monotonic. Wrapping at 1.0 made the non-integer wave
        # frequencies jump to a different shape every few seconds.
        self._phase += 0.3 / SPHERE_FPS
        self._tick += 1
        if (
            self._driven_state == "celebrating"
            and self._state_tick >= SPHERE_SUPERNOVA_CYCLE_TICKS
        ):
            self._driven_state = None
            self._tick = SPHERE_STAGE_TICKS
        if self._driven_state is not None:
            state = self._driven_state
        else:
            state = _idle_sphere_state(self._tick)
        if self._rendered_state is None:
            self._rendered_state = state
        elif state != self._rendered_state:
            self._transition_from_state = self._rendered_state
            self._rendered_state = state
            self._transition_tick = 0
            self._state_tick = 0
            if state == "celebrating":
                self._celebration_source_phase = self._phase
        frame_width, frame_height = self._frame_size()
        state_phase = _sphere_state_phase(state, self._state_tick)
        render_phase = (
            self._celebration_source_phase if state == "celebrating" else self._phase
        )
        frame = build_dot_sphere(
            width=frame_width,
            height=frame_height,
            phase=render_phase,
            state_key=state,
            state_phase=state_phase,
        )
        if self._transition_from_state is not None:
            previous_phase = (
                self._celebration_source_phase
                if self._transition_from_state == "celebrating"
                else self._phase
            )
            previous_frame = build_dot_sphere(
                width=frame_width,
                height=frame_height,
                phase=previous_phase,
                state_key=self._transition_from_state,
                state_phase=(
                    1.0 if self._transition_from_state == "celebrating" else None
                ),
            )
            raw_progress = min(
                1.0,
                (self._transition_tick + 1) / SPHERE_TRANSITION_TICKS,
            )
            eased_progress = raw_progress * raw_progress * (3 - 2 * raw_progress)
            frame = blend_dot_sphere_frames(
                previous_frame,
                frame,
                eased_progress,
                background=EVEROS_SURFACE,
            )
            self._transition_tick += 1
            if self._transition_tick >= SPHERE_TRANSITION_TICKS:
                self._transition_from_state = None
        self.update(render_dot_sphere_text(frame))
        self._state_tick += 1

        stage = _state_to_stage(state)
        if stage != self._last_stage:
            self._last_stage = stage
            self.post_message(self.StageChanged(stage))
