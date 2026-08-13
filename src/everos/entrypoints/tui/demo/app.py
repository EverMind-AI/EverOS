"""Textual TUI for ``everos demo``."""

from __future__ import annotations

from functools import partial

import anyio
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.timer import Timer
from textual.widgets import Footer, Input, Static

from everos.component.utils.datetime import today_with_timezone
from everos.entrypoints.tui.demo import cloud
from everos.entrypoints.tui.demo.data import (
    DemoStory,
    default_demo_story,
)
from everos.entrypoints.tui.demo.widgets.sphere import (
    EVEROS_AMBER,
    EVEROS_AMBER_DIM,
    EVEROS_CYAN,
    EVEROS_GREEN,
    EVEROS_ORANGE,
    EVEROS_YELLOW,
    EVEROS_YELLOW_SOFT,
    blend_dot_sphere_frames,
    build_dot_sphere,
    render_dot_sphere_text,
)

EVEROS_BLACK = "#1D1C18"
EVEROS_SURFACE = "#24231E"
EVEROS_SURFACE_RAISED = "#31302B"
EVEROS_INK = "#F5EDDC"
EVEROS_MUTED = "#918C80"
EVEROS_BORDER = "#5A5549"
SPHERE_FRAME_WIDTH = 37
SPHERE_FRAME_HEIGHT = 17
TERMINAL_CELL_HEIGHT_RATIO = 2.0
SIGNAL_RAIL_SOURCE_WIDTH = 18
# Offline default demo: how many memory -> recall rounds a user plays before the
# TUI nudges them toward the real pipeline (`--cloud` / `--live`).
DEFAULT_DEMO_ROUNDS = 3

# Sphere animation cadence. Each named state (and its highlighted trace word)
# dwells for SPHERE_STAGE_SECONDS so a viewer can read the stage it represents.
SPHERE_FPS = 24
SPHERE_STAGE_SECONDS = 3.0
SPHERE_STAGE_TICKS = round(SPHERE_FPS * SPHERE_STAGE_SECONDS)
SPHERE_TRANSITION_SECONDS = 0.35
SPHERE_TRANSITION_TICKS = round(SPHERE_FPS * SPHERE_TRANSITION_SECONDS)
SPHERE_SUPERNOVA_CYCLE_SECONDS = 8.0
SPHERE_SUPERNOVA_CYCLE_TICKS = round(SPHERE_FPS * SPHERE_SUPERNOVA_CYCLE_SECONDS)

# The four pipeline stages shown in the trace header. They line up with the four
# core sphere states, so the active word can highlight in sync with the sphere.
TRACE_STAGES = ("ingest", "extract", "index", "recall")
SPHERE_IDLE_STATES = (
    "booting",
    "ingesting",
    "extracting",
    "indexing",
    "recalling",
    "celebrating",
)

# Words a user can type in the input box to quit back to the terminal.
QUIT_COMMANDS = frozenset({"quit", "exit", ":q", "/quit", "/exit"})
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


class EverOSDemoApp(App[None]):
    """Fullscreen first-run demo cockpit."""

    TITLE = "EverOS Memory Core"
    SUB_TITLE = "dot sphere demo"
    # ctrl+c / ctrl+q are priority bindings so they quit even while the input
    # box has focus (where a bare "q" would just be typed into the field).
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True, show=False),
        Binding("ctrl+q", "quit", "Quit", priority=True),
        ("q", "quit", "Quit"),
        ("r", "replay", "Replay"),
    ]

    CSS = f"""
    Screen {{
        background: {EVEROS_BLACK};
        color: {EVEROS_INK};
    }}

    #shell {{
        width: 100%;
        height: 100%;
        padding: 1 2;
        border: round {EVEROS_BORDER};
    }}

    #command-strip {{
        height: 1;
        padding: 0 1;
        color: {EVEROS_INK};
        content-align: left middle;
    }}

    #main {{
        height: 1fr;
        margin-top: 0;
    }}

    #memory-field {{
        width: 1fr;
        border: round {EVEROS_AMBER};
        background: {EVEROS_SURFACE};
        padding: 0 2;
    }}

    #field-header {{
        height: 2;
        content-align: left middle;
    }}

    #field-answer {{
        height: 2;
        border-top: hkey {EVEROS_AMBER_DIM};
        background: {EVEROS_SURFACE_RAISED};
        padding: 0 1;
        content-align: center middle;
    }}

    #right-rail {{
        width: 48;
        height: 100%;
        margin-left: 1;
    }}

    #capabilities {{
        height: 9;
        border: panel {EVEROS_YELLOW};
        border-title-color: {EVEROS_BLACK};
        border-title-background: {EVEROS_YELLOW};
        border-title-style: bold;
        background: {EVEROS_SURFACE_RAISED};
        padding: 0 2;
        margin-bottom: 1;
    }}

    #signal-rail {{
        width: 100%;
        height: 1fr;
        border: round {EVEROS_AMBER};
        background: {EVEROS_SURFACE};
        padding: 1 2;
    }}

    #provenance-strip {{
        height: 6;
        margin-top: 1;
    }}

    #source-lock {{
        width: 1fr;
        height: 100%;
        border: round {EVEROS_CYAN};
        background: {EVEROS_SURFACE};
        padding: 0 2;
        margin-right: 1;
    }}

    #recall-lock {{
        width: 54;
        height: 100%;
        border: round {EVEROS_GREEN};
        background: {EVEROS_SURFACE};
        padding: 0 2;
    }}

    #conversation {{
        height: 6;
        overflow-y: auto;
        scrollbar-size-vertical: 1;
        scrollbar-color: {EVEROS_AMBER};
        scrollbar-background: {EVEROS_SURFACE};
        border: round {EVEROS_YELLOW};
        background: {EVEROS_SURFACE};
        color: {EVEROS_INK};
        padding: 0 1;
        margin-top: 1;
    }}

    #conversation-log {{
        height: auto;
        width: 1fr;
        background: {EVEROS_SURFACE};
        color: {EVEROS_INK};
    }}

    #console {{
        height: auto;
        margin-top: 1;
    }}

    #console-prompt {{
        height: auto;
        padding: 0 1;
    }}

    #console-input {{
        border: round {EVEROS_AMBER};
        background: {EVEROS_SURFACE};
    }}

    Footer {{
        background: {EVEROS_BLACK};
        color: {EVEROS_MUTED};
    }}

    FooterKey {{
        background: {EVEROS_BLACK};
    }}

    FooterKey > .footer-key--key {{
        color: {EVEROS_BLACK};
        background: {EVEROS_YELLOW};
        text-style: bold;
    }}

    FooterKey > .footer-key--description {{
        color: {EVEROS_INK};
        background: {EVEROS_BLACK};
    }}
    """

    def __init__(
        self,
        *,
        story: DemoStory | None = None,
        interactive: bool = False,
        base_url: str = cloud.CLOUD_API_BASE_URL,
        session_id: str = cloud.LIVE_DEMO_SESSION_ID,
        user_id: str = cloud.LIVE_DEMO_USER_ID,
        api_key: str = "",
        user_label: str = "you",
        max_rounds: int = DEFAULT_DEMO_ROUNDS,
    ) -> None:
        super().__init__()
        self._story = story or default_demo_story()
        self._interactive = interactive
        self._base_url = base_url
        self._session_id = session_id
        self._user_id = user_id
        self._api_key = api_key
        self._user_label = user_label
        self._max_rounds = max_rounds
        self._active_stage = -1
        # Each round auto-alternates two steps with no mode toggle:
        #   "memory"  -> tell EverOS one thing (stored, no answer)
        #   "query"   -> ask one question (recalls -> an answer)
        # the "*ing" variants mean a cloud call is in flight; "done" -> cap hit.
        self._conversation_phase = "memory"
        self._current_memory = ""
        self._stored_memories: list[str] = []
        self._round = 0
        self._lights = _initial_lights()
        self._log: list[tuple[str, str]] = []
        self._history_chars = 0
        self._saved_pct: int | None = None
        self._recall_celebration_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="shell"):
            yield Static(_hero_text(), id="command-strip")
            with Horizontal(id="main"):
                memory_field = Vertical(id="memory-field")
                memory_field.border_title = "memory field"
                with memory_field:
                    yield Static(
                        _field_header_text(
                            user_label=self._user_label,
                            active_stage=self._active_stage,
                        ),
                        id="field-header",
                    )
                    yield DotSphereWidget()
                    yield QueryAnswerBar(id="field-answer")
                with Vertical(id="right-rail"):
                    capabilities = Static(_capabilities_text(), id="capabilities")
                    capabilities.border_title = "EverOS strengths"
                    yield capabilities
                    signal_rail = Static(
                        _signal_rail_text(self._lights), id="signal-rail"
                    )
                    signal_rail.border_title = "signal rail"
                    yield signal_rail
            with Horizontal(id="provenance-strip"):
                source_lock = Static(_source_tree_text(), id="source-lock")
                source_lock.border_title = "source lock"
                yield source_lock
                recall_lock = Static(
                    _recall_proof_text(self._story, user_label=self._user_label),
                    id="recall-lock",
                )
                recall_lock.border_title = "recall lock"
                yield recall_lock
            # A real scroll container (not a bare Static): a Static clips but
            # never scrolls, so older turns would be unreachable once the log
            # grows past the panel height.
            conversation = VerticalScroll(id="conversation")
            conversation.border_title = "conversation"
            with conversation:
                yield Static(_conversation_text(self._log), id="conversation-log")
            if self._interactive:
                with Vertical(id="console"):
                    yield Static(
                        _prompt_memory_text(self._round, self._max_rounds),
                        id="console-prompt",
                    )
                    yield Input(
                        placeholder=(
                            "tell EverOS something & enter  ·  /live  ·  /quit"
                        ),
                        id="console-input",
                    )
            yield Footer(show_command_palette=False)

    def on_mount(self) -> None:
        if self._interactive:
            self.query_one("#console-input", Input).focus()

    @on(DotSphereWidget.StageChanged)
    def _on_stage_changed(self, event: DotSphereWidget.StageChanged) -> None:
        self._active_stage = event.stage
        self.query_one("#field-header", Static).update(
            _field_header_text(
                user_label=self._user_label,
                active_stage=self._active_stage,
            )
        )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if not self._interactive:
            return
        value = event.value.strip()
        # Quit works in any phase, even mid-round.
        if value.lower() in QUIT_COMMANDS:
            self.exit()
            return
        if self._conversation_phase in {"storing", "recalling"}:
            return  # a cloud call is in flight; ignore further input
        if value.startswith("/"):
            self._run_slash_command(value.lower())
            return
        prompt = self.query_one("#console-prompt", Static)
        field = self.query_one("#console-input", Input)
        if self._conversation_phase == "done":
            # Free rounds used up, but keep the input usable: re-show the nudge.
            field.value = ""
            prompt.update(_quota_guidance_text())
            return
        if not value:
            return  # ignore empty submissions; never substitute canned content

        if self._conversation_phase == "memory":
            # Step 1: store the line. No answer here — just remember it.
            self._record_line("you", value)
            self._conversation_phase = "storing"
            field.value = ""
            field.disabled = True
            prompt.update(_storing_text())
            self.run_worker(self._store(value), group="round", exclusive=True)
            return

        # Step 2: a question. Echo it, then recall against everything stored.
        self._record_line("ask", value)
        self._conversation_phase = "recalling"
        field.value = ""
        field.disabled = True
        prompt.update(_recalling_text())
        self.run_worker(self._ask(value), group="round", exclusive=True)

    def on_input_changed(self, event: Input.Changed) -> None:
        # Live slash-command panel: as soon as the user types "/", surface the
        # available commands; restore the phase prompt once they type real text.
        if not self._interactive or self._conversation_phase in {
            "storing",
            "recalling",
        }:
            return
        prompt = self.query_one("#console-prompt", Static)
        if event.value.startswith("/"):
            prompt.update(_commands_text())
        elif event.value:
            prompt.update(self._phase_prompt())

    def _run_slash_command(self, command: str) -> None:
        prompt = self.query_one("#console-prompt", Static)
        self.query_one("#console-input", Input).value = ""
        if command == "/live":
            prompt.update(_live_guidance_text())
        elif command == "/replay":
            self.action_replay()
            prompt.update(self._phase_prompt())
        elif command == "/clear":
            self._log.clear()
            self.query_one("#conversation-log", Static).update(
                _conversation_text(self._log)
            )
            prompt.update(self._phase_prompt())
        else:
            prompt.update(_unknown_command_text(command))

    def _phase_prompt(self) -> Text:
        if self._conversation_phase == "done":
            return _quota_guidance_text()
        if self._conversation_phase == "query":
            return _prompt_query_text()
        return _prompt_memory_text(self._round, self._max_rounds)

    async def _store(self, memory: str) -> None:
        # Step 1 of a round: store the memory. Light the pipeline as each real
        # step (add -> flush) completes. No recall happens here.
        self._reset_round_lights()
        base_url, session_id, user_id, api_key = (
            self._base_url,
            self._session_id,
            self._user_id,
            self._api_key,
        )
        try:
            await anyio.to_thread.run_sync(
                partial(
                    cloud.add_memory,
                    memory,
                    base_url=base_url,
                    session_id=session_id,
                    user_id=user_id,
                    api_key=api_key,
                )
            )
            # A successful add means the key authenticated and the memory landed.
            self._set_light("core", "ready")
            self._set_light("conversation", "captured")
            await anyio.to_thread.run_sync(
                partial(
                    cloud.flush_memory,
                    base_url=base_url,
                    session_id=session_id,
                    api_key=api_key,
                )
            )
            self._set_light("facts", "live")
            self._set_light("index", "synced")
        except cloud.CloudQuotaError:
            self._enter_done(_quota_guidance_text())
            return
        except cloud.CloudAuthError:
            self._set_light("core", "error")
            self._show_round_error(
                "demo authentication is temporarily unavailable", "memory"
            )
            return
        except cloud.CloudDemoError:
            self._set_light("core", "error")
            self._show_round_error("could not reach EverOS Cloud", "memory")
            return

        # Stored. Move to step 2 and invite the question — still no answer yet.
        # The sphere stays pinned at the last stored stage (indexing) until a
        # question actually recalls; it is not reset here on purpose.
        self._current_memory = memory
        self._stored_memories.append(memory)
        self.action_replay()
        self._conversation_phase = "query"
        self.query_one("#console-prompt", Static).update(_prompt_query_text())
        self._reenable_input()

    async def _ask(self, query: str) -> None:
        # Step 2 of a round: recall against everything stored so far.
        self._reset_recall_light()
        base_url, session_id, user_id, api_key = (
            self._base_url,
            self._session_id,
            self._user_id,
            self._api_key,
        )
        try:
            story = await anyio.to_thread.run_sync(
                partial(
                    cloud.search_recall,
                    self._current_memory,
                    query,
                    stored_memories=self._stored_memories.copy(),
                    base_url=base_url,
                    session_id=session_id,
                    user_id=user_id,
                    api_key=api_key,
                )
            )
        except cloud.CloudQuotaError:
            self._enter_done(_quota_guidance_text())
            return
        except cloud.CloudAuthError:
            self._set_light("core", "error")
            self._show_round_error(
                "demo authentication is temporarily unavailable", "query"
            )
            return
        except cloud.CloudDemoError:
            self._set_light("core", "error")
            self._show_round_error("could not reach EverOS Cloud", "query")
            return

        if story is None:
            self._set_light("recall", "miss")
            answer = "(no matching memory found)"
            self._record_line("everos", answer)
            story = DemoStory(
                owner=user_id,
                memory="",
                query=query,
                answer=answer,
                source_filename="",
                fact_filename="",
            )
        else:
            self._set_light("recall", "hit")
            self._record_line("everos", story.answer)
        self._update_savings(query, story.answer)
        self._finish_round(story)

    def _update_savings(self, query: str, answer: str) -> None:
        # Estimate (not measured): carrying the whole conversation as LLM context
        # vs. EverOS handing back only the compact recalled answer. Char counts
        # are a token proxy; the ratio is what matters, so the /4 cancels out.
        self._history_chars += len(query) + len(answer)
        if self._history_chars:
            ratio = 1 - len(answer) / self._history_chars
            self._saved_pct = max(0, min(99, round(100 * ratio)))

    def _finish_round(self, story: DemoStory) -> None:
        self._story = story
        self.query_one("#recall-lock", Static).update(
            _recall_proof_text(
                story, user_label=self._user_label, saved_pct=self._saved_pct
            )
        )
        self.action_replay()
        # Hold the white recall targets long enough to read, then celebrate.
        # There is no intermediate yellow remembered/source state.
        if self._lights.get("recall") == "hit":
            self._recall_celebration_timer = self.set_timer(
                SPHERE_STAGE_SECONDS,
                self._celebrate_recall,
            )
        self._round += 1
        if self._round >= self._max_rounds:
            self._enter_done(_quota_guidance_text())
            return
        self._conversation_phase = "memory"
        self.query_one("#console-prompt", Static).update(
            _prompt_memory_text(self._round, self._max_rounds)
        )
        self._reenable_input()

    def _reset_recall_light(self) -> None:
        self._lights["recall"] = "idle"
        self.query_one("#signal-rail", Static).update(_signal_rail_text(self._lights))
        self._sync_sphere_to_rail()

    def _reset_round_lights(self) -> None:
        if self._recall_celebration_timer is not None:
            self._recall_celebration_timer.stop()
            self._recall_celebration_timer = None
        self._lights.update(
            conversation="idle", facts="idle", index="idle", recall="idle"
        )
        self.query_one("#signal-rail", Static).update(_signal_rail_text(self._lights))
        self._sync_sphere_to_rail()

    def _celebrate_recall(self) -> None:
        self._recall_celebration_timer = None
        if self._lights.get("recall") == "hit":
            self.query_one(DotSphereWidget).drive_state("celebrating")

    def _set_light(self, key: str, state: str) -> None:
        self._lights[key] = state
        self.query_one("#signal-rail", Static).update(_signal_rail_text(self._lights))
        self._sync_sphere_to_rail()

    def _sync_sphere_to_rail(self) -> None:
        """Pin the sphere to the furthest lit pipeline stage on the rail.

        It holds there (does not advance) until the next real step lights up —
        e.g. after storing it rests at ``indexing`` and only reaches
        ``recalling`` once a question actually recalls. With no stage lit it
        free-runs the idle loop.
        """

        for key, sphere_state in _RAIL_STAGE_ORDER:
            if self._lights.get(key) in _LIGHT_YELLOW:
                self.query_one(DotSphereWidget).drive_state(sphere_state)
                return
        self.query_one(DotSphereWidget).drive_state(None)

    def _record_line(self, speaker: str, text: str) -> None:
        self._log.append((speaker, text))
        self.query_one("#conversation-log", Static).update(
            _conversation_text(self._log)
        )
        # Keep the newest line in view as the log grows past the panel height.
        self.query_one("#conversation", VerticalScroll).scroll_end(animate=False)

    def _enter_done(self, message: Text) -> None:
        # Cap reached: keep the input usable (so /live, /quit still work and the
        # user is never locked out) and show the upgrade nudge.
        self._conversation_phase = "done"
        self._sync_sphere_to_rail()
        self.query_one("#console-prompt", Static).update(message)
        self._reenable_input()

    def _show_round_error(self, message: str, phase: str) -> None:
        # A step failed (server unreachable, unhealthy, or slow). Surface the
        # reason honestly and let the user retry from the same step.
        self._conversation_phase = phase
        self._sync_sphere_to_rail()
        self.query_one("#console-prompt", Static).update(_recall_error_text(message))
        self._reenable_input()

    def _reenable_input(self) -> None:
        field = self.query_one("#console-input", Input)
        field.disabled = False
        field.focus()

    def action_replay(self) -> None:
        widget = self.query_one(DotSphereWidget)
        widget._tick = 0
        widget._phase = 0.0
        widget._advance()


def run_demo_tui(
    *,
    story: DemoStory | None = None,
    interactive: bool = False,
    base_url: str = cloud.CLOUD_API_BASE_URL,
    session_id: str = cloud.LIVE_DEMO_SESSION_ID,
    user_id: str = cloud.LIVE_DEMO_USER_ID,
    api_key: str = "",
    user_label: str = "you",
) -> None:
    EverOSDemoApp(
        story=story,
        interactive=interactive,
        base_url=base_url,
        session_id=session_id,
        user_id=user_id,
        api_key=api_key,
        user_label=user_label,
    ).run()


def _prompt_memory_text(round_index: int, total_rounds: int) -> Text:
    return Text.assemble(
        (f"round {round_index + 1}/{total_rounds}  ", EVEROS_MUTED),
        ("① tell EverOS something to remember", f"bold {EVEROS_YELLOW}"),
    )


def _prompt_query_text() -> Text:
    return Text.assemble(
        ("② now ask EverOS a question", f"bold {EVEROS_CYAN}"),
        ("  ·  it recalls what you stored", EVEROS_MUTED),
    )


def _storing_text() -> Text:
    return Text("remembering...", style=f"bold {EVEROS_ORANGE}")


def _recalling_text() -> Text:
    return Text("recalling from EverOS...", style=f"bold {EVEROS_ORANGE}")


def _recall_error_text(message: str) -> Text:
    return Text.assemble(
        (f"{message}  ", f"bold {EVEROS_ORANGE}"),
        ("· type to retry", EVEROS_MUTED),
    )


def _commands_text() -> Text:
    return Text.assemble(
        ("commands  ", f"bold {EVEROS_YELLOW}"),
        ("/live", f"bold {EVEROS_GREEN}"),
        (" use your key  ", EVEROS_MUTED),
        ("/replay", f"bold {EVEROS_GREEN}"),
        (" re-run  ", EVEROS_MUTED),
        ("/clear", f"bold {EVEROS_GREEN}"),
        (" wipe log  ", EVEROS_MUTED),
        ("/quit", f"bold {EVEROS_GREEN}"),
        (" exit", EVEROS_MUTED),
    )


def _live_guidance_text() -> Text:
    return Text.assemble(
        ("use your own key  ", f"bold {EVEROS_YELLOW}"),
        ("everos init", f"bold {EVEROS_GREEN}"),
        ("  then  ", EVEROS_MUTED),
        ("everos demo --live", f"bold {EVEROS_GREEN}"),
    )


def _unknown_command_text(command: str) -> Text:
    return Text.assemble(
        (f"unknown command {command}  ", f"bold {EVEROS_ORANGE}"),
        ("available: /live /replay /clear /quit", EVEROS_INK),
    )


def _quota_guidance_text() -> Text:
    return Text.assemble(
        ("free demo rounds used up  ", f"bold {EVEROS_YELLOW}"),
        ("configure your own key -> ", EVEROS_INK),
        ("everos init", f"bold {EVEROS_GREEN}"),
        ("  then  ", EVEROS_MUTED),
        ("everos demo --live", f"bold {EVEROS_GREEN}"),
    )


def _hero_text() -> Text:
    return Text.assemble(
        (" everos demo ", f"bold black on {EVEROS_YELLOW}"),
        ("  memory core ", f"bold {EVEROS_YELLOW}"),
        ("online", EVEROS_MUTED),
    )


def _field_header_text(*, user_label: str = "you", active_stage: int = -1) -> Text:
    parts: list[tuple[str, str]] = [
        (f"user={user_label}", f"bold {EVEROS_INK}"),
        ("  scope=local-first", f"bold {EVEROS_YELLOW_SOFT}"),
        ("  trace ", EVEROS_MUTED),
    ]
    for index, stage in enumerate(TRACE_STAGES):
        if index:
            parts.append((" · ", EVEROS_MUTED))
        if index == active_stage:
            parts.append((stage, f"bold {EVEROS_YELLOW}"))
        else:
            parts.append((stage, EVEROS_AMBER))
    return Text.assemble(*parts)


def _initial_lights() -> dict[str, str]:
    """Default signal-rail state before any round runs."""

    return {
        "core": "not_ready",
        "conversation": "idle",
        "facts": "idle",
        "index": "idle",
        "recall": "idle",
    }


# White = not ready / idle / miss; yellow = ready / active / hit; black = error.
_LIGHT_YELLOW = frozenset({"ready", "captured", "live", "synced", "hit"})

# The sphere is a progress indicator bound to the signal rail: it shows the
# *furthest* pipeline stage currently lit. Checked in furthest-first order; if
# none of these are lit the sphere free-runs its idle loop. (``core`` is just an
# "online" lamp, not a pipeline stage, so it does not drive the sphere — that is
# why an idle session keeps looping after the core comes up.)
_RAIL_STAGE_ORDER = (
    ("recall", "recalling"),
    ("index", "indexing"),
    ("facts", "extracting"),
    ("conversation", "ingesting"),
)


def _light_color(state: str) -> str:
    if state in _LIGHT_YELLOW:
        return EVEROS_YELLOW
    if state == "error":
        return EVEROS_BLACK
    return EVEROS_INK


def _light_label(state: str) -> str:
    return "not ready" if state == "not_ready" else state


_SIGNAL_ROWS = (
    ("core", "memory core      "),
    ("conversation", "conversation     "),
    ("facts", "episode -> facts "),
    ("index", "SQLite + LanceDB "),
    ("recall", "memory recall    "),
)


def _signal_rail_text(lights: dict[str, str] | None = None) -> Text:
    lights = lights or _initial_lights()
    parts: list[tuple[str, str]] = []
    for key, label in _SIGNAL_ROWS:
        state = lights.get(key, "idle")
        color = _light_color(state)
        parts.append(("● ", f"bold {color}"))
        parts.append((label, EVEROS_INK))
        parts.append((f"{_light_label(state)}\n", f"bold {color}"))
    parts.append(("\nsource route\n", EVEROS_MUTED))
    parts.append((_rail_cell(_demo_episode_name()), EVEROS_INK))
    parts.append((" attached\n", f"bold {EVEROS_YELLOW_SOFT}"))
    parts.append((_rail_cell(_demo_fact_name()), EVEROS_INK))
    parts.append((" stored", f"bold {EVEROS_ORANGE}"))
    return Text.assemble(*parts)


def _rail_cell(value: str, *, width: int = SIGNAL_RAIL_SOURCE_WIDTH) -> str:
    if len(value) > width:
        return f"{value[: width - 3]}..."
    return f"{value:<{width}}"


def _demo_episode_name() -> str:
    """Date-stamped episode filename reflecting when the demo is used."""

    return f"episode-{today_with_timezone().isoformat()}.md"


def _demo_fact_name() -> str:
    return f"atomic_fact-{today_with_timezone().isoformat()}.md"


def _capabilities_text() -> Text:
    # Real highlights from evermind.ai: the token-efficiency claim, one headline
    # SOTA benchmark, and core capabilities. No fabricated figures. (local-first
    # is dropped here because the field header already shows scope=local-first.)
    rows = (
        ("token efficiency ", "1/10 of full context", EVEROS_YELLOW),
        ("LoCoMo           ", "93.05% (SOTA)", EVEROS_GREEN),
        ("context window   ", "unlimited", EVEROS_CYAN),
        ("hybrid retrieval ", "BM25 + vector", EVEROS_ORANGE),
        ("agentic rerank   ", "on", EVEROS_YELLOW_SOFT),
        ("multimodal       ", "pdf / image / docs", EVEROS_INK),
        ("self-evolving    ", "cases -> skills", EVEROS_GREEN),
    )
    parts: list[tuple[str, str]] = []
    for label, value, color in rows:
        parts.append((label, EVEROS_MUTED))
        parts.append((f"{value}\n", f"bold {color}"))
    return Text.assemble(*parts)


def _source_tree_text() -> Text:
    return Text.assemble(
        ("episode ", EVEROS_MUTED),
        (f"{_demo_episode_name()}\n", f"bold {EVEROS_YELLOW_SOFT}"),
        ("facts   ", EVEROS_MUTED),
        (f"{_demo_fact_name()}\n", f"bold {EVEROS_ORANGE}"),
        ("index   ", EVEROS_MUTED),
        ("sqlite/system.db + lancedb/*.lance\n", EVEROS_CYAN),
        ("root    ", EVEROS_MUTED),
        ("~/.everos/default_app/demo", EVEROS_INK),
    )


def _recall_proof_text(
    story: DemoStory | None = None,
    *,
    user_label: str = "you",
    saved_pct: int | None = None,
) -> Text:
    story = story or default_demo_story()
    score = f"{story.score:.3f}" if story.score else "—"
    saved = f"~{saved_pct}% tokens (est)" if saved_pct is not None else "—"
    return Text.assemble(
        ("score   ", EVEROS_MUTED),
        (f"{score}\n", f"bold {EVEROS_GREEN}"),
        ("saved   ", EVEROS_MUTED),
        (f"{saved}\n", f"bold {EVEROS_YELLOW}"),
        ("scope   ", EVEROS_MUTED),
        (f"user={user_label} project=demo", EVEROS_INK),
    )


_SPEAKER_COLORS = {
    "you": EVEROS_CYAN,  # the memory you stored
    "ask": EVEROS_YELLOW_SOFT,  # the question you asked
    "everos": EVEROS_GREEN,  # the recalled answer
}


def _conversation_text(log: list[tuple[str, str]]) -> Text:
    if not log:
        return Text("your input and EverOS output will appear here", style=EVEROS_MUTED)
    parts: list[tuple[str, str]] = []
    for speaker, text in log:
        color = _SPEAKER_COLORS.get(speaker, EVEROS_INK)
        parts.append((f"{speaker:<7}", f"bold {color}"))
        parts.append((f"{text}\n", EVEROS_INK))
    return Text.assemble(*parts)
