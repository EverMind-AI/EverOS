"""Textual TUI for ``everos demo``."""

from __future__ import annotations

from functools import partial

import anyio
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.timer import Timer
from textual.widgets import Footer, Input, Static

from everos.component.utils.datetime import today_with_timezone
from everos.entrypoints.tui.demo import cloud
from everos.entrypoints.tui.demo.data import (
    DEFAULT_MEMORY_SEED,
    DEFAULT_QUERY,
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
SPHERE_FPS = 12
SPHERE_STAGE_SECONDS = 3.0
SPHERE_STAGE_TICKS = round(SPHERE_FPS * SPHERE_STAGE_SECONDS)

# The four pipeline stages shown in the trace header. They line up with the four
# core sphere states, so the active word can highlight in sync with the sphere.
TRACE_STAGES = ("ingest", "extract", "index", "recall")

# Words a user can type in the input box to quit back to the terminal.
QUIT_COMMANDS = frozenset({"quit", "exit", ":q", "/quit"})
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


class DotSphereWidget(Static):
    """Animated dot sphere that represents EverOS memory activity."""

    DEFAULT_CSS = """
    DotSphereWidget {
        height: 1fr;
        content-align: center middle;
    }
    """

    STATES = (
        "booting",
        "ingesting",
        "extracting",
        "indexing",
        "recalling",
        "remembered",
        "source",
        "celebrating",
    )

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
        self._animation_timer: Timer | None = None

    def on_mount(self) -> None:
        self._animation_timer = self.set_interval(1 / SPHERE_FPS, self._advance)
        self._advance()

    def pause_animation(self) -> None:
        if self._animation_timer is not None:
            self._animation_timer.pause()

    def _advance(self) -> None:
        self._phase = (self._phase + 0.025) % 1.0
        self._tick += 1
        state = self.STATES[(self._tick // SPHERE_STAGE_TICKS) % len(self.STATES)]
        frame = build_dot_sphere(
            width=SPHERE_FRAME_WIDTH,
            height=SPHERE_FRAME_HEIGHT,
            phase=self._phase,
            state_key=state,
        )
        self.update(render_dot_sphere_text(frame))

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
        height: 2;
        padding: 0 1;
        color: {EVEROS_INK};
        content-align: left middle;
    }}

    #main {{
        height: 1fr;
        margin-top: 1;
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
        height: 8;
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
        border: round {EVEROS_CYAN};
        background: {EVEROS_SURFACE};
        padding: 0 2;
        margin-right: 1;
    }}

    #recall-lock {{
        width: 54;
        border: round {EVEROS_GREEN};
        background: {EVEROS_SURFACE};
        padding: 0 2;
    }}

    #conversation {{
        height: 4;
        border-top: hkey {EVEROS_YELLOW};
        background: {EVEROS_SURFACE};
        color: {EVEROS_INK};
        padding: 0 1;
        margin-top: 1;
    }}

    #console {{
        height: 2;
        margin-top: 1;
    }}

    #console-prompt {{
        height: 1;
        padding: 0 1;
        content-align: left middle;
    }}

    #console-input {{
        height: 1;
        border: none;
        background: {EVEROS_SURFACE_RAISED};
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
        base_url: str = cloud.DEFAULT_CLOUD_DEMO_SERVER_URL,
        session_id: str = cloud.LIVE_DEMO_SESSION_ID,
        user_id: str = cloud.LIVE_DEMO_USER_ID,
        user_label: str = "you",
        max_rounds: int = DEFAULT_DEMO_ROUNDS,
    ) -> None:
        super().__init__()
        self._story = story or default_demo_story()
        self._interactive = interactive
        self._base_url = base_url
        self._session_id = session_id
        self._user_id = user_id
        self._user_label = user_label
        self._max_rounds = max_rounds
        self._round = 0
        self._active_stage = -1
        self._conversation_phase = "memory"
        self._pending_memory = ""
        self._lights = _initial_lights()
        self._log: list[tuple[str, str]] = []

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
            conversation = Static(_conversation_text(self._log), id="conversation")
            conversation.border_title = "conversation"
            yield conversation
            if self._interactive:
                with Vertical(id="console"):
                    yield Static(
                        _prompt_memory_text(self._round, self._max_rounds),
                        id="console-prompt",
                    )
                    yield Input(
                        placeholder=(
                            "type a memory and press enter  ·  'quit' or ctrl+c to exit"
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
        if not self._interactive or self._conversation_phase in {"recalling", "done"}:
            return
        value = event.value.strip()
        if value.lower() in QUIT_COMMANDS:
            self.exit()
            return
        prompt = self.query_one("#console-prompt", Static)
        field = self.query_one("#console-input", Input)
        if self._conversation_phase == "memory":
            self._pending_memory = value or DEFAULT_MEMORY_SEED
            self._conversation_phase = "query"
            prompt.update(_prompt_query_text())
            field.value = ""
            return

        # Query submitted: run the real cloud round off the event loop so the UI
        # (sphere animation, input) stays responsive while we wait on the server.
        query = value or DEFAULT_QUERY
        self._conversation_phase = "recalling"
        field.value = ""
        field.disabled = True
        prompt.update(_recalling_text())
        self.run_worker(
            self._recall(self._pending_memory, query),
            group="recall",
            exclusive=True,
        )

    async def _recall(self, memory: str, query: str) -> None:
        # Reset the per-round lights; each step below lights up as it completes,
        # so the signal rail mirrors the real add -> flush -> search pipeline.
        self._reset_round_lights()
        base_url, session_id, user_id = (
            self._base_url,
            self._session_id,
            self._user_id,
        )
        try:
            await anyio.to_thread.run_sync(
                partial(cloud.check_health, base_url=base_url)
            )
            self._set_light("core", "ready")
            await anyio.to_thread.run_sync(
                partial(
                    cloud.add_memory,
                    memory,
                    base_url=base_url,
                    session_id=session_id,
                    user_id=user_id,
                )
            )
            self._set_light("conversation", "captured")
            await anyio.to_thread.run_sync(
                partial(cloud.flush_memory, base_url=base_url, session_id=session_id)
            )
            self._set_light("facts", "live")
            self._set_light("index", "synced")
            story = await anyio.to_thread.run_sync(
                partial(
                    cloud.search_recall,
                    memory,
                    query,
                    base_url=base_url,
                    user_id=user_id,
                )
            )
        except cloud.CloudQuotaError:
            self._enter_done(_quota_guidance_text())
            return
        except cloud.CloudDemoError as exc:
            self._set_light("core", "error")
            self._show_recall_error(str(exc))
            return

        if story is None:
            self._set_light("recall", "miss")
            answer = "(no matching memory found)"
            self._record_turn(query, answer)
            story = DemoStory(
                owner=user_id,
                memory=memory,
                query=query,
                answer=answer,
                source_filename="",
                fact_filename="",
            )
        else:
            self._set_light("recall", "hit")
            self._record_turn(story.query, story.answer)
        self._finish_round(story)

    def _finish_round(self, story: DemoStory) -> None:
        self._story = story
        self.query_one("#recall-lock", Static).update(
            _recall_proof_text(story, user_label=self._user_label)
        )
        self.action_replay()
        self._round += 1
        if self._round >= self._max_rounds:
            self._enter_done(_quota_guidance_text())
            return
        self._conversation_phase = "memory"
        self.query_one("#console-prompt", Static).update(
            _prompt_memory_text(self._round, self._max_rounds)
        )
        self._reenable_input()

    def _reset_round_lights(self) -> None:
        self._lights.update(
            conversation="idle", facts="idle", index="idle", recall="idle"
        )
        self.query_one("#signal-rail", Static).update(_signal_rail_text(self._lights))

    def _set_light(self, key: str, state: str) -> None:
        self._lights[key] = state
        self.query_one("#signal-rail", Static).update(_signal_rail_text(self._lights))

    def _record_turn(self, query: str, answer: str) -> None:
        self._log.append((query, answer))
        self.query_one("#conversation", Static).update(_conversation_text(self._log))

    def _enter_done(self, message: Text) -> None:
        self._conversation_phase = "done"
        self.query_one("#console-prompt", Static).update(message)
        self.query_one("#console-input", Input).disabled = True

    def _show_recall_error(self, message: str) -> None:
        # Recall failed (server unreachable, unhealthy, or slow). Surface the
        # reason honestly and let the user retry a fresh round.
        self._conversation_phase = "memory"
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
    base_url: str = cloud.DEFAULT_CLOUD_DEMO_SERVER_URL,
    session_id: str = cloud.LIVE_DEMO_SESSION_ID,
    user_id: str = cloud.LIVE_DEMO_USER_ID,
    user_label: str = "you",
) -> None:
    EverOSDemoApp(
        story=story,
        interactive=interactive,
        base_url=base_url,
        session_id=session_id,
        user_id=user_id,
        user_label=user_label,
    ).run()


def _prompt_memory_text(round_index: int, total_rounds: int) -> Text:
    if round_index == 0:
        return Text("What should EverOS remember?", style=f"bold {EVEROS_YELLOW}")
    return Text.assemble(
        (f"round {round_index + 1}/{total_rounds}  ", EVEROS_MUTED),
        ("what should EverOS remember next?", f"bold {EVEROS_YELLOW}"),
    )


def _prompt_query_text() -> Text:
    return Text("Now ask EverOS to recall it.", style=f"bold {EVEROS_CYAN}")


def _recalling_text() -> Text:
    return Text("recalling from EverOS...", style=f"bold {EVEROS_ORANGE}")


def _recall_error_text(message: str) -> Text:
    return Text.assemble(
        ("could not reach the demo server  ", f"bold {EVEROS_ORANGE}"),
        (f"({message})  ", EVEROS_MUTED),
        ("set EVEROS_CLOUD_DEMO_URL or use --live; type to retry", EVEROS_INK),
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
    rows = (
        ("hybrid retrieval ", "BM25 + vector", EVEROS_YELLOW),
        ("agentic rerank   ", "on", EVEROS_GREEN),
        ("multimodal       ", "image / pdf / audio", EVEROS_ORANGE),
        ("md-first         ", "auditable source", EVEROS_YELLOW_SOFT),
        ("local-first      ", "runs on your machine", EVEROS_INK),
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
    story: DemoStory | None = None, *, user_label: str = "you"
) -> Text:
    story = story or default_demo_story()
    score = f"{story.score:.3f}" if story.score else "—"
    return Text.assemble(
        ("score   ", EVEROS_MUTED),
        (f"{score}\n", f"bold {EVEROS_GREEN}"),
        ("scope   ", EVEROS_MUTED),
        (f"user={user_label} project=demo", EVEROS_INK),
    )


def _conversation_text(log: list[tuple[str, str]]) -> Text:
    if not log:
        return Text("your input and EverOS output will appear here", style=EVEROS_MUTED)
    parts: list[tuple[str, str]] = []
    for query, answer in log:
        parts.append(("you    ", f"bold {EVEROS_CYAN}"))
        parts.append((f"{query}\n", EVEROS_INK))
        parts.append(("everos ", f"bold {EVEROS_GREEN}"))
        parts.append((f"{answer}\n", EVEROS_INK))
    return Text.assemble(*parts)
