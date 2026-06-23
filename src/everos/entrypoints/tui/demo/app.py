"""Textual TUI for ``everos demo``."""

from __future__ import annotations

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.timer import Timer
from textual.widgets import Footer, Static

from everos.entrypoints.tui.demo.data import DemoStory, default_demo_story
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

    def __init__(self) -> None:
        super().__init__()
        self._phase = 0.0
        self._tick = 0
        self._animation_timer: Timer | None = None

    def on_mount(self) -> None:
        self._animation_timer = self.set_interval(1 / 12, self._advance)
        self._advance()

    def pause_animation(self) -> None:
        if self._animation_timer is not None:
            self._animation_timer.pause()

    def _advance(self) -> None:
        self._phase = (self._phase + 0.025) % 1.0
        self._tick += 1
        state = self.STATES[(self._tick // 36) % len(self.STATES)]
        frame = build_dot_sphere(
            width=SPHERE_FRAME_WIDTH,
            height=SPHERE_FRAME_HEIGHT,
            phase=self._phase,
            state_key=state,
        )
        self.update(render_dot_sphere_text(frame))


class EverOSDemoApp(App[None]):
    """Fullscreen first-run demo cockpit."""

    TITLE = "EverOS Memory Core"
    SUB_TITLE = "dot sphere demo"
    BINDINGS = [
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
    }}

    #signal-rail {{
        width: 48;
        height: 100%;
        margin-left: 1;
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

    #payoff {{
        height: 2;
        border-top: hkey {EVEROS_YELLOW};
        background: {EVEROS_SURFACE};
        color: {EVEROS_INK};
        padding: 0 1;
        margin-top: 1;
        content-align: left middle;
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

    def __init__(self, *, story: DemoStory | None = None) -> None:
        super().__init__()
        self._story = story or default_demo_story()

    def compose(self) -> ComposeResult:
        with Vertical(id="shell"):
            yield Static(_hero_text(), id="command-strip")
            with Horizontal(id="main"):
                memory_field = Vertical(id="memory-field")
                memory_field.border_title = "memory field"
                with memory_field:
                    yield Static(_field_header_text(self._story), id="field-header")
                    yield DotSphereWidget()
                    yield Static(_sphere_caption(self._story), id="field-answer")
                signal_rail = Static(_signal_rail_text(self._story), id="signal-rail")
                signal_rail.border_title = "signal rail"
                yield signal_rail
            with Horizontal(id="provenance-strip"):
                source_lock = Static(_source_tree_text(self._story), id="source-lock")
                source_lock.border_title = "source lock"
                yield source_lock
                recall_lock = Static(_recall_proof_text(self._story), id="recall-lock")
                recall_lock.border_title = "recall lock"
                yield recall_lock
            yield Static(_payoff_text(self._story), id="payoff")
            yield Footer(show_command_palette=False)

    def action_replay(self) -> None:
        widget = self.query_one(DotSphereWidget)
        widget._tick = 0
        widget._phase = 0.0
        widget._advance()


def run_demo_tui(*, story: DemoStory | None = None) -> None:
    EverOSDemoApp(story=story).run()


def _hero_text() -> Text:
    return Text.assemble(
        (" everos demo ", f"bold black on {EVEROS_YELLOW}"),
        ("  memory core ", f"bold {EVEROS_YELLOW}"),
        ("online", EVEROS_MUTED),
    )


def _field_header_text(story: DemoStory | None = None) -> Text:
    story = story or default_demo_story()
    return Text.assemble(
        (f"user={story.owner}", f"bold {EVEROS_INK}"),
        ("  scope=local-first", f"bold {EVEROS_YELLOW_SOFT}"),
        ("  trace ", EVEROS_MUTED),
        ("conversation -> facts -> index", f"bold {EVEROS_YELLOW}"),
        ("  live", f"bold {EVEROS_ORANGE}"),
    )


def _sphere_caption(story: DemoStory | None = None) -> Text:
    story = story or default_demo_story()
    return Text.assemble(
        ("query  ", f"bold {EVEROS_CYAN}"),
        (f"{story.query}  ", EVEROS_INK),
        ("->  ", EVEROS_MUTED),
        ("answer ", f"bold {EVEROS_GREEN}"),
        (story.answer, f"bold {EVEROS_GREEN}"),
    )


def _signal_rail_text(story: DemoStory | None = None) -> Text:
    story = story or default_demo_story()
    return Text.assemble(
        ("● ", f"bold {EVEROS_GREEN}"),
        ("memory core        ", EVEROS_INK),
        ("ready\n", f"bold {EVEROS_GREEN}"),
        ("● ", f"bold {EVEROS_YELLOW_SOFT}"),
        ("conversation       ", EVEROS_INK),
        ("captured\n", f"bold {EVEROS_YELLOW_SOFT}"),
        ("● ", f"bold {EVEROS_ORANGE}"),
        ("episode -> facts   ", EVEROS_INK),
        ("live\n", f"bold {EVEROS_ORANGE}"),
        ("● ", f"bold {EVEROS_CYAN}"),
        ("SQLite + LanceDB   ", EVEROS_INK),
        ("synced\n", f"bold {EVEROS_CYAN}"),
        ("● ", f"bold {EVEROS_GREEN}"),
        ("memory recall      ", EVEROS_INK),
        ("hit\n", f"bold {EVEROS_GREEN}"),
        ("\nsource route\n", EVEROS_MUTED),
        (_rail_cell(story.source_filename), EVEROS_INK),
        (" attached\n", f"bold {EVEROS_YELLOW_SOFT}"),
        (_rail_cell(story.fact_filename), EVEROS_INK),
        (" 7 nodes\n", f"bold {EVEROS_ORANGE}"),
        ("lancedb orbit      ", EVEROS_INK),
        ("synced\n", f"bold {EVEROS_CYAN}"),
        ("\nrecall proof\n", EVEROS_MUTED),
        ("score              ", EVEROS_INK),
        ("0.628\n", f"bold {EVEROS_GREEN}"),
        ("source             ", EVEROS_INK),
        (f"{story.source_filename}\n", f"bold {EVEROS_CYAN}"),
        ("field integrity\n", EVEROS_MUTED),
        ("█████████░  92%\n", f"bold {EVEROS_YELLOW}"),
        ("latency            ", EVEROS_MUTED),
        ("42 ms\n", f"bold {EVEROS_GREEN}"),
        ("mode               ", EVEROS_MUTED),
        ("local-first", f"bold {EVEROS_INK}"),
    )


def _rail_cell(value: str, *, width: int = SIGNAL_RAIL_SOURCE_WIDTH) -> str:
    if len(value) > width:
        return f"{value[: width - 3]}..."
    return f"{value:<{width}}"


def _source_tree_text(story: DemoStory | None = None) -> Text:
    story = story or default_demo_story()
    return Text.assemble(
        ("episode ", EVEROS_MUTED),
        (f"{story.source_filename}\n", f"bold {EVEROS_YELLOW_SOFT}"),
        ("facts   ", EVEROS_MUTED),
        (f"{story.fact_filename}\n", f"bold {EVEROS_ORANGE}"),
        ("index   ", EVEROS_MUTED),
        ("sqlite/system.db + lancedb/*.lance\n", EVEROS_CYAN),
        ("root    ", EVEROS_MUTED),
        ("~/.everos/default_app/default_project", EVEROS_INK),
    )


def _recall_proof_text(story: DemoStory | None = None) -> Text:
    story = story or default_demo_story()
    return Text.assemble(
        ("score   ", EVEROS_MUTED),
        ("0.628\n", f"bold {EVEROS_GREEN}"),
        ("scope   ", EVEROS_MUTED),
        (f"user={story.owner} project=default\n", EVEROS_INK),
        ("answer  ", EVEROS_MUTED),
        (story.answer, f"bold {EVEROS_YELLOW}"),
    )


def _payoff_text(story: DemoStory | None = None) -> Text:
    story = story or default_demo_story()
    return Text.assemble(
        ("memory formed: ", f"bold {EVEROS_YELLOW}"),
        (
            f"EverOS recalled {story.answer} and kept the source attached.",
            f"bold {EVEROS_INK}",
        ),
    )
