"""Textual TUI for ``everos demo``."""

from __future__ import annotations

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Static

from everos.entrypoints.cli.demo_sphere import (
    EVEROS_CYAN,
    EVEROS_GREEN,
    EVEROS_ORANGE,
    EVEROS_YELLOW,
    EVEROS_YELLOW_SOFT,
    build_dot_sphere,
    render_dot_sphere_text,
)

EVEROS_BLACK = "#050500"
EVEROS_SURFACE = "#0B0900"
EVEROS_SURFACE_RAISED = "#151200"
EVEROS_INK = "#FFF8CC"
EVEROS_MUTED = "#B7A55E"


class DotSphereWidget(Static):
    """Animated dot sphere that represents EverOS memory activity."""

    DEFAULT_CSS = """
    DotSphereWidget {
        height: 27;
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
    )

    def __init__(self) -> None:
        super().__init__()
        self._phase = 0.0
        self._tick = 0

    def on_mount(self) -> None:
        self.set_interval(1 / 12, self._advance)
        self._advance()

    def _advance(self) -> None:
        self._phase = (self._phase + 0.025) % 1.0
        self._tick += 1
        state = self.STATES[(self._tick // 36) % len(self.STATES)]
        frame = build_dot_sphere(
            width=57,
            height=23,
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
        padding: 1;
    }}

    #hero {{
        height: 4;
        border: double {EVEROS_YELLOW};
        background: {EVEROS_SURFACE_RAISED};
        padding: 0 2;
        color: {EVEROS_INK};
    }}

    #main {{
        height: 1fr;
        margin-top: 1;
    }}

    #sphere-panel {{
        width: 1fr;
        border: double {EVEROS_YELLOW};
        background: {EVEROS_SURFACE};
        padding: 1 2;
    }}

    #side {{
        width: 42;
        margin-left: 1;
    }}

    .panel {{
        border: round {EVEROS_YELLOW};
        background: {EVEROS_SURFACE};
        padding: 1 2;
        margin-bottom: 1;
    }}

    .cyan-panel {{
        border: round {EVEROS_CYAN};
    }}

    .green-panel {{
        border: round {EVEROS_GREEN};
    }}

    #bottom {{
        height: 10;
        margin-top: 1;
    }}

    #markdown {{
        width: 1fr;
        margin-right: 1;
    }}

    #proof {{
        width: 58;
    }}
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="shell"):
            yield Static(_hero_text(), id="hero")
            with Horizontal(id="main"):
                with Vertical(id="sphere-panel"):
                    yield Static(
                        Text("EverOS Memory Sphere", style=f"bold {EVEROS_YELLOW}"),
                        classes="title",
                    )
                    yield DotSphereWidget()
                    yield Static(_sphere_caption(), classes="green-panel panel")
                with Vertical(id="side"):
                    yield Static(_live_run_text(), classes="panel")
                    yield Static(_source_tree_text(), classes="cyan-panel panel")
            with Horizontal(id="bottom"):
                yield Static(_markdown_preview_text(), id="markdown", classes="panel")
                yield Static(
                    _recall_proof_text(),
                    id="proof",
                    classes="green-panel panel",
                )
            yield Static(_payoff_text(), classes="panel")
            yield Footer()

    def action_replay(self) -> None:
        widget = self.query_one(DotSphereWidget)
        widget._tick = 0
        widget._phase = 0.0
        widget._advance()


def run_demo_tui() -> None:
    EverOSDemoApp().run()


def _hero_text() -> Text:
    return Text.assemble(
        ("EverOS", f"bold {EVEROS_YELLOW}"),
        ("  Memory Core Demo\n", f"bold {EVEROS_INK}"),
        ("conversation", f"bold {EVEROS_YELLOW_SOFT}"),
        ("  ->  ", EVEROS_MUTED),
        ("dot sphere", f"bold {EVEROS_YELLOW}"),
        ("  ->  ", EVEROS_MUTED),
        ("recall", f"bold {EVEROS_GREEN}"),
        ("  ->  ", EVEROS_MUTED),
        ("episode.md", f"bold {EVEROS_CYAN}"),
    )


def _sphere_caption() -> Text:
    return Text.assemble(
        ("Q ", f"bold {EVEROS_CYAN}"),
        ("Where does Alice like to climb?\n", EVEROS_INK),
        ("A ", f"bold {EVEROS_GREEN}"),
        ("Alice likes climbing in Yosemite every spring.", f"bold {EVEROS_GREEN}"),
    )


def _live_run_text() -> Text:
    return Text.assemble(
        ("Live Run\n", f"bold {EVEROS_YELLOW}"),
        ("01 ", f"bold {EVEROS_GREEN}"),
        ("Wake server                 ", EVEROS_INK),
        ("OK\n", f"bold {EVEROS_GREEN}"),
        ("02 ", f"bold {EVEROS_GREEN}"),
        ("Ingest conversation          ", EVEROS_INK),
        ("OK\n", f"bold {EVEROS_GREEN}"),
        ("03 ", f"bold {EVEROS_YELLOW}"),
        ("Extract memory               ", EVEROS_INK),
        ("LIVE\n", f"bold {EVEROS_YELLOW}"),
        ("04 ", f"bold {EVEROS_CYAN}"),
        ("Index SQLite + LanceDB       ", EVEROS_INK),
        ("SYNC\n", f"bold {EVEROS_CYAN}"),
        ("05 ", f"bold {EVEROS_GREEN}"),
        ("Recall Yosemite              ", EVEROS_INK),
        ("HIT", f"bold {EVEROS_GREEN}"),
    )


def _source_tree_text() -> Text:
    return Text.assemble(
        ("Markdown Source\n", f"bold {EVEROS_CYAN}"),
        ("~/.everos/default_app/default_project\n", EVEROS_MUTED),
        ("├── users/alice\n", f"bold {EVEROS_YELLOW}"),
        ("│   ├── episodes/\n", EVEROS_INK),
        ("│   │   └── episode-2026-06-20.md\n", f"bold {EVEROS_YELLOW_SOFT}"),
        ("│   ├── .atomic_facts/\n", EVEROS_INK),
        ("│   │   └── atomic_fact-2026-06-20.md\n", f"bold {EVEROS_ORANGE}"),
        ("│   └── user.md\n", EVEROS_INK),
        ("└── .index/\n", EVEROS_MUTED),
        ("    ├── sqlite/system.db\n", EVEROS_CYAN),
        ("    └── lancedb/*.lance", EVEROS_CYAN),
    )


def _markdown_preview_text() -> Text:
    return Text.assemble(
        ("Markdown Preview\n", f"bold {EVEROS_YELLOW}"),
        ("## ep_20260620_00000001\n", f"bold {EVEROS_YELLOW}"),
        ("### Content\n", EVEROS_MUTED),
        (
            "Alice shared that she loves climbing in Yosemite every spring.\n",
            f"bold {EVEROS_GREEN}",
        ),
        (
            "She also mentioned Blue Bottle in SOMA as a favorite coffee shop.",
            EVEROS_INK,
        ),
    )


def _recall_proof_text() -> Text:
    return Text.assemble(
        ("Recall Proof\n", f"bold {EVEROS_GREEN}"),
        ("score  ", EVEROS_MUTED),
        ("0.628\n", f"bold {EVEROS_GREEN}"),
        ("scope  ", EVEROS_MUTED),
        ("user=alice project=default\n", EVEROS_INK),
        ("source ", EVEROS_MUTED),
        ("episode-2026-06-20.md", f"bold {EVEROS_YELLOW}"),
    )


def _payoff_text() -> Text:
    return Text.assemble(
        ("SUCCESSFUL MOMENT ", f"bold black on {EVEROS_YELLOW}"),
        (
            "The sphere becomes a visible local memory field, then resolves to a "
            "recall answer and Markdown source.",
            f"bold {EVEROS_INK}",
        ),
    )
