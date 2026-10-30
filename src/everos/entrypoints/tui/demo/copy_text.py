"""Rich-text copy for every panel of the demo app, and the signal-rail lamps.

Each function returns a ready-to-render ``rich.text.Text`` from plain inputs
(round counters, rail light states, the story), so the copy and its colours
are testable without mounting the Textual app.
"""

from __future__ import annotations

from rich.text import Text

from everos.core.utils.datetime import today_with_timezone
from everos.entrypoints.tui.demo.data import (
    DemoStory,
    default_demo_story,
)
from everos.entrypoints.tui.demo.theme import (
    EVEROS_AMBER,
    EVEROS_BLACK,
    EVEROS_CYAN,
    EVEROS_GREEN,
    EVEROS_INK,
    EVEROS_MUTED,
    EVEROS_ORANGE,
    EVEROS_YELLOW,
    EVEROS_YELLOW_SOFT,
)

SIGNAL_RAIL_SOURCE_WIDTH = 18

# The four pipeline stages shown in the trace header. They line up with the four
# core sphere states, so the active word can highlight in sync with the sphere.
TRACE_STAGES = ("ingest", "extract", "index", "recall")


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
