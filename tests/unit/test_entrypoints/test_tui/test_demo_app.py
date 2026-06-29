"""EverOS demo TUI color contracts."""

from __future__ import annotations

import pytest

from everos.entrypoints.tui.demo.app import (
    SPHERE_FRAME_HEIGHT,
    SPHERE_FRAME_WIDTH,
    TERMINAL_CELL_HEIGHT_RATIO,
    TRACE_STAGES,
    DotSphereWidget,
    EverOSDemoApp,
    QueryAnswerBar,
    _capabilities_text,
    _conversation_text,
    _field_header_text,
    _hero_text,
    _recall_proof_text,
    _signal_rail_text,
    _source_tree_text,
    _state_to_stage,
)
from everos.entrypoints.tui.demo.data import DemoStory
from everos.entrypoints.tui.demo.widgets.sphere import SPHERE_STATES

_YELLOW = "#F9B91C"


def _story(memory: str, query: str, answer: str) -> DemoStory:
    return DemoStory(
        owner="you",
        memory=memory,
        query=query,
        answer=answer,
        source_filename="episode-demo.md",
        fact_filename="atomic_fact-demo.md",
    )


def test_demo_tui_uses_poster_derived_brand_palette() -> None:
    css = EverOSDemoApp.CSS

    assert "#F9B91C" in css
    assert "#31302B" in css
    assert "#F5EDDC" in css
    assert "#918C80" in css
    assert "#FFE600" not in css
    assert "#55D6FF" not in css
    assert "#73F7A7" not in css
    assert any(span.style == "bold #F9B91C" for span in _hero_text().spans)


def test_demo_tui_uses_elevated_instrument_layout() -> None:
    css = EverOSDemoApp.CSS

    assert "#command-strip" in css
    assert "#memory-field" in css
    assert "#signal-rail" in css
    assert "#capabilities" in css
    assert "#provenance-strip" in css
    assert "#conversation" in css
    assert "FooterKey" in css
    assert "background: #F9B91C" in css
    assert any("on #F9B91C" in span.style for span in _hero_text().spans)
    assert "border: double" not in css
    assert "border: heavy" not in css


def test_demo_tui_uses_balanced_panel_proportions() -> None:
    css = EverOSDemoApp.CSS

    command_strip = _css_block(css, "#command-strip")
    signal_rail = _css_block(css, "#signal-rail")
    conversation = _css_block(css, "#conversation")

    assert "height: 2;" in command_strip
    assert "border-left: thick" not in command_strip
    assert "background: #31302B" not in command_strip
    assert len(_hero_text().plain.splitlines()) == 1
    assert len(_hero_text().plain) <= 56

    assert "height: 1fr;" in DotSphereWidget.DEFAULT_CSS

    assert "height: 1fr;" in signal_rail
    assert "source route" in _signal_rail_text().plain

    # The conversation log sits below the yellow line.
    assert "border-top: hkey #F9B91C;" in conversation


def test_demo_tui_sphere_renders_round_in_terminal_cells() -> None:
    visual_ratio = (SPHERE_FRAME_WIDTH - 4) / (
        SPHERE_FRAME_HEIGHT * TERMINAL_CELL_HEIGHT_RATIO
    )

    assert visual_ratio == pytest.approx(1.0, abs=0.04)
    assert SPHERE_FRAME_WIDTH == 37
    assert SPHERE_FRAME_HEIGHT == 17


def test_demo_tui_celebrates_after_source_reveal() -> None:
    assert DotSphereWidget.STATES[-2:] == ("source", "celebrating")
    assert set(DotSphereWidget.STATES).issubset(SPHERE_STATES)


def test_signal_rail_lights_reflect_state() -> None:
    idle = _signal_rail_text().plain
    assert "memory core" in idle
    assert "not ready" in idle  # core idle => not ready
    assert "source route" in idle

    active = _signal_rail_text(
        {
            "core": "ready",
            "conversation": "captured",
            "facts": "live",
            "index": "synced",
            "recall": "hit",
        }
    ).plain
    for label in ("ready", "captured", "live", "synced", "hit"):
        assert label in active


def test_signal_rail_light_colors_follow_white_yellow_black() -> None:
    rail = _signal_rail_text(
        {
            "core": "error",
            "conversation": "idle",
            "facts": "live",
            "index": "idle",
            "recall": "idle",
        }
    )
    dot_styles = [
        span.style for span in rail.spans if "●" in rail.plain[span.start : span.end]
    ]
    assert f"bold {_YELLOW}" in dot_styles  # an active light is yellow
    assert "bold #1D1C18" in dot_styles  # the errored light is black


def test_capabilities_box_uses_real_website_numbers() -> None:
    text = _capabilities_text().plain
    # Real highlights from evermind.ai: token efficiency + one SOTA benchmark.
    assert "1/10 of full context" in text  # real token-efficiency claim
    assert "93.05%" in text  # one headline benchmark (LoCoMo)
    assert "83.00%" not in text  # only one score now
    assert "rerank" in text
    # local-first is dropped here (already shown in the field header scope).
    assert "local-first" not in text


def test_source_lock_uses_date_stamped_filenames() -> None:
    text = _source_tree_text().plain
    assert "episode-" in text and ".md" in text
    assert "atomic_fact-" in text


def test_recall_lock_shows_real_score_and_demo_scope() -> None:
    story = DemoStory(
        owner="everos_demo_abc",
        memory="m",
        query="q",
        answer="a",
        source_filename="",
        fact_filename="",
        score=0.873,
    )
    text = _recall_proof_text(story, user_label="YangtzeSeventh", saved_pct=62).plain
    assert "0.873" in text
    assert "user=YangtzeSeventh" in text  # local user, not the session id or alice
    assert "project=demo" in text
    assert "~62% tokens (est)" in text
    assert "similarity" not in text
    # No saved figure until a round has run.
    assert "saved   —" in _recall_proof_text(story, user_label="x").plain


def test_conversation_log_accumulates_turns() -> None:
    empty = _conversation_text([]).plain
    assert "will appear here" in empty

    filled = _conversation_text([("where do I climb?", "Yosemite")]).plain
    assert "you" in filled
    assert "where do I climb?" in filled
    assert "everos" in filled
    assert "Yosemite" in filled


def test_field_header_shows_local_user_and_trace_stages() -> None:
    header = _field_header_text(user_label="YangtzeSeventh", active_stage=1)

    assert "user=YangtzeSeventh" in header.plain
    assert "scope=local-first" in header.plain
    for stage in TRACE_STAGES:
        assert stage in header.plain


def test_field_header_highlights_only_the_active_stage() -> None:
    header = _field_header_text(user_label="you", active_stage=2)

    highlighted = {
        header.plain[span.start : span.end]
        for span in header.spans
        if span.style == f"bold {_YELLOW}"
    }
    assert highlighted & set(TRACE_STAGES) == {"index"}


def test_state_to_stage_maps_sphere_states_to_trace_words() -> None:
    assert _state_to_stage("ingesting") == 0
    assert _state_to_stage("extracting") == 1
    assert _state_to_stage("indexing") == 2
    assert _state_to_stage("recalling") == 3
    assert _state_to_stage("booting") == -1


def test_query_answer_bar_keeps_both_labels() -> None:
    rendered = QueryAnswerBar().render().plain

    assert "Query" in rendered
    assert "Answer" in rendered


def test_ctrl_c_is_a_priority_quit_binding() -> None:
    quit_keys = {
        binding.key
        for binding in EverOSDemoApp.BINDINGS
        if getattr(binding, "action", None) == "quit"
        and getattr(binding, "priority", False)
    }
    assert "ctrl+c" in quit_keys
    assert "ctrl+q" in quit_keys


def test_help_and_unknown_command_text() -> None:
    from everos.entrypoints.tui.demo.app import _help_text, _unknown_command_text

    help_plain = _help_text().plain
    for command in ("/help", "/live", "/replay", "/clear", "/quit"):
        assert command in help_plain
    assert "unknown command /bogus" in _unknown_command_text("/bogus").plain


def test_live_guidance_points_to_own_key_flow() -> None:
    from everos.entrypoints.tui.demo.app import _live_guidance_text

    text = _live_guidance_text().plain
    assert "everos init" in text
    assert "everos demo --live" in text


async def test_slash_live_does_not_consume_a_turn() -> None:
    from textual.widgets import Input

    app = EverOSDemoApp(
        interactive=True, base_url="http://server.test", session_id="s", user_id="u"
    )
    async with app.run_test() as pilot:
        console_input = app.query_one("#console-input", Input)
        console_input.value = "/live"
        await pilot.press("enter")
        await pilot.pause()

        assert app._conversation_phase == "memory"
        assert app._pending_memory == ""


async def test_slash_help_does_not_consume_a_turn() -> None:
    from textual.widgets import Input

    app = EverOSDemoApp(
        interactive=True, base_url="http://server.test", session_id="s", user_id="u"
    )
    async with app.run_test() as pilot:
        console_input = app.query_one("#console-input", Input)
        console_input.value = "/help"
        await pilot.press("enter")
        await pilot.pause()

        # /help is a command, not a memory: the conversation does not advance.
        assert app._conversation_phase == "memory"
        assert app._pending_memory == ""


async def test_slash_clear_wipes_the_conversation_log() -> None:
    from textual.widgets import Input

    app = EverOSDemoApp(
        interactive=True, base_url="http://server.test", session_id="s", user_id="u"
    )
    async with app.run_test() as pilot:
        app._record_turn("where do I climb?", "Yosemite")
        assert app._log

        console_input = app.query_one("#console-input", Input)
        console_input.value = "/clear"
        await pilot.press("enter")
        await pilot.pause()

        assert app._log == []


async def test_typing_quit_exits_the_app(monkeypatch) -> None:
    from textual.widgets import Input

    app = EverOSDemoApp(
        interactive=True,
        base_url="http://server.test",
        session_id="s",
        user_id="u",
    )
    async with app.run_test() as pilot:
        exited: list[bool] = []
        monkeypatch.setattr(app, "exit", lambda *a, **k: exited.append(True))
        console_input = app.query_one("#console-input", Input)
        console_input.value = "quit"
        await pilot.press("enter")
        await pilot.pause()

        assert exited == [True]


def test_demo_tui_signal_rail_keeps_source_status_columns_separate() -> None:
    rail = _signal_rail_text().plain

    assert "mdattached" not in rail
    assert "md7 nodes" not in rail
    assert "..." in rail


async def test_demo_tui_interactive_runs_cloud_round_per_input(monkeypatch) -> None:
    from textual.widgets import Input

    from everos.entrypoints.tui.demo import cloud

    monkeypatch.setattr(cloud, "add_memory", lambda *_, **__: "task-1")
    monkeypatch.setattr(cloud, "wait_task", lambda *_, **__: None)
    monkeypatch.setattr(cloud, "flush_memory", lambda *_, **__: None)

    def fake_search(
        memory: str, query: str, *, base_url: str, user_id: str, **_: object
    ) -> DemoStory:
        assert base_url == "http://server.test"
        return _story(memory, query, f"recalled<{memory}>")

    monkeypatch.setattr(cloud, "search_recall", fake_search)

    app = EverOSDemoApp(
        interactive=True,
        max_rounds=2,
        base_url="http://server.test",
        session_id="everos-demo-x",
        user_id="everos_demo_x",
    )
    async with app.run_test() as pilot:
        console_input = app.query_one("#console-input", Input)

        # Round 1: a memory, then a recall query -> a real (faked) cloud round.
        console_input.value = "我喜欢吃杨梅"
        await pilot.press("enter")
        console_input.value = "我喜欢吃什么"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        # BUG 305: the panels follow the user's own input, never Yosemite.
        assert app._story.memory == "我喜欢吃杨梅"
        assert app._story.query == "我喜欢吃什么"
        assert app._story.answer == "recalled<我喜欢吃杨梅>"
        assert "Yosemite" not in app._story.answer
        # Lights walked the full pipeline to a hit.
        assert app._lights["core"] == "ready"
        assert app._lights["recall"] == "hit"
        # A per-round token-saving estimate was computed.
        assert app._saved_pct is not None

        # Round 2 reaches the cap and locks the input behind the upgrade nudge.
        console_input.value = "I bike to work"
        await pilot.press("enter")
        console_input.value = "How do I commute?"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app._conversation_phase == "done"
        assert console_input.disabled is True


async def test_demo_tui_interactive_shows_quota_guidance(monkeypatch) -> None:
    from textual.widgets import Input

    from everos.entrypoints.tui.demo import cloud
    from everos.entrypoints.tui.demo.app import _quota_guidance_text

    def quota(*_: object, **__: object) -> None:
        raise cloud.CloudQuotaError("http://server.test")

    monkeypatch.setattr(cloud, "add_memory", quota)

    app = EverOSDemoApp(
        interactive=True,
        base_url="http://server.test",
        session_id="s",
        user_id="u",
    )
    async with app.run_test() as pilot:
        console_input = app.query_one("#console-input", Input)
        console_input.value = "a memory"
        await pilot.press("enter")
        console_input.value = "a question"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app._conversation_phase == "done"
        assert console_input.disabled is True

    assert "everos init" in _quota_guidance_text().plain


async def test_demo_tui_non_interactive_has_no_input_box() -> None:
    from textual.css.query import NoMatches
    from textual.widgets import Input

    app = EverOSDemoApp()
    async with app.run_test():
        with pytest.raises(NoMatches):
            app.query_one("#console-input", Input)


def _css_block(css: str, selector: str) -> str:
    start = css.index(f"{selector} {{")
    end = css.index("}", start)
    return css[start:end]
