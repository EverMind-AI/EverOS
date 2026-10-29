"""Textual TUI for ``everos demo``.

``EverOSDemoApp`` is the shell: it lays the panels out, runs the memory and
recall rounds through :mod:`.cloud` and keeps the signal rail, the sphere and
the copy in step. Widgets live in :mod:`.widgets`, the panel copy in
:mod:`.copy_text`, the palette in :mod:`.theme` and the stylesheet in
:mod:`._styles`.
"""

from __future__ import annotations

from functools import partial

import anyio
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.timer import Timer
from textual.widgets import Footer, Input, Static

from everos.entrypoints.tui.demo import cloud
from everos.entrypoints.tui.demo._styles import APP_CSS
from everos.entrypoints.tui.demo.copy_text import (
    _LIGHT_YELLOW,
    _capabilities_text,
    _commands_text,
    _conversation_text,
    _field_header_text,
    _hero_text,
    _initial_lights,
    _live_guidance_text,
    _prompt_memory_text,
    _prompt_query_text,
    _quota_guidance_text,
    _recall_error_text,
    _recall_proof_text,
    _recalling_text,
    _signal_rail_text,
    _source_tree_text,
    _storing_text,
    _unknown_command_text,
)
from everos.entrypoints.tui.demo.data import (
    DemoStory,
    default_demo_story,
)
from everos.entrypoints.tui.demo.widgets.answer_bar import QueryAnswerBar
from everos.entrypoints.tui.demo.widgets.dot_sphere import (
    SPHERE_STAGE_SECONDS,
    DotSphereWidget,
)

# Offline default demo: how many memory -> recall rounds a user plays before the
# TUI nudges them toward the real pipeline (`--cloud` / `--live`).
DEFAULT_DEMO_ROUNDS = 3

# Words a user can type in the input box to quit back to the terminal.
QUIT_COMMANDS = frozenset({"quit", "exit", ":q", "/quit", "/exit"})

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

    CSS = APP_CSS

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
