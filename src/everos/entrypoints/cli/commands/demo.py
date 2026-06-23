"""``everos demo`` — first-run memory sphere demo."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel

from everos.component.utils.datetime import get_utc_now
from everos.entrypoints.tui.demo.data import (
    DEFAULT_MEMORY_SEED,
    DEFAULT_QUERY,
    DemoStory,
    build_demo_story,
    default_demo_story,
)
from everos.entrypoints.tui.demo.widgets.sphere import (
    EVEROS_GREEN,
    EVEROS_YELLOW,
    build_dot_sphere,
    render_dot_sphere_text,
)

LIVE_DEMO_SERVER_URL = "http://127.0.0.1:8000"
LIVE_DEMO_SESSION_ID = "everos-demo-live"
LIVE_DEMO_USER_ID = "everos_demo_user"
LIVE_DEMO_APP_ID = "default"
LIVE_DEMO_PROJECT_ID = "default"
LIVE_DEMO_TIMEOUT_SECONDS = 10.0
LIVE_DEMO_SEARCH_ATTEMPTS = 6
LIVE_DEMO_SEARCH_INTERVAL_SECONDS = 0.5


def register(parent: typer.Typer) -> None:
    """Attach the ``demo`` command to the root CLI app."""

    @parent.command("demo")
    def demo(
        plain: bool = typer.Option(
            False,
            "--plain",
            help="Print a static terminal preview instead of launching the TUI.",
        ),
        cinematic: bool = typer.Option(
            False,
            "--cinematic",
            help="Skip prompts and launch the looping README-style demo.",
        ),
        live: bool = typer.Option(
            False,
            "--live",
            help="Connect to a running EverOS server and run add/flush/search.",
        ),
        server_url: str = typer.Option(
            LIVE_DEMO_SERVER_URL,
            "--server-url",
            help="EverOS server URL used by --live.",
        ),
    ) -> None:
        """Launch the EverOS first-memory Textual TUI."""
        if live:
            _run_live_demo(
                cinematic=cinematic,
                plain=plain or not sys.stdout.isatty(),
                base_url=server_url,
            )
            return

        if plain or not sys.stdout.isatty():
            _print_plain_demo()
            return

        _run_interactive_demo(cinematic=cinematic)


def _run_interactive_demo(*, cinematic: bool) -> None:
    run_demo_tui = _load_run_demo_tui()
    story = None if cinematic else _collect_playable_story()
    run_demo_tui(story=story)


def _run_live_demo(*, cinematic: bool, plain: bool, base_url: str) -> None:
    run_demo_tui = None if plain else _load_run_demo_tui()
    story = default_demo_story() if cinematic or plain else _collect_playable_story()
    live_story = _run_live_demo_flow(story, base_url=base_url)

    if plain:
        _print_plain_demo(live_story)
        return

    if run_demo_tui is None:  # pragma: no cover - guarded by plain branch.
        raise typer.Exit(code=1)
    run_demo_tui(story=live_story)


def _load_run_demo_tui():
    try:
        from everos.entrypoints.tui.demo.app import run_demo_tui
    except ModuleNotFoundError as exc:
        if exc.name != "textual":
            raise
        typer.secho(
            "error: Textual is required for `everos demo`; install the "
            "package with TUI dependencies or run `everos demo --plain`.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from exc

    return run_demo_tui


def _collect_playable_story() -> DemoStory:
    Console().print(
        f"[bold {EVEROS_YELLOW}]EverOS demo[/] "
        "Give it one memory, then ask for it back."
    )
    memory = typer.prompt(
        "Give EverOS one thing to remember",
        default=DEFAULT_MEMORY_SEED,
    )
    query = typer.prompt(
        "Ask EverOS to recall it",
        default=DEFAULT_QUERY,
    )
    return build_demo_story(memory, query)


def _run_live_demo_flow(
    story: DemoStory,
    *,
    base_url: str,
    request_json: Callable[..., dict[str, Any]] | None = None,
    timeout_seconds: float = LIVE_DEMO_TIMEOUT_SECONDS,
    search_attempts: int = LIVE_DEMO_SEARCH_ATTEMPTS,
    search_interval_seconds: float = LIVE_DEMO_SEARCH_INTERVAL_SECONDS,
) -> DemoStory:
    """Run the educational demo story through a live EverOS server."""

    request = request_json or _request_json
    health = request(
        "GET",
        "/health",
        base_url=base_url,
        timeout_seconds=timeout_seconds,
    )
    if health.get("status") != "ok":
        raise typer.BadParameter(
            f"EverOS server at {base_url} did not return healthy status"
        )

    timestamp_ms = int(get_utc_now().timestamp() * 1000)
    request(
        "POST",
        "/api/v1/memory/add",
        base_url=base_url,
        json_body={
            "session_id": LIVE_DEMO_SESSION_ID,
            "app_id": LIVE_DEMO_APP_ID,
            "project_id": LIVE_DEMO_PROJECT_ID,
            "messages": [
                {
                    "sender_id": LIVE_DEMO_USER_ID,
                    "role": "user",
                    "timestamp": timestamp_ms,
                    "content": story.memory,
                }
            ],
        },
        timeout_seconds=timeout_seconds,
    )
    request(
        "POST",
        "/api/v1/memory/flush",
        base_url=base_url,
        json_body={
            "session_id": LIVE_DEMO_SESSION_ID,
            "app_id": LIVE_DEMO_APP_ID,
            "project_id": LIVE_DEMO_PROJECT_ID,
        },
        timeout_seconds=timeout_seconds,
    )

    search_payload = {
        "user_id": LIVE_DEMO_USER_ID,
        "app_id": LIVE_DEMO_APP_ID,
        "project_id": LIVE_DEMO_PROJECT_ID,
        "query": story.query,
        "top_k": 5,
    }
    for attempt in range(search_attempts):
        search = request(
            "POST",
            "/api/v1/memory/search",
            base_url=base_url,
            json_body=search_payload,
            timeout_seconds=timeout_seconds,
        )
        episode = _first_live_episode(search)
        if episode is not None:
            return _story_from_live_episode(story, episode)
        if attempt < search_attempts - 1:
            time.sleep(search_interval_seconds)

    raise typer.BadParameter(
        "EverOS server accepted the memory, but search did not return it yet. "
        "Try `everos demo --live` again after indexing catches up."
    )


def _request_json(
    method: str,
    path: str,
    *,
    base_url: str,
    json_body: dict[str, object] | None = None,
    timeout_seconds: float,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    data = None if json_body is None else json.dumps(json_body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise typer.BadParameter(
            f"Could not reach EverOS server at {base_url}. "
            "Start it with `everos server start` and try again."
        ) from exc
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise typer.BadParameter(f"EverOS server returned non-object JSON: {url}")
    return parsed


def _first_live_episode(payload: dict[str, Any]) -> dict[str, Any] | None:
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    episodes = data.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        return None
    first = episodes[0]
    return first if isinstance(first, dict) else None


def _story_from_live_episode(story: DemoStory, episode: dict[str, Any]) -> DemoStory:
    facts = episode.get("atomic_facts")
    first_fact = facts[0] if isinstance(facts, list) and facts else None
    fact_id = _string_field(first_fact, "id") if isinstance(first_fact, dict) else ""
    answer = (
        _string_field(first_fact, "content") if isinstance(first_fact, dict) else ""
    )
    if not answer:
        answer = (
            _string_field(episode, "summary")
            or _string_field(episode, "episode")
            or story.answer
        )
    episode_id = _string_field(episode, "id") or "live"
    return DemoStory(
        owner=LIVE_DEMO_USER_ID,
        memory=story.memory,
        query=story.query,
        answer=answer,
        source_filename=f"episode:{episode_id}",
        fact_filename=f"fact:{fact_id or 'live'}",
    )


def _string_field(payload: dict[str, Any] | None, key: str) -> str:
    if payload is None:
        return ""
    value = payload.get(key)
    return value if isinstance(value, str) else ""


def _print_plain_demo(story: DemoStory | None = None) -> None:
    story = story or default_demo_story()
    console = Console()
    frame = build_dot_sphere(
        width=57,
        height=23,
        phase=0.18,
        state_key="remembered",
    )
    console.print(
        Panel(
            render_dot_sphere_text(frame),
            title="EverOS Memory Sphere",
            border_style=EVEROS_YELLOW,
        )
    )
    console.print(f"[bold {EVEROS_GREEN}]EverOS remembered:[/]")
    console.print(story.memory)
    console.print()
    console.print(f"[bold {EVEROS_YELLOW}]Source:[/] {story.source_filename}")
