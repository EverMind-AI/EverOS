"""``everos memory``: its standard output is the JSON answer, and nothing else.

The commands run in a subprocess over a fresh root with the optional providers
unconfigured, so building the application logs a warning per missing
provider, as it does on a real host. What these pin:

- ``history``, ``update``, ``forget`` (a dry run, then ``--apply``) and
  ``undo`` each print exactly one JSON document on standard output, which a
  script parses as is; every log line goes to standard error. (The smoke
  parsing ``history``'s output found a provider warning in front of it.)
- That holds for a warning the settings log while they load, too: a key they
  ignore (the benchmark launchers export one) is named before the settings
  that send the logs to standard error exist. (The second smoke found
  ``config_key_ignored`` in front of the JSON.)
- No command asks the OS for a timezone, so a container with none prints no
  tzlocal warning.
- The round trip itself: the update supersedes the version it names, the
  dry run marks nothing, the real forget answers the id the undo takes.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.markdown import MarkdownEpisodeStore
from everos.memory.models import Episode
from everos.memory.ports import NO_SOURCE_CHECK, Scope

from .conftest import hermetic_env

_TZ_WARNING = "Can not find any timezone configuration"


def _seed(root: Path) -> str:
    """One episode of amy in conversation s1; its id."""

    async def write() -> str:
        memory_root = MemoryRoot(root)
        memory_root.ensure()
        (ref,) = await MarkdownEpisodeStore(memory_root).append_many(
            Scope(
                app_id="default",
                project_id="default",
                owner_id="amy",
                owner_type="user",
            ),
            [
                Episode(
                    owner_id="amy",
                    episode="Amy adopted a cat named Biscuit.",
                    summary="Biscuit",
                    subject="Biscuit",
                    timestamp=1_772_323_200_000,
                    session_id="s1",
                    parent_id="mc_1",
                )
            ],
            only_if=NO_SOURCE_CHECK,
        )
        return ref.id

    return asyncio.run(write())


def _memory(
    root: Path, *args: str, env: dict[str, str] | None = None
) -> tuple[dict[str, Any], str]:
    """Run ``everos memory <args>``: its parsed stdout, and its stderr.

    ``env`` adds variables to the hermetic environment.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "everos.entrypoints.cli.main",
            "memory",
            *args,
            "--root",
            str(root),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**hermetic_env(plain_output=True), **(env or {})},
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    try:
        answer = json.loads(result.stdout)
    except ValueError as exc:
        raise AssertionError(
            f"stdout is not one JSON document: {result.stdout!r}"
        ) from exc
    return answer, result.stderr


def test_history_prints_its_json_alone_and_logs_on_stderr(tmp_path: Path) -> None:
    episode = _seed(tmp_path)

    history, stderr = _memory(tmp_path, "history", episode, "--user", "amy")

    assert [v["id"] for v in history["versions"]] == [episode]
    assert history["versions"][0]["state"] == "current"
    assert "rerank_capability_build_failed" in stderr  # the logs went here
    assert _TZ_WARNING not in stderr


def test_a_warning_while_the_settings_load_goes_to_stderr(tmp_path: Path) -> None:
    episode = _seed(tmp_path)

    history, stderr = _memory(
        tmp_path,
        "history",
        episode,
        "--user",
        "amy",
        env={"EVEROS_LLM__SEARCH_TIMEOUT_SECONDS": "60"},  # an ignored key
    )

    assert [v["id"] for v in history["versions"]] == [episode]
    assert "config_key_ignored" in stderr


def test_update_forget_and_undo_print_json_a_script_can_read(tmp_path: Path) -> None:
    episode = _seed(tmp_path)

    updated, _ = _memory(
        tmp_path, "update", episode, "--user", "amy", "--content", "A dog: Biscuit."
    )
    assert updated["supersedes"] == episode and updated["held"] is True

    plan, _ = _memory(tmp_path, "forget", "--user", "amy", "--session", "s1")
    assert plan["dry_run"] is True and plan["forget_id"] == ""
    assert plan["episodes"] == [updated["id"]]

    done, _ = _memory(tmp_path, "forget", "--user", "amy", "--session", "s1", "--apply")
    assert done["forget_id"].startswith("fg_")

    undone, stderr = _memory(tmp_path, "undo", done["forget_id"], "--user", "amy")
    assert undone["applied"] is True
    assert _TZ_WARNING not in stderr

    history, _ = _memory(tmp_path, "history", episode, "--user", "amy")
    assert [(v["id"], v["state"]) for v in history["versions"]] == [
        (updated["id"], "current"),
        (episode, "superseded"),
    ]
