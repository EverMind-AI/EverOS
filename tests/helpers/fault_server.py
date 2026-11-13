"""A real server process over the scripted model, for the crash tests.

    python -m tests.helpers.fault_server --port 18123 \
        [--pause-at POINT --marker PATH] [--cut-on-add]

Configured from the environment like any server (``EVEROS_ROOT``, ...); the
model and the embedder are the deterministic doubles of
:mod:`tests.helpers.faults`. ``POINT`` names a place the process stops at: it
writes ``PATH`` there and waits until it is killed (the test kills it with its
own process handle). The points:

* ``index_write`` -- the cascade is about to project an episode file into the
  index: the markdown is written and its ledger row claimed, the index is not;
* ``episode_write`` -- an extraction is about to write its episode: the
  memcell is committed and the buffer consumed, the markdown is not written;
* ``forget_log`` -- a forget has marked its entries and erased their rows,
  and has not logged itself yet;
* ``undo_requeue`` -- an undo has unmarked its entries in markdown, and has
  neither handed them back to the cascade nor marked the forget undone.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import uvicorn

from everos.config import load_settings
from everos.entrypoints.api.app import create_app
from everos.service import build_application

from .faults import FlakyEmbedder, Upstream, fault_providers


def _mark(marker: Path) -> None:
    marker.write_text("paused\n", encoding="utf-8")


def _pause(marker: Path) -> Callable[..., Awaitable[Any]]:
    async def stop_here(*_args: Any, **_kwargs: Any) -> Any:
        _mark(marker)
        await asyncio.Event().wait()  # until the test kills the process

    return stop_here


def _before(owner: Any, name: str, marker: Path, *, when: Any = None) -> None:
    """Make ``owner.name`` stop at ``marker`` before it runs (when ``when``)."""
    original = getattr(owner, name)
    stop = _pause(marker)

    @functools.wraps(original)
    async def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        if when is None or when(self, *args, **kwargs):
            await stop()
        return await original(self, *args, **kwargs)

    setattr(owner, name, wrapped)


def install_pause(point: str, marker: Path) -> None:
    """Install the pause ``point`` (see the module docstring)."""
    if point == "index_write":
        from everos.infra.cascade._projections._daily_log_base import (
            BaseDailyLogHandler,
        )

        _before(
            BaseDailyLogHandler,
            "handle_added_or_modified",
            marker,
            when=lambda _self, md_path, *a, **k: "/episodes/" in str(md_path),
        )
    elif point == "episode_write":
        from everos.infra.persistence.markdown import MarkdownEpisodeStore

        _before(MarkdownEpisodeStore, "append_many", marker)
    elif point == "forget_log":
        from everos.service.memory._forget_apply import ForgetWriter

        _before(ForgetWriter, "profiles", marker)
    elif point == "undo_requeue":
        from everos.service.memory._forget_undo import Undoer

        _before(Undoer, "_restore_clusters", marker)
    else:
        raise SystemExit(f"unknown pause point {point!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--pause-at")
    parser.add_argument("--marker", type=Path)
    parser.add_argument("--cut-on-add", action="store_true")
    args = parser.parse_args()
    # The token counter's warm-up fetches its encoding; answer it offline, as
    # the test suite does (tests/conftest.py).
    import everos.service._warmup as warmup

    warmup.encode = lambda _text: []  # type: ignore[assignment]
    if args.pause_at:
        install_pause(args.pause_at, args.marker)
    settings = load_settings()
    upstream = Upstream(cut_on_add=args.cut_on_add)
    application = build_application(
        settings, providers=fault_providers(upstream, FlakyEmbedder())
    )
    app = create_app(settings, application=application)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
