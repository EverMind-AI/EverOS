"""``everos cascade`` subcommand group.

One-shot operations on the cascade subsystem, all run in-process
without standing up the FastAPI app:

- ``cascade sync [PATH]`` — flush the work queue. With ``PATH`` the
  command first force-enqueues that single file (used after a manual
  md edit with no server running), then drains.
- ``cascade status`` — print the queue + LSN summary that the daemon
  sees right now.
- ``cascade fix`` — list every ``failed`` row. With ``--apply``, also
  reset ``retryable=TRUE`` rows back to ``pending`` and drain the
  worker once so the retry actually runs before the command returns.
- ``cascade backfill`` — one-shot Tier 1 → Tier 2/3 migration: re-embed
  vectors, build clusters, extract skills. See
  :func:`everos.entrypoints.cli.commands._backfill.run_backfill`
  for the phase orchestration.
- ``cascade rebuild`` — drop every business LanceDB table and re-index
  all md from scratch. Recovery for a drifted / corrupt index; safe
  because md is the source of truth and un-extracted buffered messages
  are preserved. Skips the schema-verify guard (which the drift would
  otherwise trip on startup).

CLI is in-process (12 doc §7.1 + 16 doc §9.2): each command builds the
same :class:`~everos.service.Application` as the server but never starts
it -- it opens storage for the one command through
:meth:`~everos.service.OpsService.index_session` and calls ``ops``
methods. No watcher / scanner background task is started.
"""

from __future__ import annotations

import asyncio
import enum
import os
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import typer

from everos.config import Settings, load_settings
from everos.core.errors import (
    EmbeddingProvenanceError,
    IndexSchemaMismatchError,
    LLMNotConfiguredError,
    MemoryRootLockedError,
)
from everos.core.observability.logging import get_logger
from everos.core.utils.datetime import to_display_tz
from everos.entrypoints.cli._process import configure_interactive
from everos.entrypoints.cli.commands._backfill import run_backfill
from everos.service import Application, OpsService, build_application

logger = get_logger(__name__)

app = typer.Typer(
    name="cascade",
    help="Inspect and operate the md → LanceDB sync queue",
    no_args_is_help=True,
)


@app.callback()
def _cascade_callback(
    ctx: typer.Context,
    root: str | None = typer.Option(
        None,
        "--root",
        help="Memory root directory (env: EVEROS_ROOT, default: ~/.everos)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Emit INFO-level lifecycle logs (default: WARNING only).",
    ),
) -> None:
    """Record the group-level memory root and log level for the subcommand."""
    _apply_root_env(root)
    ctx.ensure_object(dict)["verbose"] = verbose


def _apply_root_env(root: str | None) -> None:
    """Set ``EVEROS_ROOT`` env var if a ``--root`` path was passed.

    Called from both the group callback (subcommand-independent path)
    and from each subcommand (subcommand-level ``--root``). Later calls
    override earlier ones — a subcommand's ``--root`` wins over the
    group callback's when both are supplied.
    """
    if root:
        os.environ["EVEROS_ROOT"] = root


def _command_settings(
    ctx: typer.Context, *, root: str | None = None, verbose: bool | None = None
) -> Settings:
    """Settle the subcommand's root and log level; return its settings.

    A subcommand-level ``--root`` / ``--verbose`` wins over the group's
    (``None`` means the subcommand did not get its own ``--verbose``), so
    ``everos cascade --verbose status`` and ``everos cascade status
    --verbose`` behave the same. Settings are loaded only after the root is
    settled, since the root decides which ``everos.toml`` they read.
    """
    _apply_root_env(root)
    if verbose is None:
        verbose = bool(ctx.ensure_object(dict).get("verbose", False))
    return configure_interactive(verbose=verbose, settings=load_settings())


_ROOT_OPTION_HELP = (
    "Memory root directory (alias for `cascade --root`; either position works)."
)

_VERBOSE_OPTION_HELP = (
    "Emit INFO-level lifecycle logs (alias for `cascade --verbose`; "
    "either position works)."
)


# ── shared runtime context ───────────────────────────────────────────────


@asynccontextmanager
async def _runtime(
    settings: Settings,
    *,
    verify: bool = True,
    ensure: bool = True,
    exclusive: bool = False,
) -> AsyncIterator[OpsService]:
    """Build the application and open its storage for one command.

    The application is per-process: a running daemon has its own
    connection and table-handle cache, and reads its own index snapshot.
    Reads interleave safely; writes do not -- a second process upserting
    the same table cannot see what the daemon just committed (nor the
    other way round) and both insert the row, so every command that writes
    the index (``sync``, ``fix --apply``, ``rebuild``) passes
    ``exclusive=True`` and holds the OME lock for its whole run. Refused
    with exit code 3 while a server (or another exclusive CLI phase) holds
    it; a server starting meanwhile fails at its own lock instead.

    ``verify=False`` / ``ensure=False`` are for ``cascade rebuild``, which
    repairs exactly what verification rejects and what the index
    migrations raise on (see :meth:`OpsService.index_session`).
    """
    try:
        async with (
            _build_application(settings) as application,
            application.ops.index_session(
                verify=verify, ensure=ensure, exclusive=exclusive
            ),
        ):
            yield application.ops
    except MemoryRootLockedError:
        typer.echo(
            "error: another process holds this memory root's OME lock — a "
            "running `everos server`\n"
            "  (or another exclusive CLI phase). Two processes writing the "
            "same index insert rows\n"
            "  twice, so this command needs the root to itself. A server "
            "projects markdown changes\n"
            "  on its own unless it was started with [cascade] enabled = false "
            "or quiesced;\n"
            "  stop it first, then re-run.",
            err=True,
        )
        raise typer.Exit(code=3) from None
    except (IndexSchemaMismatchError, EmbeddingProvenanceError) as exc:
        # The operator has one thing to go and do, so say what drifted and
        # stop. Without this the startup guards reach the user as a bare
        # traceback and exit 1 -- and 1 already means "no everos.toml here",
        # so the two would be indistinguishable in a script.
        typer.secho(str(exc), err=True)
        raise typer.Exit(code=3) from None


# ── sync ─────────────────────────────────────────────────────────────────


@app.command("sync")
def sync(
    ctx: typer.Context,
    path: Annotated[
        Path | None,
        typer.Argument(
            help="Optional md path to force-enqueue before draining. "
            "If omitted, only the existing queue is drained.",
        ),
    ] = None,
    root: Annotated[
        str | None,
        typer.Option("--root", help=_ROOT_OPTION_HELP),
    ] = None,
    verbose: Annotated[
        bool | None,
        typer.Option("--verbose", "-v", help=_VERBOSE_OPTION_HELP),
    ] = None,
    force_all: Annotated[
        bool,
        typer.Option(
            "--force-all",
            help="Force-enqueue every indexed md before draining; entries whose "
            "rows are present and unchanged are skipped by the handlers, so "
            "only lost or edited entries get re-embedded. The first full pass "
            "after upgrading from a release before 1.4.2 does this by itself.",
        ),
    ] = False,
) -> None:
    """Drain the cascade queue (and optionally re-enqueue a path first).

    Holds the OME lock for the run (see :func:`_runtime`): a second process
    writing the same LanceDB tables inserts rows twice — the server reads
    its own table snapshot and cannot see what the CLI process just
    committed, so both decide the row is new (4-5 % duplicate rows after a
    10-hour soak with two concurrent ``cascade sync`` processes). Refused
    with exit code 3 while a server holds the root.
    """
    settings = _command_settings(ctx, root=root, verbose=verbose)

    if force_all and path is not None:
        raise typer.BadParameter("--force-all re-enqueues every md; do not pass PATH")

    async def _run() -> None:
        async with _runtime(settings, exclusive=True) as ops:
            if force_all:
                rows = await ops.cascade_force_enqueue_all()
                by_status = Counter(row.status for row in rows)
                summary = ", ".join(f"{k}={v}" for k, v in sorted(by_status.items()))
                typer.echo(
                    f"force-enqueued {len(rows)} md file(s)"
                    + (f": {summary}" if summary else "")
                )
            if path is not None:
                rel = _resolve_relative(ops, path)
                kind = await ops.cascade_force_enqueue(rel)
                if kind is None:
                    typer.echo(
                        f"error: path does not match any registered cascade "
                        f"kind: {rel}",
                        err=True,
                    )
                    raise typer.Exit(code=1)
                typer.echo(f"force-enqueued {rel} (kind={kind})")
            processed = await ops.cascade_sync()
            typer.echo(f"sync complete — processed {processed} row(s)")

    asyncio.run(_run())


# ── status ───────────────────────────────────────────────────────────────


@app.command("status")
def status(
    ctx: typer.Context,
    root: Annotated[
        str | None,
        typer.Option("--root", help=_ROOT_OPTION_HELP),
    ] = None,
    verbose: Annotated[
        bool | None,
        typer.Option("--verbose", "-v", help=_VERBOSE_OPTION_HELP),
    ] = None,
) -> None:
    """Print the queue / LSN summary."""
    settings = _command_settings(ctx, root=root, verbose=verbose)

    async def _run() -> None:
        async with _runtime(settings) as ops:
            status = await ops.cascade_status()
            summary = status.summary
            lag = max(0, summary.max_lsn - summary.last_processed_lsn)
            typer.echo("queue:")
            typer.echo(f"  pending:                  {summary.pending}")
            typer.echo(f"  done:                     {summary.done}")
            typer.echo(
                f"  failed (retryable=TRUE):  {summary.failed_retryable}"
                + (
                    "     (eligible for `cascade fix --apply`)"
                    if summary.failed_retryable
                    else ""
                )
            )
            typer.echo(
                f"  failed (retryable=FALSE): {summary.failed_permanent}"
                + (
                    "     (fix md and re-save to recover)"
                    if summary.failed_permanent
                    else ""
                )
            )
            typer.echo("lsn:")
            typer.echo(f"  max:           {summary.max_lsn}")
            typer.echo(f"  last_processed: {summary.last_processed_lsn}")
            typer.echo(f"  lag:            {lag}")
            # Counted live here, unlike /health's startup snapshot: this is
            # a command someone runs *because* they are suspicious, and the
            # answer they came for is whether the index holds anything.
            rows = status.index_rows
            typer.echo("index rows:")
            for table, count in rows.items():
                typer.echo(f"  {table + ':':<24} {count}")
            # `summary.done` would count completed deletions, so a user who
            # removed their last markdown file would be warned that their
            # index is missing and sent to rebuild it from nothing.
            live = status.live_sources
            if live > 0 and summary.pending == 0 and not any(rows.values()):
                typer.secho(
                    f"\n  {live} source file(s) are indexed but every "
                    "index table is empty.\n"
                    "  Searches will return nothing. If the index directory was "
                    "removed, run `everos cascade rebuild`\n"
                    "  to re-index from markdown -- markdown is the source of "
                    "truth and is not touched.",
                    fg=typer.colors.YELLOW,
                )

    asyncio.run(_run())


# ── fix ──────────────────────────────────────────────────────────────────


@app.command("fix")
def fix(
    ctx: typer.Context,
    apply: Annotated[
        bool,
        typer.Option(
            "--apply",
            help="Re-enqueue every `retryable=TRUE` row and drain the worker.",
        ),
    ] = False,
    root: Annotated[
        str | None,
        typer.Option("--root", help=_ROOT_OPTION_HELP),
    ] = None,
    verbose: Annotated[
        bool | None,
        typer.Option("--verbose", "-v", help=_VERBOSE_OPTION_HELP),
    ] = None,
) -> None:
    """List failed rows (default) or re-enqueue retryable ones (``--apply``)."""
    settings = _command_settings(ctx, root=root, verbose=verbose)

    async def _run() -> None:
        async with _runtime(settings, exclusive=apply) as ops:
            rows = await ops.cascade_failed()
            if not rows:
                typer.echo("no failed rows")
                return

            if not apply:
                _print_failed_table(rows)
                retryable = sum(1 for r in rows if r.retryable)
                permanent = sum(1 for r in rows if not r.retryable)
                typer.echo("")
                if retryable:
                    typer.echo(
                        f"run `everos cascade fix --apply` to re-enqueue "
                        f"the {retryable} retryable row(s)."
                    )
                if permanent:
                    typer.echo(
                        f"the {permanent} retryable=FALSE row(s) require "
                        "editing the md and re-saving."
                    )
                return

            moved, processed = await ops.cascade_fix()
            typer.echo(f"re-enqueued {moved} retryable row(s)")
            if moved:
                typer.echo(f"[worker] processed {processed} row(s) on drain")
            permanent_rows = [r for r in rows if not r.retryable]
            if permanent_rows:
                typer.echo(
                    f"{len(permanent_rows)} retryable=FALSE row(s) left untouched:"
                )
                for r in permanent_rows:
                    typer.echo(f"  {r.md_path}")

    asyncio.run(_run())


# ── backfill ─────────────────────────────────────────────────────────────


class _BackfillPhaseOption(enum.StrEnum):
    """``--phase`` choices: ``all`` plus :data:`everos.service.BACKFILL_PHASES`."""

    VECTORS = "vectors"
    CLUSTERS = "clusters"
    SKILLS = "skills"
    ALL = "all"


@app.command("backfill")
def backfill(
    ctx: typer.Context,
    phase: Annotated[
        _BackfillPhaseOption,
        typer.Option(
            "--phase",
            help=(
                "Which phase to run — vectors (re-embed rows so they become "
                "searchable by meaning, not just keyword), clusters (group "
                "re-embedded episodes/agent cases into topics), skills "
                "(extract reusable agent skills from clustered cases), or "
                "all (run every phase in order). Default: all."
            ),
        ),
    ] = _BackfillPhaseOption.ALL,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help=(
                "Auto-confirm every phase's prompt instead of pausing for a "
                "y/n answer. Without this flag, each phase shows its "
                "token-count estimate and waits for confirmation first."
            ),
        ),
    ] = False,
    root: Annotated[
        str | None,
        typer.Option("--root", help=_ROOT_OPTION_HELP),
    ] = None,
    verbose: Annotated[
        bool | None,
        typer.Option("--verbose", "-v", help=_VERBOSE_OPTION_HELP),
    ] = None,
) -> None:
    """Backfill embeddings, clusters, and agent skills after upgrading from
    Tier 1 (LLM-only) to Tier 2/3 (embedding configured).

    Runs up to three phases in order (see ``--phase``), each showing a
    token-count estimate and confirming before doing any work. The
    estimate is an input-token count only (K/M shorthand) — never a
    price, since per-token pricing is provider-specific and can change.

    Ctrl-C is safe: interrupting mid-phase prints a resume hint and exits
    130. Every row and phase already written stays written, so
    re-running the same ``--phase`` (or ``--phase all``) afterwards only
    picks up what's left — nothing is redone from scratch.
    """
    settings = _command_settings(ctx, root=root, verbose=verbose)

    async def _run() -> int:
        async with _runtime(settings) as ops:
            return await run_backfill(ops, phase=phase.value, auto_yes=yes)

    try:
        code = asyncio.run(_run())
    except KeyboardInterrupt:
        # Safety net: run_backfill already catches CancelledError and
        # returns 130 cleanly in the common case, but asyncio.Runner can
        # still re-raise a bare KeyboardInterrupt past asyncio.run() on a
        # second Ctrl-C. Without this, typer/click would convert it to
        # Abort and exit 1 with no resume hint.
        typer.secho(
            "Interrupted — partial progress was written. Resume by running:",
            fg=typer.colors.YELLOW,
        )
        typer.echo("  everos cascade backfill --phase <phase-name> --yes")
        raise typer.Exit(code=130) from None
    raise typer.Exit(code=code)


# ── rebuild ────────────────────────────────────────────────────────────────


@app.command("rebuild")
def rebuild(
    ctx: typer.Context,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip the confirmation prompt."),
    ] = False,
) -> None:
    """Rebuild the LanceDB index from markdown (recover from schema drift).

    **Stop the ``everos server`` first** — this is the one cascade command
    that is not safe alongside a live daemon. It drops and recreates the
    tables, and the daemon's cached table handles would keep writing to
    the dropped dataset; the command refuses to start while a server holds
    the OME lock.

    Drops every business LanceDB table and re-indexes all md from
    scratch. Markdown is the source of truth, so no memory content is
    lost, and this is the safe recovery from a drifted / corrupt
    index (e.g. the ``verify_business_schemas`` startup failure):

    - unlike ``rm -rf ~/.everos/.index/lancedb``, it re-populates
      already-indexed entries (that command leaves the cascade queue
      marked ``done``, so nothing re-indexes and the index comes back
      empty);
    - unlike ``rm -rf ~/.everos/.index``, it preserves SQLite state that
      is NOT rebuildable from md — notably ``unprocessed_buffer``
      (messages received but not yet extracted).
    """
    settings = _command_settings(ctx)
    if not asyncio.run(_root_is_free(settings)):
        typer.echo(
            "error: a server (or another exclusive CLI phase) is running on "
            "this memory root.\n"
            "  cascade rebuild drops and recreates the LanceDB tables; a live "
            "daemon holds cached\n"
            "  table handles and would keep writing to the dropped dataset. "
            "Stop `everos server`\n"
            "  first, then re-run.",
            err=True,
        )
        raise typer.Exit(code=3)
    if not yes:
        typer.confirm(
            "Drop all LanceDB business tables and re-index from markdown? "
            "(requires the server to be stopped)",
            abort=True,
        )

    async def _run() -> None:
        # verify=False: the on-disk schema may be exactly what we're here
        # to fix; the startup guard would abort before we could rebuild.
        # ensure=False: the pre-drop migration pass would raise on exactly
        # the damage we are here to repair (see _runtime).
        async with _runtime(
            settings, verify=False, ensure=False, exclusive=True
        ) as ops:
            result = await ops.cascade_rebuild()
            typer.echo(f"reset {result.ledger_rows_reset} cascade queue row(s)")
            dropped = result.dropped_tables
            typer.echo(
                f"dropped {len(dropped)} LanceDB table(s): "
                f"{', '.join(dropped) or '(none)'}"
            )
            typer.echo(f"rebuild complete — re-indexed {result.processed} md file(s)")

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        typer.echo(
            "\ninterrupted — the cascade queue is reset, so re-running "
            "`everos cascade rebuild` (or starting the server) resumes the "
            "re-index from where it stopped.",
            err=True,
        )
        raise typer.Exit(code=130) from None


# ── helpers ──────────────────────────────────────────────────────────────


def _build_application(settings: Settings) -> Application:
    """Build the application, or exit 3 when ``[llm]`` is not configured.

    The CLI builds the same application as the server, and that refuses to
    assemble without an extraction LLM. The error already names the toml
    section to edit; a traceback would bury it.
    """
    try:
        return build_application(settings)
    except LLMNotConfiguredError as exc:
        typer.echo(
            f"error: {exc}\n"
            "  every cascade command builds the same application as the "
            "server, which needs an\n"
            "  extraction LLM; configure it, then re-run.",
            err=True,
        )
        raise typer.Exit(code=3) from None


async def _root_is_free(settings: Settings) -> bool:
    """Whether no server (or exclusive CLI phase) holds this memory root."""
    async with _build_application(settings) as application:
        return application.ops.ome_lock_is_free()


def _resolve_relative(ops: OpsService, p: Path) -> str:
    """Translate an absolute / relative path arg into the memory-root rel form.

    The state table stores paths relative to memory root, so the CLI
    must match that convention before force-enqueueing.
    Outside-the-root inputs surface as an error in the caller.
    """
    try:
        return ops.relative_to_root(p)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _print_failed_table(rows: list) -> None:  # type: ignore[type-arg]
    headers = ("md_path", "retryable", "retries", "last_attempt", "error")
    widths = [
        max(len(headers[0]), max(len(r.md_path) for r in rows)),
        len(headers[1]),
        len(headers[2]),
        len(headers[3]),
        max(len(headers[4]), max(len(r.error or "") for r in rows)),
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    typer.echo(f"{len(rows)} failed row(s):\n")
    typer.echo(fmt.format(*headers))
    for r in rows:
        typer.echo(
            fmt.format(
                r.md_path,
                "TRUE" if r.retryable else "FALSE",
                r.retry_count,
                to_display_tz(r.last_attempt_at).isoformat()
                if r.last_attempt_at
                else "",
                r.error or "",
            )
        )
