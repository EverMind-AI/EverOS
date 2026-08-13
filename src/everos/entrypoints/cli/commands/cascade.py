"""``everos cascade`` subcommand group.

One-shot operations on the cascade subsystem, all run in-process
without standing up the FastAPI app:

- ``cascade sync [PATH]`` — flush the work queue. With ``PATH`` the
  command first force-enqueues that single file (used after a manual
  md edit when waiting for the watcher is impractical), then drains.
- ``cascade status`` — print the queue + LSN summary that the daemon
  sees right now.
- ``cascade fix`` — list every ``failed`` row. With ``--apply``, also
  reset ``retryable=TRUE`` rows back to ``pending`` and drain the
  worker once so the retry actually runs before the command returns.
- ``cascade backfill`` — one-shot Tier 1 → Tier 2/3 migration: re-embed
  vectors, build clusters, extract skills. See
  :func:`everos.entrypoints.cli.commands._backfill_cmd.run_backfill`
  for the phase orchestration.
- ``cascade rebuild`` — drop every business LanceDB table and re-index
  all md from scratch. Recovery for a drifted / corrupt index; safe
  because md is the source of truth and un-extracted buffered messages
  are preserved. Skips the schema-verify guard (which the drift would
  otherwise trip on startup).

CLI is in-process (12 doc §7.1 + 16 doc §9.2): it constructs the same
:class:`CascadeOrchestrator` as the daemon but only calls
``sync_once`` / ``drain_once`` / ``queue_summary``. No watcher /
scanner background task is started.
"""

from __future__ import annotations

import asyncio
import enum
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import typer
from sqlmodel import SQLModel

from everos.component.embedding import get_embedding_capability
from everos.component.tokenizer import build_tokenizer
from everos.component.utils.datetime import to_display_tz
from everos.core.observability.logging import get_logger
from everos.core.persistence import MemoryRoot
from everos.entrypoints.cli._log_setup import configure_cli_logging
from everos.entrypoints.cli.commands._backfill_cmd import run_backfill
from everos.infra.persistence.lancedb import (
    dispose_connection,
    drop_business_tables,
    ensure_business_indexes,
    get_connection,
    verify_business_schemas,
)
from everos.infra.persistence.sqlite import (
    dispose_engine,
    get_engine,
    md_change_state_repo,
)
from everos.memory.cascade import (
    CascadeOrchestrator,
    match_kind,
    ome_lock_is_free,
)

logger = get_logger(__name__)

app = typer.Typer(
    name="cascade",
    help="Inspect and operate the md → LanceDB sync queue",
    no_args_is_help=True,
)


@app.callback()
def _cascade_callback(
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
    """Set memory root and log level before any cascade subcommand runs."""
    configure_cli_logging(verbose=verbose)
    _apply_root_env(root)


def _apply_root_env(root: str | None) -> None:
    """Set ``EVEROS_ROOT`` env var if a ``--root`` path was passed.

    Called from both the group callback (subcommand-independent path)
    and from each subcommand (subcommand-level ``--root``). Later calls
    override earlier ones — a subcommand's ``--root`` wins over the
    group callback's when both are supplied.
    """
    if root:
        os.environ["EVEROS_ROOT"] = root


def _apply_verbose_logging(verbose: bool | None) -> None:
    """Re-apply the CLI log level from a subcommand-level ``--verbose``.

    ``None`` means the subcommand did not receive its own ``--verbose``
    flag — leave whatever level the group callback set (default
    WARNING, or INFO if the user passed ``cascade --verbose``). A
    ``True``/``False`` value came from the subcommand and wins over the
    group callback (later configuration overrides earlier).

    Mirrors :func:`_apply_root_env`'s "either position works" pattern
    for symmetry — users can write ``everos cascade --verbose status``
    or ``everos cascade status --verbose`` and get the same behaviour.
    """
    if verbose is not None:
        configure_cli_logging(verbose=verbose)


_ROOT_OPTION_HELP = (
    "Memory root directory (alias for `cascade --root`; either position works)."
)

_VERBOSE_OPTION_HELP = (
    "Emit INFO-level lifecycle logs (alias for `cascade --verbose`; "
    "either position works)."
)


# ── shared runtime context ───────────────────────────────────────────────


@asynccontextmanager
async def _runtime(  # type: ignore[no-untyped-def]
    *, verify: bool = True, ensure: bool = True
):
    """Stand up sqlite + lancedb the same way the API lifespan would.

    The CLI uses the same lazy, process-wide singletons the API lifespan
    does. They are **per-process**: a running daemon has its own
    connection and table-handle cache, so read/write traffic interleaves
    safely, but a change to the table *set* made here (drop / recreate)
    is invisible to the daemon's cached handles — which is why
    ``rebuild`` refuses to run while a server holds the OME lock.

    ``verify=False`` skips :func:`verify_business_schemas` — required by
    ``cascade rebuild``, whose whole purpose is to recover from a table
    whose schema *has* drifted; running the guard there would abort
    startup before the rebuild could fix it (chicken-and-egg).

    ``ensure=False`` additionally skips :func:`ensure_business_indexes`.
    That call runs the schema / FTS migrations against the **existing**
    tables, and on the corruption classes rebuild exists to repair (a
    missing column, an un-alterable type) it raises before the drop can
    happen — the recovery path dying on the damage it was invoked to fix.
    Rebuild recreates the tables and their indexes itself after dropping,
    so skipping the pre-drop pass loses nothing.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    await get_connection()
    if verify:
        await verify_business_schemas()
    if ensure:
        await ensure_business_indexes()
    try:
        yield
    finally:
        await dispose_connection()
        await dispose_engine()


def _build_orchestrator() -> CascadeOrchestrator:
    memory_root = MemoryRoot.resolve()
    memory_root.ensure()
    tokenizer = build_tokenizer()

    capability = get_embedding_capability()
    if capability.available:
        logger.info("cli_cascade_embed_available")
    else:
        logger.info(
            "cli_cascade_embed_unavailable",
            reason="embedding not configured; keyword-only mode",
        )

    return CascadeOrchestrator(
        memory_root=memory_root,
        tokenizer=tokenizer,
    )


# ── sync ─────────────────────────────────────────────────────────────────


@app.command("sync")
def sync(
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
) -> None:
    """Drain the cascade queue (and optionally re-enqueue a path first)."""
    _apply_root_env(root)
    _apply_verbose_logging(verbose)

    async def _run() -> None:
        async with _runtime():
            orchestrator = _build_orchestrator()
            if path is not None:
                rel = _resolve_relative(path)
                spec = match_kind(rel)
                if spec is None:
                    typer.echo(
                        f"error: path does not match any registered cascade "
                        f"kind: {rel}",
                        err=True,
                    )
                    raise typer.Exit(code=1)
                await md_change_state_repo.force_enqueue(rel, spec.name)
                typer.echo(f"force-enqueued {rel} (kind={spec.name})")
            processed = await orchestrator.sync_once()
            typer.echo(f"sync complete — processed {processed} row(s)")

    asyncio.run(_run())


# ── status ───────────────────────────────────────────────────────────────


@app.command("status")
def status(
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
    _apply_root_env(root)
    _apply_verbose_logging(verbose)

    async def _run() -> None:
        async with _runtime():
            summary = await md_change_state_repo.queue_summary()
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

    asyncio.run(_run())


# ── fix ──────────────────────────────────────────────────────────────────


@app.command("fix")
def fix(
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
    _apply_root_env(root)
    _apply_verbose_logging(verbose)

    async def _run() -> None:
        async with _runtime():
            rows = await md_change_state_repo.list_failed()
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

            moved = await md_change_state_repo.reset_retryable_to_pending()
            typer.echo(f"re-enqueued {moved} retryable row(s)")
            if moved:
                orchestrator = _build_orchestrator()
                processed = await orchestrator.drain_once()
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
    """CLI-facing ``--phase`` choices — mirrors ``BackfillPhase.slug``."""

    VECTORS = "vectors"
    CLUSTERS = "clusters"
    SKILLS = "skills"
    ALL = "all"


@app.command("backfill")
def backfill(
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
    _apply_root_env(root)
    _apply_verbose_logging(verbose)

    async def _run() -> int:
        async with _runtime():
            return await run_backfill(phase=phase.value, auto_yes=yes)

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
    if not ome_lock_is_free():
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
        async with _runtime(verify=False, ensure=False):
            # Reset the queue FIRST so every crash window converges on
            # "queue pending → next scan re-indexes". Doing it after the
            # drop leaves a window where a crash yields empty tables with
            # a fully-`done` queue: nothing re-indexes, the schema guard
            # passes, and the deployment comes up silently empty — the
            # exact state this command exists to avoid.
            cleared = await md_change_state_repo.reset_all()
            typer.echo(f"reset {cleared} cascade queue row(s)")
            dropped = await drop_business_tables()
            typer.echo(
                f"dropped {len(dropped)} LanceDB table(s): "
                f"{', '.join(dropped) or '(none)'}"
            )
            # Recreate the tables (current schema) + FTS indexes.
            await ensure_business_indexes()
            # Re-scan + drain: re-embed and re-insert every md entry.
            orchestrator = _build_orchestrator()
            processed = await orchestrator.sync_once()
            typer.echo(f"rebuild complete — re-indexed {processed} md file(s)")

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


def _resolve_relative(p: Path) -> str:
    """Translate an absolute / relative path arg into the memory-root rel form.

    The state table stores paths relative to memory root, so the CLI
    must match that convention before calling :meth:`force_enqueue`.
    Outside-the-root inputs surface as an error in the caller.
    """
    memory_root = MemoryRoot.resolve()
    absolute = p.expanduser().resolve()
    try:
        rel = absolute.relative_to(memory_root.root)
    except ValueError as exc:
        raise typer.BadParameter(
            f"path {p!s} is not under memory root {memory_root.root!s}"
        ) from exc
    return rel.as_posix()


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
