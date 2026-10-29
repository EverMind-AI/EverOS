"""The terminal half of ``everos cascade backfill``.

:meth:`OpsService.backfill <everos.service.OpsService.backfill>` runs the
phases and decides when a run stops; everything the user sees lives here:
:class:`TyperPresenter` (the copy, colours and the confirmation prompt) and
:func:`run_backfill`, which maps how the run ended to an exit code:

- ``0`` — clean success.
- ``1`` — user declined a confirmation prompt.
- ``2`` — unexpected error mid-phase, or preflight found the required
  provider not configured.
- ``3`` — Phase 2/3 short-circuited on the OME jobstore lock (another
  ``everos`` process is running).
- ``4`` — every phase ran but at least one row failed embedding even
  after per-row fallback; automation can requeue.
- ``130`` — Ctrl-C (SIGINT convention: 128 + 2), including Ctrl-C at
  the ``[y/N]`` confirmation prompt (typer/click surfaces that as
  :class:`click.exceptions.Abort`, a :class:`RuntimeError` subclass).
"""

from __future__ import annotations

import asyncio
from typing import NamedTuple

import anyio.to_thread
import click
import typer

from everos.core.errors import ProviderNotConfiguredError
from everos.core.observability.logging import get_logger
from everos.service import (
    BackfillOutcome,
    BackfillSummary,
    OpsService,
    format_tokens,
)

logger = get_logger(__name__)


class PhaseCopy(NamedTuple):
    """The header shown before one phase runs."""

    title: str
    detail: str


PHASE_COPY: dict[str, PhaseCopy] = {
    "vectors": PhaseCopy(
        title="Phase 1 — make older memories searchable by meaning",
        detail=(
            "Memories created before you configured embedding can only be "
            "found by keyword right now. This step lets them be found by "
            "meaning too."
        ),
    ),
    "clusters": PhaseCopy(
        title="Phase 2 — group related memories into topics",
        detail=(
            "Related memories get clustered together so reflection and "
            "advanced retrieval can operate on them."
        ),
    ),
    "skills": PhaseCopy(
        title="Phase 3 — build the agent skill library",
        detail=(
            "Distill reusable skills from past agent conversations. Uses "
            "your LLM — the most expensive step."
        ),
    ),
}

_EXIT_CODES: dict[BackfillOutcome, int] = {
    BackfillOutcome.COMPLETED: 0,
    BackfillOutcome.ABORTED: 1,
    BackfillOutcome.BLOCKED_BY_CAPABILITY: 2,
    BackfillOutcome.BLOCKED_BY_SERVER: 3,
    BackfillOutcome.COMPLETED_WITH_FAILURES: 4,
}


_EXIT_LABELS: dict[int, str] = {
    0: "SUCCESS",
    1: "ABORTED",
    2: "FAILED",
    3: "SERVER_RUNNING",
    4: "COMPLETED_WITH_FAILURES",
    130: "INTERRUPTED",
}
"""Maps :func:`run_backfill`'s exit codes to the summary block's ``Exit:``
label. ``3`` marks a Phase 2/3 short-circuit because another process
holds the OME jobstore lock (typically ``everos server start``). ``4``
marks a partial-success run: every phase ran to completion, but at
least one row failed embedding even after the per-row fallback —
operators consult the ``cascade_backfill_row_embed_failed`` /
``..._row_subject_embed_failed`` log events for the failed row ids.
``130`` is the SIGINT convention (128 + signal number 2)."""


def _print_aborted() -> None:
    typer.secho("Aborted by user.", fg=typer.colors.RED)


def _print_interrupted() -> None:
    """Print the Ctrl-C resume hint (see :func:`run_backfill`)."""
    typer.secho(
        "Interrupted — partial progress was written. Resume by running:",
        fg=typer.colors.YELLOW,
    )
    typer.echo("  everos cascade backfill --phase <phase-name> --yes")
    typer.echo("where <phase-name> is one of: vectors, clusters, skills, all.")


def _print_capability_missing_hint(message: str) -> None:
    """Emit the friendly toml-hint message shown when a phase's required
    provider is not configured.

    The formatted :class:`ProviderNotConfiguredError` string is built by
    the memory-layer phase runner and passed through the presenter — so
    the CLI and API stay in lock-step on the remediation copy without
    duplicating it here.
    """
    typer.secho(f"  {message}", fg=typer.colors.RED)


def _print_server_running_hint() -> None:
    """Emit the friendly error shown when preflight or ``engine.start()``
    detects another process holding the OME jobstore lock.

    Copy points the user at ``everos server start`` (the usual holder)
    and Phase 1 (``--phase=vectors``), which never touches OME and so
    stays fully concurrent with a running server — never at
    ``EVEROS_*`` env vars.
    """
    typer.secho(
        "  Backfill Phase 2/3 needs exclusive access to the OME jobstore,",
        fg=typer.colors.RED,
    )
    typer.secho(
        "  but another everos process is holding it (typically your running",
        fg=typer.colors.RED,
    )
    typer.secho(
        "  `everos server start`).",
        fg=typer.colors.RED,
    )
    typer.echo("")
    typer.echo("  Stop your everos server first, then re-run backfill.")
    typer.echo("  (Phase 1 --phase=vectors already runs concurrently with the server;")
    typer.echo("  only Phase 2/3 need offline mode. See spec §10.)")


def _print_vectors_estimate(total_rows: int, total_tokens: int) -> None:
    typer.echo(f"  memories to process:  {total_rows:,}")
    typer.echo(f"  input tokens:         {format_tokens(total_tokens)}\n")
    typer.echo(
        "  Uses your embedding provider — cost depends on its per-token pricing.\n"
    )


def _print_clusters_estimate(episode_count: int, case_count: int) -> None:
    typer.echo(f"  episodes to cluster:     {episode_count:,}")
    typer.echo(f"  agent cases to cluster:  {case_count:,}\n")
    typer.echo(
        "  Uses your embedding provider (and LLM for agent-case merges) — "
        "cost depends on provider pricing.\n"
    )


def _print_skills_estimate(cases: int, clusters: int) -> None:
    typer.echo(f"  clusters to extract skills from:  {clusters:,}")
    typer.echo(f"  agent cases to replay:            {cases:,}\n")
    typer.echo("  Uses your LLM — cost depends on its per-token pricing.\n")


def _print_summary(summary: BackfillSummary, *, exit_code: int) -> None:
    """Print the consolidated post-run summary block.

    Field labels are chosen from what each phase's result dataclass
    actually tracks — e.g. Phase 3 reports "agent cases processed"
    (``events_emitted``) rather than "clusters processed", since the
    distinct cluster count isn't retained on
    :class:`~everos.service.SkillsPhaseResult`.

    On ``exit_code == 4`` (COMPLETED_WITH_FAILURES) also emits a hint
    pointing operators at the per-row failure log events so they can
    recover the failed row ids.
    """
    typer.echo("")
    typer.secho("Backfill summary", bold=True)
    typer.echo("-" * 44)
    if summary.vectors is not None:
        r = summary.vectors
        typer.echo(
            f"  Phase 1 (vectors)   — {r.rows_processed:,} rows / "
            f"{r.rows_failed:,} failed / {format_tokens(r.tokens_embedded)}"
        )
    if summary.clusters is not None:
        c = summary.clusters
        created = c.clusters_after - c.clusters_before
        typer.echo(
            f"  Phase 2 (clusters)  — {c.events_emitted:,} events emitted / "
            f"{created:,} clusters created"
        )
    if summary.skills is not None:
        s = summary.skills
        extracted = s.skills_after - s.skills_before
        typer.echo(
            f"  Phase 3 (skills)    — {s.events_emitted:,} agent cases "
            f"processed / {extracted:,} skills extracted"
        )
    typer.echo("-" * 44)
    label = _EXIT_LABELS.get(exit_code, "UNKNOWN")
    typer.echo(f"  Exit: {label}  ({exit_code})")
    if exit_code == 4:
        typer.secho(
            f"  {summary.rows_failed:,} rows failed embedding — see log events "
            "`cascade_backfill_row_embed_failed` for row_ids and text prefixes.",
            fg=typer.colors.YELLOW,
        )


class TyperPresenter:
    """The typer-backed :class:`~everos.service.BackfillPresenter`.

    Remembers the phase in progress so an interrupt can be logged against it.
    """

    def __init__(self) -> None:
        self.current_phase: str | None = None

    def phase_started(self, phase: str) -> None:
        self.current_phase = phase
        copy = PHASE_COPY[phase]
        typer.secho(copy.title, bold=True)
        typer.echo(f"  {copy.detail}")
        typer.echo("")

    def nothing_to_backfill(self, message: str, *, scan_failed: bool = False) -> None:
        colour = typer.colors.YELLOW if scan_failed else typer.colors.GREEN
        typer.secho(message, fg=colour)

    def capability_missing(self, *, provider: str, feature: str, message: str) -> None:
        _print_capability_missing_hint(message)

    def server_running(self) -> None:
        _print_server_running_hint()

    def estimate_vectors(self, rows: int, tokens: int) -> None:
        _print_vectors_estimate(rows, tokens)

    def estimate_clusters(self, episodes: int, cases: int) -> None:
        _print_clusters_estimate(episodes, cases)

    def estimate_skills(self, cases: int, clusters: int) -> None:
        _print_skills_estimate(cases, clusters)

    async def confirm(self, prompt: str, *, auto_yes: bool) -> bool:
        """Ask before a phase spends tokens; ``--yes`` skips the prompt.

        ``typer.confirm`` blocks on stdin, so it runs in a thread; Ctrl-C at
        the prompt surfaces as an abort that :func:`run_backfill` maps to 130.
        """
        if auto_yes:
            return True
        return await anyio.to_thread.run_sync(
            lambda: typer.confirm(f"Proceed with: {prompt}", default=False)
        )

    def emit_progress(self, done: int, total: int) -> None:
        typer.echo(f"  emitted {done:,} / {total:,} event(s)")

    def row_progress(self, table_name: str, done: int, total: int) -> None:
        typer.echo(f"  {table_name}: processed {done:,} / {total:,} rows")

    def phase_1_complete(self, rows_processed: int, rows_failed: int) -> None:
        typer.secho(
            f"phase 1 complete — {rows_processed:,} memories now "
            "searchable by meaning"
            + (f" ({rows_failed:,} failed)" if rows_failed else ""),
            fg=typer.colors.GREEN,
        )

    def phase_2_complete(self, before: int, after: int, emitted: int) -> None:
        typer.secho(
            f"phase 2 complete — clusters {before:,} -> {after:,} "
            f"({emitted:,} events processed)",
            fg=typer.colors.GREEN,
        )

    def phase_3_complete(
        self, skills_before: int, skills_after: int, emitted: int
    ) -> None:
        typer.secho(
            f"phase 3 complete — skills {skills_before:,} -> {skills_after:,} "
            f"({emitted:,} events processed)",
            fg=typer.colors.GREEN,
        )


async def run_backfill(ops: OpsService, *, phase: str, auto_yes: bool) -> int:
    """Run backfill through ``ops`` interactively; return the exit code.

    ``phase`` is ``"all"`` or a phase name. Every exit path ends with the
    summary of the phases that ran.

    Both ``KeyboardInterrupt`` and ``asyncio.CancelledError`` are an
    interrupt: on Python 3.11+ ``asyncio.run`` turns SIGINT into a
    cancellation of the main task, which raises ``CancelledError`` at the
    current ``await``. Catching it here lets the run end with 130 and the
    resume hint instead of leaking out as a bare ``Abort`` (exit 1). Nothing
    else in this call tree cancels the task. Ctrl-C (or EOF) at the y/N
    prompt arrives as typer's ``Abort`` -- since typer vendors click,
    ``typer.Abort`` and ``click.exceptions.Abort`` are distinct classes, and
    both are listed.
    """
    presenter = TyperPresenter()
    summary = BackfillSummary()
    try:
        outcome = await ops.backfill(
            phase, auto_yes=auto_yes, presenter=presenter, summary=summary
        )
    except (
        KeyboardInterrupt,
        asyncio.CancelledError,
        typer.Abort,
        click.exceptions.Abort,
    ):
        logger.warning(
            "cascade_backfill_interrupted", phase=presenter.current_phase or phase
        )
        _print_interrupted()
        _print_summary(summary, exit_code=130)
        return 130
    except ProviderNotConfiguredError as exc:
        # The phases preflight their providers and report a block; reaching
        # this means a path bypassed the preflight, and the remediation
        # message beats the generic "see logs" below.
        logger.error(
            "cascade_backfill_provider_error_escaped_preflight",
            phase=presenter.current_phase or phase,
            provider=exc.provider,
        )
        typer.secho(f"  {exc}", fg=typer.colors.RED)
        _print_summary(summary, exit_code=2)
        return 2
    except Exception:
        logger.exception("cascade_backfill_phase_failed", phase=phase)
        typer.secho("Backfill failed — see logs for details.", fg=typer.colors.RED)
        _print_summary(summary, exit_code=2)
        return 2
    code = _EXIT_CODES[outcome]
    if outcome is BackfillOutcome.ABORTED:
        _print_aborted()
    _print_summary(summary, exit_code=code)
    return code
