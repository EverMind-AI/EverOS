"""CLI-side orchestration for ``everos cascade backfill``.

Presentation half of PR #361 review finding M11. The memory-layer phase
runners (``memory/cascade/_backfill.py``) are pure — every user-facing
string, colour, and confirmation prompt lives here, behind the
:class:`~everos.memory.cascade._backfill.BackfillPresenter` Protocol.
:class:`TyperPresenter` is the typer-backed implementation; tests can
substitute any structurally-compatible stub.

:func:`run_backfill` is the top-level entry point the ``cascade``
subcommand imports. It builds a :class:`TyperPresenter`, dispatches to
the three phase runners in order, catches Ctrl-C /
:class:`click.exceptions.Abort` / :class:`asyncio.CancelledError`, and
maps every terminal state to a CLI exit code:

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

import anyio.to_thread
import click
import typer

from everos.core.errors import ProviderNotConfiguredError
from everos.core.observability.logging import get_logger

# Module reference (not name-level imports of the phase runners): tests
# monkeypatch the runners on this module to inject Ctrl-C / boundary
# behaviour, and re-binding names on the source module only affects the
# entrypoints layer if we resolve them through the module attribute at
# call time.
from everos.memory.cascade import _backfill as _phase_runners
from everos.memory.cascade._backfill import (
    BackfillPhase,
    _BackfillSummary,
    _count_failed_rows,
    _format_tokens,
)

logger = get_logger(__name__)


PHASES: tuple[BackfillPhase, ...] = (
    BackfillPhase(
        number=1,
        slug="vectors",
        title="Phase 1 — make older memories searchable by meaning",
        detail=(
            "Memories created before you configured embedding can only be "
            "found by keyword right now. This step lets them be found by "
            "meaning too."
        ),
    ),
    BackfillPhase(
        number=2,
        slug="clusters",
        title="Phase 2 — group related memories into topics",
        detail=(
            "Related memories get clustered together so reflection and "
            "advanced retrieval can operate on them."
        ),
    ),
    BackfillPhase(
        number=3,
        slug="skills",
        title="Phase 3 — build the agent skill library",
        detail=(
            "Distill reusable skills from past agent conversations. Uses "
            "your LLM — the most expensive step."
        ),
    ),
)


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


def _confirm(detail: str, *, auto_yes: bool) -> bool:
    """Ask the user to confirm one phase; ``--yes`` skips the prompt.

    Kept as a module-level helper (rather than a bound method of
    :class:`TyperPresenter`) so tests can monkeypatch it on this module
    the same way the pre-M11 code was patched on ``_backfill``. The
    presenter's :meth:`~TyperPresenter.confirm` delegates here.
    """
    if auto_yes:
        return True
    return typer.confirm(f"Proceed with: {detail}", default=False)


def _print_phase_header(phase: BackfillPhase) -> None:
    typer.secho(phase.title, bold=True)
    typer.echo(f"  {phase.detail}")
    typer.echo("")


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
    typer.echo(f"  input tokens:         {_format_tokens(total_tokens)}\n")
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


def _print_summary(summary: _BackfillSummary, *, exit_code: int) -> None:
    """Print the consolidated post-run summary block.

    Field labels are chosen from what each phase's result dataclass
    actually tracks — e.g. Phase 3 reports "agent cases processed"
    (``events_emitted``) rather than "clusters processed", since the
    distinct cluster count isn't retained on
    :class:`~everos.memory.cascade._backfill._SkillPhaseResult`.

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
            f"{r.rows_failed:,} failed / {_format_tokens(r.tokens_embedded)}"
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
        total_failed = _count_failed_rows(summary)
        typer.secho(
            f"  {total_failed:,} rows failed embedding — see log events "
            "`cascade_backfill_row_embed_failed` for row_ids and text prefixes.",
            fg=typer.colors.YELLOW,
        )


class TyperPresenter:
    """typer-backed :class:`BackfillPresenter` — the CLI's user-I/O seam.

    Every method delegates to a module-level ``_print_*`` helper or to
    :func:`_confirm`. Kept as a small class (rather than a module of
    free functions) so alternative front-ends (e.g. a JSON-emitting
    CI-mode presenter) can slot in without touching the memory layer.
    """

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
        # ``typer.confirm`` is blocking (reads from stdin); offload so a
        # slow / interactive user doesn't hold the event loop. Any
        # ``click.exceptions.Abort`` raised on Ctrl-C at the prompt
        # propagates to :func:`run_backfill` which catches it and maps
        # to exit 130.
        return await anyio.to_thread.run_sync(
            lambda: _confirm(prompt, auto_yes=auto_yes)
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


async def run_backfill(*, phase: str, auto_yes: bool) -> int:
    """Run backfill phases interactively; return an exit code.

    ``phase`` is ``"all"`` or one of :data:`PHASES`'s ``slug`` values.
    Exit codes: see :data:`_EXIT_LABELS`.

    Every exit path — success, abort, error, or interrupt — ends with a
    consolidated summary block (:func:`_print_summary`) covering every
    phase that actually ran. The interrupt is only ever caught here, at
    the outer loop: a phase body mid-batch must let it propagate rather
    than swallow it, so a Ctrl-C during Phase 1 doesn't get silently
    absorbed as "phase 1 failed, try phase 2 anyway".

    Both ``KeyboardInterrupt`` and ``asyncio.CancelledError`` are caught:
    on Python 3.11+, ``asyncio.Runner`` (which backs ``asyncio.run``)
    translates a real SIGINT into ``main_task.cancel()``, which raises
    ``CancelledError`` — not ``KeyboardInterrupt`` — at the currently
    suspended ``await``. Handling it here lets the task complete
    normally with exit code 130 instead of leaking the cancellation out
    to ``asyncio.run`` (which would re-synthesize a bare
    ``KeyboardInterrupt`` that typer/click turns into ``Abort``, exit 1,
    with no resume hint printed). Catching ``CancelledError``
    unconditionally is safe here: nothing in this call tree cancels the
    task for reasons other than an interrupt.
    """
    presenter = TyperPresenter()
    selected = [p for p in PHASES if phase == "all" or p.slug == phase]
    summary = _BackfillSummary()
    current_phase: BackfillPhase | None = None
    try:
        for p in selected:
            current_phase = p
            _print_phase_header(p)
            if p.slug == "vectors":
                result = await _phase_runners._run_phase_vectors(
                    auto_yes=auto_yes, presenter=presenter
                )
                summary.vectors = result
                if result.blocked_by_capability:
                    logger.warning(
                        "cascade_backfill_blocked_by_capability",
                        phase=p.slug,
                        provider=result.blocked_by_capability,
                    )
                    _print_summary(summary, exit_code=2)
                    return 2
                if result.blocked_by_server:
                    # Fail fast BEFORE any embed API call — Phase 2 / 3
                    # already do this preflight; without it here,
                    # ``--phase all`` against a running server burns
                    # through Phase 1's real token budget and only
                    # halts at Phase 2 (see PR #361 review J10).
                    logger.warning("cascade_backfill_blocked_by_server", phase=p.slug)
                    _print_summary(summary, exit_code=3)
                    return 3
                if result.aborted:
                    # Policy (PR #361 round-3 review #11, accepted):
                    # typing ``n`` at Phase 1's confirmation aborts the
                    # ENTIRE ``--phase all`` run — not just Phase 1.
                    # Rationale: ``--phase all`` is opt-in-to-run-
                    # everything. Explicitly declining Phase 1's cost
                    # estimate is a decisive "no"; interpreting it as
                    # "skip Phase 1 but still run Phase 2/3" would spend
                    # LLM/embed budget on phases the user did not see
                    # estimates for. Users who want only Phase 2 or 3
                    # invoke ``--phase clusters`` or ``--phase skills``
                    # explicitly (and see that phase's own estimate).
                    logger.info("cascade_backfill_aborted", phase=p.slug)
                    _print_aborted()
                    _print_summary(summary, exit_code=1)
                    return 1
                continue
            if p.slug == "clusters":
                cluster_result = await _phase_runners._run_phase_clusters(
                    auto_yes=auto_yes, presenter=presenter
                )
                summary.clusters = cluster_result
                if cluster_result.blocked_by_capability:
                    logger.warning(
                        "cascade_backfill_blocked_by_capability",
                        phase=p.slug,
                        provider=cluster_result.blocked_by_capability,
                    )
                    _print_summary(summary, exit_code=2)
                    return 2
                if cluster_result.blocked_by_server:
                    logger.warning("cascade_backfill_blocked_by_server", phase=p.slug)
                    _print_summary(summary, exit_code=3)
                    return 3
                if cluster_result.aborted:
                    logger.info("cascade_backfill_aborted", phase=p.slug)
                    _print_aborted()
                    _print_summary(summary, exit_code=1)
                    return 1
                continue
            if p.slug == "skills":
                skill_result = await _phase_runners._run_phase_skills(
                    auto_yes=auto_yes, presenter=presenter
                )
                summary.skills = skill_result
                if skill_result.blocked_by_capability:
                    logger.warning(
                        "cascade_backfill_blocked_by_capability",
                        phase=p.slug,
                        provider=skill_result.blocked_by_capability,
                    )
                    _print_summary(summary, exit_code=2)
                    return 2
                if skill_result.blocked_by_server:
                    logger.warning("cascade_backfill_blocked_by_server", phase=p.slug)
                    _print_summary(summary, exit_code=3)
                    return 3
                if skill_result.aborted:
                    logger.info("cascade_backfill_aborted", phase=p.slug)
                    _print_aborted()
                    _print_summary(summary, exit_code=1)
                    return 1
                continue
    # ``typer.Abort`` fires when ``typer.confirm`` catches KeyboardInterrupt
    # or EOF at the y/N prompt and re-raises it as an ``Abort`` (a
    # ``RuntimeError`` subclass, NOT a ``KeyboardInterrupt``). Typer 0.15+
    # vendored click under ``typer._click`` — ``typer.Abort`` and the
    # standalone ``click.exceptions.Abort`` are DISTINCT classes. Catching
    # only ``click.exceptions.Abort`` would let the real typer-raised
    # abort fall through to the generic ``except Exception`` branch (exit
    # 2, rich traceback). Both are listed so the Ctrl-C-at-prompt path
    # (exit 130 with resume hint) fires regardless of which one arrives.
    except (
        KeyboardInterrupt,
        asyncio.CancelledError,
        typer.Abort,
        click.exceptions.Abort,
    ):
        logger.warning(
            "cascade_backfill_interrupted",
            phase=current_phase.slug if current_phase else phase,
        )
        _print_interrupted()
        _print_summary(summary, exit_code=130)
        return 130
    except ProviderNotConfiguredError as exc:
        # Defensive backstop: phase runners preflight capability and
        # return ``blocked_by_capability`` on their own. Reaching this
        # branch means a code path bypassed the preflight — surface the
        # remediation message instead of letting the broad ``except
        # Exception`` below swallow it into "see logs for details".
        logger.error(
            "cascade_backfill_provider_error_escaped_preflight",
            phase=current_phase.slug if current_phase else phase,
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
    total_failed = _count_failed_rows(summary)
    if total_failed > 0:
        logger.warning(
            "cascade_backfill_completed_with_failures",
            total_failed=total_failed,
        )
        _print_summary(summary, exit_code=4)
        return 4
    _print_summary(summary, exit_code=0)
    return 0
