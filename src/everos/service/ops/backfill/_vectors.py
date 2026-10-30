"""Backfill phase ``vectors``: re-embed index rows written without a vector.

Rows written while no embedding provider was configured carry
``vector IS NULL`` (and an episode may lack only its ``subject_vector``).
This phase preflights, shows the estimate of what the injected
:class:`~everos.infra.persistence.index.VectorBackfill` found, asks once,
and lets it write each row's missing side(s) back; the row-level work and
its failure isolation live with the index.
"""

from __future__ import annotations

__all__ = ["run_vectors_phase"]

import anyio
import anyio.to_thread

from everos.core.errors import ProviderNotConfiguredError

from ._deps import (
    BackfillDeps,
    BackfillPresenter,
    VectorsPhaseResult,
    format_tokens,
)


async def run_vectors_phase(
    deps: BackfillDeps, *, auto_yes: bool, presenter: BackfillPresenter
) -> VectorsPhaseResult:
    """Re-embed rows with a NULL primary vector (and, for Episode, a
    NULL subject vector) across the vector-carrying index tables.

    Preflight capability → scan + tokenize → surface the estimate via
    ``presenter`` → confirm once → embed and write back in batches. Per-row
    failures (including scan-time extraction failures) are tallied and
    logged; the phase runs to completion regardless.

    Preflight runs FIRST — before scan + tokenize + confirm — so a
    user with a large accumulated backlog and no embedding configured
    exits fast with a toml-hint instead of paying an O(N) scan they
    can't act on. The empty-backlog path also preflights, so a fresh
    install still gets the hint rather than a bare "nothing to
    backfill" green message.
    """
    provider = deps.embedding
    if provider is None:
        err = ProviderNotConfiguredError(provider="embedding", feature="backfill")
        presenter.capability_missing(
            provider="embedding", feature="backfill", message=str(err)
        )
        return VectorsPhaseResult(blocked_by_capability="embedding")

    # Phase 2 / Phase 3 both preflight the OME lock; Phase 1 must do
    # the same. Otherwise ``--phase all`` starts against a running
    # server, burns through Phase 1's embed API calls (real token
    # cost + potential commit conflict with the live cascade worker),
    # then only halts at Phase 2's preflight with exit 3. Fail fast
    # here so the user's next action is "stop the server" — not
    # "reimburse this month's embedding bill".
    if not await anyio.to_thread.run_sync(deps.ome_lock.is_free):
        presenter.server_running()
        return VectorsPhaseResult(aborted=True, blocked_by_server=True)

    backlog = await deps.vectors.scan()
    if backlog.total_rows == 0:
        if backlog.scan_failed:
            presenter.nothing_to_backfill(
                f"Nothing to backfill — {backlog.scan_failed:,} row(s) could not "
                "be read and were skipped (see logs).",
                scan_failed=True,
            )
        else:
            presenter.nothing_to_backfill(
                "Nothing to backfill. All rows already have vectors."
            )
        return VectorsPhaseResult(rows_failed=backlog.scan_failed)

    presenter.estimate_vectors(backlog.total_rows, backlog.total_tokens)
    if not await presenter.confirm(
        f"proceed with {backlog.total_rows:,} rows / "
        f"~{format_tokens(backlog.total_tokens)} tokens",
        auto_yes=auto_yes,
    ):
        return VectorsPhaseResult(aborted=True)

    counts = await deps.vectors.run(
        backlog, provider, on_progress=presenter.row_progress
    )
    result = VectorsPhaseResult(
        rows_processed=counts.rows_processed,
        rows_failed=backlog.scan_failed + counts.rows_failed,
        tokens_embedded=counts.tokens_embedded,
    )
    presenter.phase_1_complete(result.rows_processed, result.rows_failed)
    return result
