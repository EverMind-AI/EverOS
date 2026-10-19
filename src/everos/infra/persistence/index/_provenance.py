"""Startup check: does the configured embedding match this index?

Two things can silently ruin a memory store, and neither shows up until a
search comes back empty:

- the model returns **fewer** dimensions than the vector column is built for,
  so nothing can ever be stored (issue #434); or
- the model is **not the one** that produced the stored vectors, so distances
  between them mean nothing.

Both are compared here, both are reported together, and either one stops
startup. The third member of that family -- the column being the wrong width
for the current configuration -- is already caught by each backend's own
schema verification, which runs just before this.

The backend is passed in rather than looked up: the package ``__init__``
re-exports this module, so reaching back for its ``_backend()`` would be a
cycle. It also lets a test hand over a stub without patching globals.
"""

from __future__ import annotations

from typing import Protocol

from everos.component.embedding import (
    build_embedding_provider,
    get_embedding_capability,
)
from everos.config import load_settings
from everos.core.errors import EmbeddingProvenanceError
from everos.core.observability.logging import get_logger

logger = get_logger(__name__)

PROBE_TEXT = "everos embedding dimension probe"
"""Text sent to measure the model's output width.

Changing it changes what every start costs: one embed call against whatever
the provider bills for.
"""

_PROBE_DIM = 1 << 20
"""Width asked of the probe provider, far beyond any real model.

``OpenAIEmbeddingProvider`` slices its result to ``dim`` before returning, so
asking for the configured width would make a too-long vector impossible to
observe -- the slice would hand back exactly ``dim`` floats either way. A
huge value makes the slice a no-op and the real length visible.
"""

_PROBE_TIMEOUT_SECONDS = 5.0
"""Budget for the probe, separate from the write path's.

The write path can afford to wait: its 30 seconds times three retries buy a
batch of vectors. The probe buys one integer, and it sits in front of every
`everos` command -- inheriting that budget turns a blackholed endpoint into a
two-minute hang on `cascade status`.
"""

_READ_FAILED = object()
"""Sentinel: the attestation could not be read, as opposed to being absent.

Absent means "not attested yet" and is answered by writing the current
configuration. Unreadable means the old value is still out there, unseen, so
writing would bury it and every later model swap would go unnoticed.
"""

_last_probe: int | None = None


class ProvenanceStore(Protocol):
    """The two backend methods this module needs."""

    async def read_embedding_model(self) -> str | None: ...

    async def write_embedding_model(self, model: str) -> None: ...


def last_probe() -> int | None:
    """Width the probe measured during this process's startup.

    ``None`` when the probe did not run or could not reach the service. This
    is a snapshot, not a liveness check: a service that recovers after startup
    does not change it, because re-probing on every read would put a network
    call behind an endpoint meant for high-frequency polling.
    """
    return _last_probe


async def probe_embedding_dimension() -> int | None:
    """Measure the model's real output width, or ``None`` if it cannot be asked.

    Failure is deliberately not fatal: an unreachable embedding service is an
    operational condition the daemon already tolerates, and turning it into a
    refusal to start would make a transient outage look like a misconfigured
    index.
    """
    settings = load_settings().embedding.model_copy(
        update={"timeout_seconds": _PROBE_TIMEOUT_SECONDS, "max_retries": 0}
    )
    try:
        provider = build_embedding_provider(settings, dim=_PROBE_DIM)
        return len(await provider.embed(PROBE_TEXT))
    except Exception as exc:  # ponytail: narrowing means enumerating two SDKs' errors
        logger.warning("embedding_probe_failed", error=repr(exc))
        return None


async def verify_embedding_provenance(
    store: ProvenanceStore, *, may_attest: bool = True
) -> list[str]:
    """Compare configuration against what the model and the index actually say.

    Collects every mismatch before reporting, so an operator who got two
    things wrong sees both at once instead of fixing one, restarting, and
    being told about the next.

    Args:
        store: the backend that keeps this index's attestation.
        may_attest: whether an unattested index may be written to. Only the
            phases that hold the exclusive lock pass ``True``: a read-only
            command writing the index would break the invariant ``_runtime``
            documents (every command that writes passes ``exclusive=True``),
            and on a shared Milvus cluster it would be a write issued from a
            read-only phase. A read-only phase that finds no attestation says
            so and moves on.

    Returns:
        The findings that did **not** stop startup, in the order they were
        logged. Returned as well as logged because what counts as "warned"
        is this function's own answer: whether a structlog event is
        interceptable depends on how logging happens to be configured, so a
        caller (or a test) asserting through the log is reading the wrong
        layer.

    Raises:
        EmbeddingProvenanceError: on a too-narrow model or a changed model.
    """
    global _last_probe
    _last_probe = None
    settings = load_settings().embedding

    # No provider can be built -> keyword-only deployment. The condition is
    # the factory's own, not `model` alone: default.toml ships a model and a
    # base_url with an empty api_key, so gating on the model name would attest
    # a model that never produced a single vector -- and then refuse to start
    # once a real one was configured.
    if not get_embedding_capability().available:
        logger.info("embedding_provenance_skipped", reason="embedding_not_configured")
        return []

    errors: list[str] = []
    warnings: list[str] = []

    length = await probe_embedding_dimension()
    _last_probe = length
    if length is not None and length < settings.dim:
        errors.append(
            f"embedding model {settings.model!r} produces {length} dimensions, "
            f"configured dim is {settings.dim}.\n"
            f"    1. set EVEROS_EMBEDDING__DIM={length} "
            f"(or [embedding] dim = {length})\n"
            f"    2. then run: everos cascade rebuild   "
            f"(tables were already created at {settings.dim})"
        )
    elif length is not None and length > settings.dim:
        # This fires on the shipped default (a 2560-dim Matryoshka model
        # against dim=1024), so the text has to say that the common case is
        # intended -- a warning every start that reads like a defect teaches
        # operators to ignore warnings.
        warnings.append(
            f"embedding model {settings.model!r} returns {length} dimensions but "
            f"the vector column is {settings.dim} wide, so the last "
            f"{length - settings.dim} are truncated on every write. For a model "
            f"trained for it (Matryoshka -- the shipped default is one) that is "
            f"the intended usage, and cosine distance needs no renormalising; "
            f"with any other model recall degrades. To stop truncating, set "
            f"[embedding] dimensions = {settings.dim} and let the server shorten "
            f"it, or dim = {length} then run `everos cascade rebuild`."
        )
    if settings.dimensions is not None and settings.dimensions != settings.dim:
        warnings.append(
            f"[embedding] dimensions={settings.dimensions} differs from "
            f"dim={settings.dim}: tables are built at dim, while the server "
            f"truncates its output to dimensions."
        )

    try:
        recorded: object = await store.read_embedding_model()
    except Exception as exc:  # ponytail: see probe_embedding_dimension
        warnings.append(
            f"could not read this index's embedding provenance ({exc!r}); "
            f"leaving the recorded model untouched."
        )
        recorded = _READ_FAILED

    if (
        recorded is not _READ_FAILED
        and recorded is not None
        and recorded != settings.model
    ):
        errors.append(
            f"model: index built with {recorded!r}, config uses "
            f"{settings.model!r}\n"
            f"    stored vectors are not comparable across models\n"
            f"    Recover with: everos cascade rebuild   (stop the server first)"
        )

    for detail in warnings:
        logger.warning("embedding_provenance_warning", detail=detail)
    if errors:
        # Attesting is deliberately *after* this. A refused start that wrote
        # anyway would stamp the rejected configuration onto an index some
        # other model built, and the operator who then put the real model
        # back would be told their index was built by the one that never
        # wrote a vector -- and sent to rebuild for nothing.
        raise EmbeddingProvenanceError(
            "embedding configuration does not match this index:\n"
            + "\n".join(f"  - {error}" for error in errors)
        )

    if recorded is None:
        if not may_attest:
            finding = (
                "this index has no recorded embedding model; it will be "
                "recorded by the next command that holds the index lock."
            )
            # Logged as well as returned: every caller today discards the
            # return value, so returning it alone made this finding
            # unreachable -- and a read-only `cascade status` right after an
            # upgrade is exactly when someone would want to see it.
            logger.info("embedding_provenance_unrecorded", hint=finding)
            return [*warnings, finding]
        try:
            await store.write_embedding_model(settings.model or "")
        except Exception as exc:  # ponytail: see probe_embedding_dimension
            # A read-only mount, or a Milvus role with describe but not
            # alter. Not being able to write down which model built an index
            # is not a reason to refuse to open it.
            logger.warning("embedding_provenance_write_failed", error=repr(exc))
            return [*warnings, f"could not record the embedding model ({exc!r})"]
        logger.info(
            "embedding_provenance_recorded",
            model=settings.model,
            hint="if this index was built by a different model, run "
            "`everos cascade rebuild`",
        )
    return warnings
