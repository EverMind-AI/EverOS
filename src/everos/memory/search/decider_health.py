"""What the server knows about its multi-round decider.

``GET /health`` lists the decider -- the model and endpoint every search asks --
and says whether it answered one minimal request when the server started. That
is the embedding probe's rule: a snapshot taken at startup, because an endpoint
that is polled must not call a model. A decider that stops answering after
startup is still visible without any extra call: every search counts its rounds
here, so ``failed_rounds`` grows and ``last_failure`` names the reason.

The probe sends what a decider round sends -- the same client, ``[decider]``
temperature and ``extra`` -- with a one-line prompt and ``max_tokens = 1``, so a
model name the endpoint does not serve, a wrong key or an unreachable host fails
it the way it would fail every round. Only a client with a raw call
(:class:`~everos.memory.search.llm_multiround.RawChatClient`, which the
production provider is) is probed; any other reads ``reachable = None``.
"""

from __future__ import annotations

__all__ = ["PROBE_TEXT", "DeciderHealth", "DeciderStatus"]

import asyncio
import dataclasses
import datetime as dt
from collections.abc import Callable, Sequence
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from everalgo.llm import LLMClient, LLMMessage

from everos.config.settings import DeciderSettings
from everos.core.observability.logging import get_logger
from everos.core.observability.metrics import Counter as MetricCounter
from everos.core.utils.datetime import get_utc_now

from .llm_multiround import DeciderFailure, RawChatClient

logger = get_logger(__name__)

PROBE_TEXT = "Reply with OK."
"""The probe's whole prompt: about ten input tokens, and one output token."""

_PROBE_TIMEOUT_SECONDS = 15.0
"""Budget of the startup probe; it runs beside startup, never in front of it."""

_ERROR_CHARS = 300

_rounds = MetricCounter(
    name="decider_rounds_total",
    description=(
        "Multi-round decider rounds by outcome: decided, fallback (every "
        "attempt failed) or salvaged (core read from a reply cut short)."
    ),
    labelnames=("outcome",),
    namespace="everos",
)


@dataclasses.dataclass(frozen=True)
class DeciderHealth:
    """The decider as ``GET /health`` reports it.

    Attributes:
        model: The model every round names (``[decider] model``, else
            ``[llm] model``).
        base_url: Its endpoint, without credentials or query.
        reachable: Whether it answered the startup probe; ``None`` while the
            probe has not finished, or for a client that cannot be probed.
        probe_error: Why the probe failed, when it did.
        probed_at: When the probe finished.
        rounds: Rounds decided or attempted since the server started.
        failed_rounds: Of those, the rounds that fell back or were salvaged.
        last_failure: The newest failed round's reason.
        last_failure_at: When it was recorded.
    """

    model: str
    base_url: str
    reachable: bool | None
    probe_error: str | None
    probed_at: dt.datetime | None
    rounds: int
    failed_rounds: int
    last_failure: str | None
    last_failure_at: dt.datetime | None


class DeciderStatus:
    """The startup probe's verdict and the round counts since.

    Args:
        llm: The decider's client -- the one every search round uses.
        model: The model it names.
        base_url: Its endpoint; anything but scheme, host, port and path is
            dropped before it is reported.
        tuning_source: ``[decider]`` as the rounds read it (temperature,
            extra).
        timeout: Seconds the probe may take.
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        model: str,
        base_url: str,
        tuning_source: Callable[[], DeciderSettings],
        timeout: float = _PROBE_TIMEOUT_SECONDS,
    ) -> None:
        self._llm = llm
        self._model = model
        self._base_url = _public_url(base_url)
        self._tuning_source = tuning_source
        self._timeout = timeout
        self._reachable: bool | None = None
        self._probe_error: str | None = None
        self._probed_at: dt.datetime | None = None
        self._rounds = 0
        self._failed_rounds = 0
        self._last_failure: str | None = None
        self._last_failure_at: dt.datetime | None = None

    async def probe(self) -> None:
        """Send one minimal request as a round would; never raises."""
        if not isinstance(self._llm, RawChatClient):
            logger.info("decider_probe_skipped", reason="client has no raw call")
            return
        tune = self._tuning_source()
        params: dict[str, Any] = {"temperature": tune.temperature, **tune.extra}
        params["max_tokens"] = 1
        try:
            await asyncio.wait_for(
                self._llm.achat_raw(
                    [LLMMessage(role="user", content=PROBE_TEXT)], **params
                ),
                timeout=self._timeout,
            )
        except Exception as err:  # every failure is the verdict, none is fatal
            reason = (
                f"no reply within {self._timeout:g}s"
                if isinstance(err, TimeoutError)
                else f"{type(err).__name__}: {err}"
            )
            self._settle(False, reason[:_ERROR_CHARS])
            logger.error(
                "decider_probe_failed",
                model=self._model,
                base_url=self._base_url,
                error=reason[:200],
            )
            return
        self._settle(True, None)
        logger.info("decider_probe_ok", model=self._model, base_url=self._base_url)

    def record(self, rounds: int, failures: Sequence[DeciderFailure]) -> None:
        """Count one search's rounds and the ones it did not decide."""
        self._rounds += rounds
        self._failed_rounds += len(failures)
        for failure in failures:
            _rounds.labels(outcome=failure.outcome).inc()
        if rounds > len(failures):
            _rounds.labels(outcome="decided").inc(rounds - len(failures))
        if failures:
            self._last_failure = failures[-1].reason
            self._last_failure_at = get_utc_now()

    def snapshot(self) -> DeciderHealth:
        """The current state, for ``GET /health``."""
        return DeciderHealth(
            model=self._model,
            base_url=self._base_url,
            reachable=self._reachable,
            probe_error=self._probe_error,
            probed_at=self._probed_at,
            rounds=self._rounds,
            failed_rounds=self._failed_rounds,
            last_failure=self._last_failure,
            last_failure_at=self._last_failure_at,
        )

    def _settle(self, reachable: bool, error: str | None) -> None:
        self._reachable = reachable
        self._probe_error = error
        self._probed_at = get_utc_now()


def _public_url(url: str) -> str:
    """``url`` with only scheme, host, port and path: no credentials, no query."""
    if not url:
        return ""
    parts = urlsplit(url)
    host = parts.hostname or ""
    netloc = f"{host}:{parts.port}" if parts.port else host
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))
