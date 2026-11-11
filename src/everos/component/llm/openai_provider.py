"""EverOS's implementation of the everalgo ``LLMClient`` contract.

:class:`OpenAIProvider` talks to any OpenAI-compatible Chat Completions
endpoint through ``openai.AsyncOpenAI``. It is the only ``LLMClient``
implementation in EverOS; everalgo's reference ``OpenAICompatClient`` is read,
never used.

``achat`` follows the recommended flow of ``llm-contract.md`` §7, one private
method per step:

1. ``_build_request`` -- build the request once; every attempt sends it as is,
   so no retry can raise ``max_tokens`` (§3.1). ``None`` falls back to the
   configured default and is otherwise left out; a call's ``max_tokens`` can
   lower one a deployment sets in ``extra``, never raise it. Messages go out
   via ``model_dump()``, multimodal parts included (§3.2).
2. ``_send`` -- one SDK call; transport retries are the SDK's ``max_retries``.
   ``asyncio.CancelledError`` is never caught (§3.3).
3. ``_check_upstream`` -- a flagged refusal, ``length`` / ``content_filter``
   or absent content raises (§3.4, §3.5); no choice and ``abort`` / ``error``
   fail only this attempt, and so does ``length`` on a call that passes
   ``retry_cut=True`` (its cap guards against a runaway, see ``achat``).
4. ``_validate`` -- ``model_validate_json`` against the caller's
   ``response_format``; a mismatch fails only this attempt (§7.1).
5. ``_request_valid_reply`` -- at most ``1 + invalid_reply_retries`` attempts
   of the same request, each completion reported to the observer; a retry is
   sampled at ``retry_temperature`` when that is above the first attempt's.

``achat_raw`` is the one call outside that contract: the same request, sent
once, with the reply handed back unchecked as a :class:`RawReply`. It exists
for a caller that reads the reply itself -- the multi-round retrieval decider,
whose parse is tolerant on purpose -- and must not be given to an everalgo
operator, which relies on ``achat``'s guarantees.
"""

from __future__ import annotations

__all__ = ["OpenAIProvider", "RawReply", "ResponseFormatMode"]

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final, Literal, NamedTuple

import httpx
import openai
from everalgo.llm import to_openai_response_format
from pydantic import BaseModel, ValidationError

from everos.core.observability.logging import get_logger
from everos.core.observability.metrics import Counter as MetricCounter

from .errors import LLMCallError
from .observer import LLMCallObserver, NullObserver, TokenUsage

if TYPE_CHECKING:
    from everalgo.llm import LLMMessage
    from openai.types.chat import ChatCompletion

logger = get_logger(__name__)

ResponseFormatMode = Literal["json_schema", "json_object", "prompt_only"]

# The identical request hits the same output cap or the same filter again.
_CUT_SHORT_REASONS: Final = frozenset({"length", "content_filter"})

# vLLM-style gateways report these when the server gave up on a generation it
# could have finished, so the identical request may well succeed.
_ABORTED_REASONS: Final = frozenset({"abort", "error"})

_cut_replies = MetricCounter(
    name="llm_cut_replies_total",
    description=(
        "Replies cut at max_tokens (finish_reason length), by what followed: "
        "retried at retry_temperature, or the call failed."
    ),
    labelnames=("outcome",),
    namespace="everos",
)


class RawReply(NamedTuple):
    """One completion as the upstream sent it, before any check.

    What :meth:`OpenAIProvider.achat_raw` returns. Its caller reads the text
    itself, so it also gets what ``achat`` would have consumed on its behalf:
    the finish reason, and the token counts for its own trace.

    Attributes:
        text: The reply content; ``""`` when the upstream sent none.
        finish_reason: The upstream's finish reason, verbatim.
        usage: Token counts the upstream reported, or ``None``.
    """

    text: str
    finish_reason: str | None
    usage: TokenUsage | None


class _AttemptFailedError(Exception):
    """This attempt's reply is unusable, but resending the request may help.

    Args:
        message: What went wrong, without request or reply content.
        reason: The :class:`LLMCallError` reason if no attempt succeeds.
        error_count: Validation errors found, when validation is what failed.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: Literal["invalid_reply", "transport", "cut_short"],
        error_count: int | None = None,
    ) -> None:
        super().__init__(message)
        self.reason: Literal["invalid_reply", "transport", "cut_short"] = reason
        self.error_count = error_count


class OpenAIProvider:
    """``LLMClient`` over an OpenAI-compatible Chat Completions endpoint.

    Satisfies :class:`everalgo.llm.LLMClient` structurally (PEP 544) and passes
    every required and optional check of ``everalgo.testing.conformance``.

    Args:
        model: Model id every request names, unless a call's ``**extra``
            overrides ``model``.
        api_key: Bearer credential, already unwrapped from ``SecretStr``.
        base_url: OpenAI-compatible endpoint; ``None`` means the SDK default.
        timeout: Per-request timeout in seconds.
        temperature: Sent when a call passes ``temperature=None``; ``None``
            here leaves it out of the request.
        max_tokens: Sent when a call passes ``max_tokens=None``; ``None`` here
            leaves it out of the request.
        max_retries: The SDK's own transport retry count; ``None`` keeps the
            SDK default. It multiplies with every retry layered above.
        extra: Request fields merged into every call; a call's ``**extra``
            wins on a key collision. Non-standard fields must sit under
            ``extra_body`` -- the SDK rejects unknown top-level names. A
            ``max_tokens`` here is a ceiling: a call's own ``max_tokens`` can
            lower it for that call, never raise it.
        response_format_mode: How the reply format is requested --
            ``json_schema`` sends the strict schema, ``json_object`` JSON mode,
            ``prompt_only`` nothing (the prompt alone asks for JSON).
        invalid_reply_retries: Extra attempts after an attempt whose reply
            fails validation, carries no choice, or was aborted upstream.
        retry_temperature: Temperature of those extra attempts, when it is
            above the first attempt's: at a low temperature the identical
            request tends to reproduce the identical bad reply. ``None``, or a
            first attempt that sends no temperature, resends it unchanged.
        observer: Notified of every completion received, failed ones included.
        http_client: Passed to ``openai.AsyncOpenAI``; tests inject an
            ``httpx.AsyncClient`` over ``httpx.MockTransport`` here.

    Raises:
        ValueError: If ``invalid_reply_retries`` is negative.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 60.0,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int | None = None,
        extra: Mapping[str, Any] | None = None,
        response_format_mode: ResponseFormatMode = "json_schema",
        invalid_reply_retries: int = 1,
        retry_temperature: float | None = None,
        observer: LLMCallObserver | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if invalid_reply_retries < 0:
            raise ValueError(
                f"invalid_reply_retries must be >= 0, got {invalid_reply_retries}"
            )
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._http_client = http_client
        self._extra: dict[str, Any] = dict(extra or {})
        self._response_format_mode: ResponseFormatMode = response_format_mode
        self._invalid_reply_retries = invalid_reply_retries
        self._retry_temperature = retry_temperature
        self._observer: LLMCallObserver = observer or NullObserver()
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": timeout,
        }
        if max_retries is not None:
            client_kwargs["max_retries"] = max_retries
        if http_client is not None:
            client_kwargs["http_client"] = http_client
        self._client = openai.AsyncOpenAI(**client_kwargs)

    async def achat(
        self,
        messages: list[LLMMessage],
        *,
        response_format: type[BaseModel],
        temperature: float | None = None,
        max_tokens: int | None = None,
        retry_cut: bool = False,
        **extra: Any,
    ) -> str:
        """Send one chat request; see :meth:`everalgo.llm.LLMClient.achat`.

        Args:
            retry_cut: Treat a reply cut at ``max_tokens`` (``finish_reason``
                ``length``) like one that fails validation: it fails only this
                attempt, and the next one -- at most ``invalid_reply_retries``
                more, each sampled at ``retry_temperature`` -- may answer the
                call. For a call whose ``max_tokens`` guards against a runaway
                (a reply that repeats itself until the cap) rather than
                bounding a reply that is legitimately long: at temperature 0 a
                runaway tends to recur, a warmer sample does not. Off, a cut
                fails the call at once (``cut_short``), as the contract says.
            Every other argument as in the contract.

        Returns:
            The reply text, already validated against ``response_format``.

        Raises:
            LLMCallError: The upstream refused, cut the generation short or
                sent no content; the request failed in transport; or no
                attempt produced a reply that validates.
            TypeError: ``response_format_mode`` is ``json_schema`` and
                ``response_format`` is a ``RootModel``.
            ValueError: ``response_format_mode`` is ``json_schema`` and
                ``response_format`` lies outside the strict schema subset.
        """
        request = self._build_request(
            messages,
            response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=extra,
        )
        return await self._request_valid_reply(
            request, response_format, retry_cut=retry_cut
        )

    async def achat_raw(
        self,
        messages: list[LLMMessage],
        *,
        response_format: type[BaseModel] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **extra: Any,
    ) -> RawReply:
        """Send one request and hand the reply back unchecked.

        Outside the everalgo contract on purpose. ``achat`` validates the reply
        against ``response_format`` and raises on a ``length`` finish even when
        the reply's JSON is complete; a caller that parses the text itself
        wants neither. The multi-round decider is that caller: the parity check
        against the exp tree (2026-11-10, deviation R2) measured ``achat``'s
        checks failing rounds the exp decider -- which reads the raw text --
        answers from one request, each failure costing eight requests and the
        fixed fallback core.

        The request is the one ``achat`` builds, so its bytes match; the format
        is requested only as ``response_format_mode`` says, and ``None`` asks
        for none. One attempt: no ``invalid_reply_retries``, no finish-reason,
        refusal or content check -- retrying is the caller's call.

        Raises:
            LLMCallError: ``reason="transport"`` when the SDK gives up or the
                upstream answers with no choice.
        """
        request = self._build_request(
            messages,
            response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=extra,
        )
        completion = await self._send(request)
        self._observe(completion, 1)
        if not completion.choices:
            raise LLMCallError(
                f"upstream returned no choices (model={completion.model!r})",
                reason="transport",
            )
        choice = completion.choices[0]
        text = choice.message.content or ""
        if choice.finish_reason not in (None, "stop"):
            # Returned all the same; logged so a decider that keeps hitting its
            # cap is visible without reading the trace.
            logger.warning(
                "llm_raw_reply_non_stop_finish",
                finish_reason=choice.finish_reason,
                content_len=len(text),
                model=completion.model,
            )
        usage = completion.usage
        return RawReply(
            text=text,
            finish_reason=choice.finish_reason,
            usage=(
                TokenUsage(usage.prompt_tokens, usage.completion_tokens)
                if usage is not None
                else None
            ),
        )

    async def acount_prompt_tokens(
        self, messages: list[LLMMessage], **extra: Any
    ) -> tuple[int, int] | None:
        """Count ``messages`` as the endpoint would render them, and its window.

        Asks vLLM's ``/tokenize`` (at the server root, beside ``/v1``) with the model,
        the messages and the ``chat_template_kwargs`` this provider's requests carry
        under ``extra_body``, so the count is the one the endpoint checks a chat
        request against. Outside the everalgo contract, like :meth:`achat_raw`: the
        multi-round decider uses it only to fit a prompt the endpoint rejected as
        too long (``[decider].fit_to_window``).

        Returns:
            ``(prompt_tokens, max_model_len)``, or ``None`` when the endpoint has no
            such route or the call fails -- counting is best effort, never fatal.
        """
        merged = {**self._extra, **extra}
        body: dict[str, Any] = {
            "model": merged.get("model", self._model),
            "messages": [message.model_dump() for message in messages],
            "add_generation_prompt": True,
        }
        extra_body = merged.get("extra_body")
        if isinstance(extra_body, Mapping) and extra_body.get("chat_template_kwargs"):
            body["chat_template_kwargs"] = extra_body["chat_template_kwargs"]
        root = str(self._client.base_url).rstrip("/").removesuffix("/v1")
        headers = {"Authorization": f"Bearer {self._client.api_key}"}
        try:
            if self._http_client is not None:
                resp = await self._http_client.post(
                    f"{root}/tokenize", json=body, headers=headers
                )
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(
                        f"{root}/tokenize", json=body, headers=headers
                    )
            resp.raise_for_status()
            got = resp.json()
            return int(got["count"]), int(got["max_model_len"])
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "llm_tokenize_unavailable",
                error=f"{type(exc).__name__}: {exc}"[:200],
                model=self._model,
            )
            return None

    def _build_request(
        self,
        messages: list[LLMMessage],
        response_format: type[BaseModel] | None,
        *,
        temperature: float | None,
        max_tokens: int | None,
        extra: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build the request: call values beat defaults, None omitted, extras last."""
        optional: dict[str, Any] = {
            "temperature": (
                temperature if temperature is not None else self._temperature
            ),
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            "response_format": (
                to_openai_response_format(response_format, self._response_format_mode)
                if response_format is not None
                else None
            ),
        }
        request = {
            "model": self._model,
            "messages": [message.model_dump() for message in messages],
            **{key: value for key, value in optional.items() if value is not None},
            **self._extra,
            **extra,
        }
        # A max_tokens in the configured extra is the deployment's ceiling; the
        # call's own can only lower it (the picture route's runaway guard).
        if max_tokens is not None and isinstance(request.get("max_tokens"), int):
            request["max_tokens"] = min(request["max_tokens"], max_tokens)
        return request

    async def _request_valid_reply(
        self,
        request: dict[str, Any],
        response_format: type[BaseModel],
        *,
        retry_cut: bool = False,
    ) -> str:
        """Send ``request`` until a reply passes every check, a bounded number of times.

        Raises:
            LLMCallError: As in :meth:`achat`.
        """
        attempts = 1 + self._invalid_reply_retries
        failure: _AttemptFailedError | None = None
        for attempt in range(1, attempts + 1):
            sent = request if attempt == 1 else self._retry_request(request)
            try:
                return await self._attempt(
                    sent, response_format, attempt, retry_cut=retry_cut
                )
            except _AttemptFailedError as exc:
                failure = exc
                self._log_failed_attempt(exc, attempt, attempts)
                if exc.reason == "cut_short":
                    outcome = "retried" if attempt < attempts else "failed"
                    _cut_replies.labels(outcome=outcome).inc()
        # Chained to the last ValidationError, if that is what failed last.
        raise LLMCallError(
            f"no usable reply in {attempts} attempts (model={self._model!r}, "
            f"response_format_mode={self._response_format_mode!r}): {failure}",
            reason=failure.reason if failure else "invalid_reply",
        ) from (failure.__cause__ if failure else None)

    def _retry_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """``request`` as a retry sends it: at ``retry_temperature`` when that is
        above the temperature the first attempt sent, else unchanged."""
        first = request.get("temperature")
        retry = self._retry_temperature
        if retry is None or first is None or first >= retry:
            return request
        return {**request, "temperature": retry}

    async def _attempt(
        self,
        request: dict[str, Any],
        response_format: type[BaseModel],
        attempt: int,
        *,
        retry_cut: bool = False,
    ) -> str:
        """Run one attempt: send, report the completion, check it, validate it.

        Raises:
            _AttemptFailedError: This attempt's reply is unusable but resendable.
            LLMCallError: The failure is final for the whole call.
        """
        completion = await self._send(request)
        self._observe(completion, attempt)
        try:
            text = _check_upstream(
                completion, retry_cut=retry_cut, max_tokens=request.get("max_tokens")
            )
        except LLMCallError as exc:
            if exc.reason == "cut_short":
                _cut_replies.labels(outcome="failed").inc()
            raise
        _validate(text, response_format)
        return text

    async def _send(self, request: dict[str, Any]) -> ChatCompletion:
        """Issue one SDK call; the SDK applies its own transport retries.

        Raises:
            LLMCallError: ``reason="transport"`` when the SDK gives up.
        """
        try:
            return await self._client.chat.completions.create(**request)
        except openai.OpenAIError as exc:
            raise LLMCallError(str(exc), reason="transport") from exc

    def _observe(self, completion: ChatCompletion, attempt: int) -> None:
        usage = completion.usage
        self._observer.on_completion(
            model=completion.model,
            usage=(
                TokenUsage(usage.prompt_tokens, usage.completion_tokens)
                if usage is not None
                else None
            ),
            finish_reason=(
                completion.choices[0].finish_reason if completion.choices else None
            ),
            attempt=attempt,
        )

    def _log_failed_attempt(
        self, failure: _AttemptFailedError, attempt: int, attempts: int
    ) -> None:
        logger.warning(
            "llm_attempt_failed",
            reason=failure.reason,
            detail=str(failure),
            error_count=failure.error_count,
            attempt=attempt,
            attempts=attempts,
            model=self._model,
            response_format_mode=self._response_format_mode,
        )


def _check_upstream(
    completion: ChatCompletion, *, retry_cut: bool = False, max_tokens: object = None
) -> str:
    """Return the reply text unless the upstream reported it unusable.

    Raises:
        _AttemptFailedError: No choice, a generation aborted upstream, or one
            cut at ``max_tokens`` when ``retry_cut``.
        LLMCallError: A flagged refusal, a cut-short generation, or no content.
    """
    # OpenRouter answers some upstream rejections with a 200 and no choices.
    if not completion.choices:
        raise _AttemptFailedError(
            f"upstream returned no choices (model={completion.model!r})",
            reason="transport",
        )
    choice = completion.choices[0]
    # Before the content: a refusal may be phrased as JSON the operator accepts.
    if choice.message.refusal:
        raise LLMCallError(
            f"upstream refused (model={completion.model!r})", reason="refusal"
        )
    # Before the content too: a reply cut right after its closing brace parses.
    if choice.finish_reason == "length" and retry_cut:
        raise _AttemptFailedError(
            f"generation cut short (model={completion.model!r}, "
            f"finish_reason='length', max_tokens={max_tokens!r})",
            reason="cut_short",
        )
    if choice.finish_reason in _CUT_SHORT_REASONS:
        raise LLMCallError(
            f"generation cut short (model={completion.model!r}, "
            f"finish_reason={choice.finish_reason!r})",
            reason="cut_short",
        )
    if choice.finish_reason in _ABORTED_REASONS:
        raise _AttemptFailedError(
            f"generation aborted upstream (finish_reason={choice.finish_reason!r})",
            reason="transport",
        )
    if choice.message.content is None:
        raise LLMCallError(
            f"upstream returned no content (model={completion.model!r}, "
            f"finish_reason={choice.finish_reason!r})",
            reason="no_content",
        )
    return choice.message.content


def _validate(text: str, response_format: type[BaseModel]) -> None:
    """Check ``text`` against the caller's own model, as the operator will.

    Raises:
        _AttemptFailedError: ``text`` does not validate; chained to the
            ``ValidationError``.
    """
    try:
        response_format.model_validate_json(text)
    except ValidationError as exc:
        raise _AttemptFailedError(
            f"reply failed validation against {response_format.__name__}",
            reason="invalid_reply",
            error_count=exc.error_count(),
        ) from exc
