"""EverOS's implementation of the everalgo ``LLMClient`` contract.

:class:`OpenAIProvider` talks to any OpenAI-compatible Chat Completions
endpoint through ``openai.AsyncOpenAI``. It is the only ``LLMClient``
implementation in EverOS; everalgo's reference ``OpenAICompatClient`` is read,
never used.

``achat`` follows the recommended flow of ``llm-contract.md`` §7, one private
method per step:

1. ``_build_request`` -- build the request once; every attempt sends it as is,
   so no retry can raise ``max_tokens`` (§3.1). ``None`` falls back to the
   configured default and is otherwise left out. Messages go out via
   ``model_dump()``, multimodal parts included (§3.2).
2. ``_send`` -- one SDK call; transport retries are the SDK's ``max_retries``.
   ``asyncio.CancelledError`` is never caught (§3.3).
3. ``_check_upstream`` -- a flagged refusal, ``length`` / ``content_filter``
   or absent content raises (§3.4, §3.5); no choice and ``abort`` / ``error``
   fail only this attempt.
4. ``_validate`` -- ``model_validate_json`` against the caller's
   ``response_format``; a mismatch fails only this attempt (§7.1).
5. ``_request_valid_reply`` -- at most ``1 + invalid_reply_retries`` attempts
   of the identical request, each completion reported to the observer.
"""

from __future__ import annotations

__all__ = ["OpenAIProvider", "ResponseFormatMode"]

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final, Literal

import openai
from everalgo.llm import to_openai_response_format
from pydantic import BaseModel, ValidationError

from everos.core.observability.logging import get_logger

from .errors import LLMCallError
from .observer import LLMCallObserver, NullObserver, TokenUsage

if TYPE_CHECKING:
    import httpx
    from everalgo.llm import LLMMessage
    from openai.types.chat import ChatCompletion

logger = get_logger(__name__)

ResponseFormatMode = Literal["json_schema", "json_object", "prompt_only"]

# The identical request hits the same output cap or the same filter again.
_CUT_SHORT_REASONS: Final = frozenset({"length", "content_filter"})

# vLLM-style gateways report these when the server gave up on a generation it
# could have finished, so the identical request may well succeed.
_ABORTED_REASONS: Final = frozenset({"abort", "error"})


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
        reason: Literal["invalid_reply", "transport"],
        error_count: int | None = None,
    ) -> None:
        super().__init__(message)
        self.reason: Literal["invalid_reply", "transport"] = reason
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
            ``extra_body`` -- the SDK rejects unknown top-level names.
        response_format_mode: How the reply format is requested --
            ``json_schema`` sends the strict schema, ``json_object`` JSON mode,
            ``prompt_only`` nothing (the prompt alone asks for JSON).
        invalid_reply_retries: Extra attempts after an attempt whose reply
            fails validation, carries no choice, or was aborted upstream.
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
        self._extra: dict[str, Any] = dict(extra or {})
        self._response_format_mode: ResponseFormatMode = response_format_mode
        self._invalid_reply_retries = invalid_reply_retries
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
        **extra: Any,
    ) -> str:
        """Send one chat request; see :meth:`everalgo.llm.LLMClient.achat`.

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
        return await self._request_valid_reply(request, response_format)

    def _build_request(
        self,
        messages: list[LLMMessage],
        response_format: type[BaseModel],
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
            "response_format": to_openai_response_format(
                response_format, self._response_format_mode
            ),
        }
        return {
            "model": self._model,
            "messages": [message.model_dump() for message in messages],
            **{key: value for key, value in optional.items() if value is not None},
            **self._extra,
            **extra,
        }

    async def _request_valid_reply(
        self, request: dict[str, Any], response_format: type[BaseModel]
    ) -> str:
        """Send ``request`` until a reply passes every check, a bounded number of times.

        Raises:
            LLMCallError: As in :meth:`achat`.
        """
        attempts = 1 + self._invalid_reply_retries
        failure: _AttemptFailedError | None = None
        for attempt in range(1, attempts + 1):
            try:
                return await self._attempt(request, response_format, attempt)
            except _AttemptFailedError as exc:
                failure = exc
                self._log_failed_attempt(exc, attempt, attempts)
        # Chained to the last ValidationError, if that is what failed last.
        raise LLMCallError(
            f"no usable reply in {attempts} attempts (model={self._model!r}, "
            f"response_format_mode={self._response_format_mode!r}): {failure}",
            reason=failure.reason if failure else "invalid_reply",
        ) from (failure.__cause__ if failure else None)

    async def _attempt(
        self, request: dict[str, Any], response_format: type[BaseModel], attempt: int
    ) -> str:
        """Run one attempt: send, report the completion, check it, validate it.

        Raises:
            _AttemptFailedError: This attempt's reply is unusable but resendable.
            LLMCallError: The failure is final for the whole call.
        """
        completion = await self._send(request)
        self._observe(completion, attempt)
        text = _check_upstream(completion)
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


def _check_upstream(completion: ChatCompletion) -> str:
    """Return the reply text unless the upstream reported it unusable.

    Raises:
        _AttemptFailedError: No choice, or a generation aborted upstream.
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
