"""The boundary LLM failure is typed so only that step is retry-safe."""

from __future__ import annotations

import datetime as _dt

import pytest
from everalgo.llm import LLMError

from everos.core.errors import BoundaryLLMError
from everos.memory import CanonicalMessage
from everos.service._boundary import _detect


def _msg(mid: str = "m1") -> CanonicalMessage:
    return CanonicalMessage(
        message_id=mid,
        session_id="s1",
        sender_id="u1",
        sender_name="u1",
        role="user",
        timestamp=_dt.datetime.fromtimestamp(1_700_000_000, tz=_dt.UTC),
        text="hi",
    )


async def test_detect_wraps_llm_error_as_boundary_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def boom(*_: object, **__: object) -> object:
        raise LLMError("provider down")

    monkeypatch.setattr("everos.service._boundary.detect_boundaries", boom)

    with pytest.raises(BoundaryLLMError) as exc_info:
        await _detect(
            [_msg()],
            mode="chat",
            llm_client=object(),  # type: ignore[arg-type]
            prompt="boundaries",
            is_final=True,
            hard_token_limit=1000,
            hard_msg_limit=10,
        )

    assert "provider down" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, LLMError)
