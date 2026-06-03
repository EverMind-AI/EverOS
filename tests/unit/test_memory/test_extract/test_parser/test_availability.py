"""Tests for the multimodal capability guard."""

from __future__ import annotations

import pytest

from everos.core.errors import MultimodalNotEnabledError
from everos.memory.extract.parser import availability


def test_has_unparsed_multimodal_true_for_unparsed_nontext() -> None:
    items = [{"type": "text", "text": "hi"}, {"type": "image", "uri": "x"}]
    assert availability.has_unparsed_multimodal(items) is True


def test_has_unparsed_multimodal_false_when_all_text() -> None:
    items = [{"type": "text", "text": "hi"}]
    assert availability.has_unparsed_multimodal(items) is False


def test_has_unparsed_multimodal_false_when_already_parsed() -> None:
    items = [{"type": "image", "uri": "x", "parsed_content": "ocr"}]
    assert availability.has_unparsed_multimodal(items) is False


def test_require_multimodal_raises_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(availability, "multimodal_available", lambda: False)
    with pytest.raises(MultimodalNotEnabledError):
        availability.require_multimodal()


def test_require_multimodal_ok_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(availability, "multimodal_available", lambda: True)
    availability.require_multimodal()  # must not raise
