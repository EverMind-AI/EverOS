"""``parser_available`` memoisation contract.

Round-3 review finding #14: ``/health`` used to trigger a heavy
``import everalgo.parser`` on every hit — pypdf / python-docx /
tesseract pulled in on the event loop, hundreds of ms of blocking
work per request. The fix wraps :func:`parser_available` in
``functools.lru_cache(maxsize=1)`` so only the first call in the
process pays the import cost; a lifespan warms that first call at
boot so ``/health`` itself never triggers it.

These tests pin the cache: the second call must reuse the cached
answer without re-attempting the import. ``cache_clear()`` is invoked
explicitly around each mutation of ``sys.modules`` to avoid
cross-test contamination — that discipline is contractual for any
test that toggles the module's importability.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from everos.component.parser import parser_available


@pytest.fixture(autouse=True)
def _clear_cache_between_tests() -> None:
    """Drop the lru_cache before and after each test.

    Without this, ``sys.modules`` swaps inside one test leak into the
    next — the second test would see the frozen decision from the
    first even after restoring ``sys.modules``.
    """
    parser_available.cache_clear()
    yield
    parser_available.cache_clear()


def test_parser_available_second_call_hits_cache() -> None:
    """Second invocation must NOT re-run the ``import everalgo.parser``.

    Uses :meth:`functools.lru_cache.cache_info` to confirm the second
    call is a cache hit — the underlying ``import`` statement never
    ran a second time, so ``/health`` can no longer trigger a fresh
    import per request.

    We deliberately avoid spying on ``builtins.__import__``: even a
    cached ``import`` still goes through ``__import__`` for the
    ``sys.modules`` lookup, so a call-count assertion at that layer
    can't distinguish "cached" from "re-imported". The cache-info
    counters are the only reliable signal here.
    """
    first = parser_available()
    second = parser_available()

    info = parser_available.cache_info()
    assert first is second, "cached identity"
    assert info.hits >= 1, f"expected at least one cache hit; got {info}"
    assert info.misses == 1, f"expected exactly one miss (the warming call); got {info}"


def test_parser_available_caches_missing_extra_result() -> None:
    """When the extra isn't installed, the cache must remember ``False``
    just as strongly as it would remember ``True``. Otherwise ``/health``
    on a base install would re-attempt the (failing) import every hit."""
    # Simulate the missing-extra case by making the module import raise
    # ImportError. Patch `sys.modules` to insert a sentinel that Python's
    # import machinery treats as "already imported → don't re-run"; use
    # `builtins.__import__` wrapper to raise instead.
    original_import = __import__

    def _fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "everalgo.parser":
            raise ImportError("simulated missing extra")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_fake_import):
        first = parser_available()
        second = parser_available()

    # Both calls returned False — but critically only the first ran the
    # failing import; the second was a hot cache read (proved by the fact
    # that we can `parser_available.cache_info().hits >= 1`).
    assert first is False
    assert second is False
    assert parser_available.cache_info().hits >= 1


def test_parser_available_cache_clear_re_evaluates() -> None:
    """``cache_clear()`` is contractually the escape hatch for tests —
    prove it works, so future tests toggling ``sys.modules`` have a
    reliable reset primitive."""
    # Force a False decision, then clear + swap to a True decision.
    fake_module = MagicMock()

    def _import_missing(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "everalgo.parser":
            raise ImportError("first pass: missing")
        return __import__(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_import_missing):
        assert parser_available() is False

    parser_available.cache_clear()

    # Now pretend the module is importable by pre-seeding sys.modules —
    # `import everalgo.parser` will find it there and skip the loader.
    with patch.dict(sys.modules, {"everalgo.parser": fake_module}):
        assert parser_available() is True
