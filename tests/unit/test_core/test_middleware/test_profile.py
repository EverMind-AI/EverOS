"""``ProfileMiddleware`` — env gating, query-param gating, pyinstrument output."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from everos.core.middleware.profile import ProfileMiddleware, _profiling_enabled


def _build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(ProfileMiddleware)

    @app.get("/hello")
    async def hello() -> dict[str, str]:
        return {"ok": "yes"}

    return app


@pytest.fixture
def disable_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROFILING_ENABLED", raising=False)
    monkeypatch.delenv("PROFILING", raising=False)


@pytest.fixture
def enable_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROFILING_ENABLED", "true")


def test_profiling_enabled_truthy_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    for v in ("1", "true", "TRUE", "yes"):
        monkeypatch.setenv("PROFILING_ENABLED", v)
        assert _profiling_enabled() is True


def test_profiling_enabled_falsy_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    for v in ("0", "false", "no", "", "anything-else"):
        monkeypatch.setenv("PROFILING_ENABLED", v)
        assert _profiling_enabled() is False


def test_profiling_falls_back_to_legacy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROFILING_ENABLED", raising=False)
    monkeypatch.setenv("PROFILING", "yes")
    assert _profiling_enabled() is True


@pytest.fixture
async def disabled_client(disable_env: None) -> AsyncIterator[AsyncClient]:
    app = _build_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


@pytest.fixture
async def enabled_client(enable_env: None) -> AsyncIterator[AsyncClient]:
    app = _build_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


async def test_disabled_passthrough(disabled_client: AsyncClient) -> None:
    """When profiling is disabled, ``?profile=true`` is ignored — JSON returned."""
    resp = await disabled_client.get("/hello?profile=true")
    assert resp.status_code == 200
    assert resp.json() == {"ok": "yes"}


async def test_enabled_without_query_passthrough(enabled_client: AsyncClient) -> None:
    """Enabled middleware but request without ``?profile=true`` → normal response."""
    resp = await enabled_client.get("/hello")
    assert resp.status_code == 200
    assert resp.json() == {"ok": "yes"}


async def test_enabled_with_query_returns_html(enabled_client: AsyncClient) -> None:
    """With ``?profile=true`` and pyinstrument available, response is HTML."""
    try:
        import pyinstrument  # noqa: F401
    except ImportError:
        pytest.skip("pyinstrument not installed in this env")

    resp = await enabled_client.get("/hello?profile=true")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    # Pyinstrument output contains the word "pyinstrument" in its template.
    assert "pyinstrument" in resp.text.lower() or "<html" in resp.text.lower()


async def test_enabled_with_query_returns_html_when_inner_raises(
    enabled_client: AsyncClient,
) -> None:
    """An exception inside the wrapped handler is logged but still produces HTML."""
    try:
        import pyinstrument  # noqa: F401
    except ImportError:
        pytest.skip("pyinstrument not installed in this env")

    # Rebuild a tiny app whose route raises so the middleware's except branch
    # fires; the profile HTML is still emitted regardless.
    app = FastAPI()
    app.add_middleware(ProfileMiddleware)

    @app.get("/bang")
    async def bang() -> None:
        raise RuntimeError("inner exception")

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as c:
        resp = await c.get("/bang?profile=true")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")


async def test_enabled_without_pyinstrument(monkeypatch: pytest.MonkeyPatch) -> None:
    """If pyinstrument import fails, middleware degrades to passthrough."""
    monkeypatch.setenv("PROFILING_ENABLED", "true")
    # Force the import inside ProfileMiddleware.__init__ to fail.
    import builtins

    real_import = builtins.__import__

    def fail_pyinstrument(name: str, *args: object, **kwargs: object) -> object:
        if name == "pyinstrument":
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_pyinstrument)
    app = _build_app()  # ProfileMiddleware ctor runs here

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        resp = await c.get("/hello?profile=true")
    assert resp.status_code == 200
    assert resp.json() == {"ok": "yes"}
