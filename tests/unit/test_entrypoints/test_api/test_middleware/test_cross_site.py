"""CrossSiteGuardMiddleware: a browser page of another site cannot drive the API.

The server has no authentication and listens on loopback; a browser on the same
machine is the way around that. These tests pin both doors: an ``Origin`` of
another site (CSRF / cross-site reads) and a public ``Host`` (DNS rebinding).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import pytest
from httpx import ASGITransport, AsyncClient

from everos.config import load_settings
from everos.entrypoints.api.app import create_app
from tests.fakes import make_application


def _client(
    base_url: str = "http://127.0.0.1:8000",
    *,
    allowed_hosts: Sequence[str] = (),
    cors_origins: Sequence[str] = (),
) -> AsyncClient:
    base = load_settings()
    hosts, origins = list(allowed_hosts), list(cors_origins)
    api = base.api.model_copy(update={"allowed_hosts": hosts, "cors_origins": origins})
    settings = base.model_copy(update={"api": api})
    app = create_app(settings, application=make_application())
    return AsyncClient(transport=ASGITransport(app=app), base_url=base_url)


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    async with _client() as c:
        yield c


def _refused(resp: object) -> bool:
    status = resp.status_code  # type: ignore[attr-defined]
    if status != 403:
        return False
    body = resp.json()  # type: ignore[attr-defined]
    assert body["error"]["code"] == "CROSS_SITE_REFUSED"
    assert body["request_id"]
    return True


@pytest.mark.parametrize(
    ("host", "accepted"),
    [
        ("127.0.0.1:8000", True),
        ("localhost:8000", True),
        ("[::1]:8000", True),
        ("192.168.1.20:8000", True),  # bound to a LAN address
        ("everos:8000", True),  # a compose / same-namespace k8s service name
        ("gw.internal", True),  # reserved TLDs: nobody can register them
        ("server.test", True),
        ("app.localhost:8000", True),
        ("nas.home.arpa:8000", True),  # RFC 8375, not publicly registrable
        ("everos.ns.svc.cluster.local", False),  # needs [api] allowed_hosts
        ("mybox.local:8000", False),  # mDNS: anyone on the LAN can answer
        ("home.arpa", False),  # only names under it are reserved
        ("attacker.com:8000", False),  # DNS rebinding needs a public name
        ("attacker.example:8000", False),
        ("127.0.0.1.nip.io:8000", False),  # public name that resolves to loopback
        ("memory.example.com", False),
    ],
)
async def test_host_must_not_be_a_public_name(host: str, accepted: bool) -> None:
    async with _client() as c:
        resp = await c.get("/health", headers={"host": host})
    assert _refused(resp) is not accepted, resp.text


async def test_a_listed_public_name_is_accepted() -> None:
    async with _client(allowed_hosts=["Memory.Example.com"]) as c:
        listed = await c.get("/health", headers={"host": "memory.example.com"})
        other = await c.get("/health", headers={"host": "attacker.com"})
    assert listed.status_code == 200, listed.text
    assert _refused(other)


async def test_a_star_accepts_any_host() -> None:
    async with _client(allowed_hosts=["*"]) as c:
        resp = await c.get("/health", headers={"host": "attacker.com"})
    assert resp.status_code == 200, resp.text


@pytest.mark.parametrize(
    ("origin", "accepted"),
    [
        (None, True),  # curl, SDKs, MCP, plugins send no Origin
        ("http://127.0.0.1:8000", True),  # a page this server serves itself
        ("HTTP://127.0.0.1:8000", True),
        ("http://127.0.0.1:8001", False),  # same host, another port: another site
        ("http://localhost:3000", False),  # a local web UI, until it is listed
        ("https://evil.example", False),
        ("null", False),  # a sandboxed or file:// page
    ],
)
async def test_origin_must_be_this_server(
    client: AsyncClient, origin: str | None, accepted: bool
) -> None:
    headers = {"origin": origin} if origin is not None else {}
    get = await client.get("/health", headers=headers)
    post = await client.post("/api/v2/memory/search", json={}, headers=headers)
    assert _refused(get) is not accepted, get.text
    assert _refused(post) is not accepted, post.text


async def test_a_foreign_preflight_is_refused_without_cors_headers(
    client: AsyncClient,
) -> None:
    resp = await client.options(
        "/api/v2/memory/search",
        headers={
            "origin": "https://evil.example",
            "access-control-request-method": "POST",
            "access-control-request-headers": "content-type",
        },
    )
    assert _refused(resp)
    assert "access-control-allow-origin" not in resp.headers


async def test_a_listed_origin_gets_cors_and_passes() -> None:
    ui = "http://localhost:3000"
    async with _client(cors_origins=[ui]) as c:
        preflight = await c.options(
            "/api/v2/memory/search",
            headers={
                "origin": ui,
                "access-control-request-method": "POST",
                "access-control-request-headers": "content-type",
            },
        )
        get = await c.get("/health", headers={"origin": ui})
        other = await c.get("/health", headers={"origin": "https://evil.example"})
    assert preflight.status_code == 200, preflight.text
    assert preflight.headers["access-control-allow-origin"] == ui
    assert get.status_code == 200, get.text
    assert get.headers["access-control-allow-origin"] == ui
    assert _refused(other)


@pytest.mark.parametrize(
    "headers",
    [
        {"host": "[::1"},  # h11 accepts it; urlsplit raises "Invalid IPv6 URL"
        {"origin": "http://[::1"},
    ],
)
async def test_a_malformed_host_or_origin_is_refused_not_a_500(
    client: AsyncClient, headers: dict[str, str]
) -> None:
    resp = await client.get("/health", headers=headers)
    assert _refused(resp), resp.text


@pytest.mark.parametrize(
    ("site", "accepted"),
    [
        ("cross-site", False),  # <img src>, <script src>, a navigation
        ("same-origin", True),
        ("same-site", True),
        ("none", True),  # typed into the address bar
        (None, True),  # curl, SDKs, MCP: no Fetch Metadata at all
    ],
)
async def test_a_no_cors_cross_site_get_is_refused_by_fetch_metadata(
    client: AsyncClient, site: str | None, accepted: bool
) -> None:
    headers = {"sec-fetch-site": site} if site is not None else {}
    resp = await client.get("/health", headers=headers)
    assert (resp.status_code == 200) is accepted, resp.text
    if not accepted:
        assert _refused(resp)


async def test_fetch_metadata_does_not_override_a_listed_origin() -> None:
    ui = "http://localhost:3000"
    async with _client(cors_origins=[ui]) as c:
        resp = await c.get(
            "/health", headers={"origin": ui, "sec-fetch-site": "cross-site"}
        )
    assert resp.status_code == 200, resp.text


async def test_a_star_origin_passed_to_create_app_opens_nothing() -> None:
    """Settings refuse ``"*"``; the guard matches it literally as a fallback."""
    settings = load_settings()
    app = create_app(settings, application=make_application(), cors_origins=["*"])
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://127.0.0.1:8000"
    ) as c:
        purge = await c.post(
            "/api/v2/memory/purge",
            json={"scope": "space"},
            headers={"origin": "https://evil.example"},
        )
    assert _refused(purge), purge.text
