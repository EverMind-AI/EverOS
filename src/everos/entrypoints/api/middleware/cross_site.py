"""Refuse what only a browser page of another site would send.

EverOS has no authentication: its boundary is "only this machine can connect"
(``[api] host`` defaults to loopback). A browser on that machine crosses it in
two ways, and this middleware closes both:

* **Cross-site requests.** Any page the user opens can send requests to
  ``127.0.0.1``. Browsers put an ``Origin`` header on every CORS request and
  every ``POST``; curl, SDKs, MCP and the plugins send none. A request whose
  ``Origin`` is neither this server's own nor listed in ``[api] cors_origins``
  is refused. A no-cors ``GET`` (``<img>``, ``<script>``, a navigation)
  carries no ``Origin``, so it is caught by Fetch Metadata instead: browsers
  mark it ``Sec-Fetch-Site: cross-site``, and such a request without an
  allowed ``Origin`` is refused too. Browsers that predate Fetch Metadata still
  let those through; that is harmless because every ``GET`` route is read-only
  and a no-cors response is opaque to the page.
* **DNS rebinding.** A page of ``attacker.com`` re-points its own name at
  ``127.0.0.1`` and is then same-origin with this server, so ``Origin`` alone
  cannot tell it apart. Rebinding needs a public domain the attacker controls,
  so a request is refused when its ``Host`` is such a name and not listed in
  ``[api] allowed_hosts``. IP addresses, single-label names (``everos`` in a
  compose network or a k8s namespace) and names under the domains reserved
  from public registration (``localhost``, ``internal``, ``test``,
  ``invalid``, ``home.arpa``) are always accepted. Other private names are
  not: a k8s name with a namespace (``everos.ns.svc.cluster.local``) needs
  listing, and so does an mDNS ``.local`` name, which any machine on the LAN
  can answer for.

Refusals answer ``403 CROSS_SITE_REFUSED`` in the usual error envelope and
cover every route, including CORS preflights, ``/health`` and ``/metrics``. A
``Host`` or ``Origin`` that does not parse (``[::1`` without its bracket) is
refused the same way rather than raising.
"""

from __future__ import annotations

__all__ = ["CrossSiteGuardMiddleware"]

import ipaddress
from collections.abc import Collection
from urllib.parse import urlsplit

from starlette.requests import Request
from starlette.types import ASGIApp, Receive, Scope, Send

from everos.core.errors import ErrorCode
from everos.core.observability.logging import get_logger

from ..exception_handlers import error_response

logger = get_logger(__name__)

_ORIGIN_REFUSED = (
    "requests from a browser page of another site are refused; "
    "to allow that site, add its origin to [api] cors_origins "
    "(EVEROS_API__CORS_ORIGINS)"
)

_RESERVED_TLDS = frozenset({"localhost", "internal", "test", "invalid"})
# RFC 8375: home networks; like the TLDs above, nobody can register under it.
_RESERVED_SUFFIXES = (".home.arpa",)


def _host_is_private(hostname: str) -> bool:
    """Whether no one else can make a browser resolve ``hostname`` to us."""
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        return (
            "." not in hostname
            or hostname.rsplit(".", 1)[1] in _RESERVED_TLDS
            or hostname.endswith(_RESERVED_SUFFIXES)
        )
    return True


class CrossSiteGuardMiddleware:
    """Pure ASGI middleware; see the module docstring for the rules.

    Args:
        app: The wrapped ASGI app.
        allowed_hosts: ``[api] allowed_hosts``; ``"*"`` accepts any ``Host``.
        allowed_origins: ``[api] cors_origins``. Matched literally: ``"*"``
            matches no ``Origin`` (settings refuse it at load), so a ``"*"``
            passed straight to :func:`create_app` cannot open the API.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        allowed_hosts: Collection[str] = (),
        allowed_origins: Collection[str] = (),
    ) -> None:
        self._app = app
        self._hosts = {h.lower() for h in allowed_hosts}
        self._origins = {o.lower().rstrip("/") for o in allowed_origins}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        request = Request(scope)
        reason = self._refusal(request)
        if reason is None:
            await self._app(scope, receive, send)
            return
        logger.warning(
            "cross_site_refused",
            path=request.url.path,
            host=request.headers.get("host"),
            origin=request.headers.get("origin"),
        )
        response = error_response(request, 403, ErrorCode.CROSS_SITE_REFUSED, reason)
        await response(scope, receive, send)

    def _refusal(self, request: Request) -> str | None:
        host = request.headers.get("host", "")
        try:
            hostname = urlsplit(f"//{host}").hostname if host else None
        except ValueError:
            return f"malformed Host header {host!r}"
        if (
            hostname is not None
            and "*" not in self._hosts
            and hostname not in self._hosts
            and not _host_is_private(hostname)
        ):
            return (
                f"host {hostname!r} is not accepted; if this server is reached "
                "under that name, add it to [api] allowed_hosts "
                "(EVEROS_API__ALLOWED_HOSTS)"
            )
        origin = request.headers.get("origin")
        if origin is None:
            if request.headers.get("sec-fetch-site") == "cross-site":
                return _ORIGIN_REFUSED
            return None
        if origin.lower().rstrip("/") in self._origins:
            return None
        try:
            netloc = urlsplit(origin).netloc
        except ValueError:
            return f"malformed Origin header {origin!r}"
        if netloc.lower() != host.lower():
            return _ORIGIN_REFUSED
        return None
