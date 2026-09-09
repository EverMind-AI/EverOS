"""Minimal EverOS 1.0 HTTP client (stdlib only)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


class EverOSError(Exception):
    """EverOS HTTP or contract error."""


def _request(
    *,
    base_url: str,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    url = f"{base_url}{path}"
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise EverOSError(f"HTTP {exc.code} {path}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise EverOSError(f"unreachable {base_url}: {exc.reason}") from exc
    if not isinstance(payload, dict):
        raise EverOSError(f"unexpected response type from {path}")
    return payload


def health(base_url: str) -> bool:
    try:
        payload = _request(base_url=base_url, method="GET", path="/health")
    except EverOSError:
        return False
    return payload.get("status") == "ok"


def search(
    *,
    base_url: str,
    user_id: str,
    query: str,
    app_id: str,
    project_id: str,
    top_k: int,
    min_score: float,
) -> dict[str, Any]:
    body = {
        "user_id": user_id,
        "app_id": app_id,
        "project_id": project_id,
        "query": query,
        "top_k": top_k,
        "min_score": min_score,
    }
    payload = _request(
        base_url=base_url,
        method="POST",
        path="/api/v1/memory/search",
        body=body,
    )
    data = payload.get("data")
    if not isinstance(data, dict):
        raise EverOSError("search response missing data envelope")
    return data


def add_messages(
    *,
    base_url: str,
    session_id: str,
    app_id: str,
    project_id: str,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    body = {
        "session_id": session_id,
        "app_id": app_id,
        "project_id": project_id,
        "messages": messages,
    }
    payload = _request(
        base_url=base_url,
        method="POST",
        path="/api/v1/memory/add",
        body=body,
    )
    data = payload.get("data")
    if not isinstance(data, dict):
        raise EverOSError("add response missing data envelope")
    return data


def flush(
    *,
    base_url: str,
    session_id: str,
    app_id: str,
    project_id: str,
) -> dict[str, Any]:
    body = {
        "session_id": session_id,
        "app_id": app_id,
        "project_id": project_id,
    }
    payload = _request(
        base_url=base_url,
        method="POST",
        path="/api/v1/memory/flush",
        body=body,
    )
    data = payload.get("data")
    if not isinstance(data, dict):
        raise EverOSError("flush response missing data envelope")
    return data


def message_item(
    *,
    sender_id: str,
    role: str,
    content: str,
    timestamp_ms: int | None = None,
) -> dict[str, Any]:
    ts = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)
    return {
        "sender_id": sender_id,
        "role": role,
        "timestamp": ts,
        "content": content,
    }
