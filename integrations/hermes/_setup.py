"""Setup wizard for the EverOS Hermes plugin — Hermes-agnostic.

Uses only stdlib + the local ``_config._atomic_write_text`` helper. The
real Hermes provider (``__init__.py``, Phase 2) adapts ``post_setup`` here
to the ABC signature ``post_setup(self, hermes_home: str, config: dict)``.
"""

from __future__ import annotations

import json
import logging
import tomllib
from pathlib import Path
from typing import Any

from ._config import _atomic_write_text
from ._constants import _DEFAULT_AGENT_ID, _DEFAULT_API_URL, _DEFAULT_USER_ID

logger = logging.getLogger(__name__)

_TOML_FILE_MODE = 0o600
_JSON_FILE_MODE = 0o600

# Optional TOML section fields beyond model/api_key/base_url.
_TOML_OPTIONAL_FIELDS = (
    "timeout_seconds",
    "max_retries",
    "batch_size",
    "max_concurrent",
)


def _toml_section(name: str, fields: dict[str, Any]) -> str:
    """Render one ``[name]`` TOML section as a string block."""
    lines = [f"[{name}]"]
    for key in ("model", "api_key", "base_url"):
        if key in fields and fields[key] is not None:
            lines.append(f"{key} = {_toml_value(fields[key])}")
    for key in _TOML_OPTIONAL_FIELDS:
        if key in fields and fields[key] is not None:
            lines.append(f"{key} = {_toml_value(fields[key])}")
    lines.append("")
    return "\n".join(lines)


def _toml_value(value: Any) -> str:
    """Render a scalar (or inline array of scalars) as a TOML rvalue."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    # datetime and other objects — quote their str form.
    return json.dumps(str(value), ensure_ascii=False)


def write_everos_toml(
    everos_root: Path,
    *,
    llm: dict[str, Any],
    embedding: dict[str, Any],
    rerank: dict[str, Any] | None,
) -> Path:
    """Write ``everos.toml`` under ``~/.everos`` (or ``everos_root``).

    Sections: ``[llm]`` and ``[embedding]`` are always written; ``[rerank]``
    only when ``rerank`` is provided. Each section emits ``model``,
    ``api_key``, ``base_url`` plus any of the optional knobs
    (``timeout_seconds`` / ``max_retries`` / ``batch_size`` /
    ``max_concurrent``) that the caller supplied.

    The file is chmod 600. Returns the written path.
    """
    root = Path(everos_root).expanduser() if everos_root else Path.home() / ".everos"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "everos.toml"

    blocks = [
        "# EverOS configuration — written by the Hermes EverOS plugin setup.",
        "",
        _toml_section("llm", llm),
        _toml_section("embedding", embedding),
    ]
    if rerank:
        blocks.append(_toml_section("rerank", rerank))

    text = "\n".join(blocks)
    _atomic_write_text(path, text, mode=_TOML_FILE_MODE)
    return path


def _emit_table(lines: list[str], path: list[str], table: dict[str, Any]) -> None:
    """Emit one TOML table's scalars, then its sub-tables (recursively).

    Scalars belonging to the current table are emitted first (TOML requires
    them to appear before any sub-table header). Sub-tables are then emitted
    as ``[a.b.c]`` headers with their own contents.
    """
    wrote_scalar = False
    for key, value in table.items():
        if not isinstance(value, dict):
            lines.append(f"{key} = {_toml_value(value)}")
            wrote_scalar = True
    if wrote_scalar:
        lines.append("")
    for key, value in table.items():
        if isinstance(value, dict):
            sub_path = [*path, key]
            lines.append(f"[{'.'.join(sub_path)}]")
            _emit_table(lines, sub_path, value)


def _emit_toml(data: dict[str, Any]) -> str:
    """Render a parsed-TOML dict back to TOML text (section-aware).

    Top-level scalars are emitted as ``key = value``; nested dicts become
    ``[section]`` tables (recursing for sub-tables). Scalar values only —
    booleans as ``true``/``false``, strings quoted, numbers bare, lists as
    inline arrays. No ``tomli_w`` dependency.
    """
    lines: list[str] = []
    _emit_table(lines, [], data)
    return "\n".join(lines)


def seed_ome_toml(everos_root: Path, *, agent_track: bool) -> Path | None:
    """Write/merge ``ome.toml`` enabling the agent-track extractors.

    Returns ``None`` when ``agent_track`` is falsy. Otherwise ensures
    ``extract_agent_case``, ``extract_agent_skill`` and
    ``trigger_skill_clustering`` are all enabled under ``[strategies.<name>]``,
    leaving any other keys/sections the file already carries untouched.

    Reads the existing file with ``tomllib`` (stdlib) and re-emits via the
    section-aware ``_emit_toml`` helper so the on-disk shape matches the
    ``TomlRoot`` schema (``[strategies.<name>]`` tables, not flat top-level
    keys). chmod 600.
    """
    if not agent_track:
        return None

    root = Path(everos_root).expanduser() if everos_root else Path.home() / ".everos"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "ome.toml"

    data: dict[str, Any] = {}
    if path.exists():
        try:
            with path.open("rb") as fh:
                data = tomllib.load(fh)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            logger.warning("Failed to parse %s: %s; starting fresh", path, exc)
            data = {}

    strategies = data.setdefault("strategies", {})
    for name in (
        "extract_agent_case",
        "extract_agent_skill",
        "trigger_skill_clustering",
    ):
        strategies.setdefault(name, {})["enabled"] = True

    header = "# OME scheduler config — written by the Hermes EverOS plugin setup."
    body = _emit_toml(data).rstrip("\n")
    text = f"{header}\n\n{body}\n" if body else f"{header}\n"
    _atomic_write_text(path, text, mode=_TOML_FILE_MODE)
    return path


def build_everos_json(
    api_url: str,
    mode: str,
    user_id: str,
    agent_id: str,
    agent_track_enabled: bool,
    everos_root: str | None,
) -> dict[str, Any]:
    """Build the dict that gets written to ``$HERMES_HOME/everos.json``."""
    return {
        "api_url": api_url,
        "mode": mode,
        "user_id": user_id,
        "agent_id": agent_id,
        "agent_track_enabled": bool(agent_track_enabled),
        "everos_root": everos_root,
    }


def _write_everos_json(hermes_home: Path, payload: dict[str, Any]) -> Path:
    path = Path(hermes_home) / "everos.json"
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(path, text, mode=_JSON_FILE_MODE)
    return path


def post_setup(
    hermes_home: Path,
    config: dict[str, Any],
    *,
    interactive: bool = True,
    inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the setup wizard and persist the resulting config files.

    Parameters
    ----------
    hermes_home:
        The Hermes home directory; ``everos.json`` is written here.
    config:
        Existing plugin config (merged defaults + env + previous JSON).
    interactive:
        When True and ``inputs`` is None, prompt the operator on stdin for
        the values the wizard needs. When False (or when ``inputs`` is
        supplied), no prompting happens — used by tests and non-interactive
        installs.
    inputs:
        Pre-collected wizard answers for non-interactive runs. Recognised
        keys: ``mode``, ``api_url``, ``user_id``, ``agent_id``,
        ``agent_track_enabled``, ``everos_root``, ``llm``, ``embedding``,
        ``rerank``.

    Returns the final ``everos.json`` dict (also written to disk).
    """
    hermes_home = Path(hermes_home)
    hermes_home.mkdir(parents=True, exist_ok=True)

    inputs = dict(inputs or {})
    if interactive and not inputs:
        inputs = _prompt_wizard(config)

    mode = inputs.get("mode") or config.get("mode") or "platform"
    if mode not in ("platform", "oss"):
        raise ValueError(f"unknown EverOS mode: {mode!r}")

    user_id = inputs.get("user_id") or config.get("user_id") or _DEFAULT_USER_ID
    agent_id = inputs.get("agent_id") or config.get("agent_id") or _DEFAULT_AGENT_ID
    agent_track_enabled = bool(
        inputs.get("agent_track_enabled", config.get("agent_track_enabled", False))
    )
    everos_root = inputs.get("everos_root") or config.get("everos_root")

    if mode == "platform":
        api_url = inputs.get("api_url") or config.get("api_url") or _DEFAULT_API_URL
        payload = build_everos_json(
            api_url=api_url,
            mode=mode,
            user_id=user_id,
            agent_id=agent_id,
            agent_track_enabled=agent_track_enabled,
            everos_root=None,
        )
        _write_everos_json(hermes_home, payload)
        return payload

    # mode == "oss"
    if everos_root:
        oss_root = Path(everos_root).expanduser()
    else:
        oss_root = Path.home() / ".everos"
    llm = inputs.get("llm") or {}
    embedding = inputs.get("embedding") or {}
    rerank = inputs.get("rerank")
    write_everos_toml(oss_root, llm=llm, embedding=embedding, rerank=rerank)
    seed_ome_toml(oss_root, agent_track=agent_track_enabled)

    api_url = inputs.get("api_url") or config.get("api_url") or _DEFAULT_API_URL
    payload = build_everos_json(
        api_url=api_url,
        mode=mode,
        user_id=user_id,
        agent_id=agent_id,
        agent_track_enabled=agent_track_enabled,
        everos_root=str(oss_root),
    )
    _write_everos_json(hermes_home, payload)
    return payload


def _prompt_wizard(config: dict[str, Any]) -> dict[str, Any]:
    """Interactively prompt the operator for setup values on stdin."""
    inputs: dict[str, Any] = {}

    default_mode = config.get("mode") or "platform"
    mode = _prompt("Mode (platform/oss)", default=default_mode).strip().lower()
    if mode:
        inputs["mode"] = mode

    if inputs.get("mode", default_mode) == "platform":
        default_api_url = config.get("api_url") or _DEFAULT_API_URL
        api_url = _prompt("EverOS API URL", default=default_api_url).strip()
        if api_url:
            inputs["api_url"] = api_url
    else:
        default_root = config.get("everos_root") or str(Path.home() / ".everos")
        root = _prompt("EverOS root", default=default_root).strip()
        if root:
            inputs["everos_root"] = root
        inputs["llm"] = _prompt_provider("LLM", config.get("llm") or {})
        inputs["embedding"] = _prompt_provider(
            "Embedding", config.get("embedding") or {}
        )
        rerank = _prompt_provider("Rerank", config.get("rerank") or {}, allow_skip=True)
        if rerank is not None:
            inputs["rerank"] = rerank

    default_user_id = config.get("user_id") or _DEFAULT_USER_ID
    user_id = _prompt("User id", default=default_user_id).strip()
    if user_id:
        inputs["user_id"] = user_id

    default_agent_id = config.get("agent_id") or _DEFAULT_AGENT_ID
    agent_id = _prompt("Agent id", default=default_agent_id).strip()
    if agent_id:
        inputs["agent_id"] = agent_id

    default_track = "y" if config.get("agent_track_enabled") else "n"
    agent_track_raw = (
        _prompt("Enable agent track? (y/n)", default=default_track).strip().lower()
    )
    inputs["agent_track_enabled"] = agent_track_raw in ("y", "yes", "true", "1")

    return inputs


def _prompt_provider(
    label: str,
    existing: dict[str, Any],
    *,
    allow_skip: bool = False,
) -> dict[str, Any] | None:
    """Prompt for a provider section (model / api_key / base_url + knobs)."""
    if allow_skip:
        skip = (
            _prompt(f"Configure {label}? (y/n)", default="n" if not existing else "y")
            .strip()
            .lower()
        )
        if skip not in ("y", "yes", "true", "1"):
            return None
    result: dict[str, Any] = {}
    model = _prompt(f"{label} model", default=existing.get("model", "")).strip()
    if model:
        result["model"] = model
    api_key = _prompt(f"{label} api_key", default=existing.get("api_key", "")).strip()
    if api_key:
        result["api_key"] = api_key
    base_url = _prompt(
        f"{label} base_url", default=existing.get("base_url", "")
    ).strip()
    if base_url:
        result["base_url"] = base_url
    return result or (dict(existing) if existing else None)


def _prompt(label: str, default: str | None = None) -> str:
    """Read one line from stdin with a ``[default]`` hint."""
    suffix = f" [{default}]" if default else ""
    print(f"  {label}{suffix}: ", end="", flush=True)
    try:
        import sys

        val = sys.stdin.readline().rstrip("\n")
    except Exception:
        val = ""
    return val or (default or "")
