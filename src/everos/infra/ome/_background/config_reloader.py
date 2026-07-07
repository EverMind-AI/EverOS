"""Config hot-reload — watchfiles + tomllib + shallow merge.

Hot-updatable fields: enabled / max_retries / gate / cron / idle_seconds /
scan_interval_seconds. Trigger type swap (Immediate ↔ Cron ↔ Idle),
event subscription (Immediate.on / Idle.on), and Idle.event_field
remain immutable — these define strategy routing and changing them
requires a code change and redeploy.

Per-strategy two-phase commit: enabled is applied independently for
emergency-stop semantics; max_retries / gate / trigger parameters
form one atomic group that fully rolls back on any failure inside it.
"""

from __future__ import annotations

import asyncio
import tomllib
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError
from watchfiles import awatch

from everos.core.observability.logging import get_logger
from everos.infra.ome._dispatch.registry import StrategyRegistry
from everos.infra.ome.config import StrategyOverride, TomlRoot
from everos.infra.ome.decorator import StrategyMeta
from everos.infra.ome.gates import Counter
from everos.infra.ome.triggers import Cron, Idle, Trigger

if TYPE_CHECKING:
    from everos.infra.ome.engine import OfflineEngine

logger = get_logger(__name__)


class _SkipAtomicGroupError(Exception):
    """Internal sentinel raised when the non-enabled atomic group for
    one strategy must be skipped without affecting other strategies.
    """


def _apply_enabled(
    meta: StrategyMeta,
    override: StrategyOverride,
    name: str,
    registry: StrategyRegistry,
) -> StrategyMeta:
    """Step 1: apply `enabled` independently — never blocked by other fields."""
    if override.enabled is None or override.enabled == meta.enabled:
        return meta
    new_meta = replace(meta, enabled=override.enabled)
    registry.replace(name, new_meta)
    return new_meta


def _build_atomic_meta(
    meta: StrategyMeta,
    override: StrategyOverride,
) -> tuple[StrategyMeta, Trigger]:
    """Step 2 pure-compute: build (new_meta, new_trigger) from current state.

    Raises `_SkipAtomicGroupError` on type mismatches or invalid gate intros.
    No registry / engine writes happen here.
    """
    new_meta = meta
    new_trigger: Trigger = meta.trigger

    if override.max_retries is not None:
        new_meta = replace(new_meta, max_retries=override.max_retries)

    if override.gate is not None:
        # Introducing a gate on a strategy that has none requires an explicit
        # threshold — silently defaulting to 1 would mean "fire on every
        # event", which is almost certainly not what the user intended.
        if meta.gate is None and override.gate.threshold is None:
            raise _SkipAtomicGroupError(
                "introducing a gate requires explicit threshold"
            )
        base = meta.gate.model_dump() if meta.gate is not None else {}
        for k, v in override.gate.model_dump(exclude_unset=True).items():
            if v is not None:
                base[k] = v
        new_meta = replace(new_meta, gate=Counter(**base))

    if override.cron is not None:
        if not isinstance(meta.trigger, Cron):
            raise _SkipAtomicGroupError(
                f"cron given on non-Cron strategy "
                f"(actual: {type(meta.trigger).__name__})"
            )
        new_trigger = Cron(expr=override.cron)

    if override.idle_seconds is not None or override.scan_interval_seconds is not None:
        if not isinstance(meta.trigger, Idle):
            raise _SkipAtomicGroupError(
                f"idle_* given on non-Idle strategy "
                f"(actual: {type(meta.trigger).__name__})"
            )
        updates: dict[str, int] = {}
        if override.idle_seconds is not None:
            updates["idle_seconds"] = override.idle_seconds
        if override.scan_interval_seconds is not None:
            updates["scan_interval_seconds"] = override.scan_interval_seconds
        # model_validate (not model_copy) re-runs Idle._validate_event_field on
        # the merged dict; model_copy(update=...) would skip it and let an
        # invalid event_field reach the registry.
        new_trigger = Idle.model_validate({**meta.trigger.model_dump(), **updates})

    if new_trigger is not meta.trigger:
        new_meta = replace(new_meta, trigger=new_trigger)

    return new_meta, new_trigger


def _needs_aps_reschedule(old_trigger: Trigger, new_trigger: Trigger) -> bool:
    """True iff scheduler-driving fields changed (cron expr / Idle scan_interval)."""
    if new_trigger is old_trigger:
        return False
    if isinstance(new_trigger, Cron) and isinstance(old_trigger, Cron):
        return new_trigger.expr != old_trigger.expr
    if isinstance(new_trigger, Idle) and isinstance(old_trigger, Idle):
        return new_trigger.scan_interval_seconds != old_trigger.scan_interval_seconds
    return False


def _maybe_reschedule_aps(
    engine: OfflineEngine, name: str, new_trigger: Trigger
) -> None:
    """Push the new trigger's APS-relevant fields to the scheduler."""
    if isinstance(new_trigger, Cron):
        engine.reschedule_cron_job(name, new_trigger.expr)
    elif isinstance(new_trigger, Idle):
        engine.reschedule_idle_job(
            name, scan_interval_seconds=new_trigger.scan_interval_seconds
        )


def _apply_one_strategy(
    name: str,
    override: StrategyOverride,
    registry: StrategyRegistry,
    engine: OfflineEngine,
) -> None:
    """Two-phase commit for one strategy: enabled, then atomic group."""
    meta = registry.get(name)
    meta = _apply_enabled(meta, override, name, registry)

    try:
        new_meta, new_trigger = _build_atomic_meta(meta, override)
        if _needs_aps_reschedule(meta.trigger, new_trigger):
            _maybe_reschedule_aps(engine, name, new_trigger)
        registry.replace(name, new_meta)
    except Exception as e:
        # User-fixable config error (typo / type mismatch / APS runtime
        # failure) — log + skip this strategy's atomic group, never the loop.
        logger.warning(
            "strategy_atomic_group_skipped",
            strategy_name=name,
            error_type=type(e).__name__,
            exc_info=True,
        )


def apply_overrides(
    registry: StrategyRegistry,
    root: TomlRoot,
    engine: OfflineEngine,
) -> None:
    """Shallow-merge TomlRoot overrides into registry strategies in place.

    Two-phase per-strategy semantics:
      Step 1 (enabled): applied independently — emergency-stop must
        never be blocked by a typo in another field.
      Step 2 (max_retries / gate / trigger params): applied as an atomic
        group. Any failure (type mismatch, invalid cron, APS reschedule
        error, ...) rolls the whole group back to the prior values.
    """
    known = {m.name for m in registry.all()}
    for name, override in root.strategies.items():
        if name not in known:
            logger.warning("config_override_unknown_strategy", strategy_name=name)
            continue
        _apply_one_strategy(name, override, registry, engine)


class ConfigReloader:
    """Watch a TOML file and apply overrides on change."""

    def __init__(
        self,
        *,
        config_path: Path,
        registry: StrategyRegistry,
        engine: OfflineEngine,
        debounce_ms: int = 1600,
    ) -> None:
        self._path = config_path
        self._registry = registry
        self._engine = engine
        self._debounce_ms = debounce_ms
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Fire-and-forget the watch loop. Idempotent: raises on double-start."""
        if self._path is None:
            return
        if not self._path.exists():
            raise FileNotFoundError(
                f"{self._path} not found. "
                "Run `everos init` to create configuration files."
            )
        if self._task is not None and not self._task.done():
            raise RuntimeError("ConfigReloader already started")
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Cancel the watch task and await it; safe to call multiple times."""
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _loop(self) -> None:
        """Initial load + per-FS-change reload; survives single-iteration failures."""
        try:
            await self._load_once()
        except Exception:
            logger.exception("config_reload_iteration_failed")
        async for _changes in awatch(self._path, debounce=self._debounce_ms):
            try:
                await self._load_once()
            except Exception:
                logger.exception("config_reload_iteration_failed")

    async def _load_once(self) -> None:
        """Read TOML off the loop, parse + validate, apply overrides."""

        def _read_and_parse() -> TomlRoot:
            with open(self._path, "rb") as f:
                content = f.read()
            parsed = tomllib.loads(content.decode("utf-8"))
            return TomlRoot.model_validate(parsed)

        try:
            root = await asyncio.to_thread(_read_and_parse)
        except (OSError, tomllib.TOMLDecodeError, ValidationError) as e:
            logger.warning(
                "config_reload_failed",
                error_type=type(e).__name__,
                error=str(e),
                path=str(self._path),
            )
            return
        apply_overrides(self._registry, root, self._engine)
        logger.info("config_reloaded", path=str(self._path))
