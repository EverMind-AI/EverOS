"""Model-based test of OME's dispatch gates: enabled, applies_to and the Counter.

The dispatcher decides, per event and strategy, whether a run is enqueued:
the strategy must be enabled (or forced), its ``applies_to`` must hold, and
its ``Counter`` gate must pass -- the threshold-th event in a bucket passes
and resets the bucket, no sooner than ``cooldown_seconds`` after the last
pass. ``inspect`` is documented as the dry-run twin of ``dispatch``.

What is real: :class:`EventDispatcher`, :class:`StrategyRegistry`,
:class:`CounterStore` on a temp OME SQLite file. What is replaced: the
store's clock, so a cooldown can be crossed without waiting.

Rules: emit an event (bucket field and applies flag drawn), a forced
manual dispatch at the disabled strategy, moving the clock, and a hot
reload that changes a threshold (counters persist across it, so a bucket
can be over-armed).

Invariants after every rule, from the gate's and store's docstrings:

1. ``dispatch`` routes exactly the strategies the gate rules say;
2. ``inspect`` predicts ``dispatch`` for the same event (``will_run``);
3. the persisted counters are the model's: reset to 0 on a pass,
   incremented otherwise, per bucket, buckets independent;
4. ``inspect`` does not mutate the store.

The second law failed before the fix: ``inspect`` compared the counter
with the threshold but ignored the cooldown, so during a cooldown window
it reported ``will_run=True`` for an event ``dispatch`` would not route.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as dt
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any
from unittest import mock

from hypothesis import HealthCheck, currently_in_test_context, settings
from hypothesis import event as _hypothesis_event
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from everos.component.utils.datetime import get_utc_now
from everos.infra.ome import BaseEvent, Immediate, offline_strategy
from everos.infra.ome._dispatch.dispatcher import EventDispatcher
from everos.infra.ome._dispatch.registry import StrategyRegistry
from everos.infra.ome._stores import counter as counter_mod
from everos.infra.ome._stores.counter import CounterStore
from everos.infra.ome._stores.storage import OMEStorage
from everos.infra.ome.gates import Counter

settings.register_profile(
    "ci",
    max_examples=40,
    stateful_step_count=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "deep",
    max_examples=1000,
    stateful_step_count=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))

USERS = ("u1", "u2", None)
CLOCK_STEPS = (1, 30, 61, 600)


def event(label: str) -> None:
    if currently_in_test_context():
        _hypothesis_event(label)


class _Ev(BaseEvent):
    user: str | None = None
    flag: bool = True


@offline_strategy(
    name="per_user",
    trigger=Immediate(on=[_Ev]),
    emits=[],
    gate=Counter(threshold=3, cooldown_seconds=0, event_field="user"),
)
async def _per_user(event: BaseEvent, ctx: Any) -> None:
    pass


@offline_strategy(
    name="cooled",
    trigger=Immediate(on=[_Ev]),
    emits=[],
    gate=Counter(threshold=2, cooldown_seconds=60),
)
async def _cooled(event: BaseEvent, ctx: Any) -> None:
    pass


@offline_strategy(
    name="flagged",
    trigger=Immediate(on=[_Ev]),
    emits=[],
    applies_to="flag",
)
async def _flagged(event: BaseEvent, ctx: Any) -> None:
    pass


@offline_strategy(
    name="disabled",
    trigger=Immediate(on=[_Ev]),
    emits=[],
    enabled=False,
)
async def _disabled(event: BaseEvent, ctx: Any) -> None:
    pass


class _Clock:
    def __init__(self) -> None:
        self.offset = dt.timedelta()

    def now(self) -> dt.datetime:
        return get_utc_now() + self.offset


class CounterGateMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.loop = asyncio.new_event_loop()
        self.tmp = Path(tempfile.mkdtemp(prefix="counter_gate_"))
        self.clock = _Clock()
        self._patch = mock.patch.object(counter_mod, "get_utc_now", self.clock.now)
        self._patch.start()
        self.storage = OMEStorage(self.tmp / "ome.db")
        self.run(self.storage.init())
        self.store = CounterStore(storage=self.storage)
        self.registry = StrategyRegistry()
        for s in (_per_user, _cooled, _flagged, _disabled):
            self.registry.register(s)
        self.registry.validate()
        self.dispatcher = EventDispatcher(
            registry=self.registry, counter_store=self.store
        )
        # the model: (strategy, bucket) -> [counter, last_passed]
        self.buckets: dict[tuple[str, str], list[Any]] = {}
        self.threshold = {"per_user": 3, "cooled": 2}
        self.cooldown = {"per_user": 0, "cooled": 60}
        self.trace: list[str] = []

    def run(self, coro: Any) -> Any:
        return self.loop.run_until_complete(coro)

    def teardown(self) -> None:
        self._patch.stop()
        self.run(self.storage.close()) if hasattr(self.storage, "close") else None
        self.loop.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ── the model ──────────────────────────────────────────────────────────

    def _gate_passes(self, name: str, bucket: str, *, apply: bool) -> bool:
        """The Counter gate's rule; mutates the model when ``apply``."""
        now = self.clock.now()
        counter, last = self.buckets.get((name, bucket), [0, None])
        counter += 1
        cooldown = self.cooldown[name]
        cooldown_ok = (
            cooldown == 0
            or last is None
            or now - last >= dt.timedelta(seconds=cooldown)
        )
        passed = counter >= self.threshold[name] and cooldown_ok
        if apply:
            self.buckets[(name, bucket)] = [
                0 if passed else counter,
                now if passed else last,
            ]
        return passed

    def _expected_routes(self, ev: _Ev, *, apply: bool) -> set[str]:
        routes: set[str] = set()
        if self._gate_passes(
            "per_user", str(ev.user) if ev.user is not None else "__none__", apply=apply
        ):
            routes.add("per_user")
        if self._gate_passes("cooled", "__all__", apply=apply):
            routes.add("cooled")
        if ev.flag:
            routes.add("flagged")
        return routes

    # ── rules ──────────────────────────────────────────────────────────────

    @rule(user=st.sampled_from(USERS), flag=st.booleans())
    def emit(self, user: str | None, flag: bool) -> None:
        ev = _Ev(user=user, flag=flag)
        self.trace.append(f"emit user={user} flag={flag} clock=+{self.clock.offset}")
        predicted = {
            r.strategy_name: r.will_run for r in self.run(self.dispatcher.inspect(ev))
        }
        want = self._expected_routes(ev, apply=False)
        assert set(predicted) == {"per_user", "cooled", "flagged", "disabled"}
        assert {n for n, w in predicted.items() if w} == want, (
            "inspect disagrees with the gate rules",
            predicted,
            want,
            "\n".join(self.trace[-10:]),
        )
        got = {meta.name for meta, _ in self.run(self.dispatcher.dispatch(ev))}
        want = self._expected_routes(ev, apply=True)
        assert got == want, (got, want, "\n".join(self.trace[-10:]))
        for name in sorted(want):
            event(f"routed: {name}")
        if not want:
            event("routed: nothing")

    @rule()
    def forced_manual_run_of_the_disabled_strategy(self) -> None:
        ev = _Ev(user="u1", flag=True)
        got = self.run(
            self.dispatcher.dispatch(ev, force_enabled=True, strategy_filter="disabled")
        )
        assert [m.name for m, _ in got] == ["disabled"]
        event("forced disabled strategy")

    @rule(seconds=st.sampled_from(CLOCK_STEPS))
    def advance_clock(self, seconds: int) -> None:
        self.clock.offset += dt.timedelta(seconds=seconds)
        self.trace.append(f"clock +{seconds}s")
        event(f"clock +{seconds}s")

    @rule(threshold=st.integers(1, 4))
    def hot_reload_threshold(self, threshold: int) -> None:
        meta = self.registry.get("per_user")
        new_meta = dataclasses.replace(
            meta,
            gate=Counter(threshold=threshold, cooldown_seconds=0, event_field="user"),
        )
        self.registry.replace("per_user", new_meta)
        self.threshold["per_user"] = threshold
        self.trace.append(f"per_user threshold -> {threshold}")
        event("hot reload threshold")

    # ── invariants ─────────────────────────────────────────────────────────

    @invariant()
    def store_matches_model(self) -> None:
        for (name, bucket), (counter, _) in self.buckets.items():
            got = self.run(self.store.get_progress(name, bucket))
            assert got == counter, (
                name,
                bucket,
                got,
                counter,
                "\n".join(self.trace[-10:]),
            )

    @invariant()
    def inspect_is_read_only(self) -> None:
        ev = _Ev(user="u1", flag=True)
        before = {k: self.run(self.store.get_progress(*k)) for k in self.buckets}
        self.run(self.dispatcher.inspect(ev))
        after = {k: self.run(self.store.get_progress(*k)) for k in self.buckets}
        assert before == after


TestCounterGateMachine = CounterGateMachine.TestCase
