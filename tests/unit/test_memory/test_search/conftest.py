"""Shared fixtures for ``memory.search`` unit tests.

The project default is ``EVEROS_SEARCH__VECTOR_STRATEGY=maxsim_atomic`` —
that path queries both the ``atomic_fact`` table and the ``episode`` table
to do MaxSim. The existing VECTOR-route tests in ``test_manager.py`` were
written against the legacy single-vector ``episode`` path and stub only the
episode recaller (atomic_fact recaller is a no-data stub).

Force the legacy ``episode`` strategy by default for these tests so they
keep asserting against the dense-recall path they were designed to cover.
MaxSim-specific tests opt back into ``maxsim_atomic`` by overriding the env
var inside their own body.
"""

from __future__ import annotations

import pytest

from everos.config.settings import load_settings


@pytest.fixture(autouse=True)
def _force_episode_vector_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVEROS_SEARCH__VECTOR_STRATEGY", "episode")
    load_settings.cache_clear()
    yield
    load_settings.cache_clear()
