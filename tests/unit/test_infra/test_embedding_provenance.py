"""Startup refuses an index the configured embedding cannot have produced.

Three things are compared at startup, and this module owns two of them: the
model's real output width against the configured vector column, and the model
name against the one recorded when the index was built. (The third -- the
column's own width against configuration -- belongs to each backend's schema
verification.)

Every case drives ``EmbeddingProvenance.verify`` with a stub store, because
what is under test is the decision, not where a backend keeps the record;
``test_embedding_provenance_store.py`` covers the storage half against a real
table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from everos.config import load_settings
from everos.core.errors import EmbeddingProvenanceError
from everos.infra.persistence.index import EmbeddingProvenance, _provenance
from tests.fakes import UnusedPort

if TYPE_CHECKING:
    from collections.abc import Iterator


class _Store:
    """Records what was written so a test can assert nothing was."""

    def __init__(self, recorded: str | None = None, *, read_raises: bool = False):
        self._recorded = recorded
        self._read_raises = read_raises
        self.writes: list[str] = []

    async def read_embedding_model(self) -> str | None:
        if self._read_raises:
            raise OSError("backend unreachable")
        return self._recorded

    async def write_embedding_model(self, model: str) -> None:
        self.writes.append(model)
        self._recorded = model


class _Probe:
    """Controls the measured width and counts how often it was measured."""

    def __init__(self) -> None:
        self.calls = 0
        self.returns: int | None = None


class _ProbedProvenance(EmbeddingProvenance):
    """The real check with the network probe answered by a :class:`_Probe`."""

    def __init__(self, probe: _Probe, *, configured: bool) -> None:
        super().__init__(load_settings().embedding, configured=configured)
        self._probe = probe

    async def probe_dimension(self) -> int | None:
        self._probe.calls += 1
        return self._probe.returns


@pytest.fixture
def probe() -> _Probe:
    return _Probe()


@pytest.fixture(autouse=True)
def _configured(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A complete embedding configuration for the settings under check."""
    monkeypatch.setenv("EVEROS_EMBEDDING__MODEL", "fake-768")
    monkeypatch.setenv("EVEROS_EMBEDDING__API_KEY", "key")
    monkeypatch.setenv("EVEROS_EMBEDDING__BASE_URL", "http://127.0.0.1:1/v1")
    yield


def _provenance_for(probe: _Probe, *, configured: bool = True) -> _ProbedProvenance:
    """Built after the test set its env, as the bootstrap builds it at startup."""
    load_settings.cache_clear()
    return _ProbedProvenance(probe, configured=configured)


def _set_dim(monkeypatch: pytest.MonkeyPatch, dim: int) -> None:
    monkeypatch.setenv("EVEROS_EMBEDDING__DIM", str(dim))


async def test_narrower_model_aborts_naming_both_widths(
    monkeypatch: pytest.MonkeyPatch, probe
) -> None:
    """issue #434: nothing can ever be stored, so refuse to start.

    The message has to carry both numbers and the rebuild step -- the tables
    were already created at the configured width by the schema check that runs
    just before, so fixing the config alone leaves a mismatched table behind.
    """
    _set_dim(monkeypatch, 1024)
    probe.returns = 768
    with pytest.raises(EmbeddingProvenanceError) as excinfo:
        await _provenance_for(probe).verify(_Store())
    message = str(excinfo.value)
    assert "768" in message
    assert "1024" in message
    assert "cascade rebuild" in message


async def test_wider_model_warns_and_starts(
    monkeypatch: pytest.MonkeyPatch, probe
) -> None:
    """Truncation is a supported arrangement, so warn rather than refuse."""
    _set_dim(monkeypatch, 1024)
    probe.returns = 1536
    warnings = await _provenance_for(probe).verify(_Store())
    assert any("truncated" in warning for warning in warnings)


async def test_changed_model_aborts_naming_both(
    monkeypatch: pytest.MonkeyPatch, probe
) -> None:
    _set_dim(monkeypatch, 768)
    probe.returns = 768
    store = _Store(recorded="nomic-embed-text")
    with pytest.raises(EmbeddingProvenanceError) as excinfo:
        await _provenance_for(probe).verify(store)
    message = str(excinfo.value)
    assert "nomic-embed-text" in message
    assert "fake-768" in message
    assert store.writes == []


async def test_unattested_index_is_recorded(
    monkeypatch: pytest.MonkeyPatch, probe
) -> None:
    _set_dim(monkeypatch, 768)
    probe.returns = 768
    store = _Store(recorded=None)
    await _provenance_for(probe).verify(store)
    assert store.writes == ["fake-768"]


async def test_matching_model_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, probe
) -> None:
    _set_dim(monkeypatch, 768)
    probe.returns = 768
    store = _Store(recorded="fake-768")
    assert await _provenance_for(probe).verify(store) == []
    assert store.writes == []


async def test_unconfigured_embedding_skips_everything(
    monkeypatch: pytest.MonkeyPatch, probe
) -> None:
    """keyword-only: no provider can be built, so there is nothing to protect.

    The gate is the factory's own three-field condition, not the model name:
    ``default.toml`` ships a model and a base_url with an empty api_key, so a
    model-only gate would attest a model that never produced a vector -- and
    then refuse to start once a real one was configured.
    """
    store = _Store(recorded="fake-768")
    await _provenance_for(probe, configured=False).verify(store)
    assert probe.calls == 0
    assert store.writes == []


async def test_probe_failure_starts_and_keeps_the_other_checks(
    monkeypatch: pytest.MonkeyPatch, probe
) -> None:
    """An unreachable embedding service must not look like a bad index.

    The model comparison does not need the probe, so it still fires -- a
    service outage is not a way to smuggle a model change past the check.
    """
    _set_dim(monkeypatch, 768)
    probe.returns = None
    provenance = _provenance_for(probe)
    await provenance.verify(_Store(recorded="fake-768"))
    assert provenance.last_probe is None

    with pytest.raises(EmbeddingProvenanceError, match="nomic"):
        await provenance.verify(_Store(recorded="nomic"))


async def test_unreadable_record_is_not_overwritten(
    monkeypatch: pytest.MonkeyPatch, probe
) -> None:
    """Writing over a record we could not read would bury it for good.

    The old model name is still in there; overwriting it with the current
    configuration makes this and every later model change undetectable.
    """
    _set_dim(monkeypatch, 768)
    probe.returns = 768
    store = _Store(read_raises=True)
    warnings = await _provenance_for(probe).verify(store)
    assert store.writes == []
    assert any("could not read" in warning for warning in warnings)


async def test_both_mismatches_are_reported_together(
    monkeypatch: pytest.MonkeyPatch, probe
) -> None:
    """Two wrong settings should cost one restart, not two."""
    _set_dim(monkeypatch, 1024)
    probe.returns = 768
    store = _Store(recorded="nomic-embed-text")
    with pytest.raises(EmbeddingProvenanceError) as excinfo:
        await _provenance_for(probe).verify(store)
    message = str(excinfo.value)
    assert "configured dim is 1024" in message
    assert "nomic-embed-text" in message


async def test_dimensions_mismatch_only_warns(
    monkeypatch: pytest.MonkeyPatch, probe
) -> None:
    """``dimensions`` cannot override ``dim``; tables are built at ``dim``."""
    _set_dim(monkeypatch, 768)
    monkeypatch.setenv("EVEROS_EMBEDDING__DIMENSIONS", "512")
    probe.returns = 768
    store = _Store(recorded="fake-768")
    warnings = await _provenance_for(probe).verify(store)
    assert any("dimensions=512" in warning for warning in warnings)


async def test_last_probe_reports_the_measured_width(
    monkeypatch: pytest.MonkeyPatch, probe
) -> None:
    """``/health`` reads this; a stale ``None`` makes it report embed: false."""
    _set_dim(monkeypatch, 768)
    probe.returns = 768
    provenance = _provenance_for(probe)
    await provenance.verify(_Store(recorded="fake-768"))
    assert provenance.last_probe == 768


def test_probe_defeats_client_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe must ask for more than any model returns.

    ``OpenAIEmbeddingProvider`` slices to ``dim`` before returning, so asking
    for the configured width would report exactly that width no matter how
    long the real vector was -- and a too-wide model would be invisible.
    """
    captured: dict[str, int | None] = {}

    def _fake_build(settings: object, *, dim: int | None = None) -> object:
        captured["dim"] = dim
        raise RuntimeError("stop here")

    monkeypatch.setattr(_provenance, "build_embedding_provider", _fake_build)
    monkeypatch.setenv("EVEROS_EMBEDDING__DIM", "768")

    import asyncio

    provenance = EmbeddingProvenance(load_settings().embedding, configured=True)
    assert asyncio.run(provenance.probe_dimension()) is None
    assert captured["dim"] is not None
    assert captured["dim"] > 100_000


async def test_dim_disagreement_between_the_two_readers_aborts() -> None:
    """The record classes and Settings must not quietly diverge on the width.

    The record classes resolve ``[embedding] dim`` with a narrow reader of
    their own, because the width has to be known while they are imported.
    Two readers can disagree -- a different root, a key one of them cannot
    parse -- and the symptom would otherwise be tables silently built at the
    wrong width. The width is frozen at import, so the disagreement is
    constructed from the Settings side: the dim the index is built with.
    """
    from everos.infra.persistence.index import DerivedIndex
    from everos.infra.persistence.records.episode import Episode

    baked = Episode.to_arrow_schema().field("vector").type.list_size
    index = DerivedIndex(UnusedPort("index backend"), embedding_dim=baked + 256)

    with pytest.raises(EmbeddingProvenanceError) as excinfo:
        index.assert_record_dim_matches_settings()
    message = str(excinfo.value)
    assert str(baked) in message
    assert str(baked + 256) in message


async def test_matching_dim_passes() -> None:
    """The guard must not fire on the normal case."""
    from everos.infra.persistence.index import DerivedIndex
    from everos.infra.persistence.records.episode import Episode

    baked = Episode.to_arrow_schema().field("vector").type.list_size
    index = DerivedIndex(UnusedPort("index backend"), embedding_dim=baked)
    index.assert_record_dim_matches_settings()


async def test_a_refused_start_does_not_attest(
    monkeypatch: pytest.MonkeyPatch, probe
) -> None:
    """Writing on the way out of a refusal buries what built the index.

    The operator story: a wrong dim is refused, but the refusal stamps the
    rejected model onto an index some *other* model built. Putting the real
    model back is then reported as "index built with <the rejected one>" and
    sends them to rebuild for nothing.
    """
    _set_dim(monkeypatch, 1024)
    probe.returns = 512
    store = _Store(recorded=None)
    with pytest.raises(EmbeddingProvenanceError):
        await _provenance_for(probe).verify(store)
    assert store.writes == []
    assert await store.read_embedding_model() is None


async def test_read_only_phases_do_not_attest(
    monkeypatch: pytest.MonkeyPatch, probe
) -> None:
    """Only a phase holding the index lock may write.

    `_runtime` documents that every command writing the index passes
    `exclusive=True`; `status`, `fix` and `backfill` do not, and on a shared
    Milvus cluster this would be a write issued from a read-only phase.
    """
    _set_dim(monkeypatch, 768)
    probe.returns = 768
    store = _Store(recorded=None)
    findings = await _provenance_for(probe).verify(store, may_attest=False)
    assert store.writes == []
    assert any("no recorded embedding model" in finding for finding in findings)


async def test_a_failed_write_does_not_stop_startup(
    monkeypatch: pytest.MonkeyPatch, probe
) -> None:
    """A read-only mount is not a reason to refuse to open an index."""
    _set_dim(monkeypatch, 768)
    probe.returns = 768

    class _ReadOnly(_Store):
        async def write_embedding_model(self, model: str) -> None:
            raise PermissionError("read-only index mount")

    findings = await _provenance_for(probe).verify(_ReadOnly(recorded=None))
    assert any("could not record" in finding for finding in findings)


def test_the_probe_does_not_inherit_the_write_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One integer must not cost the write path's 30s x 3 on every command."""
    captured: dict[str, object] = {}

    def _fake_build(settings: object, *, dim: int | None = None) -> object:
        captured["timeout"] = settings.timeout_seconds  # type: ignore[attr-defined]
        captured["retries"] = settings.max_retries  # type: ignore[attr-defined]
        raise RuntimeError("stop here")

    monkeypatch.setattr(_provenance, "build_embedding_provider", _fake_build)
    monkeypatch.setenv("EVEROS_EMBEDDING__TIMEOUT_SECONDS", "30.0")
    monkeypatch.setenv("EVEROS_EMBEDDING__MAX_RETRIES", "3")

    import asyncio

    provenance = EmbeddingProvenance(load_settings().embedding, configured=True)
    asyncio.run(provenance.probe_dimension())
    assert captured["retries"] == 0
    assert captured["timeout"] < 30.0
