"""Resume refuses to merge two experiments, and re-runs only what actually changed.

Before this, resume skipped work on row index and an `add.done` marker alone. Change the
model, the data or the retrieval parameters in a directory that already held results and
the two runs merged into one report reproducible from neither -- and the fresh
`run_spec.json` overwrote the old one, erasing the evidence that they had differed.

The identity is two values, not one, because the halves cost differently: re-ingesting a
store is hours of LLM calls, re-reading it is minutes. These tests pin that split. In
particular `top_k` must NOT invalidate a store -- it caps how many episodes reach the
answer prompt and has no bearing on what was ingested -- and neither must an `everalgo`
version bump, whose effect on extraction quality was measured and did not move the
benchmark.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import run as run_mod  # noqa: E402
from config import (  # noqa: E402
    BenchmarkConfig,
    ServingSpec,
    identity_diff,
    ingest_identity,
    read_identity,
)


def _config(tmp_path: Path, body: str = "seed", **over: object) -> BenchmarkConfig:
    data = tmp_path / "data.json"
    data.write_text(body, encoding="utf-8")
    fields: dict[str, object] = {
        "data_path": str(data),
        "adapter": "locomo",
        "methods": "hybrid",
        "top_k": 20,
        "answer_model": "gpt-4.1-mini",
        "judge_model": "gpt-4o-mini",
    }
    fields.update(over)
    return BenchmarkConfig(**fields)  # type: ignore[arg-type]


def _serving(backbone: str = "gpt-4.1-mini", decider: str = "") -> list[ServingSpec]:
    out = [ServingSpec(role="extraction", model=backbone, endpoint="https://e/v1")]
    if decider:
        out.append(ServingSpec(role="decider", model=decider, endpoint="https://d/v1"))
    return out


def _seed_spec(out_dir: Path, cfg: BenchmarkConfig, serving: list[ServingSpec]) -> None:
    """Write the run_spec.json a previous run in this directory would have left."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_spec.json").write_text(
        json.dumps(
            {
                "run_name": "prev",
                "ingest_identity": ingest_identity(cfg, "locomo", serving),
                "read_identity": read_identity(cfg, serving),
            }
        ),
        encoding="utf-8",
    )


def _check(
    out_dir: Path, cfg: BenchmarkConfig, serving: list[ServingSpec], force: bool = False
):
    return run_mod._check_resume_identity(
        out_dir,
        ingest_identity(cfg, "locomo", serving),
        read_identity(cfg, serving),
        force,
    )


# ── the two halves are separate ────────────────────────────────────────────────
def test_an_unchanged_run_resumes_silently(tmp_path: Path) -> None:
    out = tmp_path / "run"
    cfg, srv = _config(tmp_path), _serving()
    _seed_spec(out, cfg, srv)
    assert _check(out, cfg, srv) == []


def test_a_fresh_directory_has_nothing_to_check(tmp_path: Path) -> None:
    assert _check(tmp_path / "brand-new", _config(tmp_path), _serving()) == []


@pytest.mark.parametrize(
    "over,serving_kw,expect",
    [
        ({"body": "CHANGED"}, {}, "data_digest"),
        ({"adapter": "longmemeval"}, {}, "adapter"),
        ({}, {"backbone": "other-model"}, "extraction_backbone"),
    ],
)
def test_a_different_ingest_refuses(
    tmp_path: Path, over: dict, serving_kw: dict, expect: str
) -> None:
    """Data, adapter and extraction backbone each invalidate the stored memories."""
    out = tmp_path / "run"
    _seed_spec(out, _config(tmp_path), _serving())
    body = over.pop("body", "seed")
    with pytest.raises(run_mod.ResumeIdentityError) as err:
        _check(out, _config(tmp_path, body=body, **over), _serving(**serving_kw))
    assert expect in str(err.value)


def test_the_override_is_allowed_and_recorded(tmp_path: Path) -> None:
    """A forced reuse must leave a trace; a silent one is the defect, not the flag."""
    out = tmp_path / "run"
    _seed_spec(out, _config(tmp_path), _serving())
    notes = _check(out, _config(tmp_path, body="CHANGED"), _serving(), force=True)
    assert any("FORCED" in n and "data_digest" in n for n in notes)


@pytest.mark.parametrize(
    "over,serving_kw",
    [
        ({"top_k": 5}, {}),
        ({"methods": "agentic"}, {}),
        ({"answer_model": "gpt-5.6-terra"}, {}),
        ({"judge_model": "gpt-4.1-mini"}, {}),
        ({}, {"decider": "qwen3.6-27B"}),
    ],
)
def test_a_read_change_keeps_the_store_and_re_runs_the_reads(
    tmp_path: Path, over: dict, serving_kw: dict
) -> None:
    """The store is still correct, so this must not raise -- only the rows go."""
    out = tmp_path / "run"
    _seed_spec(out, _config(tmp_path), _serving())
    conv = out / "conv0"
    conv.mkdir()
    (conv / "search_hybrid.jsonl").write_text('{"index": 0}\n', encoding="utf-8")
    (conv / "answer_hybrid.jsonl").write_text('{"index": 0}\n', encoding="utf-8")
    (conv / "add.done").write_text("{}", encoding="utf-8")

    notes = _check(out, _config(tmp_path, **over), _serving(**serving_kw))

    assert any("read identity changed" in n for n in notes)
    assert not (conv / "search_hybrid.jsonl").exists()
    assert not (conv / "answer_hybrid.jsonl").exists()
    assert (conv / "search_hybrid.superseded.jsonl").exists(), "set aside, not deleted"
    assert (conv / "add.done").exists(), "an ingest that is still valid is kept"


def test_top_k_alone_never_invalidates_the_store(tmp_path: Path) -> None:
    """Named on its own because it is the most common thing to change."""
    cfg_a, cfg_b = _config(tmp_path), _config(tmp_path, top_k=5)
    srv = _serving()
    assert ingest_identity(cfg_a, "locomo", srv) == ingest_identity(
        cfg_b, "locomo", srv
    )
    assert read_identity(cfg_a, srv) != read_identity(cfg_b, srv)


def test_the_everalgo_version_is_not_part_of_the_identity(tmp_path: Path) -> None:
    """Measured to leave extraction quality unmoved; must not discard an ingest."""
    keys = set(ingest_identity(_config(tmp_path), "locomo", _serving()))
    assert keys == {"data_digest", "adapter", "extraction_backbone"}


# ── the guard must run before the spec is overwritten ──────────────────────────
def test_the_old_spec_is_still_readable_when_the_check_runs(tmp_path: Path) -> None:
    """`_write_run_spec` overwrites unconditionally; the check has to precede it."""
    out = tmp_path / "run"
    _seed_spec(out, _config(tmp_path), _serving())
    before = (out / "run_spec.json").read_text(encoding="utf-8")
    _check(out, _config(tmp_path), _serving())
    assert (out / "run_spec.json").read_text(encoding="utf-8") == before


def test_a_spec_without_identity_is_unknown_not_mismatched(tmp_path: Path) -> None:
    """Directories written before run identity existed stay resumable."""
    out = tmp_path / "run"
    out.mkdir()
    (out / "run_spec.json").write_text(
        json.dumps({"run_name": "old"}), encoding="utf-8"
    )
    notes = _check(out, _config(tmp_path), _serving())
    assert notes == ["run_spec.json predates run identity; nothing to compare"]


def test_an_unreadable_spec_is_reported_not_fatal(tmp_path: Path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    (out / "run_spec.json").write_text("{ not json", encoding="utf-8")
    assert any("unchecked" in n for n in _check(out, _config(tmp_path), _serving()))


# ── the marker binds to the experiment ─────────────────────────────────────────
def test_the_marker_identity_omits_the_backbone(tmp_path: Path) -> None:
    """It is computed where `serving` is out of scope; the directory check covers it."""
    marker = run_mod._ingest_identity_for_marker(_config(tmp_path), "locomo")
    assert set(marker) == {"data_digest", "adapter"}


def test_the_marker_identity_tracks_the_data(tmp_path: Path) -> None:
    a = run_mod._ingest_identity_for_marker(_config(tmp_path), "locomo")
    b = run_mod._ingest_identity_for_marker(_config(tmp_path, body="CHANGED"), "locomo")
    assert a != b


# ── the diff itself ────────────────────────────────────────────────────────────
def test_a_missing_key_reads_as_unrecorded(tmp_path: Path) -> None:
    assert identity_diff({}, {"a": "1"}) == ["a: (not recorded) -> 1"]


def test_the_diff_is_ordered_so_two_runs_report_alike(tmp_path: Path) -> None:
    assert identity_diff({"b": "0", "a": "0"}, {"b": "1", "a": "1"}) == [
        "a: 0 -> 1",
        "b: 0 -> 1",
    ]


# ── a dataset can be a directory ──────────────────────────────────────────────
def test_a_directory_dataset_is_digested_not_skipped(tmp_path: Path) -> None:
    """SubtleMemory's `data_path` is a directory; hashing only files left it unchecked."""
    ddir = tmp_path / "ds"
    (ddir / "sub").mkdir(parents=True)
    (ddir / "a.json").write_text("one", encoding="utf-8")
    (ddir / "sub" / "b.json").write_text("two", encoding="utf-8")
    first = ingest_identity(
        BenchmarkConfig(data_path=str(ddir), adapter="subtlememory"),  # type: ignore[arg-type]
        "subtlememory",
        _serving(),
    )["data_digest"]
    assert not first.startswith("unavailable")

    (ddir / "sub" / "b.json").write_text("CHANGED", encoding="utf-8")
    second = ingest_identity(
        BenchmarkConfig(data_path=str(ddir), adapter="subtlememory"),  # type: ignore[arg-type]
        "subtlememory",
        _serving(),
    )["data_digest"]
    assert second != first, "a byte inside the directory must move the digest"


def test_renaming_a_file_in_a_directory_dataset_moves_the_digest(tmp_path: Path) -> None:
    """Paths are hashed alongside bytes, so a rename is a change even byte-for-byte."""
    ddir = tmp_path / "ds"
    ddir.mkdir()
    (ddir / "a.json").write_text("same", encoding="utf-8")
    cfg = BenchmarkConfig(data_path=str(ddir), adapter="subtlememory")  # type: ignore[arg-type]
    before = ingest_identity(cfg, "subtlememory", _serving())["data_digest"]
    (ddir / "a.json").rename(ddir / "b.json")
    after = ingest_identity(cfg, "subtlememory", _serving())["data_digest"]
    assert before != after


def test_a_missing_dataset_is_reported_as_unavailable(tmp_path: Path) -> None:
    cfg = BenchmarkConfig(data_path=str(tmp_path / "nope"), adapter="locomo")  # type: ignore[arg-type]
    d = ingest_identity(cfg, "locomo", _serving())["data_digest"]
    assert d.startswith("unavailable:")
