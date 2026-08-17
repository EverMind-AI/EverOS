"""Kind-scoped scanner sweeps (finding #6 support).

Phase-3 backfill needs a way to sync only ``agent_skill`` md without
walking every registered kind — a full sweep would rediscover
knowledge md, and if the current process has embedding but not rerank
those handlers are gated off, leading cascade to mark every knowledge
row as permanently failed.

The plumbing lives in :func:`_collect_scan_inputs` (walk only requested
kinds) and :meth:`CascadeScanner.scan_once` (also scope the state
snapshot so reconcile does not emit spurious ``deleted`` decisions for
unrelated kinds).
"""

from __future__ import annotations

from pathlib import Path

from everos.memory.cascade.scanner import _collect_scan_inputs


def _make_episode_md(root: Path) -> Path:
    """Create a plausible episode daily-log file under the memory root."""
    d = root / "default_app" / "default_project" / "users" / "u1" / "episodes"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "episode-2026-01-01.md"
    p.write_text("ok")
    return p


def _make_agent_skill_md(root: Path) -> Path:
    """Create a plausible SKILL.md under the memory root."""
    d = (
        root
        / "default_app"
        / "default_project"
        / "agents"
        / "a1"
        / "skills"
        / "skill_greet"
    )
    d.mkdir(parents=True, exist_ok=True)
    p = d / "SKILL.md"
    p.write_text("ok")
    return p


def test_collect_scan_inputs_default_scans_all_kinds(tmp_path: Path) -> None:
    """Default (``kinds=None``) walks every :data:`KIND_REGISTRY` path
    glob — episode + skill both surface."""
    _make_episode_md(tmp_path)
    _make_agent_skill_md(tmp_path)

    inputs = _collect_scan_inputs(tmp_path)
    kinds = {i.kind for i in inputs}
    assert "episode" in kinds
    assert "agent_skill" in kinds


def test_collect_scan_inputs_with_kinds_filter_restricts_to_named(
    tmp_path: Path,
) -> None:
    """``kinds={"agent_skill"}`` walks only the skill glob; episode md
    on disk is not stat'd or emitted."""
    episode = _make_episode_md(tmp_path)
    skill = _make_agent_skill_md(tmp_path)

    inputs = _collect_scan_inputs(tmp_path, {"agent_skill"})
    paths = {i.md_path for i in inputs}
    assert skill.relative_to(tmp_path).as_posix() in paths
    assert episode.relative_to(tmp_path).as_posix() not in paths
    # And every emitted input's kind is the requested one.
    assert {i.kind for i in inputs} == {"agent_skill"}


def test_collect_scan_inputs_with_empty_kinds_returns_nothing(
    tmp_path: Path,
) -> None:
    """Empty-set filter is a degenerate but well-defined case: nothing
    walked, nothing returned. Guards against a future refactor that
    treats ``kinds=frozenset()`` as ``None`` by mistake."""
    _make_episode_md(tmp_path)
    _make_agent_skill_md(tmp_path)

    inputs = _collect_scan_inputs(tmp_path, set())
    assert inputs == []


def test_collect_scan_inputs_unknown_kind_yields_nothing(tmp_path: Path) -> None:
    """A kind name that doesn't match any spec silently yields no inputs
    (rather than raising) — the caller is responsible for passing valid
    kinds; the walker just filters."""
    _make_episode_md(tmp_path)

    inputs = _collect_scan_inputs(tmp_path, {"not_a_real_kind"})
    assert inputs == []
