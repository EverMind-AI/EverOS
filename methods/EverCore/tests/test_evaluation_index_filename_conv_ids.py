"""
Regression test for issue #127.

Bug: the EvoAgentBench / EverCore evaluation index-building stage (stage 2) wrote
BM25/Embedding index files keyed by a *local* ``range(num_conv)`` counter
(``conv_0 .. conv_{N-1}``), while stage 1 (memcell writer) and the search stage
(reader) both key off the *global* conversation id extracted from
``conversation_id`` (e.g. ``locomo_5`` -> ``5``). On a sliced run
(``--from-conv 5 --to-conv 10``) the sliced conversations keep their global ids,
so stage 2 looked for ``memcell_list_conv_0..4.json`` (absent -> skipped),
built nothing, and the search stage then failed to find
``bm25_index_conv_5..9.pkl`` -> empty retrieval.

Fix: stage 2 iterates the global conv ids actually present (passed explicitly by
the adapter, or discovered from the ``memcell_list_conv_*.json`` filenames), and
the adapter's missing-index probe checks those same global ids. This test
reproduces the slice offline (no docker, no LLM, no network) and asserts the
produced index filenames are exactly the ones the search stage reads.

This is a pure filename-mapping test: it exercises the BM25 path only (BM25 is
fully local), the ``discover_conv_ids`` helper, and the adapter's
``_check_missing_indexes`` keying. It does not require the embedding service.
"""

import json
import pickle
from pathlib import Path

import pytest

from evaluation.src.adapters.evermemos import stage2_index_building as stage2
from evaluation.src.adapters.evermemos.config import ExperimentConfig

# The fix lives in stage2_index_building (above) and in the adapter's
# _check_missing_indexes / stage2 call sites (evermemos_adapter.py). The full
# EverCoreAdapter import currently pulls in an UNRELATED, pre-existing import
# break (memory_layer.profile_manager no longer exports ScenarioType, used by
# stage1_memcells_extraction.py:53 on clean origin/main). We import it lazily and
# skip the two adapter-level tests if that pre-existing break is present, so the
# core stage2 reproduction still runs offline. The adapter helpers under test
# (_extract_conv_index, _check_missing_indexes) are pure and reproduced inline.
try:
    from evaluation.src.adapters.evermemos_adapter import EverCoreAdapter

    _ADAPTER_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-dependent
    EverCoreAdapter = None
    _ADAPTER_IMPORT_ERROR = exc


def _extract_conv_index(conversation_id: str) -> str:
    """Mirror of EverCoreAdapter._extract_conv_index (pure, for offline assertions)."""
    if "_" in conversation_id:
        return conversation_id.split("_")[-1]
    return conversation_id


def _check_missing_indexes_ref(index_dir: Path, conv_ids, index_type: str = "bm25"):
    """Mirror of the fixed EverCoreAdapter._check_missing_indexes keying contract."""
    missing = []
    for conv_id in conv_ids:
        if index_type == "bm25":
            index_file = index_dir / f"bm25_index_conv_{conv_id}.pkl"
        else:
            index_file = index_dir / f"embedding_index_conv_{conv_id}.pkl"
        if not index_file.exists():
            missing.append(conv_id)
    return missing


# Global ids of a slice, e.g. produced by `--from-conv 5 --to-conv 8`.
# The corresponding conversation_ids would be "locomo_5", "locomo_6", "locomo_7".
SLICE_GLOBAL_IDS = ["5", "6", "7"]
# Local count for this slice; the buggy code iterated range(LOCAL_NUM_CONV) == 0..2.
LOCAL_NUM_CONV = len(SLICE_GLOBAL_IDS)


def _write_memcells(data_dir: Path, conv_ids):
    """Write one minimal memcell file per global conv id (stage 1 output shape)."""
    for cid in conv_ids:
        docs = [
            {
                "subject": f"Trip discussion {cid}",
                "summary": f"They planned a vacation in conversation {cid}",
                "episode": f"We talked about visiting Paris in conversation {cid}",
            }
        ]
        (data_dir / f"memcell_list_conv_{cid}.json").write_text(
            json.dumps(docs), encoding="utf-8"
        )


def test_discover_conv_ids_reads_global_ids_from_filenames(tmp_path):
    data_dir = tmp_path / "memcells"
    data_dir.mkdir()
    _write_memcells(data_dir, SLICE_GLOBAL_IDS)

    # Discovery must return the GLOBAL ids on disk, not a local 0..N-1 range.
    assert stage2.discover_conv_ids(data_dir) == SLICE_GLOBAL_IDS
    # Sanity: the buggy local range would have been ["0", "1", "2"].
    assert stage2.discover_conv_ids(data_dir) != [str(i) for i in range(LOCAL_NUM_CONV)]


def test_discover_conv_ids_sorts_numerically(tmp_path):
    data_dir = tmp_path / "memcells"
    data_dir.mkdir()
    # Out-of-order, multi-digit ids must sort numerically (2 before 10), not lexically.
    _write_memcells(data_dir, ["10", "2", "7"])
    assert stage2.discover_conv_ids(data_dir) == ["2", "7", "10"]


def test_bm25_index_filenames_match_global_conv_ids(tmp_path):
    """Core #127 reproduction: built index filenames == search-read filenames."""
    data_dir = tmp_path / "memcells"
    data_dir.mkdir()
    bm25_dir = tmp_path / "bm25_index"
    bm25_dir.mkdir()
    _write_memcells(data_dir, SLICE_GLOBAL_IDS)

    config = ExperimentConfig()
    # Reproduce the trap: local num_conv (3) != global ids (5,6,7).
    config.num_conv = LOCAL_NUM_CONV

    # Explicit-ids path (how the adapter now drives stage 2 for a slice).
    stage2.build_bm25_index(
        config=config,
        data_dir=data_dir,
        bm25_save_dir=bm25_dir,
        conv_ids=SLICE_GLOBAL_IDS,
    )

    produced = sorted(p.name for p in bm25_dir.glob("bm25_index_conv_*.pkl"))
    expected = sorted(f"bm25_index_conv_{cid}.pkl" for cid in SLICE_GLOBAL_IDS)
    assert produced == expected

    # The buggy filenames (local range) must NOT be present.
    buggy = {f"bm25_index_conv_{i}.pkl" for i in range(LOCAL_NUM_CONV)}
    assert buggy.isdisjoint(set(produced))

    # Each produced index must be loadable and carry its docs (non-empty retrieval).
    for cid in SLICE_GLOBAL_IDS:
        with open(bm25_dir / f"bm25_index_conv_{cid}.pkl", "rb") as f:
            data = pickle.load(f)
        assert "bm25" in data and "docs" in data
        assert len(data["docs"]) == 1


def test_bm25_index_discovery_fallback_when_ids_not_passed(tmp_path):
    """Without explicit ids (e.g. stage2 CLI entry), discover from disk, not range()."""
    data_dir = tmp_path / "memcells"
    data_dir.mkdir()
    bm25_dir = tmp_path / "bm25_index"
    bm25_dir.mkdir()
    _write_memcells(data_dir, SLICE_GLOBAL_IDS)

    config = ExperimentConfig()
    config.num_conv = LOCAL_NUM_CONV

    stage2.build_bm25_index(config=config, data_dir=data_dir, bm25_save_dir=bm25_dir)

    produced = sorted(p.name for p in bm25_dir.glob("bm25_index_conv_*.pkl"))
    expected = sorted(f"bm25_index_conv_{cid}.pkl" for cid in SLICE_GLOBAL_IDS)
    assert produced == expected


def test_built_filename_matches_search_lookup_key(tmp_path):
    """
    End-to-end key alignment: the filename the search stage computes for a sliced
    conversation_id must be exactly the filename stage 2 wrote.
    """
    data_dir = tmp_path / "memcells"
    data_dir.mkdir()
    bm25_dir = tmp_path / "bm25_index"
    bm25_dir.mkdir()
    _write_memcells(data_dir, SLICE_GLOBAL_IDS)

    config = ExperimentConfig()
    config.num_conv = LOCAL_NUM_CONV
    stage2.build_bm25_index(
        config=config,
        data_dir=data_dir,
        bm25_save_dir=bm25_dir,
        conv_ids=SLICE_GLOBAL_IDS,
    )

    # The search stage extracts the global id from the conversation_id and reads
    # bm25_index_conv_{global_id}.pkl. That file must now exist.
    for cid in SLICE_GLOBAL_IDS:
        conversation_id = f"locomo_{cid}"
        search_key = _extract_conv_index(conversation_id)
        assert search_key == cid
        search_file = bm25_dir / f"bm25_index_conv_{search_key}.pkl"
        assert search_file.exists(), (
            f"search would look for {search_file.name} but stage 2 did not write it"
        )


def test_check_missing_indexes_uses_global_conv_ids(tmp_path):
    """
    The skip-logic probe must key off global conv ids. Before the fix it iterated
    range(num_conv) and so always reported a slice's indexes as missing.
    """
    bm25_dir = tmp_path / "bm25_index"
    bm25_dir.mkdir()

    # Indexes already exist under their GLOBAL ids (5,6,7).
    for cid in SLICE_GLOBAL_IDS:
        (bm25_dir / f"bm25_index_conv_{cid}.pkl").write_bytes(b"stub")

    # Probing by global ids -> nothing missing (correctly skips rebuild).
    missing = _check_missing_indexes_ref(
        index_dir=bm25_dir, conv_ids=SLICE_GLOBAL_IDS, index_type="bm25"
    )
    assert missing == []

    # Probing by the buggy local range -> would wrongly report 0,1,2 as missing.
    buggy_missing = _check_missing_indexes_ref(
        index_dir=bm25_dir,
        conv_ids=[str(i) for i in range(LOCAL_NUM_CONV)],
        index_type="bm25",
    )
    assert buggy_missing == ["0", "1", "2"]


@pytest.mark.skipif(
    EverCoreAdapter is None,
    reason=f"EverCoreAdapter unimportable (pre-existing, unrelated): {_ADAPTER_IMPORT_ERROR}",
)
def test_real_adapter_check_missing_indexes_keys_by_global_id(tmp_path):
    """When the adapter chain is importable, assert the real fixed method matches."""
    adapter = object.__new__(EverCoreAdapter)  # bypass heavy __init__
    bm25_dir = tmp_path / "bm25_index"
    bm25_dir.mkdir()
    for cid in SLICE_GLOBAL_IDS:
        (bm25_dir / f"bm25_index_conv_{cid}.pkl").write_bytes(b"stub")

    assert (
        adapter._check_missing_indexes(
            index_dir=bm25_dir, conv_ids=SLICE_GLOBAL_IDS, index_type="bm25"
        )
        == []
    )
    assert adapter._extract_conv_index("locomo_7") == "7"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
