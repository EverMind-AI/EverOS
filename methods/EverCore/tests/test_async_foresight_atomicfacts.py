"""
tests/test_async_foresight_atomicfacts.py

Unit tests for the async fire-and-forget foresight/atomic_fact background task.

Usage:
    PYTHONPATH=src pytest tests/test_async_foresight_atomicfacts.py -v
"""

import asyncio as real_asyncio
from contextlib import ExitStack
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from api_specs.memory_types import MemCell, RawDataType, Foresight, AtomicFact
from api_specs.dtos import MemorizeRequest
from biz_layer.mem_memorize import (
    ExtractionState,
    _foresight_and_atomic_facts_with_metrics,
    _process_memories,
    process_memory_extraction,
)


def _make_memcell(raw_data_type: RawDataType = RawDataType.CONVERSATION) -> MemCell:
    mc = MagicMock(spec=MemCell)
    mc.type = raw_data_type
    mc.event_id = "evt-001"
    mc.original_data = []
    mc.participants = ["user_001"]
    mc.timestamp = datetime(2026, 3, 30, 10, 0, 0)
    return mc


def _make_request(scene: str = "solo") -> MemorizeRequest:
    req = MagicMock(spec=MemorizeRequest)
    req.group_id = "grp-001"
    req.session_id = "sess-001"
    req.scene = scene
    req.raw_data_type = RawDataType.CONVERSATION
    return req


def _make_state(is_solo: bool = True, has_episode: bool = True) -> ExtractionState:
    state = MagicMock(spec=ExtractionState)
    state.memcell = _make_memcell()
    state.request = _make_request("solo" if is_solo else "team")
    state.is_solo_scene = is_solo
    state.participants = ["user_001"]
    state.current_time = datetime(2026, 3, 30, 10, 0, 0)
    state.foresight_parent_type = "episodic_memory"
    state.atomic_fact_parent_type = "episodic_memory"
    state.parent_id = "evt-001"
    saved_ep = MagicMock()
    saved_ep.id = "ep-mongo-001"
    state.group_episode_memories = [MagicMock(id="ep-mongo-001")] if has_episode else []
    state.parent_docs_map = {"ep-mongo-001": saved_ep} if has_episode else {}
    # MagicMock doesn't evaluate properties — set explicitly to mirror property logic
    state.episode_saved = has_episode
    state.agent_case = None
    return state


@pytest.mark.asyncio
async def test_background_fn_extracts_parallel_and_saves():
    """Solo non-agent: foresight and atomic_fact extracted in parallel, then saved."""
    state = _make_state()
    mm = AsyncMock()
    foresight_mock = MagicMock(spec=Foresight)
    af_mock = MagicMock(spec=AtomicFact)

    with (
        patch(
            'biz_layer.mem_memorize._should_skip_atomic_fact_for_agent',
            return_value=False,
        ),
        patch(
            'biz_layer.mem_memorize._extract_foresights',
            new_callable=AsyncMock,
            return_value=[foresight_mock],
        ) as mock_ef,
        patch(
            'biz_layer.mem_memorize._extract_atomic_facts',
            new_callable=AsyncMock,
            return_value=[af_mock],
        ) as mock_eaf,
        patch(
            'biz_layer.mem_memorize._save_foresight_and_atomic_fact',
            new_callable=AsyncMock,
        ) as mock_save,
        patch('biz_layer.mem_memorize.record_memory_extracted'),
        patch('biz_layer.mem_memorize.record_extraction_stage'),
        patch(
            'biz_layer.mem_memorize.get_space_id_for_metrics', return_value='space_test'
        ),
    ):

        await _foresight_and_atomic_facts_with_metrics(state, mm)

        mock_ef.assert_called_once_with(state, mm)
        mock_eaf.assert_called_once_with(state, mm)
        mock_save.assert_called_once_with(state, [foresight_mock], [af_mock])


@pytest.mark.asyncio
async def test_background_fn_skips_atomic_fact_when_agent_flag_set():
    """When _should_skip_atomic_fact_for_agent returns True, atomic_fact extraction skipped."""
    state = _make_state()
    mm = AsyncMock()
    foresight_mock = MagicMock(spec=Foresight)

    with (
        patch(
            'biz_layer.mem_memorize._should_skip_atomic_fact_for_agent',
            return_value=True,
        ),
        patch(
            'biz_layer.mem_memorize._extract_foresights',
            new_callable=AsyncMock,
            return_value=[foresight_mock],
        ) as mock_ef,
        patch(
            'biz_layer.mem_memorize._extract_atomic_facts', new_callable=AsyncMock
        ) as mock_eaf,
        patch(
            'biz_layer.mem_memorize._save_foresight_and_atomic_fact',
            new_callable=AsyncMock,
        ) as mock_save,
        patch('biz_layer.mem_memorize.record_memory_extracted'),
        patch('biz_layer.mem_memorize.record_extraction_stage'),
        patch(
            'biz_layer.mem_memorize.get_space_id_for_metrics', return_value='space_test'
        ),
    ):

        await _foresight_and_atomic_facts_with_metrics(state, mm)

        mock_ef.assert_called_once_with(state, mm)
        mock_eaf.assert_not_called()
        mock_save.assert_called_once_with(state, [foresight_mock], [])


@pytest.mark.asyncio
async def test_background_fn_swallows_exception_and_records_stage():
    """Exception in extraction must not propagate; record_extraction_stage always called."""
    state = _make_state()
    mm = AsyncMock()

    with (
        patch(
            'biz_layer.mem_memorize._should_skip_atomic_fact_for_agent',
            return_value=False,
        ),
        patch(
            'biz_layer.mem_memorize._extract_foresights',
            new_callable=AsyncMock,
            side_effect=RuntimeError("LLM down"),
        ),
        patch('biz_layer.mem_memorize._extract_atomic_facts', new_callable=AsyncMock),
        patch(
            'biz_layer.mem_memorize._save_foresight_and_atomic_fact',
            new_callable=AsyncMock,
        ) as mock_save,
        patch('biz_layer.mem_memorize.record_memory_extracted'),
        patch('biz_layer.mem_memorize.record_extraction_stage') as mock_stage,
        patch(
            'biz_layer.mem_memorize.get_space_id_for_metrics', return_value='space_test'
        ),
    ):

        await _foresight_and_atomic_facts_with_metrics(state, mm)

        mock_save.assert_not_called()
        call_kwargs = mock_stage.call_args[1]
        assert call_kwargs["stage"] == "foresight_and_atomic_facts_bg"


@pytest.mark.asyncio
async def test_process_memories_does_not_call_save_foresight():
    """After refactor, _process_memories must NOT call _save_foresight_and_atomic_fact."""
    state = _make_state()

    with (
        patch('biz_layer.mem_memorize._save_episodes', new_callable=AsyncMock),
        patch('biz_layer.mem_memorize._save_agent_case', new_callable=AsyncMock),
        patch(
            'biz_layer.mem_memorize.update_status_after_memcell', new_callable=AsyncMock
        ),
        patch(
            'biz_layer.mem_memorize._save_foresight_and_atomic_fact',
            new_callable=AsyncMock,
        ) as mock_save,
        patch('biz_layer.mem_memorize._clone_episodes_for_users', return_value=[]),
    ):

        await _process_memories(state)

        mock_save.assert_not_called()


@pytest.mark.asyncio
async def test_background_fn_empty_results_skips_save_and_count_metrics():
    """When both foresight and atomic_facts return empty, save and count metrics are not called."""
    state = _make_state()
    mm = AsyncMock()

    with (
        patch(
            'biz_layer.mem_memorize._should_skip_atomic_fact_for_agent',
            return_value=False,
        ),
        patch(
            'biz_layer.mem_memorize._extract_foresights',
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            'biz_layer.mem_memorize._extract_atomic_facts',
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            'biz_layer.mem_memorize._save_foresight_and_atomic_fact',
            new_callable=AsyncMock,
        ) as mock_save,
        patch('biz_layer.mem_memorize.record_memory_extracted') as mock_record,
        patch('biz_layer.mem_memorize.record_extraction_stage') as mock_stage,
        patch(
            'biz_layer.mem_memorize.get_space_id_for_metrics', return_value='space_test'
        ),
    ):

        await _foresight_and_atomic_facts_with_metrics(state, mm)

        mock_save.assert_not_called()
        mock_record.assert_not_called()
        mock_stage.assert_called_once()  # finally block always runs


def _patch_process_memory_extraction_deps(state):
    """Patch all heavy dependencies of process_memory_extraction for unit testing."""
    return [
        patch(
            'biz_layer.mem_memorize._init_extraction_state',
            new_callable=AsyncMock,
            return_value=state,
        ),
        patch('biz_layer.mem_memorize._extract_episodes', new_callable=AsyncMock),
        patch('biz_layer.mem_memorize._extract_agent_case', new_callable=AsyncMock),
        patch(
            'biz_layer.mem_memorize._update_memcell_and_cluster', new_callable=AsyncMock
        ),
        patch(
            'biz_layer.mem_memorize._process_memories',
            new_callable=AsyncMock,
            return_value=1,
        ),
        patch('biz_layer.mem_memorize.if_memorize', return_value=True),
        patch(
            'biz_layer.mem_memorize.get_space_id_for_metrics', return_value='space_test'
        ),
        patch('biz_layer.mem_memorize.record_extraction_stage'),
        patch('biz_layer.mem_memorize.record_memory_extracted'),
        patch(
            'biz_layer.mem_memorize._save_memcell_to_database', new_callable=AsyncMock
        ),
    ]


@pytest.mark.asyncio
async def test_process_memory_extraction_fires_background_task_for_solo():
    """Solo scene with saved episode: background task must be created."""
    state = _make_state(is_solo=True, has_episode=True)
    memcell = state.memcell
    request = state.request
    mm = AsyncMock()
    current_time = datetime(2026, 3, 30, 10, 0, 0)

    task_coro_names = []
    _orig = real_asyncio.create_task

    def _tracking_create_task(coro, **kwargs):
        task_coro_names.append(getattr(coro, '__qualname__', type(coro).__name__))
        return _orig(coro, **kwargs)

    patches = _patch_process_memory_extraction_deps(state)
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        with patch.object(
            real_asyncio, 'create_task', side_effect=_tracking_create_task
        ):
            await process_memory_extraction(memcell, request, mm, current_time)

    foresight_tasks = [n for n in task_coro_names if 'foresight' in n.lower()]
    assert (
        len(foresight_tasks) == 1
    ), f"Expected 1 foresight task, got: {task_coro_names}"


@pytest.mark.asyncio
async def test_process_memory_extraction_no_background_task_for_team():
    """Team scene: background task must NOT be created."""
    state = _make_state(is_solo=False, has_episode=True)
    memcell = state.memcell
    request = state.request
    mm = AsyncMock()
    current_time = datetime(2026, 3, 30, 10, 0, 0)

    task_coro_names = []
    _orig = real_asyncio.create_task

    def _tracking_create_task(coro, **kwargs):
        task_coro_names.append(getattr(coro, '__qualname__', type(coro).__name__))
        return _orig(coro, **kwargs)

    patches = _patch_process_memory_extraction_deps(state)
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        with patch.object(
            real_asyncio, 'create_task', side_effect=_tracking_create_task
        ):
            await process_memory_extraction(memcell, request, mm, current_time)

    foresight_tasks = [n for n in task_coro_names if 'foresight' in n.lower()]
    assert (
        len(foresight_tasks) == 0
    ), f"Expected no foresight task for team, got: {task_coro_names}"


@pytest.mark.asyncio
async def test_process_memory_extraction_no_background_task_when_no_episode():
    """Solo scene but no episode saved: background task must NOT be created (no parent_doc)."""
    state = _make_state(is_solo=True, has_episode=False)
    memcell = state.memcell
    request = state.request
    mm = AsyncMock()
    current_time = datetime(2026, 3, 30, 10, 0, 0)

    task_coro_names = []
    _orig = real_asyncio.create_task

    def _tracking_create_task(coro, **kwargs):
        task_coro_names.append(getattr(coro, '__qualname__', type(coro).__name__))
        return _orig(coro, **kwargs)

    patches = _patch_process_memory_extraction_deps(state)
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        with patch.object(
            real_asyncio, 'create_task', side_effect=_tracking_create_task
        ):
            await process_memory_extraction(memcell, request, mm, current_time)

    foresight_tasks = [n for n in task_coro_names if 'foresight' in n.lower()]
    assert (
        len(foresight_tasks) == 0
    ), f"Expected no foresight task (no episode), got: {task_coro_names}"
