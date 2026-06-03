import pytest

from agentic_layer import memory_manager as memory_manager_module
from agentic_layer.memory_manager import MemoryManager
from api_specs.dtos import RetrieveMemRequest
from api_specs.memory_models import MemoryType


class _RepoA:
    async def multi_search(self, **kwargs):
        return [{'_id': 'a', '_score': 0.2}]


class _RepoB:
    async def multi_search(self, **kwargs):
        return [{'_id': 'b', '_score': 0.9}]


@pytest.mark.asyncio
async def test_keyword_search_queries_all_requested_memory_types(monkeypatch):
    repos = {_RepoA: _RepoA(), _RepoB: _RepoB()}
    monkeypatch.setattr(
        memory_manager_module,
        'ES_REPO_MAP',
        {MemoryType.EPISODIC_MEMORY: _RepoA, MemoryType.AGENT_CASE: _RepoB},
    )
    monkeypatch.setattr(
        memory_manager_module, 'get_bean_by_type', lambda repo_class: repos[repo_class]
    )

    manager = object.__new__(MemoryManager)
    request = RetrieveMemRequest(
        query='soccer',
        group_ids=['group-1'],
        top_k=10,
        memory_types=[MemoryType.EPISODIC_MEMORY, MemoryType.AGENT_CASE],
    )

    hits = await manager.get_keyword_search_results(request)

    assert [hit['memory_type'] for hit in hits] == [
        MemoryType.AGENT_CASE.value,
        MemoryType.EPISODIC_MEMORY.value,
    ]
    assert [hit['id'] for hit in hits] == ['b', 'a']


@pytest.mark.asyncio
async def test_hybrid_dedupe_keeps_same_id_from_distinct_memory_types():
    manager = object.__new__(MemoryManager)

    async def keyword_results(*args, **kwargs):
        return [{'id': 'same', 'memory_type': 'episodic_memory', 'score': 0.8}]

    async def vector_results(*args, **kwargs):
        return [
            {'id': 'same', 'memory_type': 'agent_case', 'score': 0.9},
            {'id': 'same', 'memory_type': 'episodic_memory', 'score': 0.7},
        ]

    async def rerank(query, hits, top_k, *args, **kwargs):
        return hits

    manager.get_keyword_search_results = keyword_results
    manager.get_vector_search_results = vector_results
    manager._rerank = rerank

    request = RetrieveMemRequest(
        query='soccer',
        group_ids=['group-1'],
        top_k=10,
        memory_types=[MemoryType.EPISODIC_MEMORY, MemoryType.AGENT_CASE],
    )

    hits = await manager._search_hybrid(request)

    assert hits == [
        {'id': 'same', 'memory_type': 'episodic_memory', 'score': 0.8},
        {'id': 'same', 'memory_type': 'agent_case', 'score': 0.9},
    ]
