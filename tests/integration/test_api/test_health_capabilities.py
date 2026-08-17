"""Tests for GET /health endpoint capabilities and disabled_features response."""

from __future__ import annotations

from pytest import mark, param

from everos.component.capabilities import compute_disabled_features


class TestComputeDisabledFeatures:
    """Test the compute_disabled_features helper function."""

    def test_all_capabilities_available(self) -> None:
        """When all capabilities are available, no features are disabled."""
        caps = {
            "llm": True,
            "embed": True,
            "rerank": True,
            "multimodal_llm": True,
            "parser": True,
        }
        disabled = compute_disabled_features(caps)
        assert disabled == []

    def test_embedding_unavailable_disables_vector_search_features(
        self,
    ) -> None:
        """When embedding unavailable, vector/hybrid/reflection/skill disabled."""
        caps = {
            "llm": True,
            "embed": False,
            "rerank": True,
            "multimodal_llm": True,
            "parser": True,
        }
        disabled = compute_disabled_features(caps)
        expected = {
            "vector_search",
            "hybrid_search",
            "reflection",
            "skill_extraction",
            "knowledge",
        }
        assert set(disabled) == expected

    def test_rerank_unavailable_disables_agentic_search(self) -> None:
        """When rerank is unavailable, agentic_search is disabled."""
        caps = {
            "llm": True,
            "embed": True,
            "rerank": False,
            "multimodal_llm": True,
            "parser": True,
        }
        disabled = compute_disabled_features(caps)
        assert "agentic_search" in disabled

    def test_both_embed_and_rerank_missing_disables_knowledge(self) -> None:
        """When both embed and rerank are missing, knowledge is disabled."""
        caps = {
            "llm": True,
            "embed": False,
            "rerank": False,
            "multimodal_llm": True,
            "parser": True,
        }
        disabled = compute_disabled_features(caps)
        assert "knowledge" in disabled

    def test_multimodal_and_parser_missing_disables_multimodal_upload(self) -> None:
        """When multimodal_llm is unavailable, multimodal_upload is disabled."""
        caps = {
            "llm": True,
            "embed": True,
            "rerank": True,
            "multimodal_llm": False,
            "parser": True,
        }
        disabled = compute_disabled_features(caps)
        assert "multimodal_upload" in disabled

    def test_parser_missing_disables_multimodal_upload(self) -> None:
        """When parser is unavailable, multimodal_upload is disabled."""
        caps = {
            "llm": True,
            "embed": True,
            "rerank": True,
            "multimodal_llm": True,
            "parser": False,
        }
        disabled = compute_disabled_features(caps)
        assert "multimodal_upload" in disabled

    def test_tier1_only_llm(self) -> None:
        """Tier 1 (LLM only): many features are disabled."""
        caps = {
            "llm": True,
            "embed": False,
            "rerank": False,
            "multimodal_llm": False,
            "parser": False,
        }
        disabled = compute_disabled_features(caps)
        # Tier 1 disables: vector/hybrid/agentic/reflection/skill/knowledge/multimodal
        expected = {
            "vector_search",
            "hybrid_search",
            "agentic_search",
            "reflection",
            "skill_extraction",
            "knowledge",
            "multimodal_upload",
        }
        assert set(disabled) == expected

    @mark.parametrize(
        "caps,expected_disabled",
        [
            param(
                {
                    "llm": True,
                    "embed": True,
                    "rerank": True,
                    "multimodal_llm": True,
                    "parser": True,
                },
                [],
                id="all_available",
            ),
            param(
                {
                    "llm": True,
                    "embed": False,
                    "rerank": True,
                    "multimodal_llm": True,
                    "parser": True,
                },
                [
                    "vector_search",
                    "hybrid_search",
                    "reflection",
                    "skill_extraction",
                    "knowledge",
                ],
                id="no_embed",
            ),
            param(
                {
                    "llm": True,
                    "embed": True,
                    "rerank": False,
                    "multimodal_llm": True,
                    "parser": True,
                },
                ["agentic_search", "knowledge"],
                id="no_rerank",
            ),
        ],
    )
    def test_combinations(
        self,
        caps: dict[str, bool],
        expected_disabled: list[str],
    ) -> None:
        """Parametrized tests for various capability combinations."""
        disabled = compute_disabled_features(caps)
        assert set(disabled) == set(expected_disabled)


@mark.asyncio
async def test_health_endpoint_includes_capabilities() -> None:
    """GET /health returns capabilities dict with all 5 provider fields."""
    from httpx import ASGITransport, AsyncClient

    from everos.entrypoints.api.app import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport) as client:
        response = await client.get("http://testserver/health")

    assert response.status_code == 200
    data = response.json()

    assert "status" in data
    assert "version" in data
    assert "capabilities" in data
    assert "disabled_features" in data

    caps = data["capabilities"]
    assert isinstance(caps, dict)
    assert "llm" in caps
    assert "embed" in caps
    assert "rerank" in caps
    assert "multimodal_llm" in caps
    assert "parser" in caps

    # All capability values should be boolean
    for key, value in caps.items():
        msg = f"Capability {key} should be bool, got {type(value)}"
        assert isinstance(value, bool), msg

    # disabled_features should be a list
    disabled = data["disabled_features"]
    assert isinstance(disabled, list)
    for feature in disabled:
        assert isinstance(feature, str)


@mark.asyncio
@mark.parametrize(
    ("embed", "rerank", "multimodal_llm", "parser", "expected_disabled"),
    [
        param(
            False,
            False,
            False,
            False,
            {
                "vector_search",
                "hybrid_search",
                "reflection",
                "skill_extraction",
                "agentic_search",
                "knowledge",
                "multimodal_upload",
            },
            id="tier1_llm_only",
        ),
        param(
            True,
            False,
            False,
            False,
            {"agentic_search", "knowledge", "multimodal_upload"},
            id="tier2_embed_only",
        ),
        param(
            True,
            True,
            False,
            False,
            {"multimodal_upload"},
            id="tier3_embed_rerank_only",
        ),
        param(
            True,
            True,
            True,
            True,
            set(),
            id="full_stack_all_enabled",
        ),
    ],
)
async def test_health_endpoint_disabled_features_matches_hardcoded_expectation(
    monkeypatch,
    embed: bool,
    rerank: bool,
    multimodal_llm: bool,
    parser: bool,
    expected_disabled: set[str],
) -> None:
    """GET /health disabled_features must match a hard-coded expected set.

    Pre-M12 this test asserted ``set(response['disabled_features']) ==
    set(compute_disabled_features(response['capabilities']))`` — a
    tautology, because the endpoint uses the same
    :func:`compute_disabled_features` to build the field. Any bug in
    the helper would be present on both sides of the equation and go
    undetected. Round-2 finding M12 replaces the tautology with a
    per-tier hard-coded expected set, driven by parametrisation over
    every soft-dependency capability toggle — so a regression in
    :func:`compute_disabled_features` (e.g. dropping the
    ``knowledge`` disable, mis-spelling a feature name) breaks the
    test rather than staying invisible.
    """
    from httpx import ASGITransport, AsyncClient

    from everos.component import embedding, multimodal
    from everos.component import rerank as rerank_mod
    from everos.component.parser import _core as parser_core
    from everos.entrypoints.api.app import create_app

    class _StubCapability:
        def __init__(self, available: bool) -> None:
            self.available = available

    monkeypatch.setattr(
        embedding, "get_embedding_capability", lambda: _StubCapability(embed)
    )
    monkeypatch.setattr(
        rerank_mod, "get_rerank_capability", lambda: _StubCapability(rerank)
    )
    monkeypatch.setattr(
        multimodal,
        "get_multimodal_llm_capability",
        lambda: _StubCapability(multimodal_llm),
    )
    monkeypatch.setattr(parser_core, "parser_available", lambda: parser)

    # The health route imports the accessors at module load; patch there
    # too so the values it actually reads honour the monkeypatch.
    from everos.entrypoints.api.routes import health as health_route

    monkeypatch.setattr(
        health_route, "get_embedding_capability", lambda: _StubCapability(embed)
    )
    monkeypatch.setattr(
        health_route, "get_rerank_capability", lambda: _StubCapability(rerank)
    )
    monkeypatch.setattr(
        health_route,
        "get_multimodal_llm_capability",
        lambda: _StubCapability(multimodal_llm),
    )
    monkeypatch.setattr(health_route, "parser_available", lambda: parser)

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport) as client:
        response = await client.get("http://testserver/health")

    assert response.status_code == 200
    data = response.json()

    assert set(data["disabled_features"]) == expected_disabled
