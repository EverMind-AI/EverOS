"""Knowledge settings load from TOML."""

from __future__ import annotations

from everos.config import load_settings


class TestKnowledgeSettings:
    def test_defaults(self) -> None:
        load_settings.cache_clear()
        s = load_settings()
        ks = s.knowledge
        assert ks.search.recall_n == 200
        assert ks.search.rerank_n == 50
        assert ks.search.mass_top_m == 50
        assert ks.search.lam == 0.1
        assert ks.search.top_k_cap == 100
