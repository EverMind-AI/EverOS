"""Task 25 acceptance suite — Tier 1/2/3 end-to-end + upgrade path.

Drives the real FastAPI app (``create_app()`` + full lifespan: LLM,
SQLite, LanceDB, Cascade, OME) against a per-test ``EVEROS_ROOT`` with a
controlled capability configuration, exercising the behavior every
earlier task in the embed-soft-dependency refactor (Tasks 1-24) is
supposed to produce. See ``conftest.py`` for the three tier fixtures.
"""

from __future__ import annotations
