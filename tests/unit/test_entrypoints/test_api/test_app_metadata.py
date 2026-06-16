"""FastAPI metadata must stay aligned with package metadata."""

from __future__ import annotations

from everos import __version__
from everos.entrypoints.api.app import create_app


def test_openapi_info_version_matches_package_version() -> None:
    app = create_app(lifespan_providers=[])
    assert app.openapi()["info"]["version"] == __version__
