"""``create_app`` hands its lifespan one ``Application`` to start and close.

An injected application is reachable by routes as soon as the app exists
(tests drive routes without running the lifespan) and is started on startup
and closed on shutdown like one the lifespan built itself.
"""

from __future__ import annotations

from pathlib import Path

from everos.entrypoints.api.app import create_app
from tests.fakes import make_application


def test_an_injected_application_is_on_state_before_startup(tmp_path: Path) -> None:
    application = make_application(root=tmp_path)

    app = create_app(application=application)

    assert app.state.application is application


async def test_the_lifespan_starts_and_closes_the_application(tmp_path: Path) -> None:
    events: list[str] = []
    application = make_application(root=tmp_path, lifecycle=events)
    app = create_app(application=application)

    async with app.router.lifespan_context(app):
        assert "memory.start" in events
        assert app.state.application is application
        assert "memory.aclose" not in events

    assert events[-1] == "sqlite.dispose"
    assert "memory.aclose" in events
