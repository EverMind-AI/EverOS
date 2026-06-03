"""``build_lifespan`` — provider ordering, state storage, shutdown errors."""

from __future__ import annotations

from fastapi import FastAPI

from everos.core.lifespan import LifespanProvider
from everos.core.lifespan.factory import build_lifespan


class _RecordingProvider(LifespanProvider):
    """Provider that records the order in which startup/shutdown ran."""

    def __init__(
        self,
        name: str,
        order: int,
        log: list[str],
        *,
        returns: object | None = None,
        shutdown_raises: bool = False,
    ) -> None:
        super().__init__(name=name, order=order)
        self._log = log
        self._returns = returns
        self._shutdown_raises = shutdown_raises

    async def startup(self, app: FastAPI) -> object | None:
        self._log.append(f"start:{self.name}")
        return self._returns

    async def shutdown(self, app: FastAPI) -> None:
        self._log.append(f"stop:{self.name}")
        if self._shutdown_raises:
            raise RuntimeError(f"{self.name} shutdown boom")


async def test_startup_runs_in_order_ascending() -> None:
    log: list[str] = []
    p1 = _RecordingProvider("a", order=2, log=log)
    p2 = _RecordingProvider("b", order=1, log=log)
    p3 = _RecordingProvider("c", order=3, log=log)

    app = FastAPI()
    async with build_lifespan([p1, p2, p3])(app):
        pass
    assert log[:3] == ["start:b", "start:a", "start:c"]


async def test_shutdown_runs_in_reverse_order() -> None:
    log: list[str] = []
    p1 = _RecordingProvider("a", order=1, log=log)
    p2 = _RecordingProvider("b", order=2, log=log)

    app = FastAPI()
    async with build_lifespan([p1, p2])(app):
        pass
    # shutdown phase: reverse of startup
    assert log[2:] == ["stop:b", "stop:a"]


async def test_non_none_startup_result_stored_in_state() -> None:
    sentinel = object()
    p = _RecordingProvider("x", order=1, log=[], returns=sentinel)
    app = FastAPI()
    async with build_lifespan([p])(app):
        assert app.state.lifespan_data["x"] is sentinel


async def test_none_startup_result_not_stored() -> None:
    p = _RecordingProvider("nullone", order=1, log=[], returns=None)
    app = FastAPI()
    async with build_lifespan([p])(app):
        assert "nullone" not in app.state.lifespan_data


async def test_shutdown_exception_swallowed_and_logged() -> None:
    """Failed shutdown logs but doesn't break sibling shutdown."""
    log: list[str] = []
    p1 = _RecordingProvider("a", order=1, log=log)
    p2 = _RecordingProvider("b", order=2, log=log, shutdown_raises=True)

    app = FastAPI()
    async with build_lifespan([p1, p2])(app):
        pass
    # Even though "b" raised, "a" still shut down.
    assert log[-1] == "stop:a"
    assert "stop:b" in log  # b's shutdown ran (and raised, but swallowed)
