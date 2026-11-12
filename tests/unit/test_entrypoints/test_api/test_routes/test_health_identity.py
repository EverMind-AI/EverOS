"""``/health`` names the process that answered.

A launcher that starts several servers on one machine (the benchmark fleet)
cannot tell from an HTTP 200 alone which server answered: when two launches
pick the same port, one child fails to bind and the other launch's server
answers for it. ``pid`` is what lets the launcher compare the answer with the
child it started.
"""

from __future__ import annotations

import os
from typing import Any

from everos.entrypoints.api.routes import health as health_route
from tests.fakes import make_application


async def test_health_reports_the_id_of_the_process_that_answered() -> None:
    response: Any = await health_route.health(make_application())

    assert response.pid == os.getpid()
