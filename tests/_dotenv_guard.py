"""Keep the developer's ``.env`` out of the test process.

pymilvus 3.0.1 runs ``load_dotenv()`` when it is imported
(``pymilvus/settings.py:6``), which copies the working directory's ``.env`` --
a developer's keys and settings -- into ``os.environ`` and makes the suite
pass or fail depending on whose machine runs it. Importing this module first
(``tests/conftest.py`` does) replaces the package-level ``dotenv.load_dotenv``
with a no-op, so that import-time call loads nothing.
``dotenv.main.load_dotenv`` stays real: the e2e conftest loads ``.env`` on
purpose, for the live suites.
"""

from __future__ import annotations

import dotenv


def _no_dotenv(*_args: object, **_kwargs: object) -> bool:
    return False


dotenv.load_dotenv = _no_dotenv  # type: ignore[assignment]  # the guard itself
