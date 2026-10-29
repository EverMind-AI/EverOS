"""Interactive CLI commands log WARNING by default and INFO with ``--verbose``.

The root logger level is the observable side effect -- it is what the
handler installed by the process setup gates output against.
"""

from __future__ import annotations

import logging

from everos.config import Settings
from everos.entrypoints.cli._process import configure_interactive, with_log_level


def test_interactive_default_is_warning() -> None:
    settings = configure_interactive(verbose=False)
    assert logging.getLogger().level == logging.WARNING
    assert settings.logging.level == "WARNING"


def test_interactive_verbose_is_info() -> None:
    settings = configure_interactive(verbose=True)
    assert logging.getLogger().level == logging.INFO
    assert settings.logging.level == "INFO"


def test_interactive_keeps_the_given_settings_but_the_level() -> None:
    base = Settings.model_construct()
    base = base.model_copy(update={"api": base.api.model_copy(update={"port": 9999})})
    settings = configure_interactive(verbose=False, settings=base)
    assert settings.api.port == 9999
    assert settings.logging.level == "WARNING"


def test_with_log_level_leaves_the_original_untouched() -> None:
    base = Settings.model_construct()
    changed = with_log_level(base, "DEBUG")
    assert (base.logging.level, changed.logging.level) == ("INFO", "DEBUG")
