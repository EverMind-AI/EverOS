"""Tests for :func:`everos.entrypoints.cli._log_setup.configure_cli_logging`.

Pins the interactive-CLI logging contract: default = WARNING (lifecycle
logs suppressed), ``verbose=True`` = INFO (lifecycle logs emitted).
The root logger level is the observable side effect — it is what the
handler installed by ``configure_logging()`` gates output against.
"""

from __future__ import annotations

import logging

from everos.entrypoints.cli._log_setup import configure_cli_logging


def test_configure_cli_logging_default_warning() -> None:
    configure_cli_logging(verbose=False)
    assert logging.getLogger().level == logging.WARNING


def test_configure_cli_logging_verbose_info() -> None:
    configure_cli_logging(verbose=True)
    assert logging.getLogger().level == logging.INFO
