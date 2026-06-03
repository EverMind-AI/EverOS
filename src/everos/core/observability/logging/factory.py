"""structlog logger factory.

Provides ``get_logger(__name__)`` for module-level logger acquisition.
``configure_logging()`` is called once at process startup (run.py / lifespan)
to set up the structlog processor chain and route stdlib logging through
the same formatter so output stays uniform regardless of the caller.

The configuration follows structlog's official "Foreign Log Integration"
recipe: a single ``ProcessorFormatter`` renders both everos's own
``get_logger(...)`` calls and any stdlib ``logging.getLogger(...)`` call
made by third-party libraries (uvicorn, fastapi, httpx, openai, ...).
That way all three of the previously divergent prefixes — ``INFO:``,
``[warning  ]``, plus the unconfigured no-prefix output — collapse to
one ``[level] event key=value`` shape.

Rust-side loggers (LanceDB / Lance / Arrow) live in the Rust ``log``
crate and emit straight to stderr without going through Python; this
module cannot reach them. Control their level with ``RUST_LOG`` env.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def get_logger(name: str) -> Any:
    """Return a structlog logger bound to the given module name."""
    return structlog.get_logger(name)


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog and stdlib logging once at process startup.

    After this call:

    * Every ``structlog.get_logger(...)`` and ``logging.getLogger(...)``
      message flows through the same ``ProcessorFormatter``, so output
      format is identical regardless of which logging API the caller used.
    * Root-logger handlers are replaced with a single ``StreamHandler``
      pointing at ``sys.stdout``; any previously installed handler
      (uvicorn's default ``LOGGING_CONFIG``, libraries that call
      ``logging.basicConfig``, etc.) is removed.

    The ``uvicorn.run(..., log_config=None)`` flag is the matching half
    on the server entry point — without it, uvicorn re-installs its own
    handlers on every startup and overrides what we set here.

    Args:
        level: Log level name (``DEBUG`` / ``INFO`` / ``WARNING`` / ``ERROR``).
            Unknown names silently fall back to ``INFO`` via
            ``getattr(logging, ..., INFO)``.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    # structlog's own loggers feed into stdlib's logging, so the root
    # logger handler decides where output lands and how it's rendered.
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # The single formatter shared by both pipelines:
    # * structlog events arrive already wrapped via ``wrap_for_formatter``;
    # * foreign records (stdlib LogRecord) get pushed through
    #   ``foreign_pre_chain`` so they pick up the same level / timestamp
    #   fields before hitting ``ConsoleRenderer``.
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(),
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Drop any handler we installed on a previous ``configure_logging``
    # call (identified by formatter type) so repeated invocations don't
    # produce duplicate output, but keep handlers other parties have
    # attached — pytest's caplog handler in particular has to survive,
    # otherwise tests using the ``caplog`` fixture can't see records
    # that flow through structlog.
    root = logging.getLogger()
    root.handlers = [
        h
        for h in root.handlers
        if not isinstance(h.formatter, structlog.stdlib.ProcessorFormatter)
    ]
    root.addHandler(handler)
    root.setLevel(log_level)

    # Third-party HTTP clients log every successful request at INFO level —
    # `httpx` is the worst offender (one line per call, called once per
    # LLM / embedding / rerank request). A single LoCoMo conv run easily
    # produces a thousand such lines, drowning everos's own events. They
    # are useful for debugging API failures, but failures already surface
    # via exceptions + status codes — so demote the success path to WARNING
    # and let real errors still come through.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
