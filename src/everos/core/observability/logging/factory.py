"""structlog logger factory.

Provides ``get_logger(__name__)`` for module-level logger acquisition.
``configure_logging`` is called by the application bootstrap
(``everos.service.configure_process``) to set up the structlog processor
chain and route stdlib logging through the same formatter so output stays
uniform regardless of the caller.

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
import traceback
from contextlib import AbstractContextManager
from typing import Any, Literal, TextIO

import structlog
from structlog.typing import ExcInfo

_TRACEBACK_FRAMES = 15
"""Innermost frames kept per exception; deep async stacks run to 80+."""


def _plain_traceback(sio: TextIO, exc_info: ExcInfo) -> None:
    """Write ``exc_info`` as a plain stdlib traceback, innermost frames only.

    structlog's default (rich, syntax-highlighted, with locals) is what the
    server used to render. It is CPU-bound and runs inside the event loop:
    ~290 ms per exception when first measured, ~3 s on a loaded soak host
    (2026-11), long enough to keep other requests' SQLite write
    transactions open past ``busy_timeout``. Locals are never printed: they
    can hold request payloads.
    """
    sio.write("\n")
    sio.write("".join(traceback.format_exception(*exc_info, limit=-_TRACEBACK_FRAMES)))


def get_logger(name: str) -> Any:
    """Return a structlog logger bound to the given module name."""
    return structlog.get_logger(name)


def bound_log_context(**fields: Any) -> AbstractContextManager[None]:
    """Add ``fields`` to every log event inside the ``with`` block.

    Bound through context variables, so the fields follow the current task
    and are restored to their previous values on exit.
    """
    return structlog.contextvars.bound_contextvars(**fields)  # type: ignore[return-value]  # structlog types it as a Generator; @contextmanager wraps it at runtime


def configure_logging(
    level: str = "INFO", *, stream: Literal["stdout", "stderr"] = "stdout"
) -> None:
    """Configure structlog and stdlib logging once at process startup.

    After this call:

    * Every ``structlog.get_logger(...)`` and ``logging.getLogger(...)``
      message flows through the same ``ProcessorFormatter``, so output
      format is identical regardless of which logging API the caller used.
    * Root-logger handlers are replaced with a single ``StreamHandler``
      pointing at ``sys.stdout`` (``sys.stderr`` with ``stream="stderr"``:
      a command whose standard output is its answer, such as the JSON of
      ``everos memory``, keeps its log lines out of it); any previously
      installed handler (uvicorn's default ``LOGGING_CONFIG``, libraries
      that call ``logging.basicConfig``, etc.) is removed.

    The ``uvicorn.run(..., log_config=None)`` flag is the matching half
    on the server entry point — without it, uvicorn re-installs its own
    handlers on every startup and overrides what we set here.

    Args:
        level: Log level name (``DEBUG`` / ``INFO`` / ``WARNING`` / ``ERROR``).
            Unknown names silently fall back to ``INFO`` via
            ``getattr(logging, ..., INFO)``.
        stream: ``stdout`` (default) or ``stderr``: where the lines go.
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
        # Not cached: a cached logger keeps the level and processors of the
        # configuration it first ran under, so a later call (a CLI command
        # changing the level, a test capturing logs) would not reach it.
        cache_logger_on_first_use=False,
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
            structlog.dev.ConsoleRenderer(
                # Not structlog's default ``RichTracebackFormatter``: one
                # exception in a soak run rendered 82 frames into 6423 log
                # lines, and the rendering blocks the event loop (see
                # ``_plain_traceback``).
                exception_formatter=_plain_traceback,
            ),
        ],
    )

    handler = logging.StreamHandler(sys.stderr if stream == "stderr" else sys.stdout)
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
    # LLM / embedding request). A single LoCoMo conv run easily
    # produces a thousand such lines, drowning everos's own events. They
    # are useful for debugging API failures, but failures already surface
    # via exceptions + status codes — so demote the success path to WARNING
    # and let real errors still come through.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
