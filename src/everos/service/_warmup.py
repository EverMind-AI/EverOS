"""Startup warm-up of the algorithm library's token counter.

everalgo budgets the agent-track prompts with tiktoken's ``o200k_base``
encoding (the case gate, case compression, the agent profile), and tiktoken
fetches that encoding over the network the first time a process uses it,
caching it under ``TIKTOKEN_CACHE_DIR`` (default: a ``data-gym-cache`` folder
in the temp dir). Left to the request path, the first agent ``/add`` of a
fresh deployment carried the download and answered a generic 500 when it
failed (#277). Warming the encoding at startup moves the fetch out of the
request path and turns a failure into one clear startup error.
"""

from __future__ import annotations

import anyio
from everalgo.utils import encode

from everos.core.errors import TokenizerAssetError
from everos.core.observability.logging import get_logger

logger = get_logger(__name__)

_PROBE = "warm-up"


async def warm_token_counter() -> None:
    """Load the token-counting encoding once, off the event loop.

    Raises:
        TokenizerAssetError: The encoding could not be loaded -- in practice
            the first-run download failed and no cached copy exists.
    """
    try:
        await anyio.to_thread.run_sync(encode, _PROBE)
    except Exception as exc:  # tiktoken surfaces HTTP, TLS and OS errors alike
        raise TokenizerAssetError(
            "the token-counting encoding (tiktoken o200k_base) could not be "
            "loaded. It is downloaded once on first start and cached; allow the "
            "download, or pre-populate TIKTOKEN_CACHE_DIR with a copy taken from "
            f"a machine that has it. Cause: {exc!r}"
        ) from exc
    logger.info("token_counter_warm")
