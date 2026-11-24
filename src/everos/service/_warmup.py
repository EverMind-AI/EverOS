"""Startup warm-up of the algorithm library's token counter.

everalgo budgets the agent-track prompts with tiktoken's ``o200k_base``
encoding (the case gate, case compression, the agent profile), and tiktoken
fetches that encoding over the network the first time a process uses it,
caching it under ``TIKTOKEN_CACHE_DIR`` (default: a ``data-gym-cache`` folder
in the temp dir). Left to the request path, the first agent ``/add`` of a
fresh deployment carried the download and answered a generic 500 when it
failed (#277). Warming the encoding at startup moves the fetch out of the
request path and turns a failure into one clear startup error.

The wheel carries the encoding (``hatch_build.py``): before the warm-up it is
copied to where tiktoken's cache lookup finds it, so an installed EverOS
starts without the network, also after the temp dir was cleaned. A source
checkout has no copy and downloads as before.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import os
import tempfile
from pathlib import Path

import anyio
from everalgo.utils import encode

from everos.core.errors import TokenizerAssetError
from everos.core.observability.logging import get_logger

logger = get_logger(__name__)

_PROBE = "warm-up"
_O200K_URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
"""tiktoken's source for the encoding; its cache file is named by this URL's sha1."""
_BUNDLED = "o200k_base.tiktoken"


def _tiktoken_cache_dir() -> str:
    """The directory tiktoken's ``read_file_cached`` looks in ("" = no cache)."""
    if "TIKTOKEN_CACHE_DIR" in os.environ:
        return os.environ["TIKTOKEN_CACHE_DIR"]
    if "DATA_GYM_CACHE_DIR" in os.environ:
        return os.environ["DATA_GYM_CACHE_DIR"]
    return os.path.join(tempfile.gettempdir(), "data-gym-cache")


def seed_tiktoken_cache() -> Path | None:
    """Copy the wheel's encoding into tiktoken's cache unless it is there.

    Returns the cache file it wrote, else ``None`` (no bundled copy, caching
    off, already cached, or the directory is not writable -- tiktoken then
    downloads, as it would have). tiktoken checks the file's sha256 when it
    reads it.
    """
    bundled = importlib.resources.files(__package__).joinpath(_BUNDLED)
    cache_dir = _tiktoken_cache_dir()
    if not bundled.is_file() or not cache_dir:
        return None
    target = Path(cache_dir) / hashlib.sha1(_O200K_URL.encode()).hexdigest()
    if target.exists():
        return None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        partial.write_bytes(bundled.read_bytes())
        partial.replace(target)
    except OSError:
        return None
    return target


async def warm_token_counter() -> None:
    """Load the token-counting encoding once, off the event loop.

    Raises:
        TokenizerAssetError: The encoding could not be loaded -- in practice
            the first-run download failed and no cached copy exists.
    """
    seeded = seed_tiktoken_cache()
    if seeded is not None:
        logger.info("token_counter_encoding_seeded", path=str(seeded))
    try:
        await anyio.to_thread.run_sync(encode, _PROBE)
    except Exception as exc:  # tiktoken surfaces HTTP, TLS and OS errors alike
        raise TokenizerAssetError(
            "the token-counting encoding (tiktoken o200k_base) could not be "
            "loaded. An installed wheel carries it; a source checkout downloads "
            "it once on first start and caches it. Allow the download, or "
            "pre-populate TIKTOKEN_CACHE_DIR with a copy taken from a machine "
            f"that has it. Cause: {exc!r}"
        ) from exc
    logger.info("token_counter_warm")
