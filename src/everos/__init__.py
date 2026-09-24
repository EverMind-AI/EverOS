"""everos — md-first memory extraction framework."""

from __future__ import annotations

import os
import site
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("everos")
except PackageNotFoundError:
    # Editable install without dist-info, or running from a source tree that
    # was never installed. Fall back to a sentinel rather than crash imports.
    __version__ = "0.0.0+unknown"


# Handles returned by ``os.add_dll_directory``; a collected handle would
# silently unregister its directory again, so they live for the process.
_dll_dir_handles: list[object] = []


def _register_runtime_dll_dirs(
    prefix: str = sys.prefix, user_base: str | None = None
) -> list[str]:
    """On Windows, let compiled extensions find the MSVC C++ runtime.

    ``greenlet`` -- under SQLAlchemy's async engine, so under every SQLite
    call -- is a C++ extension whose wheel does not bundle ``msvcp140.dll``,
    and a stock Windows install does not ship it: ``import greenlet`` dies
    with ``DLL load failed while importing _greenlet``. The ``msvc-runtime``
    dependency drops the runtime DLLs into ``sys.prefix`` and its ``Scripts``
    dir, but that alone is not enough: a venv's ``python.exe`` is a launcher
    (uv's is a trampoline), so the loader's "application directory" is the
    base interpreter's and the DLLs sit unseen next door. Registering the
    directories with ``os.add_dll_directory`` fixes the search path for every
    extension imported afterwards -- which is why this runs from the package
    ``__init__``: nothing in everos is imported before it.

    Feature-detected rather than platform-checked: ``os.add_dll_directory``
    exists only on Windows, so elsewhere this is a no-op. Returns the
    directories it registered.
    """
    add = getattr(os, "add_dll_directory", None)
    if add is None:
        return []
    # ``pip install`` falls back to a per-user install when site-packages is
    # not writable (Python under Program Files); the wheel's data files then
    # land under ``site.getuserbase()`` rather than ``sys.prefix``.
    if user_base is None:
        user_base = site.getuserbase()
    registered: list[str] = []
    for base in dict.fromkeys((prefix, user_base)):
        for d in (base, os.path.join(base, "Scripts")):
            if os.path.isfile(os.path.join(d, "msvcp140.dll")):
                _dll_dir_handles.append(add(d))
                registered.append(d)
    return registered


_register_runtime_dll_dirs()
