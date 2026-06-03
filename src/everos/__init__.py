"""everos — md-first memory extraction framework."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("everos")
except PackageNotFoundError:
    # Editable install without dist-info, or running from a source tree that
    # was never installed. Fall back to a sentinel rather than crash imports.
    __version__ = "0.0.0+unknown"
