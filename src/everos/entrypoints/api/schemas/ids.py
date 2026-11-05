"""Path-safe identifier types for request fields that become directories.

``app_id`` / ``project_id`` / ``sender_id`` all become directory segments
under the memory root (``sender_id`` flows through to ``owner_id`` and is
joined into the daily-log write path), so they must reject ``.`` and ``..``
(path traversal). The basic character whitelist is enforced via ``pattern``
(pydantic_core uses the Rust regex engine, which does NOT support
lookaround), and the two reserved tokens are filtered out with a follow-up
``AfterValidator``.

``@`` and ``+`` are admitted so real-world ids survive (email-style
``user@example.com``, plus-addressing ``user+tag``); both are legal,
non-separator filename chars on every target filesystem (incl. NTFS, whose
reserved set is ``< > : " / \\ | ? *``). The genuinely path-dangerous chars
(``/`` ``\\`` NUL) stay out of the whitelist, and ``.``/``..`` stay blocked
by the token filter; the markdown writer's ``_ensure_within_root`` is the
final backstop regardless.

``session_id`` never becomes a directory but is stored as one inline
markdown value, which is trimmed and ends at the line; ``SessionId`` admits
only ids that come back from that unchanged.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from typing import Annotated

from pydantic import AfterValidator

PATH_SAFE_CHARSET = r"^[a-zA-Z0-9_.@+-]+$"
_PATH_TRAVERSAL_TOKENS = frozenset({".", ".."})


_PATH_SAFE_RE = re.compile(PATH_SAFE_CHARSET)


def _reject_path_traversal(value: str) -> str:
    if value in _PATH_TRAVERSAL_TOKENS:
        raise ValueError("'.' and '..' are reserved (path traversal)")
    if not _PATH_SAFE_RE.match(value):
        raise ValueError(
            "Only alphanumerics, underscore, dot, hyphen, @, and + are allowed"
        )
    return value


def _reject_reserved_scope_dir(reserved: str) -> Callable[[str], str]:
    """Refuse the on-disk directory name of the default scope as that scope's id.

    ``app_dir_name("default")`` is ``default_app`` and ``app_id_from_dir``
    maps it back to ``default``; an app whose id is literally ``default_app``
    would share the default app's directory and have its index rows
    attributed to ``default``. The same for ``project_id`` and
    ``default_project``. Only that field/name pair aliases, so only it is
    refused: ``sender_id`` never goes through the scope mapping, and
    ``app_id="default_project"`` round-trips unchanged.
    """

    def check(value: str) -> str:
        _reject_path_traversal(value)
        if value == reserved:
            raise ValueError(
                f"{reserved!r} is the directory name of the default scope; "
                "use 'default'"
            )
        return value

    return check


def _one_trimmed_line(value: str) -> str:
    """Refuse what a daily-log entry cannot hold as given: a line break, a
    control character, leading or trailing whitespace. Stored, such an id
    reads back as another one, and a deletion of the id sent finds nothing."""
    if value != value.strip() or any(
        unicodedata.category(ch) in ("Cc", "Cs", "Zl", "Zp") for ch in value
    ):
        raise ValueError(
            "must be one line without control characters or surrounding whitespace"
        )
    return value


PathSafeId = Annotated[str, AfterValidator(_reject_path_traversal)]
SessionId = Annotated[str, AfterValidator(_one_trimmed_line)]
AppScopeId = Annotated[str, AfterValidator(_reject_reserved_scope_dir("default_app"))]
ProjectScopeId = Annotated[
    str, AfterValidator(_reject_reserved_scope_dir("default_project"))
]
