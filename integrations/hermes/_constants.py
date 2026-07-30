"""Provider-level constants and small runtime invariants.

Keep this module dependency-free: it is imported by other plugin modules and is
a contract boundary for tests.
"""

from __future__ import annotations

from collections.abc import Set

# ── Circuit breaker / async timeouts ─────────────────────────────────────────

_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120.0
_PREFETCH_WAIT_SECS = 1.5
_MAX_PREFETCH_CHARS = 4000
_ADD_BATCH_SIZE = 500

# ── Defaults ───────────────────────────────────────────────────────────────────

_DEFAULT_USER_ID = "hermes-user"
_DEFAULT_AGENT_ID = "hermes"
_DEFAULT_APP_ID = "default"
_DEFAULT_PROJECT_ID = "default"
_DEFAULT_API_URL = "http://127.0.0.1:8000"

# ── EverOS scope-id validation ─────────────────────────────────────────────────

_SCOPE_ID_MIN_LEN = 1
_SCOPE_ID_MAX_LEN = 128
# Note: docs/api.md §ScopeId lists ^[a-zA-Z0-9_.-]+$, but the server-side
# PathSafeId validator in src/everos/entrypoints/api/routes/memorize.py also
# accepts '@' and '+'. We match the server so valid ids are not rejected.
_SCOPE_ID_CHARSET = r"^[a-zA-Z0-9_.@+-]+$"
_SCOPE_TRAVERSAL_TOKENS: Set[str] = frozenset({".", ".."})

# ── Search defaults ──────────────────────────────────────────────────────────

_DEFAULT_SEARCH_METHOD = "hybrid"
# Default for the agent-facing tool only. EverOS API default is -1.
_DEFAULT_TOOL_SEARCH_TOP_K = 5

# ── Tool names ─────────────────────────────────────────────────────────────────

TOOL_SEARCH = "everos_search"
TOOL_LIST = "everos_list"
TOOL_ADD = "everos_add"
TOOL_FLUSH = "everos_flush"

# ── Memory-write mirroring scope ───────────────────────────────────────────────

# Hermes `target` values that should be mirrored into EverOS's user track.
_MIRROR_TARGETS: Set[str] = frozenset({"user"})
