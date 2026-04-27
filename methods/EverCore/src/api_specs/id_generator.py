"""
System-generated ID utilities.

All system-generated IDs share the `gen_` prefix to distinguish them from
user-supplied IDs. Sub-prefixes identify the ID type:

- `gen_solo_` — auto-generated group_id (personal / single-user scene)
- `gen_msg_` — auto-generated message_id (when client omits message_id)
- `gen_sdr_` — auto-generated sender_id (personal scene, assistant role)

User-supplied IDs MUST NOT start with `GEN_PREFIX` to avoid ambiguity.
"""

import hashlib

# ---------------------------------------------------------------------------
# Prefix constants
# ---------------------------------------------------------------------------

GEN_PREFIX = "gen_"
"""Common prefix for all system-generated IDs."""

GEN_SOLO_GROUP_PREFIX = "gen_solo_"
"""Prefix for auto-generated group IDs (single-user / personal scene)."""

GEN_MESSAGE_PREFIX = "gen_msg_"
"""Prefix for auto-generated message IDs."""

GEN_SENDER_PREFIX = "gen_sdr_"
"""Prefix for auto-generated sender IDs (personal scene, assistant role)."""

DEFAULT_SESSION_ID = "-1"
"""Sentinel value for session_id when session isolation is not applicable.

Used in two scenarios:
- Group scene: groups do not use session isolation.
- Personal scene: when the client does not provide a session_id.
"""


# ---------------------------------------------------------------------------
# ID generators
# ---------------------------------------------------------------------------


def generate_single_user_group_id(user_id: str) -> str:
    """Generate a deterministic group_id for the single-user (personal) scene.

    The result is prefixed with `gen_solo_` so it can always be distinguished
    from user-supplied group IDs. The `solo` segment maps to ScenarioType.SOLO.

    Args:
        user_id: The owner user ID.

    Returns:
        str: Generated group_id in format ``gen_solo_{md5(user_id)[:12]}``.
    """
    hash_value = hashlib.md5(user_id.encode("utf-8")).hexdigest()[:12]
    return f"{GEN_SOLO_GROUP_PREFIX}{hash_value}"


def generate_message_id(context_id: str, timestamp_ms: int) -> str:
    """Generate a deterministic message_id when the client does not provide one.

    The result is prefixed with `gen_msg_` so it can always be distinguished
    from user-supplied message IDs.

    Args:
        context_id: Context identifier (user_id for personal, group_id for group).
        timestamp_ms: Message timestamp in unix milliseconds.

    Returns:
        str: Generated message_id in format ``gen_msg_{md5(context_id + ts)[:12]}``.
    """
    raw = f"{context_id}_{timestamp_ms}"
    hash_value = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return f"{GEN_MESSAGE_PREFIX}{hash_value}"


def generate_assistant_sender_id(user_id: str) -> str:
    """Generate a deterministic sender_id for the assistant in personal scene.

    The result is prefixed with `gen_sdr_` so it can always be distinguished
    from user-supplied sender IDs.

    Args:
        user_id: The owner user ID.

    Returns:
        str: Generated sender_id in format ``gen_sdr_{md5(user_id)[:12]}``.
    """
    hash_value = hashlib.md5(user_id.encode("utf-8")).hexdigest()[:12]
    return f"{GEN_SENDER_PREFIX}{hash_value}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


# Per-field validation rules: (check_fn, error_message_template)
# Template receives {field_name} and {value} via str.format_map().
_FIELD_RULES: dict[str, list[tuple]] = {
    "group_id": [
        (
            lambda v: v.startswith(GEN_PREFIX),
            "{field_name} must not start with reserved prefix '{prefix}'. "
            "IDs starting with '{prefix}' are reserved for system-generated values.",
        )
    ],
    "message_id": [
        (
            lambda v: v.startswith(GEN_PREFIX),
            "{field_name} must not start with reserved prefix '{prefix}'. "
            "IDs starting with '{prefix}' are reserved for system-generated values.",
        )
    ],
    "sender_id": [
        (
            lambda v: v.startswith(GEN_PREFIX),
            "{field_name} must not start with reserved prefix '{prefix}'. "
            "IDs starting with '{prefix}' are reserved for system-generated values.",
        )
    ],
    "session_id": [
        (
            lambda v: v == DEFAULT_SESSION_ID,
            "{field_name} must not be '{value}'. "
            "This value is reserved for system use when session isolation is not applicable.",
        )
    ],
}


def validate_input_id(field_name: str, value: str) -> None:
    """Validate a user-supplied value against reserved patterns for the given field.

    Dispatches to field-specific rules defined in `_FIELD_RULES`.
    Fields without rules pass through silently.

    Args:
        field_name: The field being validated (e.g. "group_id", "message_id", "session_id").
        value: The user-supplied value to check.

    Raises:
        ValueError: If the value violates any rule for the given field.
    """
    rules = _FIELD_RULES.get(field_name)
    if not rules:
        return
    for check_fn, msg_template in rules:
        if check_fn(value):
            raise ValueError(
                msg_template.format(
                    field_name=field_name, value=value, prefix=GEN_PREFIX
                )
            )
