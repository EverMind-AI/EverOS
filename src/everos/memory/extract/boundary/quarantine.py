"""Set aside a session buffer the model keeps failing to segment.

One unusable boundary reply is noise. The same session failing call after
call means its buffer holds something the model cannot segment, and every
later call would merge it back in and fail the same way. After
:data:`POISON_CALLS` consecutive failed calls the buffer (and the failing
call's own messages) moves to the ``quarantine`` track and the session
starts over. Infrastructure errors are never counted.
"""

from __future__ import annotations

__all__ = ["POISON_CALLS", "SessionQuarantine"]

from collections.abc import Sequence

from everos.core.errors import SessionBufferQuarantinedError
from everos.core.observability.logging import get_logger
from everos.memory.models import CanonicalMessage
from everos.memory.ports import SessionBufferStore, SessionKey

logger = get_logger(__name__)

POISON_CALLS = 3
"""Consecutive failed calls on one session before its buffer is set aside."""


class SessionQuarantine:
    """Counts consecutive failed boundary calls per session, in memory.

    A restart starts every count again.

    Args:
        session_buffer: Where the buffer is moved to the quarantine track.
        poison_calls: Failed calls in a row that trigger the quarantine.
    """

    def __init__(
        self, session_buffer: SessionBufferStore, poison_calls: int = POISON_CALLS
    ) -> None:
        self._session_buffer = session_buffer
        self._poison_calls = poison_calls
        self._failures: dict[SessionKey, int] = {}

    def reset(self, session: SessionKey) -> None:
        """Forget the session's failures after a usable reply."""
        self._failures.pop(session, None)

    async def record_failure(
        self, session: SessionKey, unsaved: Sequence[CanonicalMessage]
    ) -> None:
        """Count one failed call; on the last allowed one, quarantine the session.

        Returns normally while the session is under the limit, so the caller
        re-raises its own error.

        Args:
            session: The session whose call failed.
            unsaved: The call's messages not yet in the buffer.

        Raises:
            SessionBufferQuarantinedError: This failure reached the limit; the
                buffer and ``unsaved`` were moved to the quarantine track.
        """
        failures = self._failures.get(session, 0) + 1
        if failures < self._poison_calls:
            self._failures[session] = failures
            return
        self._failures.pop(session, None)
        moved = await self._session_buffer.quarantine(session, unsaved)
        logger.error(
            "memorize_session_buffer_quarantined",
            session_id=session.session_id,
            app_id=session.app_id,
            project_id=session.project_id,
            messages=moved,
            failed_calls=self._poison_calls,
        )
        raise SessionBufferQuarantinedError(
            f"boundary detection failed on {self._poison_calls} consecutive calls "
            f"for session {session.session_id!r}; {moved} buffered messages were "
            "moved to the 'quarantine' track so the session can continue"
        ) from None
