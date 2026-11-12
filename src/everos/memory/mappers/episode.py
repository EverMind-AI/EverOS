"""Domain :class:`~everos.memory.models.Episode` -> everalgo ``Episode``.

The reverse direction is :meth:`Episode.from_algo`: building a domain episode
needs the owner, session and source memcell only the pipeline knows.
"""

from __future__ import annotations

__all__ = ["to_algo"]

from everalgo.types import Episode as AlgoEpisode

from everos.memory.models import Episode


def to_algo(episode: Episode) -> AlgoEpisode:
    """Algo episode carrying the narrative fields; everos context stays behind."""
    return AlgoEpisode(
        owner_id=episode.owner_id,
        episode=episode.episode,
        subject=episode.subject,
        summary=episode.summary,
        timestamp=episode.timestamp,
    )
