"""
Memory shared types for OpenHer.

These types bridge the two memory providers:
  - SoulMem (behavioral memory, always-on SQLite layer)
  - EverCore (declarative memory, cross-session persistence)

The SessionContext is the key data structure loaded from EverCore
at session start — it provides relationship priors, user profile,
episode summaries, and foresight data that expand the neural
network's perception from 8D to 12D.

Full source: https://github.com/kellyvv/OpenHer/blob/main/memory/types.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Memory:
    """A single memory entry (SoulMem behavioral layer)."""
    memory_id: int = 0
    user_id: str = ""
    persona_id: str = ""
    content: str = ""
    category: str = "conversation"  # conversation | fact | event | preference
    importance: float = 0.5
    source_turn: int = 0
    created_at: float = 0.0


@dataclass
class SessionContext:
    """
    EverCore session context (declarative memory).

    Loaded once at session start, this contains everything the
    persona needs to know about the user from past sessions:

    - user_profile:       Who they are (name, preferences, occupation)
    - episode_summary:    What happened between us (narrative history)
    - foresight_text:     What we should pay attention to (unresolved topics)
    - relationship_*:     4D vector feeding the neural network
    - has_history:        Whether there's prior interaction (gates search)

    These values feed into the ChatAgent lifecycle at multiple steps:
    - Step 0:   Session context loaded
    - Step 2:   user_profile + episode_summary inject into Critic prompt
    - Step 2.5: relationship_* feed EMA computation
    - Step 5:   4D vector enters neural network as context features
    - Step 8.5: Used as fallback when async search times out
    """
    user_id: str = ""
    persona_id: str = ""
    user_profile: str = ""
    episode_summary: str = ""
    foresight_text: str = ""
    relationship_depth: float = 0.0
    emotional_valence: float = 0.0
    trust_level: float = 0.0
    pending_foresight: float = 0.0
    has_history: bool = False
    raw_data: Optional[dict] = None
