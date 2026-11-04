"""Stateless algorithms developed alongside EverOS.

One family per subpackage: ``types`` (the data contracts every family and its
hosts share), ``llm`` (the ``LLMClient`` protocol, request types and reply
parsing), ``utils`` (prompt rendering, token budgets, timestamps), ``boundary``,
``user_memory``, ``agent_memory``, ``clustering``, ``knowledge`` and ``rank``.

Callers own persistence, configuration, clients, and workflow state. Algorithms
receive data and capabilities explicitly and keep no state between calls.
Import each algorithm family through its public facade, e.g. ``algo.rank``.
"""

from __future__ import annotations

__all__: list[str] = []
