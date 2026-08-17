---
paths:
  - "src/everos/infra/**/*.py"
  - "src/everos/memory/**/*.py"
  - "src/everos/service/**/*.py"
  - "src/everos/component/**/*.py"
  - "src/everos/core/**/*.py"
---

# Module docstring rule

Every non-trivial module in the domain/infra layers opens with a docstring that
explains **intent and contract**, not just a one-line label.

A good module docstring states:

- **What** the module is responsible for (one sentence).
- **The load-bearing invariants** — the rules a reader must know to change it
  safely (partition keys, what is/isn't written, defaults, ignored flags).
- **External usage** when the module is a package facade (a short import example).

Example (abbreviated, from `memory/search/manager.py`):

```python
"""SearchManager — top-level orchestrator for POST /api/v2/memory/search.

Hard partition by owner_type: user → episodes (+ profiles), agent →
agent_cases + agent_skills. The manager never writes to storage; it only
reads LanceDB + markdown.
"""
```

Prefer prose that would save the next engineer a debugging session over
boilerplate. If a module is genuinely trivial (a 3-line constant), a one-liner
is fine — but most modules here are not.
