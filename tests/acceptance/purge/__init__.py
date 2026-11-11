"""Purge against a real ``everos server`` process, with no paid provider.

Each test starts its own server on a fresh memory root. The LLM, embedding
and rerank calls go through the hold / delay proxy of the deletion suite to a
local fake upstream (:mod:`.fake_upstream`): it answers each LLM call with the
scripted reply its ``response_format`` names and embeds by hashing, so the
suite needs no credentials and costs nothing. Facts are read from the layer
that owns them: the md files, ``system.db``, ``ome.db`` and the index.
"""
