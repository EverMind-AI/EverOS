"""Memory deletion against a real ``everos server`` process.

Each test starts its own server on a fresh memory root, with the LLM and the
embedding provider reached through a local proxy that can hold or delay a
chosen call (so a deletion can be timed into a window), and reads the facts
from the layer that owns them: the md files, ``system.db``, ``ome.db``, the
index, ``/metrics``. Real credentials come from the shell
(``OPENROUTER_API_KEY``, ``DEEPINFRA_API_KEY``); without them every test is
skipped -- and a skip is not a pass.
"""
