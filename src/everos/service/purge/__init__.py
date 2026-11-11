"""Purge: clear a project, an app or the whole memory root while serving.

Public API:
    from everos.service.purge import PurgeRunGate
"""

from ._gate import PurgeRunGate as PurgeRunGate

__all__ = ["PurgeRunGate"]
