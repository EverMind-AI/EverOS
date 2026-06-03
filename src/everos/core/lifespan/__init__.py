"""Application lifespan composition (chassis only).

This subpackage holds the *generic* lifespan machinery — the
:class:`LifespanProvider` ABC, :func:`build_lifespan` factory, and
chassis-level providers that are independent of any storage backend
(observability metrics, etc.). Concrete storage-backend providers
(SQLite / LanceDB) live next to the entrypoint that composes them
(see :mod:`everos.entrypoints.api.lifespans`) so ``core`` stays free
of concrete-backend imports.

External usage:
    from everos.core.lifespan import (
        LifespanProvider,
        MetricsLifespanProvider,
        build_lifespan,
    )
"""

from .base import LifespanProvider as LifespanProvider
from .factory import build_lifespan as build_lifespan
from .metrics_lifespan import MetricsLifespanProvider as MetricsLifespanProvider

__all__ = [
    "LifespanProvider",
    "MetricsLifespanProvider",
    "build_lifespan",
]
