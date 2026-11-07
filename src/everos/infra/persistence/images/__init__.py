"""Picture files under the memory root.

The originals live in ``figures/`` at the root, not inside the
``<app>/<project>/`` trees: one picture is referenced by every memory that saw
it, so a per-scope copy would break the "stored once" invariant. Their
metadata and reference counts live in SQLite (``image_original``), which is
what lets a store answer "how many distinct pictures do I hold" without
walking the tree.

External usage::

    from everos.infra.persistence.images import FileImageStore
"""

from .store import FileImageStore as FileImageStore

__all__ = ["FileImageStore"]
