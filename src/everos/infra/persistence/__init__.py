"""Persistence layer: md-first storage adapters plus the memory-root primitives.

The top-level facade exposes only the memory-root path manager and its
cross-process lock; storage-specific APIs live behind each subpackage
facade (``markdown`` / ``sqlite`` / ``lancedb`` / ``records``), and
:mod:`.factories` builds the adapters behind the memory ports. The
factories stay out of this facade because the storage subpackages they
load import ``MemoryRoot`` from here.

External usage::

    from everos.infra.persistence import MemoryRoot
    from everos.infra.persistence.factories import build_stores

    stores = build_stores(MemoryRoot.resolve())
"""

from .locking import LockError as LockError
from .locking import memory_root_lock as memory_root_lock
from .memory_root import MemoryRoot as MemoryRoot
from .memory_root import app_dir_name as app_dir_name
from .memory_root import app_id_from_dir as app_id_from_dir
from .memory_root import project_dir_name as project_dir_name
from .memory_root import project_id_from_dir as project_id_from_dir

__all__ = [
    "LockError",
    "MemoryRoot",
    "app_dir_name",
    "app_id_from_dir",
    "memory_root_lock",
    "project_dir_name",
    "project_id_from_dir",
]
