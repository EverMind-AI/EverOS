"""Concrete derived-index adapters.

Each module here implements the ports in
:mod:`everos.infra.persistence.index.protocols` for one storage engine, and
owns everything specific to it: predicate rendering, physical schema, and the
connection lifecycle. LanceDB, Milvus, and SeekDB are the registered engines.

They live outside :mod:`everos.infra.persistence.index` on purpose. That
package is the boundary outer layers depend on, and mixing adapters into it
made "which of these is the abstraction?" a question a reader had to answer by
opening files. Adding another backend means adding a module here, not editing
the abstraction.

Nothing outside ``persistence`` should import these directly — go through the
``index`` facade, which resolves the configured backend at call time.
"""

from __future__ import annotations
