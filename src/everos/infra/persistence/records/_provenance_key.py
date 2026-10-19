"""The key an index records its embedding model under.

Lives with the record declarations because both backends already import them
and ``records`` imports no sibling persistence package, so there is no cycle
to route around -- the value is one string and does not need two copies that
can drift.

It is a storage-layout detail, not part of a table's shape: each backend still
decides *where* the key hangs (Arrow field metadata on a column, a collection
property), the port only says the model travels with the index.
"""

from __future__ import annotations

EMBEDDING_MODEL_KEY = "everos.embedding_model"
