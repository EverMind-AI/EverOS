"""The derived index the application holds: one backend plus the checks
every opening of it runs.

:class:`DerivedIndex` wraps the configured engine's backend (LanceDB or
Milvus). The backend knows its engine; this class knows what is the same on
both -- primary-key declarations, the record-width check, the configuration
check against an :class:`EmbeddingProvenance`, row counts -- so the server
and every CLI phase run one sequence instead of each repeating it.
"""

from __future__ import annotations

__all__ = ["DerivedIndex", "check_primary_keys"]

from collections.abc import Iterable
from typing import Any

from everos.core.errors import EmbeddingProvenanceError

from ..records import Episode
from ._provenance import EmbeddingProvenance
from .protocols import IndexBackend
from .repositories import IndexRepositories
from .schema import schema_for


class DerivedIndex:
    """The configured derived-index backend and its shared lifecycle.

    Args:
        backend: The engine's backend, built over the application's
            connection.
        embedding_dim: ``[embedding] dim`` as the application's settings
            report it, checked against the width the record classes baked in.
    """

    def __init__(self, backend: IndexBackend, *, embedding_dim: int) -> None:
        self._backend = backend
        self._embedding_dim = embedding_dim

    @property
    def name(self) -> str:
        """The engine's name (``lancedb`` / ``milvus``)."""
        return self._backend.name

    @property
    def repositories(self) -> IndexRepositories:
        """One repository per business table."""
        return self._backend.repositories

    async def connect(self) -> Any:
        return await self._backend.connect()

    async def shutdown(self) -> None:
        await self._backend.shutdown()

    def assert_record_dim_matches_settings(self) -> None:
        """Fail loud if the record classes and ``Settings`` disagree on the width.

        The record classes resolve ``[embedding] dim`` with a narrow reader of
        their own, because the width has to be known while they are being
        imported (``Vector(dim)`` bakes it into the pydantic core schema) and
        building ``Settings`` on that path would make a malformed config break
        every CLI command. Two readers means they can diverge -- a different
        memory root, a key one of them cannot parse -- and the symptom would
        otherwise be tables quietly built at the wrong width.

        Public rather than private because it has to run on paths that skip
        the rest of verification: ``cascade rebuild`` passes ``verify=False``,
        and it is the one command that unconditionally creates tables --
        exactly where a disagreement would bake the wrong width into fresh
        tables.

        Raises:
            EmbeddingProvenanceError: The two readers disagree.
        """
        baked = Episode.to_arrow_schema().field("vector").type.list_size
        if baked != self._embedding_dim:
            raise EmbeddingProvenanceError(
                f"the index record classes were built for dim={baked} but "
                f"configuration reports dim={self._embedding_dim}.\n"
                f"  The two readers disagree -- check EVEROS_ROOT and "
                f"<root>/everos.toml, then start again."
            )

    async def verify_configuration(
        self, provenance: EmbeddingProvenance, *, may_attest: bool = True
    ) -> None:
        """Every "does this index match the configuration" check, in order.

        One call so the server and the CLI phases cannot run different
        subsets: :meth:`startup` uses it, and so does the CLI's own connect /
        verify / ensure sequence, which exists separately because ``cascade
        rebuild`` has to skip the middle step.

        Raises:
            EmbeddingProvenanceError: the record classes and ``Settings``
                disagree on the vector width, or the configured embedding
                cannot have produced the stored vectors.
            IndexSchemaMismatchError: a stored table's columns drifted from the
                current schema.
        """
        self.assert_record_dim_matches_settings()
        await self._backend.verify_business_schemas()
        await provenance.verify(self._backend, may_attest=may_attest)

    async def startup(self, provenance: EmbeddingProvenance) -> Any:
        """Open the index and refuse to use one the configuration does not match.

        Order is load-bearing. Schema verification goes first because it is
        the most specific complaint (a column of the wrong type names both
        widths), and because it is also what creates missing tables -- so
        provenance runs against tables that exist. Index creation goes last:
        there is no point building indexes on a store that is about to be
        rejected.
        """
        check_primary_keys(repo.schema for repo in self.repositories.all())
        connection = await self.connect()
        await self.verify_configuration(provenance)
        await self.ensure_business_indexes()
        return connection

    async def ensure_business_indexes(self) -> None:
        check_primary_keys(repo.schema for repo in self.repositories.all())
        await self._backend.ensure_business_indexes()

    async def verify_business_schemas(self) -> None:
        await self._backend.verify_business_schemas()

    async def drop_business_tables(self) -> list[str]:
        return await self._backend.drop_business_tables()

    async def read_embedding_model(self) -> str | None:
        """Model the backend says this index was built with."""
        return await self._backend.read_embedding_model()

    async def write_embedding_model(self, model: str) -> None:
        """Attest ``model`` as the producer of this index's vectors."""
        await self._backend.write_embedding_model(model)

    async def count_business_rows(self) -> dict[str, int]:
        """Rows currently held per business table, keyed by table name.

        Backend-neutral: ``count`` is on the repository port, so this answers
        for Milvus as well as LanceDB. Cheap on LanceDB (dataset metadata) and
        one round trip per collection on Milvus -- which is why callers take
        it as a snapshot rather than on every request.

        The number it answers is "what the index actually holds", and the only
        thing that can contradict a cascade ledger claiming every file is
        indexed (issue #501).
        """
        return {repo.table_name: await repo.count() for repo in self.repositories.all()}


def check_primary_keys(schemas: Iterable[type[Any]]) -> None:
    """Fail at startup, not at first write, when a table has no PRIMARY_KEY.

    For an undeclared table ``schema_for`` raises the same "has no
    PRIMARY_KEY" message the LanceDB write path raises, so it is caught by
    whichever runs first, on either backend; the write-path check stays as
    the last line of defence.
    """
    for schema in schemas:
        schema_for(schema)
