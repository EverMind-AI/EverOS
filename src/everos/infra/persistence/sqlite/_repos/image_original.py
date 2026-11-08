"""Repository for ``image_original`` — the reference-counted picture files.

Two operations need a transaction rather than a read-then-write: recording an
arrival and retiring a reference. Both are expressed as single UPDATE
statements so two concurrent writers converge on the right count instead of
overwriting each other, which is the failure a read-then-write would produce
when the same picture arrives twice at once.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import func, select, update

from everos.core.utils.datetime import get_utc_now
from everos.infra.persistence.sqlite._base.repository import RepoBase
from everos.infra.persistence.sqlite._base.session import session_scope

from .._tables import ImageOriginal


class ImageOriginalRepo(RepoBase[ImageOriginal]):
    model = ImageOriginal

    async def record_arrival(
        self,
        *,
        image_id: str,
        media_type: str,
        byte_size: int,
        width: int,
        height: int,
        original_path: str,
    ) -> ImageOriginal:
        """Count one arrival of ``image_id``; insert the row on the first one.

        An existing row keeps its ``first_seen_at`` and bumps ``ref_count``.
        Uses a conditional UPDATE rather than a Python-side increment so two
        concurrent arrivals of the same picture add one reference each instead
        of both writing back the same number.
        """
        async with session_scope(self._factory) as s:
            result = await s.execute(
                update(ImageOriginal)
                .where(ImageOriginal.image_id == image_id)  # type: ignore[arg-type]
                .values(ref_count=ImageOriginal.ref_count + 1)
            )
            if result.rowcount:  # type: ignore[attr-defined]
                await s.commit()
                row = await s.get(ImageOriginal, image_id)
            else:
                row = ImageOriginal(
                    image_id=image_id,
                    media_type=media_type,
                    byte_size=byte_size,
                    width=width,
                    height=height,
                    original_path=original_path,
                    first_seen_at=get_utc_now(),
                    ref_count=1,
                )
                s.add(row)
                await s.commit()
                await s.refresh(row)
        return row

    async def record_unreferenced(
        self,
        *,
        image_id: str,
        media_type: str,
        byte_size: int,
        width: int,
        height: int,
        original_path: str,
    ) -> ImageOriginal:
        """Insert ``image_id`` at zero references if it is not known yet.

        The question-picture case: the file must exist and be addressable, but
        no memory points at it, so the row must not claim one. An existing row
        is left exactly as it is — a real arrival already counted, and raising
        the count from a query would make a forget under-release.
        """
        async with session_scope(self._factory) as s:
            row = await s.get(ImageOriginal, image_id)
            if row is not None:
                return row
            row = ImageOriginal(
                image_id=image_id,
                media_type=media_type,
                byte_size=byte_size,
                width=width,
                height=height,
                original_path=original_path,
                first_seen_at=get_utc_now(),
                ref_count=0,
            )
            s.add(row)
            try:
                await s.commit()
            except Exception:
                # Two queries with the same new picture at once: the loser
                # reads the winner's row instead of failing the search.
                await s.rollback()
                existing = await s.get(ImageOriginal, image_id)
                if existing is None:
                    raise
                return existing
            await s.refresh(row)
        return row

    async def release_reference(self, image_id: str) -> int:
        """Drop one reference to ``image_id``, floored at zero.

        Returns the reference count left. Zero means nothing points at the
        picture any more; the file itself is left on disk, because undo has to
        be able to bring the memory back without a restore.
        """
        async with session_scope(self._factory) as s:
            await s.execute(
                update(ImageOriginal)
                .where(
                    ImageOriginal.image_id == image_id,  # type: ignore[arg-type]
                    ImageOriginal.ref_count > 0,
                )
                .values(ref_count=ImageOriginal.ref_count - 1)
            )
            await s.commit()
            remaining = await s.execute(
                select(ImageOriginal.ref_count).where(
                    ImageOriginal.image_id == image_id  # type: ignore[arg-type]
                )
            )
        return int(remaining.scalar_one_or_none() or 0)

    async def mark_forgotten(self, image_id: str) -> bool:
        """Take ``image_id`` out of circulation; whether the row existed.

        Idempotent: forsetting an already-forgotten picture keeps the first
        timestamp, so the undo window is not silently extended by a repeat
        call.
        """
        async with session_scope(self._factory) as s:
            row = await s.get(ImageOriginal, image_id)
            if row is None:
                return False
            if row.forgotten_at is None:
                row.forgotten_at = get_utc_now()
                await s.commit()
        return True

    async def restore(self, image_id: str) -> bool:
        """Undo :meth:`mark_forgotten`; whether the row existed."""
        async with session_scope(self._factory) as s:
            row = await s.get(ImageOriginal, image_id)
            if row is None:
                return False
            row.forgotten_at = None
            await s.commit()
        return True

    async def count_rows(self) -> int:
        """Distinct pictures stored."""
        return await self.count()

    async def total_bytes(self) -> int:
        """Bytes held by stored originals (not their derived copies)."""
        async with session_scope(self._factory) as s:
            stmt = select(func.coalesce(func.sum(ImageOriginal.byte_size), 0))
            return int((await s.execute(stmt)).scalar_one())

    async def list_unreferenced(
        self, *, older_than: dt.datetime
    ) -> list[ImageOriginal]:
        """Pictures nothing references, first seen before ``older_than``.

        A sweep target, not a delete: the caller decides whether such a file is
        garbage or an undo waiting to happen.
        """
        async with session_scope(self._factory) as s:
            stmt = select(ImageOriginal).where(
                ImageOriginal.ref_count == 0,  # type: ignore[arg-type]
                ImageOriginal.first_seen_at < older_than,  # type: ignore[arg-type]
            )
            return list((await s.execute(stmt)).scalars().all())
