"""``everos export`` / ``verify`` / ``import`` — portable memory archive commands.

Package a memory root's source-of-truth Markdown and configuration into a
self-contained ``.tar.gz`` archive with a SHA-256 manifest, verify archive
integrity without touching any live root, and restore into a target root with
automatic derived-index rebuild and before/after entry-count verification.

Mounted as top-level leaf commands on the root CLI app (matching ``everos init``
and ``everos demo``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated

import typer

from everos.core.errors import (
    ArchiveConflictError,
    ArchiveIntegrityError,
    InvalidInputError,
    MemoryRootLockedError,
    NotFoundError,
)
from everos.entrypoints.cli._process import configure_interactive
from everos.service import (
    export_memory_root,
    import_memory_archive,
    verify_memory_archive,
)


def _echo_entry_counts(label: str, counts: Mapping[str, int]) -> None:
    typer.echo(f"{label}:")
    for kind, count in counts.items():
        typer.echo(f"  {kind + ':':<22} {count}")
    typer.echo(f"  {'total:':<22} {sum(counts.values())}")


def register(parent: typer.Typer) -> None:
    """Attach ``export``, ``verify``, and ``import`` commands to the root CLI app."""

    @parent.command("export")
    def export_cmd(
        out: Annotated[
            Path,
            typer.Argument(
                help="Output archive path (for example: backup.tar.gz).",
            ),
        ],
        root: Annotated[
            str | None,
            typer.Option(
                "--root",
                help="Memory root directory (env: EVEROS_ROOT, default: ~/.everos).",
            ),
        ] = None,
        include_input_log: Annotated[
            bool,
            typer.Option(
                "--include-input-log",
                help=(
                    "Include raw .input_log/ JSONL files in the archive "
                    "(excluded by default)."
                ),
            ),
        ] = False,
        force: Annotated[
            bool,
            typer.Option(
                "--force",
                "-f",
                help="Overwrite the output archive if it already exists.",
            ),
        ] = False,
        verbose: Annotated[
            bool,
            typer.Option(
                "--verbose",
                "-v",
                help="Emit INFO-level lifecycle logs (default: WARNING only).",
            ),
        ] = False,
    ) -> None:
        """Export a memory root to a portable .tar.gz archive with a manifest.

        Packages Markdown source-of-truth files, root configuration files, and
        a manifest.json containing per-file SHA-256 checksums and entry
        counts. Excludes derived .index/ state and, unless --include-input-log
        is passed, raw .input_log/ files.
        """
        configure_interactive(verbose=verbose)
        try:
            result = export_memory_root(
                out,
                root=Path(root) if root else None,
                include_input_log=include_input_log,
                force=force,
            )
        except (
            ArchiveConflictError,
            ArchiveIntegrityError,
            InvalidInputError,
            NotFoundError,
        ) as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from None

        typer.secho(f"exported: {result.archive_path}", fg=typer.colors.GREEN)
        typer.echo(f"everos version: {result.manifest.everos_version}")
        typer.echo(f"files: {result.file_count} ({result.total_bytes} bytes)")
        input_log_label = "yes" if result.manifest.includes_input_log else "no"
        typer.echo(f"input log included: {input_log_label}")
        _echo_entry_counts("entry counts", result.manifest.entry_counts)

    @parent.command("verify")
    def verify_cmd(
        archive: Annotated[
            Path,
            typer.Argument(
                help="Path to the portable .tar.gz archive to verify.",
            ),
        ],
        verbose: Annotated[
            bool,
            typer.Option(
                "--verbose",
                "-v",
                help="Emit INFO-level lifecycle logs (default: WARNING only).",
            ),
        ] = False,
    ) -> None:
        """Verify an archive's manifest, layout, checksums, and entry counts.

        Runs completely offline without touching any live memory root.
        """
        configure_interactive(verbose=verbose)
        try:
            result = verify_memory_archive(archive)
        except (ArchiveIntegrityError, InvalidInputError, NotFoundError) as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from None

        typer.secho(f"verified: {result.archive_path}", fg=typer.colors.GREEN)
        typer.echo(f"everos version: {result.manifest.everos_version}")
        typer.echo(f"created at: {result.manifest.created_at}")
        typer.echo(f"files: {result.file_count} ({result.total_bytes} bytes)")
        input_log_label = "yes" if result.manifest.includes_input_log else "no"
        typer.echo(f"input log included: {input_log_label}")
        _echo_entry_counts("entry counts", result.entry_counts)

    @parent.command("import")
    def import_cmd(
        archive: Annotated[
            Path,
            typer.Argument(
                help="Path to the portable .tar.gz archive to restore.",
            ),
        ],
        root: Annotated[
            str | None,
            typer.Option(
                "--root",
                help=(
                    "Target memory root directory "
                    "(env: EVEROS_ROOT, default: ~/.everos)."
                ),
            ),
        ] = None,
        force: Annotated[
            bool,
            typer.Option(
                "--force",
                "-f",
                help=(
                    "Replace an existing non-empty memory root instead of "
                    "refusing to overwrite."
                ),
            ),
        ] = False,
        no_rebuild: Annotated[
            bool,
            typer.Option(
                "--no-rebuild",
                help=(
                    "Skip rebuilding the derived index after extraction "
                    "(run `everos cascade rebuild` later to populate search)."
                ),
            ),
        ] = False,
        verbose: Annotated[
            bool,
            typer.Option(
                "--verbose",
                "-v",
                help="Emit INFO-level lifecycle logs (default: WARNING only).",
            ),
        ] = False,
    ) -> None:
        """Verify and restore a portable archive into a memory root.

        Verifies every checksum before writing anything. Refuses to overwrite
        a non-empty target root unless --force is given. Rebuilds the
        derived index from restored Markdown and compares entry counts before
        and after.
        """
        configure_interactive(verbose=verbose)
        try:
            result = asyncio.run(
                import_memory_archive(
                    archive,
                    root=Path(root) if root else None,
                    force=force,
                    rebuild=not no_rebuild,
                )
            )
        except MemoryRootLockedError as exc:
            typer.echo(
                f"error: {exc}\n"
                "  stop `everos server` (or any running CLI index operation) "
                "before importing into this root.",
                err=True,
            )
            raise typer.Exit(code=3) from None
        except (
            ArchiveConflictError,
            ArchiveIntegrityError,
            InvalidInputError,
            NotFoundError,
        ) as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from None

        typer.secho(f"imported into: {result.target_root}", fg=typer.colors.GREEN)
        typer.echo(f"files restored: {result.files_restored}")
        _echo_entry_counts("entry counts (manifest)", result.entry_counts_before)
        _echo_entry_counts("entry counts (restored)", result.entry_counts_after)
        if result.rebuilt:
            typer.echo(f"index rebuilt: {result.processed_md_files} md file(s) indexed")
            typer.echo("index rows:")
            for table, count in result.index_rows.items():
                typer.echo(f"  {table + ':':<22} {count}")
        else:
            typer.echo(
                "index rebuild skipped (--no-rebuild); run "
                "`everos cascade rebuild --yes` before searching."
            )
