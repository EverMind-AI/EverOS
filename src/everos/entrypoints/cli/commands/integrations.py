"""``everos integrations`` — install EverOS bundles into third-party tools.

Currently ships the Hermes memory-provider bundle (``integrations/hermes/``).
``install`` symlinks the bundle into ``$HERMES_HOME/plugins/everos`` so Hermes
picks it up as a user-namespace memory plugin; ``uninstall`` removes the
symlink. Real directories at the target path are never deleted automatically
— ``install`` will only ``rmtree`` one when ``--force`` (or an accepted
prompt) explicitly authorises it, and ``uninstall`` refuses outright.

This module is ``everos.entrypoints`` code: it manipulates paths only and
does not import the bundle (``integrations/hermes/*``).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import typer

import everos
from everos.core.observability.logging import get_logger

app = typer.Typer(
    name="integrations",
    help="Install EverOS integrations into third-party tools.",
    no_args_is_help=True,
)

logger = get_logger(__name__)

_BUNDLE_REL = Path("integrations") / "hermes"
_PLUGIN_SUBDIR = Path("plugins") / "everos"
_SUPPORTED_TARGETS = frozenset({"hermes"})
_ENV_SOURCE = "EVEROS_HERMES_PLUGIN_SOURCE"
_ENV_HERMES_HOME = "HERMES_HOME"


def _resolve_bundle_source(explicit: str | None) -> Path:
    """Resolve the bundle source directory.

    Priority: ``EVEROS_HERMES_PLUGIN_SOURCE`` env > ``--source`` flag > walk
    up from ``everos.__file__`` to a repo root containing
    ``integrations/hermes/``.
    """
    from_env = os.environ.get(_ENV_SOURCE)
    if from_env:
        return Path(from_env).expanduser().resolve()

    if explicit:
        return Path(explicit).expanduser().resolve()

    here = Path(everos.__file__).resolve()
    for parent in here.parents:
        candidate = parent / _BUNDLE_REL
        if candidate.is_dir():
            return candidate.resolve()

    raise typer.BadParameter(
        f"Could not locate {_BUNDLE_REL} by walking up from {here}. "
        f"Pass --source PATH or set {_ENV_SOURCE}."
    )


def _resolve_hermes_home() -> Path:
    """Resolve the Hermes home directory (``HERMES_HOME`` env or ``~/.hermes``)."""
    from_env = os.environ.get(_ENV_HERMES_HOME)
    if from_env:
        return Path(from_env).expanduser().resolve()
    return Path("~/.hermes").expanduser().resolve()


def _target_path(hermes_home: Path) -> Path:
    return hermes_home / _PLUGIN_SUBDIR


@app.command("install")
def install(
    target: str = typer.Argument(
        "hermes",
        help="Integration target. Currently only 'hermes' is supported.",
    ),
    source: str | None = typer.Option(
        None,
        "--source",
        help="Override the bundle source path (defaults to env/walk-up).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite an existing real directory at the target path.",
    ),
) -> None:
    """Symlink the EverOS bundle into a third-party tool's plugin directory."""
    if target not in _SUPPORTED_TARGETS:
        raise typer.BadParameter(
            f"Unsupported integration target: {target!r}. "
            f"Supported: {', '.join(sorted(_SUPPORTED_TARGETS))}."
        )

    bundle_src = _resolve_bundle_source(source)
    if not bundle_src.is_dir():
        raise typer.BadParameter(f"Bundle source not found: {bundle_src}")

    hermes_home = _resolve_hermes_home()
    plugins_dir = hermes_home / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    target_path = _target_path(hermes_home)

    if target_path.is_symlink():
        target_path.unlink()
    elif target_path.exists() and target_path.is_dir():
        # Real directory — refuse unless explicitly authorised.
        if not force:
            confirm = typer.confirm(
                f"{target_path} is a real directory. Replace it with a symlink "
                "to the EverOS bundle? (Its contents will be deleted.)",
                default=False,
            )
            if not confirm:
                logger.info(
                    "everos.integration.skipped",
                    target=str(target_path),
                    reason="real_dir_no_force",
                )
                typer.echo("Aborted; target left untouched.")
                raise typer.Exit(code=1)
        shutil.rmtree(target_path)
    elif target_path.exists():
        # A file (not a dir, not a symlink) — refuse; too surprising to clobber.
        typer.echo(
            f"Refusing to replace {target_path}: not a directory or symlink. "
            "Remove it manually and re-run."
        )
        raise typer.Exit(code=1)

    target_path.symlink_to(bundle_src, target_is_directory=True)

    logger.info(
        "everos.integration.installed",
        target=str(target_path),
        source=str(bundle_src),
        integration=target,
    )
    typer.secho(f"linked: {target_path} -> {bundle_src}", fg=typer.colors.GREEN)
    typer.echo(
        "\nNext steps:\n"
        "  1. Ensure an EverOS server is running (everos server start).\n"
        "  2. Run `hermes memory setup` and select 'everos' to activate it.\n"
        "  3. Verify with `hermes everos status`."
    )


@app.command("uninstall")
def uninstall(
    target: str = typer.Argument(
        "hermes",
        help="Integration target. Currently only 'hermes' is supported.",
    ),
    source: str | None = typer.Option(
        None,
        "--source",
        help="Override the bundle source path for the ownership check "
        "(defaults to env/walk-up). Use when the bundle was installed via "
        "--source and that path is still reachable.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Skip the ownership check and unlink the symlink directly. "
        "Prompts for confirmation unless --yes is also given.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the confirmation prompt implied by --force.",
    ),
) -> None:
    """Remove the EverOS bundle symlink from a third-party tool's plugin dir."""
    if target not in _SUPPORTED_TARGETS:
        raise typer.BadParameter(
            f"Unsupported integration target: {target!r}. "
            f"Supported: {', '.join(sorted(_SUPPORTED_TARGETS))}."
        )

    hermes_home = _resolve_hermes_home()
    target_path = _target_path(hermes_home)

    if not target_path.exists() and not target_path.is_symlink():
        typer.echo(f"Nothing to remove: {target_path} does not exist.")
        return

    if target_path.is_dir() and not target_path.is_symlink():
        # Real directory — refuse to delete.
        typer.echo(
            f"Refusing to remove {target_path}: it is a real directory, not a "
            "symlink. Remove it manually if you really intend to."
        )
        raise typer.Exit(code=1)

    if not target_path.is_symlink():
        typer.echo(f"Refusing to remove {target_path}: not a symlink.")
        raise typer.Exit(code=1)

    resolved = target_path.resolve()

    if force:
        if not yes and not typer.confirm(
            f"Remove {target_path} without verifying which bundle it points at?",
            default=False,
        ):
            logger.info(
                "everos.integration.uninstall.skipped",
                target=str(target_path),
                reason="force_not_confirmed",
            )
            typer.echo("Aborted; symlink left in place.")
            raise typer.Exit(code=1)
        bundle_src: Path | None = None
    else:
        try:
            bundle_src = _resolve_bundle_source(source)
        except typer.BadParameter:
            typer.echo(
                f"Could not re-resolve the EverOS bundle source to verify "
                f"{target_path} (it points at {resolved}). Pass --source PATH "
                "to point at the bundle directory, or --force to unlink "
                "without checking."
            )
            raise typer.Exit(code=1) from None
        if resolved != bundle_src:
            typer.echo(
                f"Refusing to remove {target_path}: it points at {resolved}, "
                f"not the EverOS bundle ({bundle_src}). Pass --force to "
                "unlink anyway."
            )
            raise typer.Exit(code=1)

    target_path.unlink()
    logger.info(
        "everos.integration.uninstalled",
        target=str(target_path),
        integration=target,
        forced=force,
        source=str(bundle_src) if bundle_src is not None else None,
    )
    typer.secho(f"removed: {target_path}", fg=typer.colors.GREEN)
    typer.echo(
        "\nThe `memory.provider` setting in Hermes config was left untouched. "
        "If you want to clear it, run:\n"
        "  hermes config set memory.provider ''"
    )
