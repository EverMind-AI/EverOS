"""``everos config show`` — display effective configuration."""

from __future__ import annotations

import os

import typer
from pydantic import BaseModel

from everos.config.settings import resolve_root
from everos.entrypoints.cli._process import (
    configure_interactive,
    load_settings_or_exit,
)

app = typer.Typer(
    name="config",
    help="Configuration management",
    no_args_is_help=True,
)


@app.callback()
def _config_callback(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Emit INFO-level lifecycle logs (default: WARNING only).",
    ),
) -> None:
    """Set log level before any config subcommand runs."""
    configure_interactive(verbose=verbose)


def _mask(value: str) -> str:
    """Mask a secret value, keeping first/last 4 chars if long enough."""
    if len(value) <= 8:
        return "****"
    return value[:4] + "****" + value[-4:]


def _shown(name: str, value: object) -> str:
    """A value as printed: a secret masked, never echoed."""
    if hasattr(value, "get_secret_value"):  # a SecretStr
        secret = value.get_secret_value()  # type: ignore[attr-defined]
        return _mask(secret) if secret else ""
    if name.endswith("_key") and isinstance(value, str) and value:
        return _mask(value)
    return str(value)


def _echo_section(label: str, section: BaseModel) -> None:
    """``[label]`` with its keys, then each sub-table as ``[label.sub]``."""
    tables: list[tuple[str, BaseModel]] = []
    typer.secho(f"[{label}]", bold=True)
    for name in type(section).model_fields:
        value = getattr(section, name)
        if isinstance(value, BaseModel):
            tables.append((f"{label}.{name}", value))
            continue
        typer.echo(f"  {name} = {_shown(name, value)}")
    typer.echo()
    for sub_label, sub in tables:
        _echo_section(sub_label, sub)


@app.command("show")
def show(
    root: str | None = typer.Option(
        None,
        "--root",
        help="Memory root directory",
    ),
) -> None:
    """Print the effective configuration."""
    if root:
        os.environ["EVEROS_ROOT"] = root

    resolved = resolve_root(root)
    typer.echo(f"Root: {resolved}")
    typer.echo()

    everos_toml = resolved / "everos.toml"
    if everos_toml.is_file():
        typer.echo(f"Config: {everos_toml}")
    else:
        typer.echo("Config: (no everos.toml found, using defaults)")

    ome_toml = resolved / "ome.toml"
    if ome_toml.is_file():
        typer.echo(f"Strategy: {ome_toml}")
    typer.echo()

    from everos.config import load_settings

    load_settings.cache_clear()
    settings = load_settings_or_exit()

    # Every section, in the order Settings declares them; secrets masked.
    for section_name in type(settings).model_fields:
        _echo_section(section_name, getattr(settings, section_name))
