"""``everos config show`` — display effective configuration.

The ``[llm.<task>]`` sections are not dumped raw: each task is shown as it
resolves -- model, endpoint host and the section every field came from -- the
same view the server logs at startup.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import typer

from everos.config import LLMTask
from everos.config.settings import resolve_root
from everos.entrypoints.cli._process import configure_interactive
from everos.service import describe_llm_tasks

if TYPE_CHECKING:
    from pydantic import BaseModel

    from everos.service import LLMTaskResolution

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


_SECRET_FIELDS = {"api_key"}

_SECTION_NAMES = (
    "memory",
    "api",
    "sqlite",
    "lancedb",
    "llm",
    "multimodal",
    "embedding",
    "rerank",
    "boundary_detection",
    "memorize",
    "clustering",
    "search",
    "knowledge",
    "logging",
)


def _mask(value: str) -> str:
    """Mask a secret value, keeping first/last 4 chars if long enough."""
    if len(value) <= 8:
        return "****"
    return value[:4] + "****" + value[-4:]


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
    settings = load_settings()

    for section_name in _SECTION_NAMES:
        section = getattr(settings, section_name, None)
        if section is None:
            continue
        typer.secho(f"[{section_name}]", bold=True)
        for field_name, value in _shown_fields(section_name, section).items():
            typer.echo(f"  {field_name} = {_display(field_name, value)}")
        typer.echo()

    _echo_llm_tasks(describe_llm_tasks(settings))


def _display(field_name: str, value: object) -> str:
    """A field's value as printed, secrets masked."""
    if field_name in _SECRET_FIELDS and value:
        return _mask(str(value))
    return str(value)


def _shown_fields(section_name: str, section: BaseModel) -> dict[str, Any]:
    """A section's fields as printed; ``[llm]``'s task sections print resolved."""
    exclude = {task.value for task in LLMTask} if section_name == "llm" else None
    return section.model_dump(exclude=exclude)


def _echo_llm_tasks(resolutions: Sequence[LLMTaskResolution]) -> None:
    """Print each task's model and host, and its fields grouped by source."""
    typer.secho("[llm tasks]", bold=True)
    typer.echo("  # each field comes from llm.<task>, llm, decider or default")
    for resolution in resolutions:
        typer.echo(
            f"  {resolution.task}: model = {resolution.settings.model}, "
            f"host = {resolution.host or '(none)'}"
        )
        by_source: dict[str, list[str]] = {}
        for field_name, source in resolution.sources.items():
            value = getattr(resolution.settings, field_name)
            shown = f"{field_name}={_display(field_name, value)}"
            by_source.setdefault(source, []).append(shown)
        for source, fields in by_source.items():
            typer.echo(f"    {source}: {', '.join(fields)}")
    if any(resolution.deprecated_fields for resolution in resolutions):
        typer.echo(
            "  # the LLM keys of [decider] are deprecated: move them to [llm.decider]"
        )
    typer.echo()
