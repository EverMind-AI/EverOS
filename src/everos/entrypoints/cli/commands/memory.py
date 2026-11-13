"""``everos memory`` -- change, forget and read the versions of one memory.

A thin layer over the same service functions the HTTP API calls
(``POST /memory/update``, ``POST /memory/forget`` and its undo, ``GET
/memory/history``; docs/api.md), run in-process against a memory root:

- ``memory history ID`` -- every version of one memory, current first;
- ``memory update ID`` -- write a person's new version (held unless
  ``--release``);
- ``memory forget`` -- by records, a conversation or messages of it,
  everything, a kind and a time range, or a picture (a group: records, a
  session or messages of it, a member's messages, a picture). A dry run
  unless ``--apply``, as the API's default;
- ``memory undo FORGET_ID`` -- undo one forget by the id it answered.

A user's memory is named by ``--user``; a group's by ``--group`` and
``--asker``. Each command prints its answer as JSON on standard output, and
its log lines on standard error. ``history`` only reads
and runs beside a server; the others write, and -- as ``cascade sync`` --
need the memory root to themselves (exit 3 while a server holds it; use the
HTTP API then).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

import typer

from everos.config import Settings
from everos.core.errors import (
    AppError,
    EmbeddingProvenanceError,
    IndexSchemaMismatchError,
    LLMNotConfiguredError,
    MemoryRootLockedError,
    StoreVersionError,
)
from everos.entrypoints.cli._process import (
    configure_interactive,
    load_settings_or_exit,
    with_log_stream,
)
from everos.service import (
    ForgetTarget,
    MemoryChanges,
    MemoryService,
    build_application,
)

app = typer.Typer(
    name="memory",
    help="Change, forget and read the versions of one memory",
    no_args_is_help=True,
)

_ROOT = typer.Option(
    "--root", help="Memory root directory (env: EVEROS_ROOT, default: ~/.everos)"
)
_USER = typer.Option("--user", help="The user whose memory it is.")
_GROUP = typer.Option("--group", help="The group whose memory it is.")
_ASKER = typer.Option("--asker", help="Who asks (a group's member).")
_APP = typer.Option("--app-id", help="The app scope.")
_PROJECT = typer.Option("--project-id", help="The project scope.")
_VERBOSE = typer.Option("--verbose", "-v", help="Emit INFO-level lifecycle logs.")


@app.command("history")
def history(
    record_id: Annotated[str, typer.Argument(help="Any version's id.")],
    user: Annotated[str | None, _USER] = None,
    group: Annotated[str | None, _GROUP] = None,
    asker: Annotated[str | None, _ASKER] = None,
    app_id: Annotated[str, _APP] = "default",
    project_id: Annotated[str, _PROJECT] = "default",
    root: Annotated[str | None, _ROOT] = None,
    verbose: Annotated[bool, _VERBOSE] = False,
) -> None:
    """Every version of one memory, current first (read-only)."""
    _check_owner(user, group)
    settings = _settings(root, verbose)

    async def run(memory: MemoryService) -> object:
        return await memory.memory_history(
            app_id=app_id,
            project_id=project_id,
            user_id=user,
            group_id=group,
            asker_id=asker,
            record_id=record_id,
        )

    _print(asyncio.run(_run(settings, run, exclusive=False)))


@app.command("update")
def update(
    record_id: Annotated[str, typer.Argument(help="The version to change.")],
    user: Annotated[str | None, _USER] = None,
    group: Annotated[str | None, _GROUP] = None,
    asker: Annotated[str | None, _ASKER] = None,
    subject: Annotated[str | None, typer.Option(help="Episode or scene.")] = None,
    summary: Annotated[str | None, typer.Option(help="Episode or scene.")] = None,
    content: Annotated[
        str | None, typer.Option(help="Episode body or scene narrative.")
    ] = None,
    description: Annotated[
        str | None, typer.Option(help="Picture memory or profile item.")
    ] = None,
    ocr: Annotated[str | None, typer.Option(help="Picture memory.")] = None,
    expected_version: Annotated[
        str | None,
        typer.Option(help="The version you read (default: the id given)."),
    ] = None,
    release: Annotated[
        bool,
        typer.Option(
            "--release",
            help="Do not hold the new version: the model may change it later. "
            "Alone, releases a held version.",
        ),
    ] = False,
    app_id: Annotated[str, _APP] = "default",
    project_id: Annotated[str, _PROJECT] = "default",
    root: Annotated[str | None, _ROOT] = None,
    verbose: Annotated[bool, _VERBOSE] = False,
) -> None:
    """Write a person's new version of one memory; the old one is superseded."""
    _check_owner(user, group)
    settings = _settings(root, verbose)

    async def run(memory: MemoryService) -> object:
        return await memory.update_memory(
            app_id=app_id,
            project_id=project_id,
            user_id=user,
            group_id=group,
            asker_id=asker,
            record_id=record_id,
            expected_version=expected_version,
            changes=MemoryChanges(
                subject=subject,
                summary=summary,
                content=content,
                description=description,
                ocr=ocr,
            ),
            hold=not release,
        )

    _print(asyncio.run(_run(settings, run, exclusive=True)))


@app.command("forget")
def forget(
    user: Annotated[str | None, _USER] = None,
    group: Annotated[str | None, _GROUP] = None,
    asker: Annotated[str | None, _ASKER] = None,
    record: Annotated[
        list[str] | None,
        typer.Option("--record", help="A record id (repeatable)."),
    ] = None,
    session: Annotated[
        str | None, typer.Option("--session", help="A conversation.")
    ] = None,
    message: Annotated[
        list[str] | None,
        typer.Option(
            "--message",
            help="A message of the --session conversation, by id (repeatable).",
        ),
    ] = None,
    owner: Annotated[
        bool, typer.Option("--owner", help="Everything of the user.")
    ] = False,
    types: Annotated[
        str | None,
        typer.Option(
            "--types",
            help="A condition: kinds, comma-separated (episode, image_memory, "
            "scene, profile), with --start and/or --end.",
        ),
    ] = None,
    start: Annotated[
        int | None, typer.Option("--start", help="Epoch ms, inclusive.")
    ] = None,
    end: Annotated[
        int | None, typer.Option("--end", help="Epoch ms, exclusive.")
    ] = None,
    image: Annotated[
        str | None, typer.Option("--image", help="A picture's content hash.")
    ] = None,
    member: Annotated[
        str | None,
        typer.Option("--member", help="A group: every message of one member."),
    ] = None,
    reason: Annotated[str | None, typer.Option(help="Kept in the forget log.")] = None,
    apply: Annotated[
        bool,
        typer.Option("--apply", help="Forget for real (default: a dry run)."),
    ] = False,
    app_id: Annotated[str, _APP] = "default",
    project_id: Annotated[str, _PROJECT] = "default",
    root: Annotated[str | None, _ROOT] = None,
    verbose: Annotated[bool, _VERBOSE] = False,
) -> None:
    """Forget by target, reversibly; a dry run unless --apply."""
    _check_owner(user, group)
    chosen = [
        bool(record),
        session is not None,
        owner,
        types is not None,
        image is not None,
        member is not None,
    ]
    if sum(chosen) != 1:
        raise typer.BadParameter(
            "name one target: --record, --session (with --message for messages), "
            "--owner, --types, --image or (a group) --member"
        )
    if group is not None and (owner or types is not None):
        raise typer.BadParameter("a group forget takes no --owner or --types")
    if group is None and member is not None:
        raise typer.BadParameter("--member is a group's target")
    target = ForgetTarget(
        kind="record"
        if record
        else "message"
        if message
        else "session"
        if session is not None
        else "owner"
        if owner
        else "condition"
        if types is not None
        else "member"
        if member is not None
        else "image",
        record_ids=tuple(record or ()),
        session_id=session,
        message_ids=tuple(message or ()),
        member_id=member,
        image_id=image,
        memory_types=tuple(t.strip() for t in (types or "").split(",") if t.strip()),  # type: ignore[arg-type]
        start=start,
        end=end,
    )
    if target.kind == "condition" and start is None and end is None:
        raise typer.BadParameter("--types needs --start, --end or both")
    settings = _settings(root, verbose)

    async def run(memory: MemoryService) -> object:
        if group is not None:
            return await memory.forget_in_group(
                app_id=app_id,
                project_id=project_id,
                group_id=group,
                asker_id=asker,
                target=target,
                reason=reason,
                dry_run=not apply,
            )
        assert user is not None
        return await memory.forget(
            app_id=app_id,
            project_id=project_id,
            user_id=user,
            asker_id=asker,
            target=target,
            reason=reason,
            dry_run=not apply,
        )

    _print(asyncio.run(_run(settings, run, exclusive=apply)))


@app.command("undo")
def undo(
    forget_id: Annotated[str, typer.Argument(help="The id a forget answered.")],
    user: Annotated[str | None, _USER] = None,
    group: Annotated[str | None, _GROUP] = None,
    asker: Annotated[str | None, _ASKER] = None,
    app_id: Annotated[str, _APP] = "default",
    project_id: Annotated[str, _PROJECT] = "default",
    root: Annotated[str | None, _ROOT] = None,
    verbose: Annotated[bool, _VERBOSE] = False,
) -> None:
    """Undo one forget by its id."""
    _check_owner(user, group)
    settings = _settings(root, verbose)

    async def run(memory: MemoryService) -> object:
        if group is not None:
            return await memory.undo_group_forget(
                app_id=app_id,
                project_id=project_id,
                group_id=group,
                asker_id=asker,
                forget_id=forget_id,
            )
        assert user is not None
        return await memory.undo_forget(
            app_id=app_id,
            project_id=project_id,
            user_id=user,
            asker_id=asker,
            forget_id=forget_id,
        )

    _print(asyncio.run(_run(settings, run, exclusive=True)))


# ── runtime ──────────────────────────────────────────────────────────────


def _check_owner(user: str | None, group: str | None) -> None:
    if (user is None) == (group is None):
        raise typer.BadParameter("name exactly one of --user and --group")


def _settings(root: str | None, verbose: bool) -> Settings:
    """The command's settings: its root, its log level, and its logs on stderr.

    Standard output carries the JSON answer alone, so a script can parse it;
    every log line goes to stderr: a provider's warning at build, and one the
    settings log while they load (``config_key_ignored`` for a key they
    ignore, which launchers still export). That one comes before the settings
    that name the stream exist, so the logs are sent to stderr first, on the
    defaults, and again on the settings loaded.
    """
    if root:
        os.environ["EVEROS_ROOT"] = root
    configure_interactive(
        verbose=verbose,
        settings=with_log_stream(Settings.model_construct(), "stderr"),
    )
    settings = with_log_stream(load_settings_or_exit(), "stderr")
    return configure_interactive(verbose=verbose, settings=settings)


async def _run(settings: Settings, step: Any, *, exclusive: bool) -> object:
    async with _memory(settings, exclusive=exclusive) as memory:
        try:
            return await step(memory)
        except AppError as exc:
            typer.secho(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from None


@asynccontextmanager
async def _memory(
    settings: Settings, *, exclusive: bool
) -> AsyncIterator[MemoryService]:
    """Build the application and open its storage for one command.

    A command that writes (``update``, ``forget --apply``, ``undo``) takes the
    root's lock for its run, as ``cascade sync`` does: a second process
    writing beside a server could erase index rows the server's cascade
    writes back, and two forgets of one user would not see each other.
    """
    try:
        application = build_application(settings)
    except LLMNotConfiguredError as exc:
        typer.echo(
            f"error: {exc}\n  `everos memory` builds the same application as the "
            "server, which needs an extraction LLM; configure it, then re-run.",
            err=True,
        )
        raise typer.Exit(code=3) from None
    try:
        async with (
            application,
            application.ops.index_session(
                verify=True, ensure=True, exclusive=exclusive
            ),
        ):
            yield application.memory
    except MemoryRootLockedError:
        typer.echo(
            "error: another process holds this memory root -- a running `everos "
            "server`.\n  Change or forget through its HTTP API (POST "
            "/api/v2/memory/update, /api/v2/memory/forget), or stop it first.",
            err=True,
        )
        raise typer.Exit(code=3) from None
    except (
        IndexSchemaMismatchError,
        EmbeddingProvenanceError,
        StoreVersionError,
    ) as exc:
        typer.secho(str(exc), err=True)
        raise typer.Exit(code=3) from None


def _print(result: object) -> None:
    data = dataclasses.asdict(result) if dataclasses.is_dataclass(result) else result
    typer.echo(json.dumps(data, ensure_ascii=False, indent=2, default=str))
