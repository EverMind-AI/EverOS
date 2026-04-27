"""
Tenant Command Interceptor

Intercepts ALL MongoDB commands at the PyMongo network layer by occupying
the client._encrypter hook point.

PyMongo 4.x has 3 data-sending paths, all converging through _encrypter when it exists:

  Path 1: conn.command() → network.command() → _encrypter.encrypt()
      Used by: insert_one, update_one/many, replace_one, delete_one/many,
               find_one_and_*, aggregate, distinct, count, estimated_document_count

  Path 2: cursor._refresh() → server.run_operation() → server.operation_to_command()
           → _encrypter.encrypt()
      Used by: find(), find_one() (via cursor), async for iteration, getMore

  Path 3: _AsyncBulk → _EncryptedBulkWriteContext.batch_command()
           → conn.command() → network.command() → _encrypter.encrypt()
      Used by: insert_many(), bulk_write()
      Note: documents arrive as RawBSONDocument (immutable) — interceptor handles this

  Blocked: find_raw_batches() and aggregate_raw_batches() raise InvalidOperation
           when _encrypter is set. These are low-level APIs not used in this project.

Trade-offs:
    - Pro: Single interception point for 100% command coverage
    - Con: Occupies the _encrypter slot, cannot coexist with CSFLE
    - Con: Uses private API (client._encrypter), may break on PyMongo major upgrades
    - Con: find_raw_batches/aggregate_raw_batches become unavailable
"""

from typing import Any, Mapping, MutableMapping, Optional, Set

from bson.codec_options import CodecOptions
from bson.raw_bson import RawBSONDocument

from core.observation.logger import get_logger
from core.tenants.tenant_config import get_tenant_config
from core.tenants.tenant_constants import TENANT_ID_FIELD
from core.tenants.tenant_contextvar import get_current_tenant_id


logger = get_logger(__name__)

# Data-plane commands that MUST have tenant_id injection
_DATA_COMMANDS: Set[str] = {
    "insert",
    "update",
    "delete",
    "find",
    "findAndModify",
    "aggregate",
    "count",
    "distinct",
}

# Control-plane commands that should NOT be intercepted
_PASSTHROUGH_COMMANDS: Set[str] = {
    # Connection & auth
    "hello",
    "ismaster",
    "isMaster",
    "saslStart",
    "saslContinue",
    "authenticate",
    "getnonce",
    "logout",
    # Server admin
    "ping",
    "buildInfo",
    "buildinfo",
    "serverStatus",
    "hostInfo",
    "getLog",
    "replSetGetStatus",
    "currentOp",
    "killCursors",
    "killOp",
    "getMore",
    # Index management
    "createIndexes",
    "dropIndexes",
    "listIndexes",
    # Collection management
    "create",
    "drop",
    "renameCollection",
    "listCollections",
    "collStats",
    "collMod",
    # Database management
    "listDatabases",
    "dbStats",
    # Transaction
    "commitTransaction",
    "abortTransaction",
    "endSessions",
    # Search index
    "createSearchIndexes",
    "updateSearchIndex",
    "dropSearchIndex",
    "listSearchIndexes",
}


class TenantCommandInterceptor:
    """
    Masquerades as PyMongo's _Encrypter to intercept MongoDB commands.

    Occupies the client._encrypter hook to inject tenant_id into all
    data-plane commands (find, insert, update, delete, aggregate, etc.)
    while passing through control-plane commands (ping, auth, index ops).

    Usage:
        real_client = tenant_aware_client.get_real_client()
        real_client._encrypter = TenantCommandInterceptor()

    Or with collection exclusion:
        interceptor = TenantCommandInterceptor(
            excluded_collections={"system.profile", "migrations"}
        )
    """

    def __init__(self, excluded_collections: Optional[Set[str]] = None):
        # Required by PyMongo: must be False for the hook to fire
        self._bypass_auto_encryption = False
        self._closed = False

        self._excluded_collections = excluded_collections or set()

    def _get_tenant_id(self) -> Optional[str]:
        """Get current tenant_id from context."""
        return get_current_tenant_id()

    def _should_intercept(self, cmd_name: str, cmd: dict) -> bool:
        """
        Determine if this command should be intercepted.

        Three categories:
        - _DATA_COMMANDS: must intercept, inject tenant_id
        - _PASSTHROUGH_COMMANDS: skip, no injection needed
        - Unknown: reject — refuse to let unrecognized commands through silently

        Raises:
            TenantIsolationViolation: If cmd_name is not in either whitelist
        """
        if cmd_name in _PASSTHROUGH_COMMANDS:
            return False

        if cmd_name not in _DATA_COMMANDS:
            # Unknown command — refuse to let it through silently
            collection_name = cmd.get(cmd_name, "unknown")
            msg = (
                "\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                "!!  UNKNOWN MONGODB COMMAND — TENANT ISOLATION RISK        !!\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                f"!!  Command:    {cmd_name}\n"
                f"!!  Collection: {collection_name}\n"
                "!!  Action:     Command not in _DATA_COMMANDS or _PASSTHROUGH_COMMANDS.\n"
                "!!              Add to the appropriate set in tenant_field_command_interceptor.py\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )
            logger.error(msg)
            raise TenantIsolationViolation(msg)

        # Check collection exclusion
        collection_name = cmd.get(cmd_name)
        if (
            isinstance(collection_name, str)
            and collection_name in self._excluded_collections
        ):
            return False

        return True

    async def encrypt(
        self, database: str, cmd: Mapping[str, Any], codec_options: CodecOptions
    ) -> dict[str, Any]:
        """
        Called before every command is serialized to BSON.

        This is the single interception point. The command dict structure
        follows the MongoDB wire protocol:
            insert:         {"insert": "coll", "documents": [...]}
            update:         {"update": "coll", "updates": [{"q": filter, "u": doc}]}
            delete:         {"delete": "coll", "deletes": [{"q": filter}]}
            find:           {"find": "coll", "filter": {...}}
            findAndModify:  {"findAndModify": "coll", "query": {...}}
            aggregate:      {"aggregate": "coll", "pipeline": [...]}
            count:          {"count": "coll", "query": {...}}
            distinct:       {"distinct": "coll", "key": "...", "query": {...}}
        """
        # Convert to mutable dict
        cmd = dict(cmd)

        tid = self._get_tenant_id()
        if not tid:
            if get_tenant_config().app_ready:
                cmd_name = next(iter(cmd))
                collection_name = cmd.get(cmd_name, "unknown")
                raise TenantIsolationViolation(
                    f"Missing tenant_id for MongoDB command '{cmd_name}' "
                    f"on collection '{collection_name}'. "
                    f"Ensure tenant context is set before data operations."
                )
            return cmd

        cmd_name = next(iter(cmd))

        if not self._should_intercept(cmd_name, cmd):
            return cmd

        # All modes: inject tenant_id filter on ALL operations (read + write).
        # Even in exclusive mode (physical isolation), filter injection is needed
        # so that compound indexes with tenant_id prefix can be utilized.
        shared = True

        # ---- INSERT: inject tenant_id into each document (always) ----
        # Use force-set (not setdefault) because Pydantic models serialize
        # tenant_id=None as an explicit key — setdefault would skip it.
        if cmd_name == "insert":
            documents = cmd.get("documents", [])
            new_docs = []
            needs_replace = False
            for i, doc in enumerate(documents):
                if isinstance(doc, RawBSONDocument):
                    mutable_doc = dict(doc)
                    mutable_doc[TENANT_ID_FIELD] = tid
                    new_docs.append(mutable_doc)
                    needs_replace = True
                elif isinstance(doc, MutableMapping):
                    doc[TENANT_ID_FIELD] = tid
                    new_docs.append(doc)
                elif isinstance(doc, Mapping):
                    mutable_doc = dict(doc)
                    mutable_doc[TENANT_ID_FIELD] = tid
                    new_docs.append(mutable_doc)
                    needs_replace = True
                else:
                    collection_name = cmd.get("insert", "unknown")
                    raise TenantIsolationViolation(
                        f"insert command on '{collection_name}': "
                        f"documents[{i}] is {type(doc).__name__}, expected Mapping. "
                        f"Cannot inject tenant_id into unknown document type."
                    )
            if needs_replace:
                cmd["documents"] = new_docs

        # ---- UPDATE / REPLACE ----
        elif cmd_name == "update":
            for update_spec in cmd.get("updates", []):
                # Filter injection: shared mode only
                if shared:
                    q = update_spec.get("q", {})
                    update_spec["q"] = {TENANT_ID_FIELD: tid, **q}

                # Replacement doc injection: always (write operation)
                u = update_spec.get("u", {})
                if isinstance(u, Mapping) and not any(k.startswith("$") for k in u):
                    if isinstance(u, MutableMapping):
                        u[TENANT_ID_FIELD] = tid
                    else:
                        mutable_u = dict(u)
                        mutable_u[TENANT_ID_FIELD] = tid
                        update_spec["u"] = mutable_u

        # ---- DELETE: filter injection, shared mode only ----
        elif cmd_name == "delete":
            if shared:
                for del_spec in cmd.get("deletes", []):
                    q = del_spec.get("q", {})
                    del_spec["q"] = {TENANT_ID_FIELD: tid, **q}

        # ---- FIND: filter injection, shared mode only ----
        elif cmd_name == "find":
            if shared:
                f = cmd.get("filter") or {}
                cmd["filter"] = {TENANT_ID_FIELD: tid, **f}

        # ---- FIND AND MODIFY ----
        elif cmd_name == "findAndModify":
            # Query filter: shared mode only
            if shared:
                q = cmd.get("query") or {}
                cmd["query"] = {TENANT_ID_FIELD: tid, **q}
            # Replacement doc: always (write operation)
            update = cmd.get("update")
            if isinstance(update, Mapping) and not any(
                k.startswith("$") for k in update
            ):
                if isinstance(update, MutableMapping):
                    update[TENANT_ID_FIELD] = tid
                else:
                    mutable_update = dict(update)
                    mutable_update[TENANT_ID_FIELD] = tid
                    cmd["update"] = mutable_update

        # ---- AGGREGATE: prepend $match, shared mode only ----
        elif cmd_name == "aggregate":
            if shared:
                pipeline = list(cmd.get("pipeline", []))
                cmd["pipeline"] = [{"$match": {TENANT_ID_FIELD: tid}}] + pipeline

        # ---- COUNT: filter injection, shared mode only ----
        elif cmd_name == "count":
            if shared:
                q = cmd.get("query") or {}
                cmd["query"] = {TENANT_ID_FIELD: tid, **q}

        # ---- DISTINCT: filter injection, shared mode only ----
        elif cmd_name == "distinct":
            if shared:
                q = cmd.get("query") or {}
                cmd["query"] = {TENANT_ID_FIELD: tid, **q}

        return cmd

    async def decrypt(self, response: bytes) -> bytes:
        """
        Called after every response is received.

        Pass-through: we don't need to modify responses.
        The raw bytes are returned as-is to be decoded normally.
        """
        return response

    async def close(self) -> None:
        """Cleanup. Nothing to clean up for this interceptor."""
        self._closed = True


# ============================================================
# Guard: CommandListener-based double check
# ============================================================


class TenantGuardListener:
    """
    CommandListener that verifies tenant_id is present in all data-plane commands.

    This is a SECOND line of defense, independent of TenantCommandInterceptor.
    It hooks into PyMongo's event publishing system (a different code path from _encrypter),
    covering ALL 3 data-sending paths including the bulk non-encrypted path.

    Hook points comparison:
        _encrypter.encrypt():  network.py:137, server.py:129 (misses bulk non-encrypted path 3b)
        CommandListener:       network.py:182, server.py:194, bulk.py:272 (covers ALL paths)

    Behavior when tenant_id is missing:
        - mode="block": raises TenantIsolationViolation (caught by PyMongo, printed to stderr,
          but the command has ALREADY been serialized — so this is a post-hoc alarm, not a true block)
        - mode="log": logs error but allows command to proceed

    IMPORTANT: CommandListener.started() exceptions are swallowed by PyMongo's _handle_exception()
    (printed to stderr only). For true blocking, the interceptor (_encrypter) is the primary defense.
    This guard serves as:
        1. Detection of _encrypter bypass (e.g. bulk path if _encrypter was somehow removed)
        2. Audit trail for tenant isolation compliance
        3. Alerting mechanism for operations team
    """

    def __init__(self, excluded_collections: Optional[Set[str]] = None):
        """
        Args:
            excluded_collections: Collection names to skip checking
        """
        self._excluded_collections = excluded_collections or set()
        self._violation_count = 0

    @property
    def violation_count(self) -> int:
        """Number of detected violations since creation."""
        return self._violation_count

    def started(self, event: Any) -> None:
        """Called by PyMongo before each command is sent."""
        cmd_name = event.command_name

        if cmd_name not in _DATA_COMMANDS:
            return

        # Check tenant context
        tid = get_current_tenant_id()
        if not tid:
            return

        # Check collection exclusion
        cmd = event.command
        collection_name = cmd.get(cmd_name)
        if (
            isinstance(collection_name, str)
            and collection_name in self._excluded_collections
        ):
            return

        # Verify tenant_id is present in the command
        missing = self._check_tenant_id(cmd_name, cmd, tid)
        if missing:
            self._violation_count += 1
            msg = (
                "\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                "!!  TENANT ISOLATION VIOLATION — DATA LEAK RISK            !!\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                f"!!  Command:    {cmd_name}\n"
                f"!!  Collection: {collection_name}\n"
                f"!!  Expected:   tenant_id={tid}\n"
                f"!!  Violation:  {missing}\n"
                "!!  Cause:      Interceptor (_encrypter) was bypassed\n"
                f"!!  Count:      #{self._violation_count}\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )
            logger.error(msg)

            # Always raise — PyMongo swallows the exception via _handle_exception()
            # and prints the full traceback to stderr, making it highly visible
            # in both application logs (logger.error) and process stderr output.
            raise TenantIsolationViolation(msg)

    def _check_tenant_id(
        self, cmd_name: str, cmd: Any, expected_tid: str
    ) -> Optional[str]:
        """
        Check if tenant_id is correctly present in the command.

        Returns None if OK, or a description of what's missing.
        """
        if cmd_name == "insert":
            docs = cmd.get("documents", [])
            for i, doc in enumerate(docs):
                if not isinstance(doc, Mapping):
                    continue
                if doc.get(TENANT_ID_FIELD) != expected_tid:
                    return f"documents[{i}] has tenant_id={doc.get('tenant_id')!r}, expected {expected_tid!r}"
            return None

        elif cmd_name == "find":
            f = cmd.get("filter", {})
            return self._check_filter(f, expected_tid, "filter")

        elif cmd_name == "update":
            updates = cmd.get("updates", [])
            for i, u in enumerate(updates):
                q = u.get("q", {})
                result = self._check_filter(q, expected_tid, f"updates[{i}].q")
                if result:
                    return result
            return None

        elif cmd_name == "delete":
            deletes = cmd.get("deletes", [])
            for i, d in enumerate(deletes):
                q = d.get("q", {})
                result = self._check_filter(q, expected_tid, f"deletes[{i}].q")
                if result:
                    return result
            return None

        elif cmd_name == "findAndModify":
            q = cmd.get("query", {})
            return self._check_filter(q, expected_tid, "query")

        elif cmd_name == "aggregate":
            pipeline = cmd.get("pipeline", [])
            if not pipeline:
                return "empty pipeline, no $match for tenant_id"
            first = pipeline[0]
            match = first.get("$match", {})
            if match.get(TENANT_ID_FIELD) != expected_tid:
                return f"pipeline[0].$match.tenant_id={match.get('tenant_id')!r}, expected {expected_tid!r}"
            return None

        elif cmd_name in ("count", "distinct"):
            q = cmd.get("query", {})
            return self._check_filter(q, expected_tid, "query")

        return None

    def _check_filter(
        self, filter_doc: Any, expected_tid: str, location: str
    ) -> Optional[str]:
        """Check if a filter dict contains the correct tenant_id."""
        if not isinstance(filter_doc, Mapping):
            return f"{location} is not a mapping: {type(filter_doc)}"
        if filter_doc.get(TENANT_ID_FIELD) != expected_tid:
            return f"{location}.tenant_id={filter_doc.get('tenant_id')!r}, expected {expected_tid!r}"
        return None

    def succeeded(self, event: Any) -> None:
        pass

    def failed(self, event: Any) -> None:
        pass


class TenantIsolationViolation(Exception):
    """Raised when a command bypasses tenant isolation."""

    pass


# ============================================================
# Installation
# ============================================================


def install_tenant_interceptor(
    client: Any, excluded_collections: Optional[Set[str]] = None
) -> TenantCommandInterceptor:
    """
    Install the tenant command interceptor AND guard listener on a PyMongo AsyncMongoClient.

    Two-layer defense:
        Layer 1 (_encrypter): Intercepts and modifies commands before serialization
        Layer 2 (CommandListener): Verifies commands after serialization, independent hook point

    On violation, Layer 2 outputs a highly visible error to both logger and stderr:
        - logger.error() with banner format
        - TenantIsolationViolation exception (caught by PyMongo, traceback printed to stderr)

    Args:
        client: An AsyncMongoClient instance (the real client, not the tenant-aware proxy)
        excluded_collections: Collection names to skip tenant filtering

    Returns:
        The installed interceptor instance

    Raises:
        RuntimeError: If the client already has an _encrypter (CSFLE enabled)
    """
    existing = getattr(client, "_encrypter", None)
    if existing is not None and not isinstance(existing, TenantCommandInterceptor):
        raise RuntimeError(
            "Cannot install TenantCommandInterceptor: client already has an _encrypter "
            f"({type(existing).__name__}). This interceptor cannot coexist with CSFLE."
        )

    # Layer 1: Command interceptor (_encrypter hook)
    interceptor = TenantCommandInterceptor(excluded_collections=excluded_collections)
    client._encrypter = interceptor

    # Layer 2: Guard listener (CommandListener hook — independent code path)
    guard = TenantGuardListener(excluded_collections=excluded_collections)
    listeners = getattr(client, "_event_listeners", None)
    if listeners is not None:
        # Python name mangling: __command_listeners → _EventListeners__command_listeners
        listeners._EventListeners__command_listeners.append(guard)
        listeners._EventListeners__enabled_for_commands = True
        logger.info("Tenant guard listener installed")
    else:
        logger.warning(
            "Cannot install tenant guard listener: client has no _event_listeners. "
            "Guard verification will not be available."
        )

    logger.info(
        "Tenant command interceptor installed on client %s "
        "(excluded_collections=%s)",
        type(client).__name__,
        excluded_collections or "none",
    )
    return interceptor
