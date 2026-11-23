"""MemTrace file-tracking tracer for EverOS.

This tracer combines **file tracking** with captured pipeline events. The
application injects it into :class:`everos.service.memory.MemoryService`, whose
add, flush and search methods call it at each request boundary. It diffs the
memory-root markdown files at every request
boundary. File observations capture terminal disk changes; they cannot recover
every intermediate construction/update/deletion between scans. Actual boundary
and episode extractor inputs/results come from request-local construction events.

The ``/memory/search`` hook additionally records the per-stage candidate
pools of the retrieval pipeline (sparse / dense recall, fusion, rerank)
collected by :mod:`everos.memory.search.trace` — a zero-dependency
``ContextVar`` collector in the memory layer, so the ``smartcomment``
dependency stays confined to this package.

All observations are recorded into `smartcomment`_ execution graphs keyed by
``(app_id, project_id, user_id, session_id)``: one long-lived graph per user
per session and business scope, exported as MemTraceBench-compatible JSON to
``<memory-root>/memtrace/`` (default; ``EVEROS_MEMTRACE_DIR`` overrides).
Exports are batched: a request that dirties the same graph several times
(message recording plus an N-entry file diff) writes it out once per phase,
not once per change.

The exported graphs double as the tracker's restart-recovery state. The
first post-restart diff restores unchanged entry snapshots from them and
reconciles downtime deletions — a tracked entry/artifact whose file
disappeared (or vanished from its file) while the process was down gets its
deletion mutation then. The message-content index is process-local; the
memcell rows persisted by boundary detection
(:class:`everos.infra.persistence.sqlite.Memcell`) carry the exact boundary
message ids, so post-restart extractions still re-link their buffer
messages (as identity-only stubs where the content index no longer covers
them).

Configuration (environment variables, no settings-schema change):

* ``EVEROS_MEMTRACE_ENABLED`` — off by default; ``1``/``true``/``yes``
  records. Tracing is a debugging and benchmarking aid, not a production
  setting: every request diffs the whole memory root under one
  process-wide lock (the cost grows with the markdown file count and
  concurrent requests wait their turn), and the exported graphs keep raw
  message and memory text that ``/memory/delete`` and purge do not remove.
* ``EVEROS_MEMTRACE_DIR`` — export directory (default ``<root>/memtrace``).

Identity / category vocabulary (aligned with MemTraceBench and the earlier
inline-instrumentation experiment):

* variables: ``message`` (identity ``m_<session>_<ts_ms>_<idx>`` — the
  ingest-side ``gen_message_id`` format, so ``/add`` messages stitch onto
  the in-flight buffer messages returned by ``/search``),
  ``message_buffer`` (``buffer-<session>-<owner>``), ``memory_entry``
  (memcell ``mc_*`` bare; entries ``{owner}_{entry_id}``; profile
  ``profile-{owner}``), ``message & query`` + class ``query``,
  ``memory_context`` + classes ``context`` / ``answer_context``, and request-local
  ``retrieval_result`` + class ``retrieval_hit``;
* candidate pools: ``memory_context`` variables with identity
  ``candidates_{kind}_{phase}-{query_id}`` — one per recorded search stage
  (``sparse`` / ``dense`` / ``single_route`` / ``maxsim_atomic`` recall,
  ``rrf`` / ``hierarchy`` / ``filtered`` / ``fusion`` / ``bridged_dense``
  fusion, ``agentic``); the pool snapshot carries the capped
  ``ids`` / ``scores`` / ``sources`` lists. Only the last pool of each kind
  expands candidates into request-local ``retrieval_hit`` nodes (identity-only
  observations when content is unknown). Returned hits share membership edges;
  a returned entry absent from the snapshot gets a ``response_only`` edge,
  without changing the real pool's candidate list. Each pool also carries
  a native-style ``class_name``
  (``sparse_retrieval_results`` / ``dense_retrieval_results`` /
  ``hybrid_retrieval_results`` / ``reranked_results`` / …, see
  ``_POOL_CLASS_NAMES``) so downstream attribution can group pools the way
  the native MemTraceBench graphs do. The retrieval chain runs
  ``query -> recall pools -> fusion pools -> last pool -> hits``; stage-less
  searches keep the legacy ``query -> hits`` chain;
* pending session messages: one ``retrieval_result`` variable of class
  ``retrieval_results`` (produced from the query and linked into the
  assembled context) carries the pending messages without expanding message
  nodes; profile tracing is disabled while
  actual profile response content remains in the context snapshot;
* construction inputs: ``memcell -> cell-input`` is shared by episode and
  foresight extraction, with separate ``cell-input -> extracted -> episode``
  and ``cell-input -> foresight`` branches and no extraction-start placeholders;
* boundary results: each boundary event records one ``boundary_outcome`` /
  ``boundary_detection_result`` variable (identity
  ``boundary-result-<session>-<digest>``, content-addressed by the sorted
  boundary message ids, so a deferred add and the flush that later decides
  over the same messages become versions of one node instead of duplicate
  results on the same buffer version) carrying the detection status
  plus the carved cell/message ids, chained
  ``message_buffer -> boundary_detection_result -> memcell`` under a single
  event-level ``boundary_detection`` operation (mirroring the native
  membase/EverMemOS graphs); a detection that splits nothing ends in a
  ``no_split_marker`` ``marker`` node; deferred adds also record an outcome
  with ``decision=deferred`` and ``detection_performed=false``;
* operations: ``initialization`` (memory_add), ``boundary_detection``,
  ``extraction`` (entry extractions and the request-level ``memory_flush``
  op), ``update`` / ``deletion`` (mutations), ``retrieval``
  (memory_search / context_assembly plus one explicit op per recorded
  search stage — ``{kind}_{phase}_{seq}`` — with the stage's real
  inputs/outputs, mirroring the native graphs). Failed requests still leave
  a request-level op (``memory_add_*`` / ``memory_flush_*`` /
  ``memory_search_*``) carrying ``status: "error"`` plus the exception
  type / message, so a failed add / flush / search is visible in the graph.
* QA chain (native MemTraceBench answer/evaluation vocabulary):
  :meth:`MemTraceTracker.on_answer` appends ``query (+ context + prompt) ->
  prediction`` (``llm_response`` / class ``prediction`` node,
  ``answer_generation_*`` op of category ``response``) and
  :meth:`MemTraceTracker.on_evaluation` appends ``query ->
  golden_answers`` (``inject_golden_answers_*``) and ``prediction +
  golden_answers -> judge_response`` (``judge_answer_*``), both of
  category ``evaluation``. Both hooks resolve the query node by request id
  (honouring the ``-NNNN`` collision suffix) and are fail-safe no-ops for
  unrecorded searches. For out-of-process QA (e.g. the LoCoMo drive
  script), :mod:`everos.infra.memtrace.qa` injects the same chain into an
  already-exported graph offline; ``examples/memtrace/inject_qa.py`` is the
  CLI wrapper.

Operation metadata vocabulary: every ``comment_op`` / ``comment_op_scope``
/ ``comment_mutation`` carries ``event_seq`` (a process-local monotonic
sequence restoring the true observation order of the post-hoc records) and
``observed_at`` (UTC ISO timestamp). Ops produced by the file diff
additionally carry ``source: "file_diff"`` and ``trigger`` (``add`` /
``flush`` / ``search`` / ``shutdown`` — the hook that drove the diff);
request-level ops carry the request's ``request_id``. Stage pools carry
``total_count`` (pre-truncation pool size) and ``truncated`` so consumers
can tell a capped pool from a complete one.

Graph files are named with one ``{safe}-{digest}`` fragment per key
component, joined by ``__``: ``safe`` keeps the name readable, the SHA-1
``digest`` suffix keeps colliding ids (CJK names, ``a@b`` vs ``a_b``) in
distinct files. The naming scheme changed after the first instrumented
drop; files written with the old collision-prone names are orphans (the
raw ids cannot be recovered from them) — delete or ignore them, there is
no migration.

Version-semantics notes:

* Memcell nodes carry a ``membership`` marker: ``exact`` when the message
  list came from the memcell's SQLite row, ``approximate`` when the
  session heuristic had to substitute (row missing or unparsable), and
  ``lookup_failed`` when the SQLite read itself failed — the heuristic
  result is then marked as suspect rather than merely approximate. A
  non-``exact`` node also carries ``membership_detail`` with the raw
  status (``row_missing`` / ``invalid_payload`` / ``lookup_failed``).
  SQLite also supplies their immutable sender ids. Each boundary is recorded
  once — buffer (or message fallback) → ``boundary_detection_result`` →
  memcell — then reused by entries.
  Restart recovery reuses the exported boundary; an exact lookup can upgrade
  an approximation without a later failed read downgrading it.
* Candidates are always recorded ``identity_only``; a search hit creates a
  rich snapshot node only when the identity has none yet, otherwise it
  reuses the existing node and the score rides on the edge metadata. A
  later hit therefore never silently forks a candidate stub into a second
  version.
* An entry that only moves files (log rotation, OME rewrite) updates the
  file index without producing a new version — ``path`` never participates
  in change detection.
* The in-memory message index is capped per session; the graph registry
  itself grows with the number of scopes (an eviction policy is future
  work).

Every hook is fail-safe: tracing errors are logged and never propagate into
the memory request path.

Deployment constraint: graph exports are guarded by an inter-process
``portalocker`` file lock, so concurrent writers cannot corrupt or
last-write-trample a file — but each process still holds its own in-memory
graph instance, so multi-worker deployments should either run a single
writer or point each worker at its own ``EVEROS_MEMTRACE_DIR``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import portalocker

from everos.core.observability.logging import get_logger
from everos.core.utils.datetime import from_iso_format, get_utc_now
from everos.infra.persistence.memory_root import MemoryRoot

from .descriptions import describe_sentinels, stage_comments
from .parsing import MemoryFileEntry, MemoryFileSnapshot, parse_memory_file

if TYPE_CHECKING:
    from everos.memory.search.dto import (
        SearchRequest,
        SearchResponse,
        UnprocessedMessageDTO,
    )
    from everos.memory.search.trace import SearchStageRecord

logger = get_logger(__name__)

try:  # smartcomment is an optional tracing dependency.
    from smartcomment import (
        comment_graph,
        comment_link,
        comment_mutation,
        comment_op,
        comment_op_scope,
        comment_session,
        comment_variable,
        is_tracing_enabled,
    )
    from smartcomment.runtime import ExecNetwork

    _SMARTCOMMENT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without smartcomment
    _SMARTCOMMENT_AVAILABLE = False

# Directories that never hold memory markdown (indexes / staging / exports).
_SKIP_DIR_NAMES = {".index", ".tmp", "memtrace"}

# Per-session cap on the tracked message index; the oldest identities (and
# their snapshots / binding state) are dropped when a session outgrows it.
_MAX_TRACKED_MESSAGES_PER_SESSION = 10_000

# Error messages recorded on failed-request ops are capped so a pathological
# exception cannot bloat the exported graph.
_MAX_ERROR_MESSAGE_CHARS = 500

_ENTRY_KIND_COMMENT = {
    "episode": "A narrative memory extracted from a conversation.",
    "atomic_fact": "A factual statement extracted from a conversation episode.",
    "foresight": "A future expectation inferred from a conversation.",
    "agent_case": "A stored account of an agent's approach to a task.",
}

_ENTRY_FILE_KINDS = frozenset({"episode", "atomic_fact", "foresight", "agent_case"})

# app_id, project_id, owner_id, session bucket
GraphKey = tuple[str, str, str, str]
# app_id, project_id, smartcomment identity
ScopedIdentity = tuple[str, str, str]


def _encode(value: Any) -> str:
    """Deterministically encode a variable snapshot as sorted JSON."""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _encode_without_path(value: dict[str, Any]) -> str:
    """Encode an entry snapshot for change detection, ignoring its location.

    A pure file move (log rotation, OME rewrite) must not register as an
    entry update, so ``path`` never participates in the comparison.
    """
    return _encode({key: item for key, item in value.items() if key != "path"})


def _by_id(value: dict[str, Any]) -> str:
    """Identity strategy: every traced variable is a dict carrying ``id``."""
    return str(value["id"])


def _safe_name(raw: str) -> str:
    """Make an id safe for use in file names."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def _scoped_name(raw: str) -> str:
    """Readable-but-unique file-name fragment for one graph-key component.

    ``_safe_name`` alone folds non-ASCII ids, or ``a@b`` and ``a_b``, onto
    the same graph file; the SHA-1 digest suffix keeps every distinct
    raw id in its own file while the truncated safe prefix stays readable.
    """
    digest = hashlib.sha1(raw.encode()).hexdigest()[:8]
    return f"{_safe_name(raw)[:40]}-{digest}"


def _hit_identity(owner: str, raw_id: str) -> str:
    """Map a search-hit id to the tracker's entry identity.

    Entry identities are ``{owner}_{entry_id}``; some EverOS versions already
    prefix the owner onto response ids, so only add the prefix when missing.
    """
    return raw_id if raw_id.startswith(f"{owner}_") else f"{owner}_{raw_id}"


def _boundary_result_identity(session: str, message_ids: Sequence[str]) -> str:
    """Content-addressed identity for a boundary-detection result.

    The same buffered messages can be observed twice: a deferred add records
    an outcome without running detection and the flush that later drains them
    records the real decision. Addressing the result by its sorted message
    ids makes the second observation a new version of the same node instead
    of a duplicate result hanging off the same buffer version.
    """
    digest = hashlib.sha256(json.dumps(sorted(message_ids)).encode()).hexdigest()
    return f"boundary-result-{session}-{digest[:16]}"


def resolve_query_node(
    graph: ExecNetwork,
    *,
    owner: str,
    request_id: str,
    query_full_node_id: str | None = None,
) -> tuple[str, Any] | None:
    """Resolve the recorded query node of one search by its request id.

    Repeated request ids require an exact ``query_full_node_id`` selector.
    Return ``None`` for missing or ambiguous matches rather than choosing an
    arbitrary execution. Matching uses snapshots, including after namespacing.
    """
    matches = []
    for node in graph.search_variables(class_name="query"):
        if query_full_node_id is not None and node.full_node_id != query_full_node_id:
            continue
        try:
            snapshot = json.loads(node.value)
        except (ValueError, TypeError):
            continue
        if (
            isinstance(snapshot, dict)
            and snapshot.get("id")
            and snapshot.get("request_id") == request_id
            and snapshot.get("user_id") == owner
        ):
            matches.append(node)
    if len(matches) != 1:
        return None
    node = matches[0]
    return str(json.loads(node.value)["id"]), node


def _latest_handle(graph: ExecNetwork, full_name: str) -> Any | None:
    """Return the latest-version handle of ``full_name``, or ``None``."""
    versions = graph.get_versions(full_name)
    return versions[-1] if versions else None


def _qa_prompt_snapshot(query_id: str, prompt: str) -> dict[str, Any]:
    """Snapshot of the answer-generation prompt of one query."""
    return {"id": f"prompt-{query_id}", "query_id": query_id, "prompt": prompt}


def _qa_prediction_snapshot(query_id: str, answer: str) -> dict[str, Any]:
    """Snapshot of the predicted answer of one query."""
    return {
        "id": f"prediction-{query_id}",
        "query_id": query_id,
        "answer": answer,
    }


def _qa_golden_snapshot(query_id: str, golden_answers: Sequence[str]) -> dict[str, Any]:
    """Snapshot of the reference answers of one query."""
    return {
        "id": f"golden-{query_id}",
        "query_id": query_id,
        "answers": list(golden_answers),
    }


def _qa_judge_snapshot(
    query_id: str, judge_response: str | None, judge_score: float | None
) -> dict[str, Any]:
    """Snapshot of the judge verdict of one query."""
    return {
        "id": f"judge-{query_id}",
        "query_id": query_id,
        "response": judge_response,
        "score": judge_score,
    }


# Stage phases that draw candidates straight from a recall route — these
# pools hang directly off the query node. Every other phase (``rrf``,
# ``hierarchy``, ``filtered``, ``fusion``, ``bridged_dense``) is a fusion /
# filter stage and chains off the previous pools of the same kind. Mirrors
# the vocabulary in :mod:`everos.memory.search.trace`.
_RECALL_PHASES = frozenset(
    {"sparse", "dense", "single_route", "maxsim_atomic", "agentic"}
)

# Class names for stage-pool variables, aligned with the retrieval-result
# vocabulary of the native MemTraceBench graphs (the pool's ``category`` stays
# ``memory_context`` so existing consumers keep working; the class name lets
# downstream attribution group sparse / dense / fused / reranked pools the way
# the native graphs do).
_POOL_CLASS_NAMES: dict[str, str] = {
    "sparse": "sparse_retrieval_results",
    "dense": "dense_retrieval_results",
    "rrf": "hybrid_retrieval_results",
    "fusion": "hybrid_retrieval_results",
    "bridged_dense": "hybrid_retrieval_results",
    "hierarchy": "reranked_results",
    "filtered": "reranked_results",
    "single_route": "retrieval_results",
    "maxsim_atomic": "retrieval_results",
    "agentic": "agentic_results",
}

# Per-input options used when a request-level flush op wires the session
# buffer in as its input (the unchanged snapshot reuses the buffer's
# existing node version).
_BUFFER_INPUT_OPTIONS: dict[str, Any] = {
    "id_strategy": _by_id,
    "encoding_fn": _encode,
    "category": "message_buffer",
    "comment": "The session message buffer drained by this flush.",
}


def _unprocessed_message_snapshot(message: UnprocessedMessageDTO) -> dict[str, Any]:
    """Snapshot one in-flight buffer message returned alongside a search."""
    content = message.content
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    return {
        "id": message.id,
        "content": content,
        "name": message.sender_name or message.sender_id,
        "role": message.role,
        "session_id": message.session_id,
        "timestamp": message.timestamp.isoformat(),
        "source": "unprocessed_buffer",
    }


class MemTraceTracker:
    """File-tracking tracer that maintains one execution graph per
    ``(app_id, project_id, user_id, session_id)``.

    The tracker keeps a snapshot index of every memory markdown entry under
    the memory root. :meth:`on_memorize` and :meth:`on_search` are called by
    the service layer at each request boundary; both first diff the markdown
    tree (catching asynchronous OME writes) and then record the request
    itself. Graphs are exported as MemTraceBench-compatible JSON, batched so
    each graph is written at most once per request phase; the exports double
    as restart-recovery and downtime-reconciliation state (see the module
    docstring).
    """

    def __init__(self) -> None:
        self._memcell_repo: Any = None
        self._root: Path = MemoryRoot.resolve().root
        env_enabled = os.environ.get("EVEROS_MEMTRACE_ENABLED", "").lower() in (
            "1",
            "true",
            "yes",
        )
        if env_enabled and not _SMARTCOMMENT_AVAILABLE:
            logger.warning(
                "memtrace_smartcomment_missing",
                hint=(
                    "smartcomment is not installed; MemTrace hooks are no-ops "
                    "— install with `uv sync --extra memtrace`"
                ),
            )
        self._enabled: bool = _SMARTCOMMENT_AVAILABLE and env_enabled
        self._out_dir = Path(
            os.environ.get("EVEROS_MEMTRACE_DIR", str(self._root / "memtrace"))
        )
        # (app_id, project_id, owner_id, session_bucket) -> graph / lock.
        self._graphs: dict[GraphKey, ExecNetwork] = {}
        self._locks: dict[GraphKey, asyncio.Lock] = {}
        # Graphs dirtied since the last batched export.
        self._dirty_keys: set[GraphKey] = set()
        # Serialises the global filesystem index and its in-memory mirrors.
        self._sync_lock = asyncio.Lock()
        # Whether the first post-start diff (and with it the downtime
        # deletion reconciliation) has run.
        self._first_scan_done = False
        # Scoped message index: (app, project, identity) -> snapshot dict.
        self._messages: dict[ScopedIdentity, dict[str, Any]] = {}
        # (app, project, session) -> ordered message identities ever added.
        self._session_messages: dict[tuple[str, str, str], list[str]] = {}
        # Message identities already bound to a memcell by boundary detection.
        self._bound_messages: set[ScopedIdentity] = set()
        # Graph key -> session-buffer snapshot dict (mutation target).
        self._buffers: dict[GraphKey, dict[str, Any]] = {}
        # Scoped entry identity -> snapshot dict / source file.
        self._entries: dict[ScopedIdentity, dict[str, Any]] = {}
        self._entry_file: dict[ScopedIdentity, str] = {}
        # Scoped whole-file artifact identity -> snapshot dict / source file.
        self._artifacts: dict[ScopedIdentity, dict[str, Any]] = {}
        self._artifact_file: dict[ScopedIdentity, str] = {}
        # File path -> (mtime_ns, size) at last parse.
        self._file_state: dict[str, tuple[int, int]] = {}
        self._query_seq: dict[GraphKey, int] = {}
        # Process-local monotonic operation counter: restores the true
        # observation order of the post-hoc records for downstream
        # attribution (see ``_op_metadata``).
        self._event_seq = 0
        self._entry_observations: dict[ScopedIdentity, dict[str, Any]] = {}
        # Which hook is driving the current file diff (``add`` / ``flush`` /
        # ``search`` / ``shutdown`` / ``manual``); set by ``_sync_files``,
        # read by the file-diff op metadata.
        self._trigger = "manual"

    # ── public hooks ─────────────────────────────────────────────────────

    def set_memcell_repo(self, repo: Any) -> None:
        """Inject the application's memcell repository for exact boundary recovery."""
        self._memcell_repo = repo

    @property
    def enabled(self) -> bool:
        """Whether recording is active (smartcomment present + env toggle)."""
        return self._enabled and is_tracing_enabled()

    @property
    def out_dir(self) -> Path:
        """Directory the MemTraceBench-compatible graph JSONs are exported to."""
        return self._out_dir

    @property
    def smartcomment_available(self) -> bool:
        """Whether the optional ``smartcomment`` dependency is importable."""
        return _SMARTCOMMENT_AVAILABLE

    def _op_metadata(self, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Build operation metadata with the shared ordering fields.

        Every ``comment_op`` / ``comment_op_scope`` / ``comment_mutation``
        goes through this helper so downstream attribution can sort the
        post-hoc records back into true observation order (``event_seq``)
        and wall-clock time (``observed_at``).
        """
        self._event_seq += 1
        metadata: dict[str, Any] = {
            "event_seq": self._event_seq,
            "observed_at": get_utc_now().isoformat(),
        }
        if extra:
            metadata.update(extra)
        return metadata

    def _file_diff_metadata(
        self, extra: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Operation metadata for ops produced by the markdown file diff."""
        base: dict[str, Any] = {
            "source": "file_diff",
            "trigger": self._trigger,
            "observation_only": True,
        }
        if extra:
            base.update(extra)
        return self._op_metadata(base)

    async def on_memorize(
        self,
        payload: dict[str, Any],
        *,
        is_final: bool,
        defer_extraction: bool,
        result: Any,
        request_id: str | None = None,
        construction_events: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        """Record one ``/add`` or ``/flush`` request into the session graphs.

        Args:
            payload: The service-layer payload ``{"session_id", "app_id",
                "project_id", "messages": [...]}``.
            is_final: ``True`` for flush requests.
            defer_extraction: ``True`` when the add only buffered messages.
            result: The :class:`MemorizeResult` returned by the pipeline.
            request_id: The propagated request id (written onto the
                request-level ops).
        """
        if not self.enabled:
            return
        try:
            session_id = str(payload.get("session_id", ""))
            app_id = str(payload.get("app_id") or "default")
            project_id = str(payload.get("project_id") or "default")
            messages = payload.get("messages") or []
            buffer_loaded = next(
                (e for e in construction_events or [] if e["phase"] == "buffer_loaded"),
                None,
            )
            buffer_persisted = next(
                (
                    e
                    for e in reversed(construction_events or [])
                    if e["phase"] == "buffer_persisted"
                ),
                None,
            )
            if messages:
                # Record the request's own messages first: an /add may
                # already run boundary detection and extraction (writing
                # episode markdown synchronously), and the
                # entry -> memcell -> message links recorded by the file
                # diff below need the message index populated.
                await self._record_messages(
                    app_id,
                    project_id,
                    session_id,
                    messages,
                    defer_extraction=defer_extraction,
                    request_id=request_id,
                    buffer_loaded=buffer_loaded,
                )
            # Diff the markdown tree; also catches background (OME) writes
            # that landed since the previous request. A flush writes its
            # episode markdown synchronously inside ``_memorize_locked``, so
            # one diff after it observes everything.
            await self._record_construction_events(construction_events, request_id)
            if buffer_persisted is not None and not is_final:
                await self._record_buffer_state(buffer_persisted, request_id=request_id)
            await self._sync_files(trigger="flush" if is_final else "add")
            if is_final:
                # A flush is its own request-level event: record it even
                # when the diff found nothing, so an empty flush stays
                # distinguishable from "no request happened".
                await self._record_flush(
                    app_id,
                    project_id,
                    session_id,
                    result=result,
                    request_id=request_id,
                    buffer_state=buffer_persisted
                    or (
                        buffer_loaded
                        if buffer_loaded and not buffer_loaded["message_ids"]
                        else None
                    ),
                )
        except Exception as exc:
            logger.warning("memtrace_on_memorize_failed", error=str(exc))

    async def on_memorize_error(
        self,
        payload: dict[str, Any],
        *,
        is_final: bool,
        defer_extraction: bool,
        error: BaseException,
        request_id: str | None = None,
        construction_events: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        """Record a failed ``/add`` or ``/flush`` request.

        The business path re-raises after this hook; the graph keeps a
        request-level op carrying ``status: "error"`` plus the exception
        type/message, so failed extractions stay attributable. A best-effort
        file diff runs first — the pipeline may have written partial
        markdown before raising.
        """
        if not self.enabled:
            return
        try:
            session_id = str(payload.get("session_id", ""))
            app_id = str(payload.get("app_id") or "default")
            project_id = str(payload.get("project_id") or "default")
            await self._record_construction_events(construction_events, request_id)
            for event in construction_events or []:
                if event["phase"] == "buffer_persisted":
                    await self._record_buffer_state(event, request_id=request_id)
            await self._sync_files(trigger="flush" if is_final else "add")
            error_metadata: dict[str, Any] = {
                "session_id": session_id,
                "is_final": is_final,
                "defer_extraction": defer_extraction,
                "status": "error",
                "error_type": type(error).__name__,
                "error_message": str(error)[:_MAX_ERROR_MESSAGE_CHARS],
                "request_id": request_id,
            }
            if is_final:
                op_name = f"memory_flush_{session_id}"
                category = "extraction"
                keys = self._flush_graph_keys(app_id, project_id, session_id)
            else:
                op_name = f"memory_add_{session_id}"
                category = "initialization"
                owners = {
                    str(m.get("sender_id") or "unknown")
                    for m in payload.get("messages", [])
                }
                keys = [
                    (app_id, project_id, owner, session_id)
                    for owner in sorted(owners or {"unknown"})
                ]
            for key in keys:
                buffer = self._buffers.get(key)

                def record(buf: dict[str, Any] | None = buffer) -> None:
                    attempted = [
                        comment_variable(
                            self._message_snapshot(
                                app_id,
                                project_id,
                                session_id,
                                str(message.get("sender_id") or "unknown"),
                                message,
                                idx,
                            ),
                            to_runtime=True,
                            id_strategy=_by_id,
                            encoding_fn=_encode,
                            category="message",
                            comment=(
                                "An input message submitted in a failed memory request."
                            ),
                            metadata={"acceptance": "attempted"},
                        )
                        for idx, message in enumerate(payload.get("messages") or [])
                    ]
                    comment_op(
                        op_name=op_name,
                        category=category,
                        comment=(
                            "The request failed; any partial markdown writes "
                            "were captured by the accompanying file diff."
                        ),
                        inputs=(
                            attempted
                            + (
                                [(buf, dict(_BUFFER_INPUT_OPTIONS))]
                                if buf is not None
                                else []
                            )
                        ),
                        outputs=[],
                        metadata=self._op_metadata(error_metadata),
                    )

                await self._update_graph(
                    key,
                    session_name=op_name,
                    session_category="memory_construction",
                    record=record,
                )
        except Exception as exc:
            logger.warning("memtrace_on_memorize_error_failed", error=str(exc))

    async def on_search(
        self,
        req: SearchRequest,
        resp: SearchResponse,
        *,
        stages: Sequence[SearchStageRecord] | None = None,
        request_id: str | None = None,
    ) -> None:
        """Record one ``/search`` request: query -> pools -> hits -> context.

        Args:
            req: The validated search request.
            resp: The search response returned to the caller.
            stages: Ordered per-stage candidate pools collected by
                :func:`everos.memory.search.trace.collect_search_trace`
                around the manager call; ``None``/empty keeps the legacy
                ``query -> hits`` chain.
            request_id: The propagated request id; forms the query node's
                identity (``query-{owner}-{request_id}``).
        """
        if not self.enabled:
            return
        try:
            await self._sync_files(trigger="search")
            await self._record_search(req, resp, stages=stages, request_id=request_id)
        except Exception as exc:
            logger.warning("memtrace_on_search_failed", error=str(exc))

    async def on_search_error(
        self,
        req: SearchRequest,
        *,
        stages: Sequence[SearchStageRecord] | None = None,
        error: BaseException,
        request_id: str | None = None,
    ) -> None:
        """Record a failed ``/search`` request.

        The query node and whatever stage pools the collector gathered
        before the failure (possibly none, possibly partial) are recorded
        under a ``status: "error"`` retrieval op, so a failed query is
        visible in the graph. The business path re-raises after this hook.
        """
        if not self.enabled:
            return
        try:
            await self._sync_files(trigger="search")
            await self._record_search_error(
                req, stages=stages, error=error, request_id=request_id
            )
        except Exception as exc:
            logger.warning("memtrace_on_search_error_failed", error=str(exc))

    async def sync_now(self) -> None:
        """Final sync for shutdown: one last diff plus a dirty-graph flush.

        Catches asynchronous writes (OME) that landed after the last
        request and exports every graph dirtied since the previous flush.
        Fail-safe like every other hook.
        """
        if not self.enabled:
            return
        try:
            await self._sync_files(trigger="shutdown")
            await self._flush_dirty_graphs()
        except Exception as exc:
            logger.warning("memtrace_sync_now_failed", error=str(exc))

    # ── QA recording (answer / evaluation) ─────────────────────────────

    async def on_answer(
        self,
        *,
        app_id: str,
        project_id: str,
        owner: str,
        request_id: str,
        answer: str,
        prompt: str | None = None,
        answer_context: str | None = None,
        bucket: str = "global",
        query_full_node_id: str | None = None,
    ) -> None:
        """Record the answer stage of a previously recorded search.

        Appends the QA chain ``query (+ context + prompt) -> prediction``
        to the search's graph, using the native MemTraceBench vocabulary
        (``llm_response`` / class ``prediction`` node, ``response``
        operation), so a retrieval-correct-but-answer-wrong failure stays
        attributable (``ResponseError``).

        Args:
            app_id: The app scope of the original search.
            project_id: The project scope of the original search.
            owner: The user the original search ran for.
            request_id: The request id of the original search (the query
                node identity is ``query-{owner}-{request_id}``).
            answer: The answer text generated from the retrieved context.
            prompt: The answer-generation prompt, when the caller has it.
            bucket: The session bucket of the original search (``global``
                unless it ran with a ``session_id`` filter).
        """
        if not self.enabled:
            return
        try:
            key = (app_id, project_id, owner, bucket)
            await self._update_graph(
                key,
                session_name=f"answer_generation_{owner}_{request_id}",
                session_category="memory_retrieval",
                record=lambda: self._record_answer(
                    key,
                    owner=owner,
                    request_id=request_id,
                    answer=answer,
                    prompt=prompt,
                    answer_context=answer_context,
                    query_full_node_id=query_full_node_id,
                ),
            )
        except Exception as exc:
            logger.warning("memtrace_on_answer_failed", error=str(exc))

    def _record_answer(
        self,
        key: GraphKey,
        *,
        owner: str,
        request_id: str,
        answer: str,
        prompt: str | None,
        answer_context: str | None = None,
        query_full_node_id: str | None = None,
    ) -> None:
        """Record one ``answer_generation`` op inside the graph context."""
        graph = self._graphs[key]
        resolved = resolve_query_node(
            graph,
            owner=owner,
            request_id=request_id,
            query_full_node_id=query_full_node_id,
        )
        if resolved is None:
            logger.warning(
                "memtrace_qa_query_not_found",
                owner=owner,
                request_id=request_id,
                bucket=key[3],
            )
            return
        query_id, query_handle = resolved
        context_handle = _latest_handle(graph, f"context:context-{query_id}")
        from .answer_context import record_answer_context

        context_handle = record_answer_context(
            query_id=query_id,
            query_handle=query_handle,
            retrieved=context_handle,
            answer_context=answer_context,
            request_id=request_id,
        )
        prompt_handle = (
            comment_variable(
                _qa_prompt_snapshot(query_id, prompt),
                to_runtime=True,
                id_strategy=_by_id,
                encoding_fn=_encode,
                category="prompt",
                comment="The answer-generation prompt for this query.",
            )
            if prompt is not None
            else None
        )
        prediction_handle = comment_variable(
            _qa_prediction_snapshot(query_id, answer),
            to_runtime=True,
            id_strategy=_by_id,
            encoding_fn=_encode,
            category="llm_response",
            class_name="prediction",
            comment="The answer generated from the retrieved memory context.",
        )
        comment_op(
            inputs=[
                handle
                for handle in (query_handle, context_handle, prompt_handle)
                if handle is not None
            ],
            outputs=[prediction_handle],
            op_name=f"answer_generation_{query_id}",
            category="response",
            comment="Generate the final answer from the retrieved memory context.",
            metadata=self._op_metadata(
                {"request_id": request_id, "query_id": query_id}
            ),
        )

    async def on_evaluation(
        self,
        *,
        app_id: str,
        project_id: str,
        owner: str,
        request_id: str,
        golden_answers: Sequence[str],
        judge_response: str | None = None,
        judge_score: float | None = None,
        bucket: str = "global",
        query_full_node_id: str | None = None,
    ) -> None:
        """Record the evaluation stage of a previously recorded search.

        Appends ``query -> golden_answers`` (``inject_golden_answers``)
        and, when a verdict is supplied,
        ``prediction + golden_answers -> judge_response`` (``judge_answer``)
        to the search's graph — the native MemTraceBench ``evaluation``
        operation vocabulary.
        """
        if not self.enabled:
            return
        try:
            key = (app_id, project_id, owner, bucket)
            await self._update_graph(
                key,
                session_name=f"evaluation_{owner}_{request_id}",
                session_category="memory_retrieval",
                record=lambda: self._record_evaluation(
                    key,
                    owner=owner,
                    request_id=request_id,
                    golden_answers=golden_answers,
                    judge_response=judge_response,
                    judge_score=judge_score,
                    query_full_node_id=query_full_node_id,
                ),
            )
        except Exception as exc:
            logger.warning("memtrace_on_evaluation_failed", error=str(exc))

    def _record_evaluation(
        self,
        key: GraphKey,
        *,
        owner: str,
        request_id: str,
        golden_answers: Sequence[str],
        judge_response: str | None,
        judge_score: float | None,
        query_full_node_id: str | None = None,
    ) -> None:
        """Record one ``inject_golden_answers`` / ``judge_answer`` pair."""
        graph = self._graphs[key]
        resolved = resolve_query_node(
            graph,
            owner=owner,
            request_id=request_id,
            query_full_node_id=query_full_node_id,
        )
        if resolved is None:
            logger.warning(
                "memtrace_qa_query_not_found",
                owner=owner,
                request_id=request_id,
                bucket=key[3],
            )
            return
        query_id, query_handle = resolved
        golden_handle = comment_variable(
            _qa_golden_snapshot(query_id, golden_answers),
            to_runtime=True,
            id_strategy=_by_id,
            encoding_fn=_encode,
            category="golden_answers",
            class_name="golden_answers",
            comment="Reference answers injected by the evaluation batch.",
        )
        comment_op(
            inputs=[query_handle],
            outputs=[golden_handle],
            op_name=f"inject_golden_answers_{query_id}",
            category="evaluation",
            comment="Reference answers injected by the evaluation batch.",
            metadata=self._op_metadata(
                {"request_id": request_id, "query_id": query_id}
            ),
        )
        if judge_response is None and judge_score is None:
            return
        judge_handle = comment_variable(
            _qa_judge_snapshot(query_id, judge_response, judge_score),
            to_runtime=True,
            id_strategy=_by_id,
            encoding_fn=_encode,
            category="judge_response",
            class_name="judge_response",
            comment="The judge verdict for this query's answer.",
        )
        prediction_handle = _latest_handle(graph, f"prediction:prediction-{query_id}")
        comment_op(
            inputs=[
                handle
                for handle in (prediction_handle, golden_handle, query_handle)
                if handle is not None
            ],
            outputs=[judge_handle],
            op_name=f"judge_answer_{query_id}",
            category="evaluation",
            comment="Judge the predicted answer against the reference answers.",
            metadata=self._op_metadata(
                {"request_id": request_id, "query_id": query_id}
            ),
        )

    async def _record_construction_events(
        self,
        events: Sequence[Mapping[str, Any]] | None,
        request_id: str | None,
    ) -> None:
        """Replay captured boundary inputs and actual extraction results.

        Event timestamps describe the actual boundary/extractor return, unlike
        file observations. Background evolution still requires independent events.
        """
        for event in events or []:
            phase = event["phase"]
            if phase == "extraction_attempt":
                continue
            if phase == "buffer_loaded":
                await self._record_buffer_state(event, request_id=request_id)
                continue
            if phase == "buffer_persisted":
                # Replay tails after their boundary/extraction inputs. Successful
                # flushes use the request-level flush mutation for this step.
                continue
            if phase == "boundary":
                await self._record_boundary_event(event, request_id)
                continue
            app, project, session = (
                str(event[field]) for field in ("app_id", "project_id", "session_id")
            )
            for cell in [event]:
                owners = cell.get("owners", [])
                for owner in owners or ["unknown"]:
                    key = (app, project, owner, session)
                    mid = str(cell.get("id") or cell.get("memcell_id"))
                    metadata = {
                        "source": "construction_event",
                        "request_id": request_id,
                        "observed_at": event["observed_at"],
                        "phase": phase,
                    }

                    def record(
                        cell: Mapping[str, Any] = cell,
                        key: GraphKey = key,
                        mid: str = mid,
                        phase: str = phase,
                        metadata: dict[str, Any] = metadata,
                    ) -> None:
                        graph = self._graphs[key]
                        result_id = f"extracted-{mid}-{key[2]}-{request_id or 'direct'}"
                        if phase == "persisted_episode":
                            result = _latest_handle(graph, result_id)
                            if result is not None:
                                identity = f"{key[2]}_{cell['entry_id']}"
                                self._entry_observations[(key[0], key[1], identity)] = {
                                    **metadata,
                                    "source_node": result.full_node_id,
                                }
                        elif phase == "extraction_result":
                            parent = _latest_handle(graph, f"cell-input-{mid}")
                            result = comment_variable(
                                {
                                    "id": result_id,
                                    "content": cell.get("content"),
                                    "status": phase,
                                },
                                to_runtime=True,
                                id_strategy=_by_id,
                                encoding_fn=_encode,
                                category="extraction_result",
                                comment=(
                                    "The episode text extracted from the selected "
                                    "conversation content."
                                ),
                            )
                            comment_op(
                                inputs=[parent] if parent is not None else [],
                                outputs=[result],
                                op_name=f"{phase}_{mid}",
                                category="extraction",
                                comment=(
                                    "Extract an episode from the selected "
                                    "conversation content."
                                ),
                                metadata=self._op_metadata(metadata),
                            )

                    await self._update_graph(
                        key,
                        session_name=f"construction_{phase}",
                        session_category="memory_construction",
                        record=record,
                        export=False,
                    )

    async def _record_boundary_event(
        self,
        event: Mapping[str, Any],
        request_id: str | None,
    ) -> None:
        """Record the boundary decision and its message-dependent cell creation.

        Connect the consumed buffer through its structured boundary result to
        each produced memcell. Model decisions stay on the result; deferred
        requests retain a service outcome without implying a model call.
        """
        app, project, session = (
            str(event[field]) for field in ("app_id", "project_id", "session_id")
        )
        cells = list(event.get("cells", []))
        by_owner: dict[str, list[Mapping[str, Any]]] = {}
        for cell in cells:
            for sender in cell.get("sender_ids", []) or ["unknown"]:
                by_owner.setdefault(str(sender), []).append(cell)
        if not by_owner:
            # A no-split detection has no cells to derive owners from; record
            # it in every graph that tracks a buffer for this session.
            by_owner = {
                key[2]: [] for key in self._flush_graph_keys(app, project, session)
            }
        tag = request_id or uuid.uuid4().hex
        deferred = event.get("reason") == "deferred"
        detection = event.get("detection") or {}
        for owner in sorted(by_owner):
            key = (app, project, owner, session)
            owner_cells = by_owner[owner]

            def record(
                key: GraphKey = key,
                owner: str = owner,
                owner_cells: list[Mapping[str, Any]] = owner_cells,
            ) -> None:
                graph = self._graphs[key]
                metadata: dict[str, Any] = {
                    "source": "construction_event",
                    "request_id": request_id,
                    "observed_at": event["observed_at"],
                    "phase": "boundary",
                    "status": event.get("status"),
                    "reason": event.get("reason"),
                    "detection_performed": event.get("detection_performed"),
                }
                message_ids = detection.get("message_ids") or list(
                    dict.fromkeys(
                        message_id
                        for cell in owner_cells
                        for message_id in cell.get("message_ids", [])
                    )
                )
                if not owner_cells:
                    current_buffer = _latest_handle(graph, f"buffer-{session}-{owner}")
                    if current_buffer is not None:
                        message_ids = list(
                            json.loads(current_buffer.raw_value)["message_ids"]
                        )
                result_identity = _boundary_result_identity(session, message_ids)
                with comment_op_scope(
                    op_name=f"boundary_detection_{session}_{tag}",
                    category="boundary_detection",
                    comment=(
                        "Keep the session messages buffered because extraction "
                        "was deferred."
                        if deferred
                        else "Determine whether the buffered conversation can be "
                        "split into memory units."
                    ),
                    metadata=self._op_metadata(metadata),
                ):
                    result = comment_variable(
                        {
                            "id": result_identity,
                            "session_id": session,
                            "request_id": request_id,
                            "status": event.get("status"),
                            "decision": (
                                "deferred"
                                if deferred
                                else "split"
                                if owner_cells
                                else "no_split"
                            ),
                            "detection_performed": event.get("detection_performed"),
                            "cell_ids": [
                                str(cell.get("id") or cell.get("memcell_id"))
                                for cell in owner_cells
                            ],
                            "message_ids": message_ids,
                            **(
                                {
                                    "model_result": detection["model_result"],
                                    "is_final": detection["is_final"],
                                    "mode": detection["mode"],
                                }
                                if detection
                                else {}
                            ),
                        },
                        to_runtime=True,
                        id_strategy=_by_id,
                        encoding_fn=_encode,
                        category=(
                            "llm_response"
                            if detection.get("model_result") is not None
                            else "boundary_outcome"
                        ),
                        class_name="boundary_detection_result",
                        comment=(
                            "The messages remain buffered until a later request "
                            "runs boundary detection."
                            if deferred
                            else "The model's boundary decision and the resulting "
                            "message groups, including any final flush closure."
                            if detection.get("model_result") is not None
                            else "The boundary stage outcome records whether "
                            "the conversation was split into memory units."
                        ),
                    )
                    buffer_handle = next(
                        (
                            handle
                            for handle in reversed(
                                graph.get_versions(f"buffer-{session}-{owner}")
                            )
                            if message_ids
                            and set(message_ids).issubset(
                                json.loads(handle.raw_value)["message_ids"]
                            )
                        ),
                        None,
                    )
                    if buffer_handle is None and not message_ids:
                        buffer_handle = _latest_handle(
                            graph, f"buffer-{session}-{owner}"
                        )
                    if buffer_handle is not None:
                        comment_op(
                            inputs=[buffer_handle],
                            outputs=[result],
                            reuse_op=True,
                            comment=(
                                "Record that boundary detection was deferred "
                                "for these buffered messages."
                                if deferred
                                else "Assess conversation boundaries for these "
                                "buffered session messages."
                            ),
                            metadata={"boundary_input": True},
                        )
                    else:
                        # Mirror the memcell fallback: link the analysed
                        # messages directly when no covering buffer version
                        # was recorded.
                        linked: set[str] = set()
                        for message_id in message_ids:
                            if message_id in linked:
                                continue
                            linked.add(message_id)
                            snapshot = self._messages.get((app, project, message_id))
                            message_handle = comment_variable(
                                snapshot
                                if snapshot is not None
                                else {"id": message_id},
                                to_runtime=True,
                                id_strategy=_by_id,
                                encoding_fn=_encode,
                                identity_only=snapshot is None,
                                category="message",
                                comment=(
                                    "A session message analysed by boundary detection."
                                ),
                            )
                            comment_link(
                                source=message_handle,
                                target=result,
                                comment=(
                                    "Feed this buffered message into boundary "
                                    "detection."
                                ),
                            )
                    if not owner_cells:
                        marker = comment_variable(
                            {
                                "id": f"no-split-{session}-{tag}",
                                "reason": "deferred" if deferred else "no_split",
                            },
                            to_runtime=True,
                            id_strategy=_by_id,
                            encoding_fn=_encode,
                            category="marker",
                            class_name="no_split_marker",
                            comment=(
                                "No memory unit was created, so the current "
                                "messages remain buffered."
                            ),
                        )
                        comment_link(
                            source=result,
                            target=marker,
                            comment=(
                                "Retain the current messages for later processing "
                                "because no memory unit was created."
                            ),
                        )
                    for cell in owner_cells:
                        mid = str(cell.get("id") or cell.get("memcell_id"))
                        entry = MemoryFileEntry(
                            entry_id=mid,
                            app_id=key[0],
                            project_id=key[1],
                            owner_id=key[2],
                            session_id=key[3],
                            timestamp=None,
                            parent_type="memcell",
                            parent_id=mid,
                        )
                        parent = self._record_parent(
                            entry,
                            key[2],
                            message_ids=list(cell["message_ids"]),
                            sender_ids=list(cell["sender_ids"]),
                            observed_event=metadata,
                            boundary_result=result,
                        )
                        self._record_cell_input(
                            key,
                            mid,
                            parent,
                            content=cell["content"],
                            metadata=self._op_metadata(metadata),
                        )

            await self._update_graph(
                key,
                session_name="construction_boundary",
                session_category="memory_construction",
                record=record,
                export=False,
            )

    def _record_cell_input(
        self,
        key: GraphKey,
        memcell_id: str,
        parent: Any,
        *,
        content: Any,
        metadata: Mapping[str, Any],
    ) -> Any:
        """Reuse a cell's raw input across its independent memory extractors."""
        existing = _latest_handle(self._graphs[key], f"cell-input-{memcell_id}")
        if existing is not None and (
            content is None or json.loads(existing.raw_value).get("content") == content
        ):
            return existing
        value = comment_variable(
            {"id": f"cell-input-{memcell_id}", "content": content},
            to_runtime=True,
            id_strategy=_by_id,
            encoding_fn=_encode,
            category="construction_input",
            comment=(
                "The original conversation content supplied to memory extraction."
                if content is not None
                else "The original extraction input is unavailable for this memcell."
            ),
            metadata={"content_observed": content is not None},
        )
        comment_op(
            inputs=[parent],
            outputs=[value],
            op_name=f"observe_cell_{memcell_id}",
            category="boundary_detection",
            comment="Record the conversation content selected for memory extraction.",
            metadata=dict(metadata),
        )
        return value

    # ── request recording ────────────────────────────────────────────────

    async def _record_messages(
        self,
        app_id: str,
        project_id: str,
        session_id: str,
        messages: list[dict[str, Any]],
        *,
        defer_extraction: bool = False,
        request_id: str | None = None,
        buffer_loaded: Mapping[str, Any] | None = None,
    ) -> None:
        """Record the complete add batch in each participating sender's graph.

        A session buffer contains the conversation, including assistant
        replies. Owner partitioning must not drop other senders' inputs.
        """
        senders: dict[str, None] = {}
        snapshots: list[dict[str, Any]] = []
        for idx, message in enumerate(messages):
            sender = str(message.get("sender_id") or "unknown")
            snapshot = self._message_snapshot(
                app_id, project_id, session_id, sender, message, idx
            )
            scoped_id = (app_id, project_id, snapshot["id"])
            self._messages[scoped_id] = snapshot
            known = self._session_messages.setdefault(
                (app_id, project_id, session_id), []
            )
            if snapshot["id"] not in known:
                known.append(snapshot["id"])
                if len(known) > _MAX_TRACKED_MESSAGES_PER_SESSION:
                    stale = known[:-_MAX_TRACKED_MESSAGES_PER_SESSION]
                    del known[:-_MAX_TRACKED_MESSAGES_PER_SESSION]
                    for stale_id in stale:
                        self._messages.pop((app_id, project_id, stale_id), None)
                        self._bound_messages.discard((app_id, project_id, stale_id))
            senders[sender] = None
            snapshots.append(snapshot)

        for sender in senders:
            key = (app_id, project_id, sender, session_id)

            def record(
                snaps: list[dict[str, Any]] = snapshots,
                graph_key: GraphKey = key,
            ) -> None:
                handles = [
                    comment_variable(
                        snap,
                        to_runtime=True,
                        id_strategy=_by_id,
                        encoding_fn=_encode,
                        category="message",
                        comment=(
                            "An input message fed into the EverOS memory "
                            "pipeline via /memory/add."
                        ),
                    )
                    for snap in snaps
                ]
                op_metadata: dict[str, Any] = {
                    "session_id": graph_key[3],
                    "message_count": len(snaps),
                    "request_id": request_id,
                }
                if defer_extraction:
                    op_metadata["defer_extraction"] = True
                buffer = self._ensure_buffer(
                    graph_key,
                    initial_ids=list(buffer_loaded["before_message_ids"])
                    if buffer_loaded
                    else None,
                )
                with comment_mutation(
                    target=buffer,
                    inputs=handles,
                    id_strategy=_by_id,
                    encoding_fn=_encode,
                    category="message_buffer",
                    comment="The session messages held in the buffer at this point.",
                    mutation_name=f"memory_add_{graph_key[3]}",
                    mutation_category="initialization",
                    mutation_comment="Buffer input messages via /memory/add.",
                    mutation_metadata=self._op_metadata(op_metadata),
                ):
                    buffer["message_ids"] = (
                        list(buffer_loaded["message_ids"])
                        if buffer_loaded
                        else list(
                            dict.fromkeys(
                                buffer["message_ids"] + [s["id"] for s in snaps]
                            )
                        )
                    )

            await self._update_graph(
                key,
                session_name=f"memory_add_{session_id}",
                session_category="memory_construction",
                record=record,
                export=False,
            )

    def _ensure_buffer(
        self,
        key: GraphKey,
        *,
        initial_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Restore the latest buffer or register its real initial state."""
        buffer = self._buffers.get(key)
        if buffer is None:
            existing = _latest_handle(self._graphs[key], f"buffer-{key[3]}-{key[2]}")
            if existing is not None:
                buffer = json.loads(existing.raw_value)
            else:
                buffer = {
                    "id": f"buffer-{key[3]}-{key[2]}",
                    "app_id": key[0],
                    "project_id": key[1],
                    "session_id": key[3],
                    "owner_id": key[2],
                    "message_ids": list(initial_ids or []),
                }
                comment_variable(
                    buffer,
                    to_runtime=True,
                    id_strategy=_by_id,
                    encoding_fn=_encode,
                    category="message_buffer",
                    comment="The session messages held in the buffer at this point.",
                )
            self._buffers[key] = buffer
        return buffer

    async def _record_buffer_state(
        self,
        event: Mapping[str, Any],
        *,
        request_id: str | None,
    ) -> None:
        """Reconcile a real loaded buffer or record its persisted tail."""
        app, project, session = (
            str(event[field]) for field in ("app_id", "project_id", "session_id")
        )
        owners = {key[2] for key in self._flush_graph_keys(app, project, session)}
        owners.update(str(s or "unknown") for s in event.get("sender_ids", []))
        if len(owners) > 1:
            owners.discard("unknown")
        for owner in sorted(owners):
            key = (app, project, owner, session)

            def record(key: GraphKey = key) -> None:
                buffer = self._ensure_buffer(
                    key,
                    initial_ids=list(event.get("before_message_ids", [])),
                )
                ids = list(event["message_ids"])
                if buffer["message_ids"] == ids:
                    return
                with comment_mutation(
                    target=buffer,
                    id_strategy=_by_id,
                    encoding_fn=_encode,
                    category="message_buffer",
                    comment="The session messages held in the buffer at this point.",
                    mutation_name=f"{event['phase']}_{session}",
                    mutation_category="update",
                    mutation_comment="Record the real boundary-detection buffer state.",
                    mutation_metadata=self._op_metadata(
                        {
                            "source": "construction_event",
                            "request_id": request_id,
                            "observed_at": event["observed_at"],
                        }
                    ),
                ):
                    buffer["message_ids"] = ids

            await self._update_graph(
                key,
                session_name=f"buffer_state_{session}",
                session_category="memory_construction",
                record=record,
                export=False,
            )

    def _message_snapshot(
        self,
        app_id: str,
        project_id: str,
        session_id: str,
        sender: str,
        message: dict[str, Any],
        idx: int,
    ) -> dict[str, Any]:
        """Build the deterministic snapshot dict for one input message.

        The identity replicates ``gen_message_id(session_id, ts_ms, idx)``
        from ``everos.memory.extract.ingest.id_gen`` — duplicated here
        because the layered architecture forbids infra importing the memory
        layer — so ``/add`` messages share the buffer-message nodes recorded
        by the ``/search`` hook. ``idx`` is the message's position in the
        request payload, matching ingest. Numeric timestamps auto-detect
        seconds vs milliseconds and ISO-8601 strings are parsed, all
        normalising onto the same millisecond identity. A message without a
        parseable timestamp gets a stable fallback identity
        (``m_<session>_invalid_<idx>``) flagged with ``timestamp_invalid``
        and the raw value, instead of being silently dropped and breaking
        the memcell-membership chain.
        """
        content = message.get("content")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        raw_timestamp = message.get("timestamp")
        try:
            moment = from_iso_format(raw_timestamp)
        except (TypeError, ValueError):
            logger.warning(
                "memtrace_message_without_timestamp",
                session_id=session_id,
                idx=idx,
            )
            return {
                "id": f"m_{session_id}_invalid_{idx:03d}",
                "app_id": app_id,
                "project_id": project_id,
                "content": content,
                "name": sender,
                "role": message.get("role", "user"),
                "timestamp": None,
                "timestamp_invalid": True,
                "raw_timestamp": (
                    raw_timestamp
                    if raw_timestamp is None
                    or isinstance(raw_timestamp, str | int | float | bool)
                    else str(raw_timestamp)
                ),
            }
        timestamp_ms = int(moment.timestamp() * 1000)
        return {
            "id": f"m_{session_id}_{timestamp_ms}_{idx:03d}",
            "app_id": app_id,
            "project_id": project_id,
            "content": content,
            "name": sender,
            "role": message.get("role", "user"),
            "timestamp": moment.isoformat(),
        }

    def _flush_graph_keys(
        self, app_id: str, project_id: str, session_id: str
    ) -> list[GraphKey]:
        """Graph keys a flush of ``session_id`` should be recorded into.

        Every tracked sender buffer of the session gets the flush op; a
        session with no known buffer (empty flush) falls back to the
        ``unknown``-owner session graph so the event is still visible.
        """
        keys = [
            key
            for key in self._buffers
            if key[0] == app_id and key[1] == project_id and key[3] == session_id
        ]
        return keys or [(app_id, project_id, "unknown", session_id)]

    async def _record_flush(
        self,
        app_id: str,
        project_id: str,
        session_id: str,
        *,
        result: Any,
        request_id: str | None,
        buffer_state: Mapping[str, Any] | None = None,
    ) -> None:
        """Record the request-level ``memory_flush`` op for one flush."""
        op_name = f"memory_flush_{session_id}"
        op_metadata = {
            "session_id": session_id,
            "is_final": True,
            "status": getattr(result, "status", None),
            "message_count": getattr(result, "message_count", None),
            "request_id": request_id,
        }
        for key in self._flush_graph_keys(app_id, project_id, session_id):
            buffer = self._buffers.get(key)

            def record(buf: dict[str, Any] | None = buffer) -> None:
                if buf is not None and buffer_state is not None:
                    # Even an empty flush gets its own transition/version. Never
                    # infer a drain from response.message_count (fresh rows only).
                    with comment_mutation(
                        target=buf,
                        id_strategy=_by_id,
                        encoding_fn=_encode,
                        category="message_buffer",
                        comment=(
                            "The session messages held in the buffer at this point."
                        ),
                        mutation_name=op_name,
                        mutation_category="extraction",
                        mutation_comment=(
                            "Flush the buffer and persist its actual tail."
                        ),
                        mutation_metadata=self._op_metadata(
                            {
                                **op_metadata,
                                "source": "construction_event",
                                "observed_at": buffer_state["observed_at"],
                            }
                        ),
                    ):
                        buf["message_ids"] = list(buffer_state["message_ids"])
                    return
                receipt = comment_variable(
                    {
                        "id": (
                            f"flush-result-{session_id}-"
                            f"{request_id or uuid.uuid4().hex}"
                        ),
                        **op_metadata,
                        "buffer_state_observed": False,
                    },
                    to_runtime=True,
                    id_strategy=_by_id,
                    encoding_fn=_encode,
                    category="request_result",
                    class_name="flush_result",
                    comment="Completion of one flush without a captured buffer state.",
                )
                comment_op(
                    op_name=op_name,
                    category="extraction",
                    comment=(
                        "Flush the session buffer: boundary detection and "
                        "extraction ran over the buffered messages."
                    ),
                    inputs=(
                        [(buf, dict(_BUFFER_INPUT_OPTIONS))] if buf is not None else []
                    ),
                    outputs=[receipt],
                    metadata=self._op_metadata(op_metadata),
                )

            await self._update_graph(
                key,
                session_name=op_name,
                session_category="memory_construction",
                record=record,
            )

    def _search_scope(self, req: SearchRequest) -> tuple[str, str, str, str, GraphKey]:
        """Resolve the ``(owner, app, project, bucket, key)`` of one search."""
        owner = req.owner_id
        app_id = req.app_id
        project_id = req.project_id
        bucket = "global"
        if req.filters is not None:
            extra = req.filters.model_extra or {}
            session_id = extra.get("session_id")
            if isinstance(session_id, str) and session_id:
                bucket = session_id
        return owner, app_id, project_id, bucket, (app_id, project_id, owner, bucket)

    def _next_query_identity(
        self, key: GraphKey, owner: str, request_id: str | None
    ) -> tuple[str, int]:
        """Deterministic query identity: ``query-{owner}-{request_id}``.

        The request id keeps the identity reproducible across replays and
        joinable with the request logs. A collision (the identity already
        has a version in the live graph, e.g. a retried request) gets a
        sequence suffix instead of a random component. Without a request id
        (direct / CLI callers) the per-key sequence itself keeps the
        identity unique within the process.
        """
        seq = self._query_seq.get(key, 0) + 1
        self._query_seq[key] = seq
        if request_id:
            query_id = f"query-{owner}-{request_id}"
        else:
            query_id = f"query-{owner}-{seq:04d}"
        graph = self._graphs[key]
        base = query_id
        if graph.get_versions(f"query:{query_id}"):
            query_id = f"{base}-{seq:04d}"
            while graph.get_versions(f"query:{query_id}"):
                seq += 1
                query_id = f"{base}-{seq:04d}"
        self._query_seq[key] = seq
        return query_id, seq

    def _record_query_node(
        self,
        req: SearchRequest,
        *,
        owner: str,
        query_id: str,
        request_id: str,
    ) -> Any:
        """Record the ``query`` variable for one search and return its handle."""
        return comment_variable(
            {
                "id": query_id,
                "query": req.query,
                "user_id": owner,
                "app_id": req.app_id,
                "project_id": req.project_id,
                "method": str(req.method),
                "top_k": req.top_k,
                "request_id": request_id,
            },
            to_runtime=True,
            id_strategy=_by_id,
            encoding_fn=_encode,
            category="message & query",
            class_name="query",
            comment="A query issued to the EverOS /memory/search endpoint.",
            metadata={"request_id": request_id},
        )

    def _record_stage_pools(
        self,
        stages: Sequence[SearchStageRecord] | None,
        *,
        query_id: str,
        query_handle: Any,
        seq: int,
        request_id: str,
    ) -> tuple[list[tuple[SearchStageRecord, Any]], dict[str, Any]]:
        """Record every stage pool and its chain edges.

        Explicit input ids take precedence, including cross-kind dependencies.
        Older collectors fall back to the previous pools of their kind.
        Returns the
        ``(stage, pool_handle)`` pairs (candidates are recorded later) and
        the last pool per kind (the hits' source).

        Every stage is an explicit ``retrieval`` operation with real
        inputs and outputs (not a bare link), mirroring the native
        MemTraceBench graphs where each recall / fusion / rerank step is
        its own operation; the resulting edge topology is unchanged.
        """
        last_pool: dict[str, Any] = {}
        stage_pools: list[tuple[SearchStageRecord, Any]] = []
        stage_handles: dict[str, Any] = {}
        pending: dict[str, list[Any]] = {}
        for stage in stages or []:
            # Profile retrieval remains active; its graph branch is disabled.
            if stage.kind == "profile":
                continue
            pool_handle = self._record_stage_pool(stage, query_id=query_id)
            stage_pools.append((stage, pool_handle))
            if stage.input_stage_ids is not None:
                sources = [stage_handles[sid] for sid in stage.input_stage_ids]
                sources = sources or [query_handle]
                pending[stage.kind] = [pool_handle]
            elif stage.phase in _RECALL_PHASES:
                sources = [query_handle]
                pending.setdefault(stage.kind, []).append(pool_handle)
            else:
                sources = pending.get(stage.kind) or [query_handle]
                pending[stage.kind] = [pool_handle]
            comment_op(
                inputs=sources,
                outputs=[pool_handle],
                op_name=f"{stage.phase}_{seq:04d}"
                if stage.phase.startswith("_")
                else f"{stage.kind}_{stage.phase}_{seq:04d}",
                category="retrieval",
                comment=stage_comments(
                    stage.phase,
                    stage.kind,
                    status=str(stage.metadata.get("status", "success")),
                )[0],
                metadata=self._op_metadata(
                    {
                        "phase": stage.phase,
                        "kind": stage.kind,
                        "request_id": request_id,
                        "query_id": query_id,
                        **{k: v for k, v in stage.metadata.items() if k != "items"},
                    }
                ),
            )
            last_pool[stage.kind] = pool_handle
            if stage.stage_id:
                stage_handles[stage.stage_id] = pool_handle
        return stage_pools, last_pool

    async def _record_search(
        self,
        req: SearchRequest,
        resp: SearchResponse,
        *,
        stages: Sequence[SearchStageRecord] | None = None,
        request_id: str | None = None,
    ) -> None:
        """Record one search request into the ``(owner, session)`` graph.

        With ``stages`` (the request's candidate-pool trace collected by
        :func:`everos.memory.search.trace.collect_search_trace`) the
        retrieval chain runs ``query -> recall pools -> fusion pools ->
        last pool -> hits``, so a graph consumer can tell whether a piece
        of evidence was recalled, survived fusion, and was returned.
        Without stages the legacy ``query -> hits`` chain is recorded
        unchanged.
        """
        owner, app_id, project_id, bucket, key = self._search_scope(req)
        original_request_id = request_id
        request_id = request_id or "unknown"

        def record() -> None:
            query_id, seq = self._next_query_identity(key, owner, original_request_id)
            query_handle = self._record_query_node(
                req, owner=owner, query_id=query_id, request_id=request_id
            )
            # (kind, identity, score, text) per hit.
            hits: list[tuple[str, str, float, str]] = []
            for episode in resp.data.episodes:
                hits.append(
                    (
                        "episode",
                        _hit_identity(owner, episode.id),
                        episode.score,
                        episode.episode,
                    )
                )
                for fact in episode.atomic_facts:
                    hits.append(
                        (
                            "episode",
                            _hit_identity(owner, fact.id),
                            fact.score,
                            fact.content,
                        )
                    )
            for case in resp.data.agent_cases:
                hits.append(
                    (
                        "agent_case",
                        _hit_identity(owner, case.id),
                        case.score,
                        case.approach,
                    )
                )
            for skill in resp.data.agent_skills:
                hits.append(
                    (
                        "agent_skill",
                        _hit_identity(owner, skill.id),
                        skill.score,
                        skill.content,
                    )
                )

            hit_handles = []
            with comment_op_scope(
                op_name=f"memory_search_{seq:04d}",
                category="retrieval",
                comment="Retrieve stored memories relevant to the query.",
                metadata=self._op_metadata(
                    {
                        "session_id": bucket,
                        "hit_count": len(hits),
                        "request_id": request_id,
                        "query_id": query_id,
                    }
                ),
            ):
                stage_pools, last_pool = self._record_stage_pools(
                    stages,
                    query_id=query_id,
                    query_handle=query_handle,
                    seq=seq,
                    request_id=request_id,
                )
                terminal_stages = {stage.kind: stage for stage, _pool in stage_pools}
                terminal_ids = {
                    kind: {_hit_identity(owner, c.id) for c in stage.candidates}
                    for kind, stage in terminal_stages.items()
                }
                linked_hits: set[tuple[str, str]] = set()

                for kind, identity, score, text in hits:
                    source_memory = self._memory_source(key, identity, text=text)
                    hit_handle = comment_variable(
                        {
                            "id": self._result_identity(query_id, kind, identity),
                            "memory_identity": identity,
                            "query_id": query_id,
                            "owner_id": owner,
                            "content": text,
                            "source": "search_response",
                            "source_revision": source_memory.full_node_id
                            if source_memory
                            else None,
                        },
                        to_runtime=True,
                        id_strategy=_by_id,
                        encoding_fn=_encode,
                        category="retrieval_result",
                        class_name="retrieval_hit",
                        comment="The exact text returned by this search request.",
                        metadata={
                            "score": score,
                            "request_id": request_id,
                            "revision_resolved": source_memory is not None,
                        },
                    )
                    if source_memory is not None:
                        comment_link(
                            source=source_memory,
                            target=hit_handle,
                            comment="Return the text of this stored memory revision.",
                            edge_metadata={"source_revision": True},
                        )
                    # Kinds without a recorded stage (or stage-less
                    # recording) keep the legacy query -> hit edge.
                    if (kind, identity) not in linked_hits:
                        source = last_pool.get(kind)
                        if source is None:
                            comment_link(
                                source=query_handle,
                                target=hit_handle,
                                comment="Return this memory as a match for the query.",
                                edge_metadata={"score": score},
                            )
                        elif identity not in terminal_ids[kind]:
                            # Response shaping can wrap atomic facts in an
                            # episode; a capped pool can also omit a hit.
                            # Keep that output connected without inventing
                            # another candidate in the actual pool snapshot.
                            comment_link(
                                source=source,
                                target=hit_handle,
                                comment="Include this returned memory in the results.",
                                edge_metadata={
                                    "score": score,
                                    "phase": terminal_stages[kind].phase,
                                    "kind": kind,
                                    "membership": True,
                                    "response_only": True,
                                },
                            )
                        # Listed hits receive their single membership edge
                        # below, after their rich snapshot has been recorded.
                        linked_hits.add((kind, identity))
                    hit_handles.append(hit_handle)

                # Candidate stubs come last so a candidate that was also
                # returned reuses the hit's richer snapshot (identity_only)
                # instead of forking a stub-only version of the node.
                for stage, pool_handle in stage_pools:
                    if stage.kind in {"profile", "unprocessed"}:
                        continue
                    if pool_handle.full_node_id != last_pool[stage.kind].full_node_id:
                        continue
                    self._record_stage_candidates(
                        stage,
                        owner=owner,
                        app_id=app_id,
                        project_id=project_id,
                        pool_handle=pool_handle,
                        query_id=query_id,
                        graph_key=key,
                    )

                # Older callers may provide a response without collected stages.
                if "unprocessed" not in last_pool and resp.data.unprocessed_messages:
                    messages = resp.data.unprocessed_messages
                    phase = "_load_unprocessed"
                    operation_comment, result_comment = stage_comments(
                        phase, "unprocessed"
                    )
                    result = comment_variable(
                        {
                            "id": f"candidates_unprocessed_{phase}-{query_id}",
                            "query_id": query_id,
                            "phase": phase,
                            "kind": "unprocessed",
                            "count": len(messages),
                            "total_count": len(messages),
                            "ids": [m.id for m in messages],
                            "scores": [0.0] * len(messages),
                            "sources": [""] * len(messages),
                            "items": [m.model_dump(mode="json") for m in messages],
                        },
                        to_runtime=True,
                        id_strategy=_by_id,
                        encoding_fn=_encode,
                        category="retrieval_result",
                        class_name="retrieval_results",
                        comment=result_comment,
                        metadata={"total_count": len(messages), "truncated": False},
                    )
                    comment_op(
                        inputs=[query_handle],
                        outputs=[result],
                        op_name=f"{phase}_{seq:04d}",
                        category="retrieval",
                        comment=operation_comment,
                        metadata=self._op_metadata(
                            {"query_id": query_id, "request_id": request_id}
                        ),
                    )
                    last_pool["unprocessed"] = result

            context = {
                "id": f"context-{query_id}",
                "query": req.query,
                "passages": [
                    {"id": identity, "score": score, "text": text}
                    for _kind, identity, score, text in hits
                ]
                + [
                    {
                        "id": f"profile-{owner}",
                        "score": profile.score or 0.0,
                        "text": json.dumps(profile.profile_data, ensure_ascii=False),
                    }
                    for profile in resp.data.profiles
                ],
            }
            context["unprocessed_messages"] = [
                _unprocessed_message_snapshot(message)
                for message in resp.data.unprocessed_messages
            ]
            context_handle = comment_variable(
                context,
                to_runtime=True,
                id_strategy=_by_id,
                encoding_fn=_encode,
                category="memory_context",
                class_name="context",
                comment="The memory context assembled from the search hits.",
            )
            with comment_op_scope(
                op_name=f"context_assembly_{seq:04d}",
                category="retrieval",
                comment="Assemble the retrieved memory entries into a context.",
                metadata=self._op_metadata(
                    {"request_id": request_id, "query_id": query_id}
                ),
            ):
                if "unprocessed" in last_pool:
                    pending_status = next(
                        (
                            stage.metadata.get("status", "success")
                            for stage, _pool in reversed(stage_pools)
                            if stage.kind == "unprocessed"
                        ),
                        "success",
                    )
                    if resp.data.unprocessed_messages:
                        pending_comment = (
                            "Include the pending session messages in the context."
                        )
                    elif pending_status == "skipped":
                        pending_comment = (
                            "Add no pending messages because no session was specified."
                        )
                    elif pending_status == "error":
                        pending_comment = (
                            "Add no pending messages because their retrieval failed."
                        )
                    else:
                        pending_comment = (
                            "Add no pending messages because the session buffer "
                            "is empty."
                        )
                    comment_link(
                        source=last_pool["unprocessed"],
                        target=context_handle,
                        comment=pending_comment,
                    )
                for hit_handle in hit_handles:
                    comment_link(
                        source=hit_handle,
                        target=context_handle,
                        comment="Include this returned memory in the search context.",
                    )

        await self._update_graph(
            key,
            session_name=f"memory_search_{bucket}_{request_id}",
            session_category="memory_retrieval",
            record=record,
        )

    async def _record_search_error(
        self,
        req: SearchRequest,
        *,
        stages: Sequence[SearchStageRecord] | None,
        error: BaseException,
        request_id: str | None,
    ) -> None:
        """Record a failed search: query node, partial pools, error op.

        The stage collector may hold a partially filled pool list at the
        failure point; whatever was collected is recorded so the graph
        shows how far the retrieval pipeline got. No hits / context are
        recorded — the request never produced a response.
        """
        owner, app_id, project_id, bucket, key = self._search_scope(req)
        original_request_id = request_id
        request_id = request_id or "unknown"

        def record() -> None:
            query_id, seq = self._next_query_identity(key, owner, original_request_id)
            query_handle = self._record_query_node(
                req, owner=owner, query_id=query_id, request_id=request_id
            )
            with comment_op_scope(
                op_name=f"memory_search_{seq:04d}",
                category="retrieval",
                comment=(f"EverOS {req.method} search failed."),
                metadata=self._op_metadata(
                    {
                        "session_id": bucket,
                        "status": "error",
                        "error_type": type(error).__name__,
                        "error_message": str(error)[:_MAX_ERROR_MESSAGE_CHARS],
                        "request_id": request_id,
                        "query_id": query_id,
                    }
                ),
            ):
                stage_pools, last_pool = self._record_stage_pools(
                    stages,
                    query_id=query_id,
                    query_handle=query_handle,
                    seq=seq,
                    request_id=request_id,
                )
                for stage, pool_handle in stage_pools:
                    if stage.kind in {"profile", "unprocessed"}:
                        continue
                    if pool_handle.full_node_id != last_pool[stage.kind].full_node_id:
                        continue
                    self._record_stage_candidates(
                        stage,
                        owner=owner,
                        app_id=app_id,
                        project_id=project_id,
                        pool_handle=pool_handle,
                        query_id=query_id,
                        graph_key=key,
                    )

        await self._update_graph(
            key,
            session_name=f"memory_search_{bucket}_{request_id}",
            session_category="memory_retrieval",
            record=record,
        )

    def _record_stage_pool(self, stage: SearchStageRecord, *, query_id: str) -> Any:
        """Record one search-stage candidate pool as a ``memory_context``
        variable and return its runtime handle.

        The pool identity is namespaced by the query so every request
        records its own stage chain; the snapshot carries the capped
        ``ids`` / ``scores`` / ``sources`` lists.
        """
        snapshot = {
            "id": f"candidates_{stage.kind}_{stage.phase}-{query_id}",
            "query_id": query_id,
            "phase": stage.phase,
            "kind": stage.kind,
            "count": len(stage.candidates),
            "total_count": stage.total_count,
            "ids": [candidate.id for candidate in stage.candidates],
            "scores": [candidate.score for candidate in stage.candidates],
            "sources": [candidate.source for candidate in stage.candidates],
        }
        if "items" in stage.metadata:
            snapshot["items"] = stage.metadata["items"]
        stage_metadata = {
            key: value
            for key, value in stage.metadata.items()
            if isinstance(value, str | int | float | bool)
        }
        # Truncation markers: the pool snapshot is capped at
        # ``STAGE_CANDIDATE_CAP``, so downstream attribution must know when
        # it is looking at a partial recall set.
        stage_metadata["total_count"] = stage.total_count
        stage_metadata["truncated"] = stage.total_count > len(stage.candidates)
        return comment_variable(
            snapshot,
            to_runtime=True,
            id_strategy=_by_id,
            encoding_fn=_encode,
            category="retrieval_result"
            if stage.kind == "unprocessed"
            else "memory_context",
            class_name=_POOL_CLASS_NAMES.get(stage.phase, "retrieval_results"),
            comment=stage_comments(
                stage.phase,
                stage.kind,
                status=str(stage.metadata.get("status", "success")),
            )[1],
            metadata=stage_metadata or None,
        )

    @staticmethod
    def _result_identity(query_id: str, kind: str, identity: str) -> str:
        digest = hashlib.sha256(_encode([kind, identity]).encode()).hexdigest()[:16]
        return f"hit-{query_id}-{digest}"

    def _memory_source(
        self, key: GraphKey, identity: str, *, text: str | None = None
    ) -> Any:
        """Copy a canonical revision without allocating a target-local version.

        Response text must match a recorded content revision. Unknown or stale
        index values remain unresolved rather than borrowing the current disk value.
        """
        target = self._graphs[key]
        matches = []
        for source_key, graph in self._graphs.items():
            if source_key[:2] != key[:2]:
                continue
            for node in graph.get_versions(identity):
                value = json.loads(node.value)
                if (
                    not isinstance(value, dict)
                    or "kind" not in value
                    or value.get("deleted")
                ):
                    continue
                if text is not None and value.get("content") != text:
                    continue
                matches.append((node, graph))
        if not matches:
            return None
        node, source_graph = max(
            matches, key=lambda pair: (pair[0].created_at, pair[0].version)
        )
        existing = next(
            (
                n
                for n in target.get_versions(identity)
                if n.full_node_id == node.full_node_id
            ),
            None,
        )
        if existing is not None:
            if existing.value != node.value:
                raise ValueError(
                    "Conflicting legacy memory revisions; "
                    "collect into a fresh trace directory."
                )
            return existing
        if node.session_id not in {
            session.session_id for session in target.get_all_sessions()
        }:
            target.add_session(source_graph.get_session(node.session_id))
        target.add_variable(node)
        return target.get_variable(node.full_node_id)

    def _record_stage_candidates(
        self,
        stage: SearchStageRecord,
        *,
        owner: str,
        app_id: str,
        project_id: str,
        pool_handle: Any,
        query_id: str,
        graph_key: GraphKey,
    ) -> None:
        """Record request-local candidate observations, never memory mutations."""
        seen: set[str] = set()
        for candidate in stage.candidates:
            identity = _hit_identity(owner, candidate.id)
            if identity in seen:
                continue
            seen.add(identity)
            candidate_handle = comment_variable(
                {
                    "id": self._result_identity(query_id, stage.kind, identity),
                    "memory_identity": identity,
                    "query_id": query_id,
                    "source": "candidate_pool",
                    "identity_only": True,
                },
                to_runtime=True,
                id_strategy=_by_id,
                encoding_fn=_encode,
                identity_only=True,
                category="retrieval_result",
                class_name="retrieval_hit",
                comment="A candidate reference with unobserved contents.",
                metadata={"score": candidate.score, "phase": stage.phase},
            )
            comment_link(
                source=pool_handle,
                target=candidate_handle,
                comment="Select this candidate from the retrieved memory pool.",
                edge_metadata={
                    "score": candidate.score,
                    "phase": stage.phase,
                    "kind": stage.kind,
                    "membership": True,
                },
            )

    # ── file tracking ────────────────────────────────────────────────────

    async def _sync_files(self, *, trigger: str = "manual") -> None:
        """Diff the memory-root markdown tree and record every change.

        ``_file_state`` and the entry/artifact mirrors are process-global,
        so scan + apply run as one transaction under ``_sync_lock``. The
        apply phase is split in two: snapshots are applied first (collecting
        every entry identity seen this round), then deletions are decided
        globally — an entry that moved files inside one diff round is a
        move, not a delete + recreate. The first diff after process start
        additionally reconciles downtime deletions against the exported
        graphs. Every graph dirtied by the round is exported exactly once
        at the end. Ops recorded by the diff carry ``source: "file_diff"``
        and the ``trigger`` (the hook that drove this diff) in metadata.
        """
        async with self._sync_lock:
            self._trigger = trigger
            committed_state = dict(self._file_state)
            try:
                changed, deleted = await asyncio.to_thread(self._scan)
                present: set[ScopedIdentity] = set()
                present_by_file: dict[str, set[ScopedIdentity]] = {}
                for path in deleted:
                    await self._apply_deleted_artifacts(path)
                for snapshot in changed:
                    file_present = await self._apply_snapshot(snapshot)
                    present |= file_present
                    present_by_file[snapshot.path] = file_present
                # Decide deletions across all touched files so moves survive.
                touched = set(deleted) | present_by_file.keys()
                for scoped_identity, path in list(self._entry_file.items()):
                    if path in touched and scoped_identity not in present:
                        await self._apply_deletion(scoped_identity)
                if not self._first_scan_done:
                    await self._reconcile_after_restart(present, present_by_file)
                await self._flush_dirty_graphs()
                self._first_scan_done = True
            except BaseException:
                # Apply/export failures must retry even when files are unchanged.
                self._file_state = committed_state
                raise

    def _scan(self) -> tuple[list[MemoryFileSnapshot], list[str]]:
        """Scan the memory root (runs in a worker thread).

        Returns:
            `tuple[list[MemoryFileSnapshot], list[str]]`:
                Parsed snapshots of new/changed files and the relative paths
                of deleted files.
        """
        changed: list[MemoryFileSnapshot] = []
        seen: set[str] = set()
        for path in self._root.rglob("*.md"):
            parts = set(path.relative_to(self._root).parts)
            if parts & _SKIP_DIR_NAMES:
                continue
            if path.is_relative_to(self._out_dir):
                continue
            rel = str(path.relative_to(self._root))
            seen.add(rel)
            stat = path.stat()
            state = (stat.st_mtime_ns, stat.st_size)
            if self._file_state.get(rel) == state:
                continue
            try:
                snapshot = parse_memory_file(path, relative_to=self._root)
            except Exception as exc:
                logger.warning("memtrace_parse_failed", path=rel, error=str(exc))
                # Do not commit the new state: retry the same file at the
                # next request boundary even when its mtime/size is unchanged.
                continue
            self._file_state[rel] = state
            changed.append(snapshot)
        deleted = [rel for rel in self._file_state if rel not in seen]
        for rel in deleted:
            del self._file_state[rel]
        return changed, deleted

    async def _apply_snapshot(
        self, snapshot: MemoryFileSnapshot
    ) -> set[ScopedIdentity]:
        """Route one parsed file's entries/artifact into their graphs.

        Returns the scoped identities of the entries present in the file
        (empty for whole-file artifacts); ``_sync_files`` uses the union
        per round to decide deletions globally.
        """
        if snapshot.kind in _ENTRY_FILE_KINDS:
            present: set[ScopedIdentity] = set()
            for entry in snapshot.entries:
                identity = self._entry_identity(entry)
                scoped_identity = (entry.app_id, entry.project_id, identity)
                present.add(scoped_identity)
                await self._apply_entry(entry, identity, scoped_identity)
            return present
        await self._apply_artifact(snapshot)
        return set()

    def _entry_identity(self, entry: MemoryFileEntry) -> str:
        """Stable cross-graph identity for one markdown entry."""
        owner = entry.owner_id or "unknown"
        return f"{owner}_{entry.entry_id}"

    async def _apply_entry(
        self,
        entry: MemoryFileEntry,
        identity: str,
        scoped_identity: ScopedIdentity,
    ) -> None:
        """Record a new entry or a content change of an existing entry."""
        owner = entry.owner_id or "unknown"
        bucket = entry.session_id or "global"
        key = (entry.app_id, entry.project_id, owner, bucket)
        snapshot = {
            "id": identity,
            "entry_id": entry.entry_id,
            "app_id": entry.app_id,
            "project_id": entry.project_id,
            "owner_id": owner,
            "session_id": entry.session_id,
            "timestamp": entry.timestamp,
            "parent_type": entry.parent_type,
            "parent_id": entry.parent_id,
            "sender_ids": entry.sender_ids,
            "subject": entry.subject,
            "summary": entry.summary,
            "content": entry.content,
            "path": entry.path,
            "kind": entry.kind,
        }
        previous = self._entries.get(scoped_identity)
        if previous is None:
            previous = await self._restored_snapshot(key, identity)
            if previous is not None:
                previous["path"] = entry.path
                self._entries[scoped_identity] = previous
                self._entry_file[scoped_identity] = entry.path
        if previous is not None and _encode_without_path(
            previous
        ) == _encode_without_path(snapshot):
            if previous.get("path") != entry.path:
                # A pure move (log rotation, OME rewrite) is not an update:
                # keep the node version, follow the entry to its new file.
                previous["path"] = entry.path
                self._entry_file[scoped_identity] = entry.path
            return

        if previous is None:
            message_ids, sender_ids, membership_status = await self._memcell_boundary(
                entry
            )
            cell_content = None
            if entry.kind == "foresight" and entry.parent_type == "memcell":
                graph = self._graphs[key]
                cell_input = _latest_handle(graph, f"cell-input-{entry.parent_id}")
                if (
                    cell_input is None
                    or json.loads(cell_input.raw_value).get("content") is None
                ):
                    cell_content = await self._memcell_payload(entry)

            def record() -> None:
                parent_handle = self._record_parent(
                    entry,
                    owner,
                    message_ids=message_ids,
                    sender_ids=sender_ids,
                    membership_status=membership_status,
                )
                if (
                    entry.kind == "foresight"
                    and entry.parent_type == "memcell"
                    and parent_handle is not None
                ):
                    parent_handle = self._record_cell_input(
                        key,
                        entry.parent_id,
                        parent_handle,
                        content=cell_content,
                        metadata=self._file_diff_metadata(
                            {"content_source": "memcell_archive"}
                        ),
                    )
                observation = self._entry_observations.get(scoped_identity)
                if observation:
                    extracted = self._graphs[key].get_variable(
                        observation["source_node"]
                    )
                    if (
                        json.loads(extracted.value).get("content")
                        == snapshot["content"]
                    ):
                        parent_handle = extracted
                    else:
                        # A later disk rewrite must not masquerade as the actual
                        # extractor/persist output captured before it.
                        observation = None
                with comment_op_scope(
                    op_name=f"{entry.kind}_extraction_{entry.entry_id}",
                    category="extraction",
                    comment=(
                        f"Record a new {entry.kind.replace('_', ' ')} memory "
                        "derived from this input."
                    ),
                    metadata=(
                        self._op_metadata(observation)
                        if observation
                        else self._file_diff_metadata({"path": entry.path})
                    ),
                ):
                    entry_handle = comment_variable(  #
                        snapshot,
                        to_runtime=True,
                        id_strategy=_by_id,
                        encoding_fn=_encode,
                        category="memory_entry",
                        comment=_ENTRY_KIND_COMMENT.get(
                            entry.kind, "A memory entry parsed from markdown."
                        ),
                    )
                    if parent_handle is not None:
                        comment_link(source=parent_handle, target=entry_handle)

            await self._update_graph(
                key,
                session_name=f"{entry.kind}_extraction_{entry.entry_id}",
                session_category="memory_construction",
                record=record,
                export=False,
            )
            self._entries[scoped_identity] = snapshot
            self._entry_file[scoped_identity] = entry.path
            self._entry_observations.pop(scoped_identity, None)
        else:

            def record_update() -> None:
                with comment_mutation(
                    target=previous,
                    id_strategy=_by_id,
                    encoding_fn=_encode,
                    category="memory_entry",
                    comment=_ENTRY_KIND_COMMENT.get(
                        entry.kind, "A memory entry parsed from markdown."
                    ),
                    mutation_name=f"update_{entry.entry_id}",
                    mutation_category="update",
                    mutation_comment=(
                        f"The {entry.kind} entry was rewritten in {entry.path}."
                    ),
                    mutation_metadata=self._file_diff_metadata({"path": entry.path}),
                ):
                    previous.clear()
                    previous.update(snapshot)

            before = dict(previous)
            try:
                await self._update_graph(
                    key,
                    session_name=f"update_{entry.entry_id}",
                    session_category="memory_construction",
                    record=record_update,
                    export=False,
                )
            except BaseException:
                previous.clear()
                previous.update(before)
                raise
            self._entry_file[scoped_identity] = entry.path

    async def _memcell_boundary(
        self, entry: MemoryFileEntry
    ) -> tuple[list[str] | None, list[str] | None, str]:
        """Fetch the immutable boundary message and sender ids of a memcell.

        Boundary detection persists every memcell's exact message-id list in
        SQLite (``Memcell.message_ids_json``) and the markdown entry's
        ``parent_id`` is the memcell row's primary key, so the real
        membership is recoverable even across restarts.

        Returns:
            `tuple[list[str] | None, list[str] | None, str]`:
                The exact message/sender lists (``None`` when unavailable) and the
                membership status: ``exact`` (real boundary data),
                ``row_missing`` (no matching memcell row), ``lookup_failed``
                (the SQLite read itself failed), or ``invalid_payload``
                (the row's ``message_ids_json`` is unparsable).
        """
        if entry.parent_type != "memcell" or not entry.parent_id:
            return None, None, "row_missing"
        try:
            if self._memcell_repo is None:
                raise RuntimeError("Memcell repository is not configured")
            rows = await self._memcell_repo.find_by_ids([entry.parent_id])
        except Exception as exc:
            logger.warning(
                "memtrace_memcell_lookup_failed",
                memcell_id=entry.parent_id,
                error=str(exc),
            )
            return None, None, "lookup_failed"
        row = rows[0] if rows else None
        if (
            row is None
            or row.app_id != entry.app_id
            or row.project_id != entry.project_id
        ):
            return None, None, "row_missing"
        try:
            message_ids = json.loads(row.message_ids_json)
        except (TypeError, ValueError):
            return None, None, "invalid_payload"
        if not isinstance(message_ids, list):
            return None, None, "invalid_payload"
        try:
            sender_ids = json.loads(getattr(row, "sender_ids_json", "null"))
        except (TypeError, ValueError):
            sender_ids = None
        if not isinstance(sender_ids, list):
            sender_ids = None
        return (
            [str(message_id) for message_id in message_ids],
            [str(sender_id) for sender_id in sender_ids]
            if sender_ids is not None
            else None,
            "exact",
        )

    async def _memcell_payload(self, entry: MemoryFileEntry) -> Any:
        """Recover the archived raw input without borrowing another scope's cell."""
        try:
            if self._memcell_repo is None:
                raise RuntimeError("Memcell repository is not configured")
            rows = await self._memcell_repo.find_by_ids([entry.parent_id])
            row = rows[0] if rows else None
            if (
                row is None
                or row.app_id != entry.app_id
                or row.project_id != entry.project_id
                or row.session_id != entry.session_id
            ):
                return None
            payload = json.loads(row.payload_json)
            return payload if isinstance(payload, dict) else None
        except Exception as exc:
            logger.warning(
                "memtrace_memcell_payload_unavailable",
                memcell_id=entry.parent_id,
                error_type=type(exc).__name__,
            )
            return None

    def _session_message_ids_heuristic(self, entry: MemoryFileEntry) -> list[str]:
        """Approximate memcell membership from the session message index.

        Used only when the memcell's SQLite row is unavailable: prefer the
        session's not-yet-bound messages, falling back to every indexed
        session message when all of them are already bound.
        """
        session_key = (entry.app_id, entry.project_id, entry.session_id or "")
        indexed = [
            message_id
            for message_id in self._session_messages.get(session_key, [])
            if (entry.app_id, entry.project_id, message_id) in self._messages
        ]
        unbound = [
            message_id
            for message_id in indexed
            if (entry.app_id, entry.project_id, message_id) not in self._bound_messages
        ]
        return unbound or indexed

    def _record_parent(
        self,
        entry: MemoryFileEntry,
        owner: str,
        *,
        message_ids: list[str] | None = None,
        sender_ids: list[str] | None = None,
        membership_status: str = "row_missing",
        observed_event: Mapping[str, Any] | None = None,
        boundary_result: Any = None,
    ) -> Any:
        """Record the parent chain of a new entry and return its handle.

        A ``memcell`` parent records its boundary once and reuses it for all
        derived entries. Sender ids come from SQLite, not child-specific
        Markdown metadata. Its input is the buffer version consumed by the
        boundary. Message inputs are a fallback when that version is missing.
        A later exact lookup can upgrade an
        approximation; a failed lookup never downgrades an exact record.
        Membership uses the exact id list persisted by boundary detection
        (``membership: exact``), otherwise a session heuristic over the
        process-local message index. The heuristic result is marked by how
        the lookup failed: ``approximate`` for a missing / unparsable row,
        ``lookup_failed`` when the SQLite read itself failed — a harder
        warning than a mere approximation. A non-``exact`` node also
        carries ``membership_detail`` with the raw status. Messages the
        content index no longer covers (recorded before a restart) are
        linked as identity-only stubs so they stitch onto their pre-restart
        nodes. Any other parent kind is re-recorded as a lightweight
        identity-only node.

        With ``boundary_result`` (the caller's event-level
        ``boundary_detection`` scope is active) the memcell hangs off the
        structured detection result, which already depends on the consumed
        buffer. File-only observations use a recovered grouping result instead.
        """
        if not entry.parent_id:
            return None
        if entry.parent_type == "memcell":
            key = (entry.app_id, entry.project_id, owner, entry.session_id or "global")
            graph = self._graphs[key]
            existing = _latest_handle(graph, entry.parent_id)
            previous = json.loads(existing.raw_value) if existing is not None else None
            if (
                existing is not None
                and existing.metadata.get("membership") == "exact"
                and message_ids is None
            ):
                message_ids = previous["message_ids"]
                sender_ids = previous["sender_ids"]
            membership = "exact"
            if message_ids is None:
                membership = (
                    "lookup_failed"
                    if membership_status == "lookup_failed"
                    else "approximate"
                )
                message_ids = (
                    previous["message_ids"]
                    if previous is not None
                    else self._session_message_ids_heuristic(entry)
                )
            membership_metadata: dict[str, Any] = {"membership": membership}
            if membership != "exact":
                membership_metadata["membership_detail"] = membership_status
            snapshots = [
                self._messages.get((entry.app_id, entry.project_id, message_id))
                for message_id in message_ids
            ]
            memcell = {
                "id": entry.parent_id,
                "app_id": entry.app_id,
                "project_id": entry.project_id,
                "session_id": entry.session_id,
                "sender_ids": (
                    sender_ids
                    if sender_ids is not None
                    else previous["sender_ids"]
                    if previous is not None
                    else entry.sender_ids
                ),
                "message_ids": message_ids,
            }
            for message_id in message_ids:
                self._bound_messages.add((entry.app_id, entry.project_id, message_id))
            unchanged = (
                existing is not None
                and previous == memcell
                and existing.metadata.get("membership") == membership
            )
            if unchanged and boundary_result is None:
                return existing
            if boundary_result is not None:
                # The caller's event-level boundary_detection scope is active:
                # group the memcell off the structured detection result.
                if unchanged:
                    memcell_handle = existing
                elif existing is None:
                    memcell_handle = comment_variable(
                        memcell,
                        to_runtime=True,
                        id_strategy=_by_id,
                        encoding_fn=_encode,
                        category="memory_entry",
                        comment="A memcell produced by boundary detection.",
                        metadata=membership_metadata,
                    )
                else:
                    with comment_mutation(
                        target=previous,
                        id_strategy=_by_id,
                        encoding_fn=_encode,
                        category="memory_entry",
                        comment="A memcell produced by boundary detection.",
                        metadata=membership_metadata,
                        reuse_op=True,
                        mutation_comment=(
                            "Update the message membership of this memcell."
                        ),
                    ) as mutation:
                        previous.clear()
                        previous.update(memcell)
                    memcell_handle = mutation.result
                comment_op(
                    inputs=[boundary_result],
                    outputs=[memcell_handle],
                    reuse_op=True,
                    comment=(
                        "Based on the structured boundary decision, "
                        "the memory system groups these session messages into "
                        "a new memcell."
                    ),
                    metadata=(
                        self._op_metadata(observed_event) if observed_event else None
                    ),
                )
                return memcell_handle
            with comment_op_scope(
                op_name=f"boundary_detection_{entry.parent_id}",
                category="boundary_detection",
                comment=(
                    "Boundary detection carved this memcell out of the "
                    "session messages."
                ),
                metadata=(
                    self._op_metadata(observed_event)
                    if observed_event
                    else self._file_diff_metadata(
                        {"session_id": entry.session_id or "", **membership_metadata}
                    )
                ),
            ):
                if existing is None:
                    memcell_handle = comment_variable(
                        memcell,
                        to_runtime=True,
                        id_strategy=_by_id,
                        encoding_fn=_encode,
                        category="memory_entry",
                        comment="A memcell produced by boundary detection.",
                        metadata=membership_metadata,
                    )
                else:
                    with comment_mutation(
                        target=previous,
                        id_strategy=_by_id,
                        encoding_fn=_encode,
                        category="memory_entry",
                        comment="A memcell produced by boundary detection.",
                        metadata=membership_metadata,
                        reuse_op=True,
                        mutation_comment=(
                            "Update the message membership of this memcell."
                        ),
                    ) as mutation:
                        previous.clear()
                        previous.update(memcell)
                    memcell_handle = mutation.result
                buffer_handle = next(
                    (
                        handle
                        for handle in reversed(
                            graph.get_versions(f"buffer-{entry.session_id}-{owner}")
                        )
                        if message_ids
                        and set(message_ids).issubset(
                            json.loads(handle.raw_value)["message_ids"]
                        )
                    ),
                    None,
                )
                recovered_result = comment_variable(
                    {
                        "id": (
                            f"boundary-observed-{entry.parent_id}"
                            f"-v{memcell_handle.version}"
                        ),
                        "session_id": entry.session_id,
                        "decision": "split",
                        "detection_performed": None,
                        "cell_ids": [entry.parent_id],
                        "message_ids": message_ids,
                    },
                    to_runtime=True,
                    id_strategy=_by_id,
                    encoding_fn=_encode,
                    category="boundary_outcome",
                    class_name="boundary_detection_result",
                    comment=(
                        "The message grouping recovered for an existing memory unit."
                    ),
                    metadata={"observation_only": True, **membership_metadata},
                )
                comment_link(
                    source=recovered_result,
                    target=memcell_handle,
                    comment=(
                        "Associate the recovered message grouping "
                        "with this memory unit."
                    ),
                )
                if buffer_handle is not None and message_ids:
                    comment_link(
                        source=buffer_handle,
                        target=recovered_result,
                        comment=(
                            "Identify the buffered messages belonging to "
                            "the recovered group."
                        ),
                        edge_metadata={"boundary_input": True},
                    )
                    return memcell_handle
                linked: set[str] = set()
                for message_id, snapshot in zip(message_ids, snapshots, strict=True):
                    if message_id in linked:
                        continue
                    linked.add(message_id)
                    message_handle = comment_variable(
                        snapshot if snapshot is not None else {"id": message_id},
                        to_runtime=True,
                        id_strategy=_by_id,
                        encoding_fn=_encode,
                        identity_only=snapshot is None,
                        category="message",
                        comment=(
                            "A session message grouped into this memcell by "
                            "boundary detection."
                        ),
                    )
                    comment_link(
                        source=message_handle,
                        target=recovered_result,
                        comment="Identify a message belonging to the recovered group.",
                    )
            return memcell_handle
        parent_identity = f"{owner}_{entry.parent_id}"
        return comment_variable(
            {"id": parent_identity},
            to_runtime=True,
            id_strategy=_by_id,
            encoding_fn=_encode,
            identity_only=True,
            category="memory_entry",
            comment="A lightweight reference to the parent memory entry.",
        )

    async def _apply_artifact(self, snapshot: MemoryFileSnapshot) -> None:
        """Record a whole-file artifact (profile / agent record / skill)."""
        owner = snapshot.owner_id or "unknown"
        if snapshot.kind == "profile":
            identity = f"profile-{owner}"
        elif snapshot.kind == "agent_skill" and snapshot.frontmatter.get("name"):
            # Search results use the LanceDB primary-key convention
            # ``{owner}_{skill_name}``; keep the markdown-side identity equal
            # so retrieval reuses the write-path node.
            identity = f"{owner}_{snapshot.frontmatter['name']}"
        else:
            identity = f"artifact-{owner}-{snapshot.path}"
        scoped_identity = (snapshot.app_id, snapshot.project_id, identity)
        key = (snapshot.app_id, snapshot.project_id, owner, "global")
        body = {
            "id": identity,
            "app_id": snapshot.app_id,
            "project_id": snapshot.project_id,
            "owner_id": owner,
            "path": snapshot.path,
            "kind": snapshot.kind,
            "content": snapshot.body,
        }
        # A file whose semantic identity changed (for example a hand-edited
        # skill name) deletes its old artifact before creating the new one.
        for old_identity, path in list(self._artifact_file.items()):
            if path == snapshot.path and old_identity != scoped_identity:
                await self._apply_deletion(old_identity)

        previous = self._artifacts.get(scoped_identity)
        if previous is None:
            previous = await self._restored_snapshot(key, identity)
            if previous is not None:
                previous["path"] = snapshot.path
                self._artifacts[scoped_identity] = previous
                self._artifact_file[scoped_identity] = snapshot.path
        if previous is not None and _encode(previous) == _encode(body):
            return
        if previous is None:

            def record() -> None:
                value = comment_variable(
                    body,
                    to_runtime=True,
                    id_strategy=_by_id,
                    encoding_fn=_encode,
                    category="memory_entry",
                    comment=f"A {snapshot.kind} memory artifact ({snapshot.path}).",
                )
                comment_op(
                    inputs=[],
                    outputs=[value],
                    op_name=f"observe_{snapshot.kind}_creation",
                    category="extraction",
                    comment="Record a newly observed memory artifact in the store.",
                    metadata=self._file_diff_metadata({"path": snapshot.path}),
                )

            await self._update_graph(
                key,
                session_name=f"artifact_{snapshot.kind}",
                session_category="memory_construction",
                record=record,
                export=False,
            )
            self._artifacts[scoped_identity] = body
            self._artifact_file[scoped_identity] = snapshot.path
        else:

            def record_update() -> None:
                with comment_mutation(
                    target=previous,
                    id_strategy=_by_id,
                    encoding_fn=_encode,
                    category="memory_entry",
                    comment=f"A {snapshot.kind} memory artifact ({snapshot.path}).",
                    mutation_name=f"update_{snapshot.kind}",
                    mutation_category="update",
                    mutation_comment=(f"The {snapshot.kind} artifact was rewritten."),
                    mutation_metadata=self._file_diff_metadata({"path": snapshot.path}),
                ):
                    previous.clear()
                    previous.update(body)

            before = dict(previous)
            try:
                await self._update_graph(
                    key,
                    session_name=f"update_{snapshot.kind}",
                    session_category="memory_construction",
                    record=record_update,
                    export=False,
                )
            except BaseException:
                previous.clear()
                previous.update(before)
                raise

    async def _apply_deleted_artifacts(self, path: str) -> None:
        """Record deletions for the artifact of a removed whole-file.

        Entry deletions are decided globally in ``_sync_files`` instead: a
        removed file's entries may have moved to another file inside the
        same diff round, which is a move, not a deletion.
        """
        for identity, artifact_path in list(self._artifact_file.items()):
            if artifact_path == path:
                await self._apply_deletion(identity)

    async def _apply_deletion(self, scoped_identity: ScopedIdentity) -> None:
        """Record a deletion mutation for one entry/artifact identity."""
        target = self._entries.get(scoped_identity)
        if target is None:
            target = self._artifacts.get(scoped_identity)
        if target is None:
            return
        app_id, project_id, identity = scoped_identity
        owner = str(target.get("owner_id") or "unknown")
        bucket = str(target.get("session_id") or "global")
        key = (app_id, project_id, owner, bucket)

        def record() -> None:
            with comment_mutation(
                target=target,
                id_strategy=_by_id,
                encoding_fn=_encode,
                category="memory_entry",
                comment=_ENTRY_KIND_COMMENT.get(
                    str(target.get("kind") or ""),
                    f"A {target.get('kind')} memory artifact ({target.get('path')}).",
                ),
                mutation_name=f"delete_{identity}",
                mutation_category="deletion",
                mutation_comment=(
                    "The memory entry was deleted from the markdown store."
                ),
                mutation_metadata=self._file_diff_metadata(),
            ):
                target["deleted"] = True

        before = dict(target)
        try:
            await self._update_graph(
                key,
                session_name=f"delete_{identity}",
                session_category="memory_construction",
                record=record,
                export=False,
            )
        except BaseException:
            target.clear()
            target.update(before)
            raise
        self._entries.pop(scoped_identity, None)
        self._entry_file.pop(scoped_identity, None)
        self._artifacts.pop(scoped_identity, None)
        self._artifact_file.pop(scoped_identity, None)

    # ── graph registry / export ──────────────────────────────────────────

    async def _restored_snapshot(
        self, key: GraphKey, identity: str
    ) -> dict[str, Any] | None:
        """Recover the last write-path snapshot after a process restart.

        The identity is the join key; the caller adopts the file location
        from the current scan (entries may have moved files — ``path``
        never participates in change detection). A snapshot whose latest
        version is a deletion stays deleted: a reappearing entry starts a
        fresh extraction record instead of mutating a deleted node.
        """
        graph = await self._get_graph(key)
        versions = graph.get_versions(identity)
        if not versions:
            return None
        try:
            restored = json.loads(versions[-1].raw_value)
        except (TypeError, ValueError):
            return None
        if not isinstance(restored, dict) or restored.get("deleted"):
            return None
        return restored

    async def _reconcile_after_restart(
        self,
        present: set[ScopedIdentity],
        present_by_file: dict[str, set[ScopedIdentity]],
    ) -> None:
        """Record deletions for entries/artifacts removed during downtime.

        Runs once, after the first post-start diff: that diff re-parses
        every existing file (``_file_state`` is process-local), so a tracked
        write-path snapshot whose file is gone — or whose entry no longer
        appears in its file — was deleted while the process was down and
        never got a deletion mutation.
        """
        tracked = await asyncio.to_thread(self._collect_tracked_snapshots)
        done: set[ScopedIdentity] = set()
        for snapshot in tracked:
            identity = str(snapshot.get("id") or "")
            path = str(snapshot.get("path") or "")
            if not identity or not path:
                continue
            scoped_identity = (
                str(snapshot.get("app_id") or "default"),
                str(snapshot.get("project_id") or "default"),
                identity,
            )
            if scoped_identity in done:
                continue
            is_entry = "entry_id" in snapshot
            if is_entry and scoped_identity in present:
                continue  # Still tracked — possibly moved files.
            file_present = present_by_file.get(path)
            if file_present is None:
                missing = True  # The whole file was removed during downtime.
            elif is_entry:
                missing = scoped_identity not in file_present
            else:
                missing = False  # Whole-file artifact: the file exists.
            if not missing:
                continue
            done.add(scoped_identity)
            if is_entry:
                self._entries.setdefault(scoped_identity, snapshot)
                self._entry_file.setdefault(scoped_identity, path)
            else:
                self._artifacts.setdefault(scoped_identity, snapshot)
                self._artifact_file.setdefault(scoped_identity, path)
            await self._apply_deletion(scoped_identity)

    def _collect_tracked_snapshots(self) -> list[dict[str, Any]]:
        """Read the latest live write-path snapshots from every exported graph.

        Runs in a worker thread during the first post-start diff. Only the
        latest version of each ``memory_entry`` node is considered; nodes
        without a ``path`` (candidate stubs, search-response hits, memcells)
        and versions already marked ``deleted`` are skipped.
        """
        collected: list[dict[str, Any]] = []
        if not self._out_dir.exists():
            return collected
        for path in self._out_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            latest: dict[str, dict[str, Any]] = {}
            for node in data.get("data", {}).get("nodes") or []:
                if node.get("category") != "memory_entry":
                    continue
                name = node.get("name")
                if not name:
                    continue
                current = latest.get(name)
                if current is not None and (current.get("version") or 0) >= (
                    node.get("version") or 0
                ):
                    continue
                latest[name] = node
            for node in latest.values():
                try:
                    snapshot = json.loads(node.get("value") or "")
                except (TypeError, ValueError):
                    continue
                if (
                    isinstance(snapshot, dict)
                    and snapshot.get("path")
                    and not snapshot.get("deleted")
                ):
                    collected.append(snapshot)
        return collected

    def _graph_path(self, key: GraphKey) -> Path:
        """On-disk JSON path for one scoped owner/session graph.

        Every component is a ``{safe}-{digest}`` fragment (see
        ``_scoped_name``): readable, short, and collision-proof. Renamed
        after the first instrumented drop — pre-rename files are orphans
        (the raw ids cannot be recovered from the old collision-prone
        names); delete or ignore them, there is no migration.
        """
        app_id, project_id, owner, bucket = key
        scope = ""
        if (app_id, project_id) != ("default", "default"):
            scope = f"{_scoped_name(app_id)}__{_scoped_name(project_id)}__"
        return self._out_dir / (
            f"{scope}{_scoped_name(owner)}__{_scoped_name(bucket)}.json"
        )

    async def _get_graph(self, key: GraphKey) -> ExecNetwork:
        """Return the live graph for a key, restoring it from disk if needed."""
        if key in self._graphs:
            return self._graphs[key]
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            return await self._get_graph_unlocked(key)

    async def _get_graph_unlocked(self, key: GraphKey) -> ExecNetwork:
        """Get-or-create the graph for ``key``.

        The caller must hold the key's lock so concurrent first touches
        cannot each create an instance and silently lose one side's
        updates.
        """
        if key in self._graphs:
            return self._graphs[key]
        path = self._graph_path(key)
        if await asyncio.to_thread(path.exists):
            graph = await asyncio.to_thread(self._load_graph, path)
        else:
            app_id, project_id, owner, bucket = key
            with comment_graph(
                graph_id=(
                    "everos-filetrace-"
                    f"{_scoped_name(app_id)}-{_scoped_name(project_id)}-"
                    f"{_scoped_name(owner)}-{_scoped_name(bucket)}"
                ),
                user_id=owner,
                project_id=project_id,
                storage_path=str(path),
                metadata={
                    "strategy": "file_tracking",
                    "app_id": app_id,
                    "project_id": project_id,
                    "trace_schema": 2,
                    "history_complete": False,
                    "session_bucket": bucket,
                },
            ) as graph:
                pass
        self._graphs[key] = graph
        return graph

    @staticmethod
    def _load_graph(path: Path) -> ExecNetwork:
        """Load one exported graph without blocking the event loop."""
        return ExecNetwork.import_graph(json.loads(path.read_text(encoding="utf-8")))

    async def _update_graph(
        self,
        key: GraphKey,
        *,
        session_name: str,
        session_category: str,
        record: Callable[[], None],
        export: bool = True,
    ) -> None:
        """Apply one recorded update to a graph.

        With ``export=False`` the on-disk write is deferred: the key is
        marked dirty and ``_flush_dirty_graphs`` exports it once, so a diff
        round dirtying the same graph for N entries costs one full
        serialisation instead of N.
        """
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            graph = await self._get_graph_unlocked(key)
            with (
                comment_graph(graph=graph),
                comment_session(
                    session_name=session_name,
                    category=session_category,
                ),
            ):
                try:
                    record()
                finally:
                    # Preserve retryability even if recording or export fails.
                    self._dirty_keys.add(key)
            if export:
                await asyncio.to_thread(self._export, key, graph)
                self._dirty_keys.discard(key)

    async def _flush_dirty_graphs(self) -> None:
        """Export every graph dirtied since the last flush, once per graph."""
        while self._dirty_keys:
            key = next(iter(self._dirty_keys))
            lock = self._locks.setdefault(key, asyncio.Lock())
            async with lock:
                graph = self._graphs.get(key)
                if graph is not None:
                    await asyncio.to_thread(self._export, key, graph)
                self._dirty_keys.discard(key)

    def _export(self, key: GraphKey, graph: ExecNetwork) -> None:
        """Export one graph as MemTraceBench-compatible JSON (atomic write).

        The whole read-annotations / write-tmp / replace sequence runs under
        an inter-process ``portalocker`` lock (``<graph>.json.lock``) so
        concurrent writers (e.g. multiple uvicorn workers pointing at the
        same export dir) serialise instead of interleaving a read-modify-
        write. The lock does not merge graphs — each process still exports
        its own in-memory instance — so multi-writer deployments should
        still prefer a single writer or per-worker export dirs.
        """
        data = graph.export_graph()
        describe_sentinels(data)
        payload = data.get("data", {})
        # Op-only graphs are meaningful too: an empty flush or a failed
        # request records a request-level op whose inputs/outputs collapse
        # onto the NONE sentinel, leaving no variable nodes behind.
        if not (payload.get("nodes") or payload.get("operations")):
            return
        path = self._graph_path(key)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(".json.lock")
        with portalocker.Lock(str(lock_path), mode="w", timeout=30):
            annotations: list[Any] = []
            if path.exists():
                try:
                    previous = json.loads(path.read_text(encoding="utf-8"))
                    annotations = previous.get("data", {}).get("annotations") or []
                except (OSError, json.JSONDecodeError):
                    annotations = []
            data.setdefault("data", {})["annotations"] = annotations
            tmp_path = path.with_suffix(".json.tmp")
            tmp_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            tmp_path.replace(path)
