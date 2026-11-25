# Memory lifecycle invariants

EverOS has four kinds of state:

- **Markdown** — the source of truth;
- **SQLite** — memcells, queues, ledgers, purge marks;
- **The derived index** — LanceDB or Milvus, rebuildable from Markdown;
- **The input log** — the raw `/add` payloads kept for audit and deletion.

This page states the invariants that connect them, names the test that pins
each one, and lists the gaps that are not guaranteed yet. It is deliberately
short: if an invariant cannot be tied to a test or a reproducible check, it
does not belong here.

## I1 — one message belongs to exactly one memcell

Boundary detection is the only memorize stage that runs before the memcell
and session-buffer write. A failure there leaves no state behind; a failure
after it cannot be retried the same way.

- A session cut writes the memcells, the `last_memcell_ts`, and the remaining
  buffer in one SQLite transaction.
- A message must not appear in two memcells, or in both a memcell and the
  buffer.

Pinned by `tests/unit/test_properties/test_boundary_stateful.py` and the
session-cut atomicity tests. The boundary-specific retry budget is in review
(#577).

## I2 — Markdown is the truth; the index is a projection

- An episode exists in Markdown before `/add` or `/flush` returns
  `extracted`.
- The index may lag; a read may miss a just-written record.
- A tombstone is written only for a real deletion, not for a scan/watcher
  race.
- A stale claim cannot overwrite a newer claim's index rows.
- `cascade rebuild` recreates the index from Markdown.

The BTree index on the business-table `id` is in review (#578).

Pinned by `tests/unit/test_infra/test_cascade/test_worker_stateful.py`,
`tests/unit/test_infra/test_cascade/test_fence_permutations.py`, and
`tests/unit/test_properties/test_cascade_projection_stateful.py`.

## I3 — deletion removes the closure, everywhere

A deletion must not leave a reachable copy of the deleted data.

- A session deletion removes its memcells, buffer, status, episodes, facts,
  and cluster membership.
- A by-id deletion removes the requested episodes, what was merged into them,
  their facts, and their cluster membership.
- A whole-owner deletion removes the owner's directory, name, clusters, and
  Reflection reports.
- A purge covers the inbox, Markdown, the index, SQLite, the OME run history,
  and `md_change_state`; it also removes the covered `.input_log/` directories.
- An input-log session deletion removes that session's lines; a failure is
  reported as incomplete, not as success.

Pinned by
`tests/unit/test_properties/test_memory_delete_jitter.py` (invariants I1a,
J1a, J2),
`tests/unit/test_service/test_delete_session.py`,
`tests/unit/test_service/test_purge/`, and
`tests/unit/test_service/test_memory/test_input_log.py` (session delete and
purge).

## I4 — reads survive a concurrent deletion

A read that raced a delete must behave like a missing file, not a 500.

- Profile, daily-log, knowledge-index, and taxonomy reads treat a file that
  disappears between `is_file()`/`exists()` and the read as absent.
- Parsing errors, permission errors, and schema-validation errors still
  propagate.

Pinned by
`tests/unit/test_infra/test_markdown/test_readers/test_profile_reader.py`,
`test_base.py`, `test_knowledge_store.py`, and `test_taxonomy_reader.py`.

## I5 — the audit log is append-only and deletable

- One line per accepted `/add`, in
  `<root>/<app>/<project>/.input_log/YYYY-MM-DD.jsonl`, outside `.index/`.
- Text is kept verbatim; a large `base64` attachment is replaced by its byte
  length and SHA-256.
- A failed log write is reported and never fails `/add`.
- Retention removes old daily files when the scope writes.
- A session deletion removes that session's lines; a purge removes the
  covered log directories.

Pinned by `tests/unit/test_service/test_memory/test_input_log.py` and
`tests/unit/test_service/test_delete_cancellation.py`.

## In review

These are being addressed but are not guaranteed on `main` yet:

- boundary LLM retry budget — #577;
- BTree index on the business-table `id` — #578;
- owner-only deletion log redaction — #640;
- private `.input_log` directory/file modes — #636;
- idle-scope retention sweep at startup — #637.

## Known gaps

These are intentionally not claimed as invariants yet:

1. `/memory/device/add` does not write an input-log line.
2. A request rejected by queue admission (429) can still have been logged.

Each gap needs a contract decision before it becomes a fix. The point of this
list is to keep the guaranteed and unguaranteed parts separate.

## How to use this page

- If you change a write, delete, purge, or read path, update the matching
  invariant and its test.
- If you add a new state store, add its deletion and retention rule here.
- If an invariant cannot be tested deterministically, it is a design note,
  not an invariant.
