# Device assistant: one agent, many people

A hardware device runs **one** EverOS agent instance and is spoken to by
**several** people in the same room. That breaks the assumption today's API
is written around — "one agent, one human user" — so this page fixes the
identity rules and the request surface the hardware path has to honour.

Status: **write route shipped**. `POST /api/v2/memory/device/add` accepts a
device batch ([`schemas/device.py`](../src/everos/entrypoints/api/schemas/device.py))
and projects it onto `/memory/add`; reads and deletes use the existing routes
with the composed owner id as `user_id`. The open questions at the end are
still open.

## The problem

Every write path in EverOS keys memory by one owner id per message:

- `POST /memory/add` takes `messages[]`, each with `sender_id` (a path-safe
  id) and an optional `sender_name` (a label). The user-track pipeline
  writes **one episode copy per distinct `role="user"` `sender_id`** of a
  memcell ([`user_memory.py`](../src/everos/memory/extract/pipeline/user_memory.py)),
  each copy under that sender's owner directory; the profile is split per
  owner the same way (`split_episode_by_owner` → `extract_user_profile`).
- `POST /memory/search`, `/memory/get` and `/memory/delete` address memory by
  `user_id` **or** `agent_id`, never both.

On a device there are two identities in every interaction — the device, and
the person talking — and only one of them may name the memory owner. A
device that sends its own id as `sender_id` for everyone puts every
household member into one profile; a device that sends bare voiceprint
slots collides across devices in the same app/project.

## 1. Identity mapping

| Concept | Where it travels | What it means |
|---|---|---|
| Device | `DeviceAddRequest.device_id` (path-safe without `@`, ≤ 128) | The one agent instance: the owner of utterances nobody was identified for, and the **prefix** of every per-person owner id |
| Conversation | `DeviceAddRequest.session_id` | As `/memory/add` takes it: the utterances accumulate under one session, and boundary detection cuts memcells over all speakers |
| Speaker | `DeviceUtterance.speaker.slot` | The device's stable per-person id — voiceprint slot, roster index. **Not** a name |
| Speaker label | `DeviceUtterance.speaker.display_name` | How the person is called in the room. A label only: it never selects a directory |
| Consent | `DeviceUtterance.speaker.consented` | `false` means their utterances are not written at all |
| Per-person memory owner | derived: `speaker_owner_id(device_id, slot)` = `"<device_id>@<slot>"` | The `user_id` / `sender_id` of that person's episodes, profile and clusters **on that device** |

**The rule, in one line:** the device is the scope, the speaker is the
owner, and the owner id is `<device_id>@<slot>`.

Why a composed id rather than the bare slot:

- **No cross-device collision.** App/project are shared by every device in a
  household; `spk_1` on the kitchen device and `spk_1` in the study would
  otherwise be one memory. Per-device resolution also matches how a roster
  is assigned (voiceprints are enrolled device by device).
- **An ordinary owner id everywhere.** `@` is inside the `sender_id` /
  `user_id` charset (`^[a-zA-Z0-9_.@+-]+$`,
  [`ids.py`](../src/everos/entrypoints/api/schemas/ids.py)), so the composed
  id is read, searched and deleted through the existing routes unchanged, and
  it is a valid directory name on every platform. An earlier draft used
  `::`, which no route accepted (nothing could read or delete it) and which
  is not a valid Windows file name. Neither a device id nor a slot may
  contain `@`, so one owner id names exactly one (device, speaker) pair.
  The price: a client can send such an id to `/memory/add` itself — like any
  owner id, it is not a credential.
- **Deterministic and re-derivable.** The same (device, slot) always yields
  the same owner, so a re-registered speaker finds their memory again, and a
  renamed person (`display_name`) keeps it.

Rules at the edges:

| Situation | Result | Where it is enforced |
|---|---|---|
| `speaker` is `None` (nobody identified) | The utterance is filed under the **device** (`sender_id = device_id`). Never guessed onto a person | `DeviceAddRequest.projected_messages()` |
| Speaker known, `consented = false` | Not written; no owner is produced for them | `projected_messages()`, `speaker_owners()` |
| `slot` is a name, contains `/`, `..`, `@`, or is longer than 63 | 422 at validation | `DeviceSpeaker.slot` pattern |
| `device_id` has a path separator or `@`, or is `.` / `..` | 422 at validation | `DeviceAddRequest.device_id` pattern, `PathSafeId` |
| `device_id` + `@` + `slot` would exceed 128 characters | 422 for the whole batch, before any write | `DeviceAddRequest` after-validator |
| Two utterances by one speaker in a batch | One owner (`speaker_owners()` de-duplicates, first-spoken order) | `speaker_owners()` |
| Two speakers in one memcell | Two episode copies, one per owner — the existing fan-out | `_unique_user_senders` |

### Registering a new speaker

The roster is the device's own state, not EverOS's. A device enrolls a
voiceprint (or seats a roster index), assigns a slot, and only then starts
sending `speaker.slot`. EverOS learns a person the first time their slot
appears in a batch: the first episode copy under
`users/<device>@<slot>/` creates the directory, and `extract_user_profile`
writes the profile from that episode on. Nothing needs a registration call;
there is nothing to invalidate when a slot is retired (its memory stays
readable until deleted — see the deletion question below).

### Why this is not group chat

A group has **group semantics**: a group id owns scene summaries, member
profiles *within the group*, and group-scoped reads. A device has none of
that: it is one agent serving N independent people, and the right model is
N one-to-one memories plus the device's own. Nothing in the mapping above
creates a shared record; the only shared state is the `session_id` (needed
for boundary detection) and the device's own records. If a household later
wants shared memory ("remember, we're out of milk"), that is a device-owned
record — `user_id = device_id` — not a group.

## 2. Where the records live

For device `D`, speaker slot `S`, in scope `app_id`/`project_id`:

```
<root>/<app>/<project>/
├── users/
│   ├── D@S/                            # S's own memory on D
│   │   ├── user.md                      # S's profile (as ever)
│   │   └── episodes/episode-<date>.md   # S's episode copies
│   └── D/                               # the device's own records
│       ├── user.md                      # device profile: room, settings answers
│       └── episodes/episode-<date>.md   # utterances nobody was identified for
└── .index/                              # sqlite + lancedb: one row set, owner-scoped
```

- The episode **text** of a memcell is shared (one model call), but its
  **copies** are per owner: what S said and what T said land in two
  directories, each readable on its own.
- The device directory holds exactly what belongs to no person: unattributed
  utterances, and anything the device itself decides to remember (room
  context, device settings). A device that has none of that simply never
  writes to `users/D/`.
- Clusters (used by `AGENTIC` search and Reflection) are per owner, so S's
  memory consolidates without touching T's.
- Entity records that only make sense per device (settings, room context)
  are **not** a new storage kind: they are the device owner's `user.md` and
  episodes, i.e. the existing single-file and daily-log patterns. A new kind
  would need a cascade projection and an index table, and buys nothing the
  owner directory does not already give.

The owner limit (128 characters) applies to `<device_id>@<slot>`; a device
id and a slot that would not fit is refused at request time (see the table
in section 1) rather than truncated — a truncated id would silently land in
someone else's or nowhere's memory.

## 3. The request / response surface

The design keeps **one** write path, not a parallel API: the device route
projects onto `/memory/add`, and reads and deletes are the existing routes,
so the memory, storage, search and delete code is unchanged.

### 3.1 Add — `POST /api/v2/memory/device/add`

| | |
|---|---|
| Request | `DeviceAddRequest` (section 1 table) |
| Success | `200` `{"request_id": ..., "data": {"message_count": N, "status": "accumulated" \| "extracted", "owners": ["D@S", ...]}}` (`DeviceAddResponseData`) |
| Queued | `202` with `async_mode: true`, or when the session already has queued batches: `/memory/add`'s body plus `owners` (`DeviceAcceptedResponseData`) |
| All declined | `200` with `message_count: 0`, `status: "accumulated"`, `owners: []`; nothing is written |
| Errors | `422` — no utterances, a slot or device id outside its charset, an owner id over 128 characters; `409` `SESSION_BUSY` / `500` `PAYLOAD_TOO_LARGE`, inherited from `/memory/add` |
| Projection | `DeviceAddRequest.projected_messages()` → the `messages[]` the pipeline consumes: `sender_id` = `D@S` (or `D`), `sender_name` = label, `role` = `"user"` |

`sender_name` missing: the episode copy's `owner_name` falls back to the
**sender id** (existing `_sender_label` behaviour), so the narrative says
`D@S` — ugly but unambiguous. Devices should always send a
`display_name` they know; the profile is not affected either way.

### 3.2 Flush — `POST /api/v2/memory/flush`

Unchanged: `{"session_id": ...}` (`MemorizeFlushRequest`). The device's
session id is the only thing the device needs to remember between calls.
`status: "extracted"` means the episode copies are on disk — the per-person
episodes become searchable after the cascade indexes them, and the profiles
after `split_episode_by_owner` → `extract_user_profile` run.

### 3.3 Search — `POST /api/v2/memory/search`

Whose memory is searched is expressed by `user_id` alone, so nothing new is
needed on the wire:

| Scope | `user_id` sent | Meaning |
|---|---|---|
| `speaker` | `D@S` | one person's memory on this device |
| `device` | `D` | the device's own records |
| `both` | `D@S`, then `D` | the person's memory first, then the device's |

For `both` the client issues the two searches and merges by score (the
response items carry `score` on one scale, HYBRID/AGENTIC only — see
[api.md](api.md)); a server-side `both` is a later convenience, and it must
not become a scope that reads a third person's memory.

Cross-person leakage is the one property to keep: every response item
carries `user_id`, so a client can assert none of them is another slot's.

### 3.4 Forget / delete — `POST /api/v2/memory/delete`, `/memory/delete_by_ids`

| Need | Request | Effect |
|---|---|---|
| Forget one person on one device | `{"user_id": "D@S"}` | That person's directory, profile, clusters and index rows |
| Forget one conversation | `{"session_id": ..., "user_id": "D@S"}` | Their copies of that session only |
| Forget the device | `{"user_id": "D"}` | The device's own records; **not** the people's |
| Forget one episode | `/memory/delete_by_ids` with `user_id: "D@S"` | As today, scoped to that owner |

The 128-character owner limit applies to `user_id` here too, and the same
validation runs.

### 3.5 Profile — `POST /api/v2/memory/get`

`{"user_id": "D@S", "memory_type": "profile"}` returns that person's
profile. `{"user_id": "D", "memory_type": "profile"}` returns the device's.
A device that wants to greet a person by name reads the profile's summary;
it should not parse `explicit_info` items as facts to repeat aloud.

### 3.6 Backwards compatibility

Existing clients are unaffected because **nothing in an existing request
changed**: no field renamed, re-typed, given a new validator, or made
required. `tests/unit/test_entrypoints/test_api/test_schemas/test_device_request_parity.py`
pins the request bodies of `/memory/add`, `/memory/flush`, `/memory/delete`,
`/memory/delete_by_ids`, `/memory/get` and `/memory/search` as frozen
values: each one must validate and round-trip with every value a client sent
coming back unchanged (defaults the client omitted are the server's to
fill). The device route is additive, and the same test asserts the
projection and the composed owner pass `/memory/add`, `/memory/delete` and
`/memory/search` validation unchanged.

## 4. Open questions the implementation must settle

1. **Speaker identification upstream.** Who decides `slot`, and how good is
   it? A wrong identification writes one person's words into another's
   memory, which no later query can undo. Options: require `consented` and
   a confidence the device reports; file low-confidence utterances under the
   device; or ask before writing. The schema has `consented` but no
   confidence field yet.
2. **Consent and privacy on a shared device.** Per-person memory on a
   device every household member can talk to is a consent problem, not a
   storage one. The mapping gives `consented = false` no owner, but cannot
   enforce that the *device* stops sending. Required: what the device tells
   people, how a person's memory is exported or wiped on request, and
   whether the device's own records may name a non-consenting person.
3. **Deletion semantics when a device changes hands.** Today a deletion
   removes one owner. Selling or resetting a device should remove every
   `D@*` owner plus `D` — but "every owner with prefix `D@`" is not a
   deletion the API expresses (prefix scans cross owner directories). Likely
   answer: a device-scoped deletion that resolves the prefix through the
   sqlite index and then deletes each owner in turn, with the same crash-safe
   steps as `/memory/delete`.
4. **What the profile means for a device.** A person's profile is
   straightforward. The device's own profile is not: is it a summary of what
   the device knows about its room, a preferences file for its own voice, or
   nothing at all? Recommendation: the device profile stays the ordinary
   profile of the device owner (it is what `extract_user_profile` writes
   anyway), and device *settings* live in the device's configuration, not in
   memory.
5. **Two devices, one person.** The mapping deliberately gives the same
   person two memories. Whether the product wants a cross-device person
   identity (say `person@<slot>` independent of device) is a product
   decision; it would be a second owner id scheme, not a change to this one.
6. **Session length.** A device session that never flushes grows until the
   buffer is cut; a household's conversations also interleave. The device
   should flush on silence or at a size, and a session should not span two
   different rosters.

## 5. Cost

Per 1,000 utterances, basis: EverOS pricing drivers are the episode call
(one per memcell), the summary call (one per memcell, offline), the profile
call (one per owner-episode, offline), and the embedding of every written
episode. **Estimated** from the call counts in
[`user_memory.py`](../src/everos/memory/extract/pipeline/user_memory.py) and
`default_ome.toml`; not measured for this path (no billed run was made for
this design).

| Side | What runs | Per 1,000 utterances (assume ~100 memcells of 10 utterances) |
|---|---|---|
| Write | boundary detection: 1 call per add batch | 1–10 calls, depending on batch size |
| Write | episode: **1 call per memcell**, shared by all speakers | 100 calls |
| Write | summary: **1 call per memcell**, offline | 100 calls |
| Write | profile: 1 call per owner × memcell the owner sent in | ≤ 100 calls, one per (speaker, memcell) pair that the speaker joined, up to the number of distinct speakers |
| Write | embedding: 1 per episode copy (one per owner) + 1 per episode for clustering | ≈ 100 + number of speaker-memcell pairs |
| Read | search: 1 query embedding + the decider calls (HYBRID needs no LLM; AGENTIC adds an LLM per round) | 1–many per query, by method |

The mapping itself adds **no** model call: a device batch with one speaker
costs exactly what the same batch costs today with that `sender_id`. The
cost of *many* speakers is the per-owner fan-out the pipeline already does
(one profile call and one embedding per owner per memcell), so a household
of four is roughly four times the profile cost and one times the episode
cost. Read-side cost is a client choice (`keyword` and `hybrid` need no
LLM), and `both` scope doubles the number of searches, not their price.
