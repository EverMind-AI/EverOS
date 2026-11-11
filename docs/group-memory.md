# Group and Project Memory

Design for EverOS 2.0. Group chats and projects become first-class memory
owners: one group episode per conversation segment, every memory says who said
it and whom it is about, the assistant's own messages are marked as such, and
every read is checked against the asker's group membership.

Status: design. Code references point at commit `08bb031` (2.0 development
line). Line numbers in the EverMemOS reuse map refer to the EverMemOS snapshot
the map was made against (package `memsys` 0.1.0,
[github.com/EverMind-AI/EverMemOS](https://github.com/EverMind-AI/EverMemOS)).

## Table of contents

- [1. Summary](#1-summary)
- [2. The code today](#2-the-code-today)
- [3. Goals and non-goals](#3-goals-and-non-goals)
- [4. Data model](#4-data-model)
- [5. API](#5-api)
- [6. Storage](#6-storage)
- [7. Write path](#7-write-path)
- [8. Read path](#8-read-path)
- [9. Visibility](#9-visibility)
- [10. One-to-one conversations stay unchanged](#10-one-to-one-conversations-stay-unchanged)
- [11. Where the prompts live](#11-where-the-prompts-live)
- [12. Reuse map from EverMemOS](#12-reuse-map-from-evermemos)
- [13. Evaluation](#13-evaluation)
- [14. Task list](#14-task-list)
- [15. Risks](#15-risks)
- [16. References](#16-references)

---

## 1. Summary

Group memory ships in two steps. Both are part of 2.0.

| Step | Scope |
|---|---|
| **Step 1** | The `group` owner, roster (members, roles, join and leave times) and group meta; sessions bound to a group; one group episode per segment with speaker, subject and origin; pictures stored once per group; the member-in-group profile; group-scoped and asker-conditioned search over one or several groups; the membership-window check on every read path; EverMemBench rebuilt with real member ids; GroupAssistBench L1. |
| **Step 2** | The versioned group profile (topics, decisions, dissent, roles); relational expansion in multi-round search; the constraint block in search responses; partial-memory handling; who-knew-what; forget and delete propagation inside groups; project roll-up across groups; GroupAssistBench L2; GateMem. |

The member-in-group profile is in Step 1, not with the rest of the group state,
because the EverMemBench per-member run depends on it and it needs no new
prompt (it reuses the existing profile operator, [Section 7.4](#74-member-in-group-profile)).

### Decisions

"vs EverMemOS" says how each decision relates to the predecessor. **Overrides**
marks the places where 2.0 replaces an EverMemOS design.

| # | Decision | vs EverMemOS |
|---|---|---|
| D1 | A third owner type, `group`, with its own directory `groups/<group_id>/` inside the existing `<app>/<project>/` partition. | Adapted: EverMemOS put `group_id` on user-owned records and used `user_id = None` for group records. |
| D2 | A roster of membership intervals (join, leave, rejoin), access roles and titles, kept as an append-only log in `group.md` and mirrored into SQLite. | Adapted: EverMemOS `user_details` had names and job roles but no join or leave times. |
| D3 | One group episode per segment. No per-member copies, no per-member episodes. | **Overrides**: EverMemOS wrote one group episode plus one personal episode per participant, `k + 1` model calls whose prompts are textually identical in English. |
| D4 | Every record carries `sender_ids` (who said it), `subject_ids` (whom it is about) and `origin` (member, assistant or mixed). | New. |
| D5 | Assistant lines are rendered as the assistant's; a decision needs at least one member message as evidence. | New. |
| D6 | The member-in-group profile lives under the group and holds only content about that member. A member's personal profile is never read in a group scope. | **Overrides**: EverMemOS returned the group profile and the global profile of a user together (`CombinedProfileModel`). |
| D7 | The group profile is append-only: a change is a new item that links to the item it supersedes; topics are never evicted. | **Overrides**: EverMemOS replaced topics past a cap and rewrote profile documents with an `is_latest` flag. |
| D8 | One group-profile call per batch of segments, covering topics, decisions, dissent and roles. | **Overrides**: EverMemOS ran two parallel calls (content and behaviour) and had no decision or dissent records. |
| D9 | One access policy decides every read. Requests carry `asker_id`; a group read requires current membership and returns only content inside the asker's membership windows; anything invisible behaves as absent. | **Overrides**: EverMemOS had no access check and allowed wildcard owner filters. |
| D10 | Search takes a group scope and an asker; the asker conditions visibility, the question rendering and the attached profiles. | Adapted from EverMemOS search by `user_id` and/or `group_id`. |
| D11 | Relational expansion follows stored links (scene, decision chain, subjects), not the request's wording, and keeps the decider's candidate format. | New. |
| D12 | Search returns a constraint block (subject, scope, status, source quote) built first from write-time verified items. | New. |
| D13 | Forget and delete travel along sources; forgetting a message re-extracts the segment without it. | Adapted from the 2.0 picture forget. |
| D14 | Without `group_id` the existing one-to-one path runs unchanged. | Adapted: EverMemOS used an `assistant` scene for one-to-one chats. |
| D15 | Step 1 adds no prompt text; Step 2's two new prompts live in EverOS's prompt registry for 2.0 and move to everalgo afterwards. | Adapted: EverMemOS kept its group prompts in-tree. |
| D16 | Stores that ingested a group under one user are rebuilt from raw conversations, not converted. | n/a |
| D17 | Profile items read from a picture go to the picture's poster only, on every path. | n/a (fixes a 2.0 development-line leak). |
| D18 | A session belongs to at most one group; boundary detection stays linear per session; a roster change closes the open buffer first. | Adapted: EverMemOS carried `group_id` per message. |

---

## 2. The code today

What exists at `08bb031`, and what this design changes:

| Fact | Where |
|---|---|
| `/memory/add` accepts batches with several senders, but carries no group id and no roster. | `src/everos/entrypoints/api/schemas/memorize.py:46-111` |
| Owners are users or agents only. | `src/everos/memory/ports/query.py:56` |
| `app_id` / `project_id` are directory partitions, nothing more. | `src/everos/infra/persistence/memory_root.py:35-51` |
| A segment's episode is extracted once, then written once per human sender; pictures are written once per sender as well. | `src/everos/memory/extract/pipeline/user_memory.py:145-197`, `:383-389` |
| Profile items read from a picture are merged into every human sender's profile in the segment, whoever posted the picture. | `user_memory.py:163-170`, `src/everos/memory/extract/pipeline/profile_delta.py:280-332` |
| The profile strategy treats the event's owner as the profile subject. | `src/everos/memory/strategies/extract_user_profile.py:40-43` |
| Search names exactly one owner; the base predicate pins owner, type, app and project. There is no access check. | `src/everos/memory/search/dto.py:120-124`, `:167-172`; `src/everos/memory/search/filters.py:51-72` |
| A session-filtered search returns every buffered message of that session, whoever sent it. | `src/everos/memory/search/manager.py:334-351` |
| Picture bytes are served by content hash without an owner check. | `src/everos/entrypoints/api/routes/images.py:233-264` |
| The cascade maps every scope directory other than `agents/` to a user owner, so a `groups/` file would index as a user with an empty id. | `src/everos/infra/cascade/_projections/_common.py:41-58` |
| Knowledge is per (app, project) and upload-only. | `src/everos/memory/models.py:588-633` |
| Boundary detection is linear per session; a memcell records all of its senders. | `src/everos/memory/extract/boundary/messages.py:84-98`; `src/everos/memory/extract/boundary/detection.py:261-273` |

Two consequences matter most. Cost grows with the number of people speaking in
a segment (one stored copy, two embeddings and one profile extraction per
sender). And anything a sender's copy carries (the whole segment narrative,
picture-derived profile items) becomes "their" memory, so it can surface in
answers to that person about things another member said.

---

## 3. Goals and non-goals

Goals:

- Model a group (and its project) as an owner with a roster that changes over time.
- Extract one group record per segment, with attribution that keeps speaker and subject apart.
- Keep group state (topics, decisions, dissent, roles) as versioned, append-only items.
- Answer group questions per asker: the same question can have different answers for different members.
- Turn memories into explicit constraints before an answer is written.
- Enforce visibility with one check on every read path, and propagate forgetting to derived records.
- Leave one-to-one conversations exactly as they are.

Non-goals for 2.0:

- Authentication. EverOS trusts its caller ([SECURITY.md](../SECURITY.md)); the
  asker identity is asserted by the caller or by a gateway header ([Section 9.2](#92-identity)).
- Per-message access lists inside a group. The unit of visibility is the membership window.
- Reply threads as a graph. Messages keep their order; `reply_to` is not modelled.
- Merging identities across groups. Member ids are caller-provided and stable.
- Group-owned working memory. Agent cases and skills stay per agent ([Section 9.3](#93-the-check-and-where-it-runs)).
- Dreaming for group owners. Dreaming runs for user and agent owners only.

---

## 4. Data model

### 4.1 Owners and identifiers

| Owner type | Id | Directory |
|---|---|---|
| `user` | `user_id` | `users/<user_id>/` |
| `agent` | `agent_id` | `agents/<agent_id>/` |
| `group` (new) | `group_id` | `groups/<group_id>/` |

`OwnerType` becomes `Literal["user", "agent", "group"]`. A `group_id` uses the
same path-safe charset and length limits as `user_id`, and is unique within one
`(app_id, project_id)`.

A **member id** is the `sender_id` the member's messages carry. It lives in the
same id space as `user_id`: a person who also talks to the assistant one-to-one
has `users/<member_id>/` for that personal memory. An assistant member's id is
its `agent_id`.

### 4.2 Group

| Field | Type | Notes |
|---|---|---|
| `group_id` | string | path-safe |
| `name`, `description` | string | shown to the extraction model in the roster block |
| `created_at` | epoch ms | first message time when the group is created implicitly |
| `created_by` | member id or `null` | `null` for implicit creation |
| `extra` | object | stored, never interpreted |

### 4.3 Roster

A roster is a set of **membership intervals**. Each interval belongs to one
member and is half-open: `[joined_at, left_at)`, with `left_at = null` while the
member is in the group. A member who leaves and rejoins has two intervals.

| Field | Type | Notes |
|---|---|---|
| `member_id` | string | the sender id |
| `kind` | `human` \| `assistant` | `assistant` for role=assistant senders |
| `display_name` | string | the label the extraction model sees |
| `aliases` | list of strings | other names members use for this person (subject matching) |
| `role` | `owner` \| `admin` \| `member` \| `assistant` | access role ([Section 9.1](#91-rules)) |
| `title` | string | declared job or function, for example "Product owner"; shown to the model |
| `joined_at`, `left_at` | epoch ms | the interval |
| `source` | `api` \| `inline` \| `implicit` | how the interval was created |

The roster changes only through **roster events**, appended in order and never
rewritten:

| Event | Effect |
|---|---|
| `join` | opens an interval at `at` (rejected with 409 if one is open) |
| `leave` | closes the open interval at `at` (409 if none is open) |
| `update` | changes `display_name`, `aliases`, `role` or `title` from `at` on |

The roster at time `t` is the fold of all events with `at <= t`. Events may be
backdated (an importer replays a history); backdating is restricted to owners
and admins.

Two kinds of role exist and are kept apart:

- the **access role** above, declared by the caller, which decides who may
  manage the roster and forget group content;
- **observed roles** in the group profile (decision maker, coordinator and so
  on, [Section 7.5](#75-group-profile-step-2)), extracted with evidence and used
  when an answer has to weigh decision authority.

### 4.4 Project

A project is the existing `project_id` partition, used as a collaboration
space. It holds:

- the groups whose `project_id` it is;
- its members: the union of the current members of those groups (derived, never stored separately);
- the knowledge base already kept per `(app_id, project_id)`.

The project adds no new owner. Project-scope reads (Step 2) run over the groups
the asker currently belongs to ([Section 8.7](#87-project-roll-up-step-2)).

### 4.5 Speaker, subject, origin

Every group-owned record carries three attribution fields:

| Field | Meaning | Set by |
|---|---|---|
| `sender_ids` | who said it: the distinct senders behind the record (episodes already have this column) | code, from the memcell |
| `subject_ids` | whom it is about | code in Step 1 ([Section 7.2](#72-group-episode)); the model's evidence for group-profile items |
| `origin` | `member`, `assistant` or `mixed` | code, from the senders' roster kind |

The fields also exist on one-to-one records; there they keep today's values
(`subject_ids` empty, `origin` unset), so nothing reads differently.

---

## 5. API

All routes are under `/api/v2`. New fields are optional unless a group scope is
used; one-to-one requests are accepted exactly as today.

### 5.1 Groups and roster

**Create or update a group**: `PUT /api/v2/groups/{group_id}`

```json
{
  "app_id": "default",
  "project_id": "spring-offsite",
  "asker_id": "u_chen",
  "name": "Trip team",
  "description": "Planning the spring offsite",
  "created_at": 1772323200000,
  "roster": [
    {"op": "join", "member_id": "u_chen", "display_name": "Chen", "role": "owner",
     "title": "Organiser", "at": 1772323200000},
    {"op": "join", "member_id": "u_betty", "display_name": "Betty", "role": "member",
     "at": 1772323200000},
    {"op": "join", "member_id": "bot_ava", "kind": "assistant", "display_name": "Ava",
     "role": "assistant", "at": 1772323200000}
  ]
}
```

Creating a group whose roster names no owner makes the asker its owner.
Updating meta requires an owner or admin.

**Change the roster**: `POST /api/v2/groups/{group_id}/roster`

```json
{
  "app_id": "default",
  "project_id": "spring-offsite",
  "asker_id": "u_chen",
  "events": [
    {"op": "join", "member_id": "u_evan", "display_name": "Evan", "role": "member",
     "title": "Designer", "at": 1772704800000}
  ]
}
```

Owners and admins may send any event; any member may send `leave` for
themselves. The response returns the current roster and the number of events
applied.

**Read a group**: `GET /api/v2/groups/{group_id}?app_id=&project_id=&asker_id=&include_history=false`
returns the meta and the current roster (with every interval when
`include_history=true`). Only current members may read it.

**Read a project** (Step 2): `GET /api/v2/projects/{project_id}?app_id=&asker_id=`
returns the groups the asker belongs to and the project members.

Roster updates therefore arrive three ways: the roster route, the `roster` field
of `/memory/add` ([Section 5.2](#52-add)), and implicit joins for unknown
senders (configurable).

### 5.2 Add

`POST /api/v2/memory/add` gains three things:

| Field | Meaning |
|---|---|
| `group_id` | the group the session belongs to. The first batch that names it binds the session; a later batch without `group_id` inherits the binding; a batch naming a different group gets **409**. An unknown `group_id` creates the group (`[groups] implicit_create`). |
| `roster` | roster events applied before the messages, same shape as the roster route. Lets an importer send membership with history. A batch that creates the group may set its initial roster; for an existing group the batch's `asker_id` must be an owner or admin. |
| `messages[].message_id` | optional client id, unique within the session. Used by message-level forget and in source quotes. |
| `asker_id` | needed only when the batch carries roster events for a group that already exists; must be an owner or admin of it (403 otherwise) |

A message from a sender who is not a member opens an interval at that message's
timestamp with `source = "implicit"` and role `member` (or `assistant` for
role=assistant). With `[groups] implicit_join = false` such a batch is rejected
with 422.

```json
{
  "app_id": "default",
  "project_id": "spring-offsite",
  "group_id": "trip-team",
  "session_id": "trip-team-main",
  "messages": [
    {"message_id": "m-102", "sender_id": "u_chen", "sender_name": "Chen", "role": "user",
     "timestamp": 1772442120000,
     "content": "May 1 flights are sold out. Let's leave on April 30 instead."},
    {"message_id": "m-103", "sender_id": "u_betty", "sender_name": "Betty", "role": "user",
     "timestamp": 1772442300000, "content": "Fine by me. I'll book the hotel from April 30."},
    {"message_id": "m-104", "sender_id": "bot_ava", "sender_name": "Ava", "role": "assistant",
     "timestamp": 1772442360000, "content": "There are two April 30 flights: 08:10 and 13:40."},
    {"message_id": "m-105", "sender_id": "u_chen", "sender_name": "Chen", "role": "user",
     "timestamp": 1772442480000, "content": "Take the 08:10. I'll book flights for all nine of us."}
  ]
}
```

`/memory/flush` is unchanged: it names a session, and the session's binding
gives the group.

### 5.3 Search

`POST /api/v2/memory/search` accepts exactly one owner scope:

| Scope | Fields |
|---|---|
| one-to-one user | `user_id` (unchanged); `asker_id` optional and, if given, must equal `user_id` |
| agent | `agent_id` (unchanged) |
| one group | `group_id` + `asker_id` |
| several groups | `group_ids` + `asker_id` |
| project (Step 2) | `project_scope: true` + `asker_id`: every group in the project the asker belongs to |

Further fields for group scopes:

| Field | Default | Meaning |
|---|---|---|
| `include_profile` | `false` | attach the asker's member-in-group profile and those of members named in the query (at most three) |
| `include_group_profile` | `true` for group scopes (Step 2) | attach the visible current group-profile items, capped |
| `constraints` | `true` for group scopes (Step 2) | return the constraint block |
| `history` | `false` | the 2.0 history mode: superseded and forgotten versions may enter the candidates |

Response additions (a sketch; fields marked Step 2 are absent before it):

```json
{
  "scope": {
    "owner_type": "group",
    "group_ids": ["trip-team"],
    "asker_id": "u_betty",
    "windows": [{"group_id": "trip-team", "from": 1772323200000, "to": null}]
  },
  "episodes": [
    {
      "id": "trip-team_ep_20260302_00000001",
      "owner_type": "group",
      "group_id": "trip-team",
      "sender_ids": ["u_chen", "u_betty", "bot_ava"],
      "subject_ids": ["u_chen", "u_betty"],
      "origin": "mixed",
      "source": {"memcell_id": "mc_7f3a", "from": 1772442120000, "to": 1772442480000},
      "subject": "Departure moved to April 30; Chen books the 08:10 flight",
      "episode": "Chen said the May 1 flights were sold out and proposed leaving on April 30. ...",
      "timestamp": "2026-03-02T09:08:00Z",
      "score": 0.91
    }
  ],
  "profiles": [{"owner_type": "group", "group_id": "trip-team", "member_id": "u_betty", "profile_data": {}}],
  "group_profile": {"topics": [], "decisions": [], "dissent": [], "roles": []},
  "constraints": [],
  "coverage": "complete"
}
```

`group_profile`, `constraints` and `coverage` are Step 2
([Sections 8.4-8.5](#84-constraint-block-step-2)). One-to-one responses carry
the new fields with empty defaults.

### 5.4 Get and history

`POST /api/v2/memory/get` takes the same group scopes and adds memory types for
groups:

| `memory_type` | Group scope |
|---|---|
| `episode`, `image_memory`, `scene` | as for users |
| `member_profile` | requires `member_id` |
| `group_profile` (Step 2) | items, with `include_superseded` |

The 2.0 history route returns a record's version chain; for group records only
the versions the asker can see.

### 5.5 Forget, undo, delete

Group content is forgotten with the 2.0 forget route (reversible, `dry_run`
first). Group targets (Step 2):

```json
{
  "app_id": "default",
  "project_id": "spring-offsite",
  "group_id": "trip-team",
  "asker_id": "u_betty",
  "target": {"kind": "message", "session_id": "trip-team-main", "message_ids": ["m-103"]},
  "reason": "sent by mistake",
  "dry_run": true
}
```

| `target.kind` | Addresses | Who may |
|---|---|---|
| `message` | `session_id` + `message_ids` | the message's sender; owners and admins |
| `member` | `member_id`: all of that member's messages in the group | the member; owners and admins |
| `session` | `session_id` | owners and admins |
| `record` | episode, image memory, scene or profile item ids | owners and admins; a member for records derived only from their own messages |
| `image` | `image_id` (content hash) | the poster; owners and admins |

The response returns `forget_id` and what was touched. Undo takes the
`forget_id` and the same authorisation. `/memory/delete` with a group scope is
owner/admin only and applies the same propagation, removing instead of marking.
Leaving a group is not a forget: a member's messages stay visible to the
others.

Propagation is described in [Section 9.4](#94-forget-and-delete-propagation).
Before Step 2, group-scoped forget requests are rejected with 422 so that no
partial forget can run.

### 5.6 Errors

| Status | When |
|---|---|
| 403 `forbidden` | the asker is not a current member, or may not perform the action |
| 404 | a record or picture the asker cannot see, exactly as for one that does not exist |
| 409 | a session bound to another group; a roster event that conflicts with the open interval |
| 422 | invalid fields; group forget before Step 2; unknown sender with `implicit_join = false` |

---

## 6. Storage

### 6.1 Markdown

Markdown stays the source of truth:

```
<memory-root>/<app>/<project>/
├── users/ ...                         unchanged
├── agents/ ...                        unchanged
├── knowledge/ ...                     unchanged (the project's knowledge base)
└── groups/
    └── <group_id>/
        ├── group.md                   meta (frontmatter) + roster log (append-only entries)
        ├── episodes/
        │   └── episode-<YYYY-MM-DD>.md    one entry per segment
        ├── images/
        │   └── image-<YYYY-MM-DD>.md      one entry per picture occurrence
        ├── scenes/
        │   └── <scene_id>.md              2.0 scene versions
        ├── members/
        │   └── <member_id>.md             member-in-group profile
        └── profile/                       Step 2
            └── group_profile-<YYYY-MM-DD>.md   group-profile items, append-only
```

A new frontmatter mixin, `GroupScopedFrontmatter`, carries `group_id`,
`track = "group"` and `SCOPE_DIR = "groups"`, next to the user and agent mixins.

`group.md`:

```markdown
---
id: group_trip-team
type: group
schema_version: 1
group_id: trip-team
track: group
name: Trip team
description: Planning the spring offsite
created_at: '2026-03-01T00:00:00+00:00'
---
<!-- entry:gr_20260301_00000001 -->
**op**: join
**member_id**: u_chen
**kind**: human
**display_name**: Chen
**role**: owner
**title**: Organiser
**at**: 2026-03-01T00:00:00+00:00
**source**: api
<!-- /entry:gr_20260301_00000001 -->
<!-- entry:gr_20260305_00000001 -->
**op**: join
**member_id**: u_evan
**kind**: human
**display_name**: Evan
**role**: member
**at**: 2026-03-05T10:00:00+00:00
**source**: api
<!-- /entry:gr_20260305_00000001 -->
```

A group episode entry uses the existing episode contract plus three inline
fields:

```markdown
<!-- entry:ep_20260302_00000001 -->
**owner_id**: trip-team
**session_id**: trip-team-main
**timestamp**: 2026-03-02T09:08:00+00:00
**source_start**: 2026-03-02T09:02:00+00:00
**parent_type**: memcell
**parent_id**: mc_7f3a
**sender_ids**: [u_chen, u_betty, bot_ava]
**subject_ids**: [u_chen, u_betty]
**origin**: mixed

### Subject
Departure moved to April 30; Chen books the 08:10 flight

### Summary
The group moved departure from May 1 to April 30; Chen takes the 08:10 flight for nine, Betty books the hotel.

### Content
Chen said the May 1 flights were sold out and proposed leaving on April 30 (2026-04-30) instead; Betty agreed and said she would book the hotel from April 30. The assistant, Ava, listed two April 30 flights, at 08:10 and 13:40. Chen chose the 08:10 flight and said he would book flights for all nine travellers.
<!-- /entry:ep_20260302_00000001 -->
```

`timestamp` keeps its meaning (the segment's last message); `source_start` is
the segment's first message, so a membership window can contain the whole
segment ([Section 9.1](#91-rules)).

A group-profile item (Step 2) is a daily-log entry with prefix `gp`:

```markdown
<!-- entry:gp_20260302_00000002 -->
**kind**: decision
**topic_id**: tp_4c1e
**decided_by**: [u_chen]
**agreed_by**: [u_betty]
**supersedes**: gp_20260301_00000001
**subject_ids**: [u_chen, u_betty]
**origin**: member
**sources**: [mc_7f3a/m-102, mc_7f3a/m-103]
**source_start**: 2026-03-02T09:02:00+00:00
**source_end**: 2026-03-02T09:05:00+00:00

### Statement
Depart on April 30 instead of May 1.

### Evidence
- m-102 (Chen): "Let's leave on April 30 instead."
- m-103 (Betty): "Fine by me."
<!-- /entry:gp_20260302_00000002 -->
```

The member-in-group profile `members/<member_id>.md` uses the 2.0 profile format
(items with stable ids, per-item source memcells, a history part for replaced
items, a new value linked to the item it replaced; section 7.4) with
`group_id` and `member_id` in its frontmatter.

### 6.2 SQLite

New tables in `system.db`:

| Table | Key | Columns |
|---|---|---|
| `group_meta` | `(app_id, project_id, group_id)` | `name`, `description`, `created_at`, `created_by`, `md_path`, `updated_at` |
| `group_member` | `(app_id, project_id, group_id, member_id, seq)` | `kind`, `display_name`, `aliases_json`, `role`, `title`, `joined_at`, `left_at`, `source`, `updated_at`; one row per interval, `seq` counts a member's intervals |
| `session_binding` | `(app_id, project_id, session_id)` | `group_id`, `bound_at` |
| `message_forget` (Step 2) | `(app_id, project_id, session_id, message_id, forget_id)` | `created_at`, `undone_at` |

Changed:

| Table | Change |
|---|---|
| `forget_log` | `owner_type` takes `group`; new nullable column `requested_by` |

Roster writes go to `group.md` and to `group_meta` / `group_member` in the same
locked operation, so a removed member loses access immediately rather than after
the cascade catches up. `everos cascade rebuild` re-derives both tables from
`group.md`. New columns are nullable and are added by the existing forward-only
mechanism (`src/everos/infra/persistence/sqlite/schema.py:33`).

### 6.3 LanceDB

| Table | Change |
|---|---|
| `episode` | `owner_type` takes `group`, `owner_id = group_id`; new columns `subject_ids: list<string>`, `origin: string`, `source_start: timestamp`, `source_end: timestamp`. `sender_ids` is the speaker list it already is. |
| `image_memory` | new `sender_ids` (the poster), `subject_ids`, `origin`, `source_start`, `source_end` |
| `scene` (2.0) | the same four columns; `source_start` / `source_end` span the member episodes |
| `user_profile` | rows for member-in-group profiles have `owner_type = group`, `owner_id = group_id` and a new column `subject_id = member_id` |
| `group_profile_item` (Step 2, new) | `item_id`, `kind`, `topic_id`, `statement`, `statement_tokens` (BM25), `status`, `supersedes`, `superseded_by`, `decided_by`, `agreed_by`, `held_by`, `subject_ids`, `sender_ids`, `origin`, `sources`, `source_start`, `source_end`, `observed_at`, `md_path`. No vector column. |

`status` and `superseded_by` are computed when the index syncs (an item is
superseded when a later item names it in `supersedes`), as for the other 2.0
versioned kinds. Speaker, subject and window fields are scalar columns; nothing
new is embedded.

### 6.4 Migration

| Store | What happens |
|---|---|
| Stores built by EverOS 1.x | 2.0 does not open them (the 2.0 store-version rule). Nothing to migrate. |
| 2.0 stores built before group support | Open as before. SQLite gains the new tables and nullable columns at startup; LanceDB gains the new columns in one versioned step of `migrate_table_schemas` (`src/everos/infra/persistence/lancedb/backend.py:257`), fail-loud, with `everos cascade rebuild` as recovery. Existing rows read `subject_ids = []`, `origin = null`, `source_start = source_end = timestamp`. The step is table schema version 4 (`_TABLE_SCHEMA_VERSION`). Milvus has no such step yet: a Milvus store reports schema drift and needs `everos cascade rebuild`. |
| Stores where a whole group chat was ingested under one user id (for example benchmark stores built with one owner per conversation) | Not converted. The per-sender copies and cross-owner items cannot be split back reliably, and a converter would have to guess membership. Rebuild from the raw conversations with `group_id`. |

The cascade's owner resolution must learn `groups/` (owner type `group`, id from
the `group_id` frontmatter field) before any group file is written; today it
would index such a file as a user ([Section 2](#2-the-code-today)).

---

## 7. Write path

### 7.1 Flow

For a batch with `group_id`, the 2.0 write steps run as follows:

1. **Receive.** Validate; bind the session to the group (409 on conflict);
   apply inline roster events; open implicit intervals for unknown senders;
   store pictures in `figures/` as today.
2. **Segment.** Boundary detection runs per session, linear, as today. When a
   roster event arrives for a group that has buffered messages, the buffer is
   cut first (the same finalisation `/flush` performs), so a new segment never
   spans a membership change. Correctness does not depend on this: the check uses
   `source_start` and `source_end` ([Section 9.1](#91-rules)); the cut only keeps a
   new member's first segment visible to them.
3. **Route.** The 2.0 router decides chat or working per segment, unchanged.
4. **Extract.** One group episode per segment ([Section 7.2](#72-group-episode)),
   with picture memories in the same call when the segment carries pictures
   ([Section 7.3](#73-pictures)).
5. **Write.** One entry under `groups/<group_id>/`, one index row, one set of embeddings.
6. **Post-process** (background, per group): scene update; member-in-group
   profile updates, throttled ([Section 7.4](#74-member-in-group-profile)); the
   group's vector state; the group profile in Step 2, batched
   ([Section 7.5](#75-group-profile-step-2)).

The user-profile strategy never runs for a group owner: group episodes are owned
by the group, so they never enter a user's episode index, and personal profiles
stay fed by one-to-one conversations only.

### 7.2 Group episode

The extraction call is the one that runs today; only what it is given changes.

- **Speaker labels.** When the memcell items are built for a group session,
  each sender is labelled with their roster `display_name`, and assistant
  members as `"<display_name> (assistant)"`. The transcript line format is
  unchanged; it already names the speaker on every line.
- **Roster block.** A short block is appended after the operator's custom
  instructions for group segments. It lists the segment's speakers and the
  members named in it, with titles, and states two rules:

  ```
  This conversation is a group chat in "Trip team". People in this part: Chen (owner,
  Organiser), Betty (member). "Ava (assistant)" is the group's AI assistant.
  Attribute every statement to the person who made it. What the assistant says is the
  assistant's reply or suggestion; it is something the group agreed only when a member
  says or confirms it.
  ```

  The block is a versioned template in the prompt registry (`chat.group_roster_block`),
  so a model trained on registry prompts sees exactly this text.

After the call, code sets the attribution fields:

| Field | Rule |
|---|---|
| `sender_ids` | distinct senders of the segment's chat messages (from the memcell) |
| `subject_ids` | roster members (at the segment's time, former members included) whose `display_name` or an alias occurs in the episode's title or content: whole-word and case-insensitive for alphabetic scripts, substring for scripts without word spacing. If none occurs, the human senders. |
| `origin` | `member` if every chat message is from a human member, `assistant` if every one is from an assistant member, otherwise `mixed` |

Name matching is deliberate: it is cheap, testable and explainable. Its
precision and recall are measured on a labelled sample
([Section 13.5](#135-built-in-checks)) and it is the first thing to replace if
the sample shows misses.

### 7.3 Pictures

A picture in a group segment is stored once in `figures/` (unchanged) and gets
one image memory per occurrence under `groups/<group_id>/images/`, with
`sender_ids = [poster]`. Profile items the picture call returns are merged into
the **poster's** member-in-group profile only. The same routing (poster only)
also applies on the one-to-one and legacy paths, which removes the leak at
`user_memory.py:163-170` without changing any single-sender result.

### 7.4 Member-in-group profile

`groups/<group_id>/members/<member_id>.md` holds what the group's conversations
say about one member: preferences, constraints, commitments, expertise.

- **Operator.** The existing profile operator (everalgo `ProfileExtractor`),
  called with `sender_id = member_id` over the member's new segments in this
  group. Its prompt already scopes extraction to the target speaker and tells
  the model not to treat assistant suggestions as the person's traits.
- **Input.** Only this group's segments in which the member spoke. Never the
  member's personal memory, never another group.
- **Throttle.** A member's profile is updated after `[groups] member_profile_every`
  new segments in which they spoke (default 5), and when the group's buffer is
  flushed. Episodes are searchable immediately; only the profile lags.
- **Lock.** One lock per `(group_id, member_id)`, so updates for different members
  run in parallel and two updates for one member never lose each other's items.
- **Sources.** Every item records its source memcells (the 2.0 profile format).
  A check rejects an item none of whose sources contains a message by the member.
- **Versions.** When the operator's UPDATE rewrites a fact (`explicit_info`)
  by its index, the new item is a new value of the old one. It records the
  old item's id (`supersedes`) and the time of the update (`since`, the end
  of the update's span); the old item moves to the history with
  `replaced_at` and the new one's id (`superseded_by`). The link is exact:
  everalgo keeps a host's keys on the item an UPDATE op patches, so the
  rewritten item still carries the old `item_id`. No text is compared, and
  the prompt and the model's output are unchanged. A rewrite that only
  re-punctuates is not a new value, and an implicit trait is an inferred
  pattern, not a stated value: a rewrite of one sharpens the inference and is
  not linked. COMPACT and REGROUP write items without host keys and a dedupe
  keeps the item already on file, so neither is shown as a value change; a
  `delete` with an `add` is not linked either, since the reply does not say
  they are one fact, and neither is an `add` the model chose where an
  `update` would have fit. When COMPACT or REGROUP runs in the same call as
  the UPDATE that rewrote an item, the rewritten item comes back without host
  keys and is not linked.
- **Reading versions.** The cascade puts the values an item replaced on the
  item in the index row: `previous`, newest first, at most two, each with
  `item_id`, `description`, `until` and the old item's span. A read shows
  them up to the first one outside the asker's windows; an item left with
  none shows no `since` or `supersedes`, as a decision whose newer version
  the asker cannot see names none. A value a forget took out of the history
  is not projected, so it is never shown again, as a previous value either.
  A forget by record id reaches a replaced value by the id a read lists. The
  history keeps 200 entries and never drops one a current item still shows.
- **Limit.** Facts about a member stated only by others reach the group profile
  (Step 2) and the episodes, not this profile, because the operator reads the
  member's own lines.

### 7.5 Group profile (Step 2)

The group profile is a set of append-only items. A change never edits an item:
it appends a new one that names the item it replaces.

| Kind | Fields | Status values |
|---|---|---|
| `topic` | `topic_id` (stable across versions), `name`, `summary`, `status` | `exploring`, `disagreement`, `consensus`, `implemented` |
| `decision` | `statement`, `topic_id`, `decided_by`, `agreed_by`, `assignees`, `scope` (`group` or member ids), `supersedes` | `current`, `superseded` (computed) |
| `dissent` | `statement`, `held_by`, `about` (decision or topic id), `resolved_by` | `open`, `resolved` (computed) |
| `role` | `member_id`, `role` (one of `decision_maker`, `opinion_leader`, `topic_initiator`, `execution_promoter`, `core_contributor`, `coordinator`, `info_summarizer`), `confidence` (`strong`, `weak`) | `active`, `retired` |

Every item also carries `sources` (memcell id, message id, speaker id, verbatim
quote), `subject_ids`, `sender_ids`, `origin`, `source_start`, `source_end` and
`observed_at`.

**Update call.** A background strategy runs per group after
`[groups] group_profile_batch` new segments (default 5) or after
`[groups] group_profile_idle_seconds` without new segments (default 600), and at
flush. One call per batch. Its input:

- the roster block;
- the current items, rendered with their ids: active topics (at most 20, most
  recently active first; replaced topics stay stored but are not rendered),
  their current decisions, open dissent and active roles;
- the new segments, each opened by `=== SEGMENT <memcell_id> ===`, one line per
  message: `[<message_id>] <time> <speaker>: <text>`.

Its output is a list of operations: `topic.new` (with a label of its own in
`topic_id`: `new_1`, `new_2`, ...), `topic.update` (with the old `topic_id`),
`decision.new`, `decision.supersede` (with the replaced item id),
`dissent.open`, `dissent.resolve`, `role.add`, `role.retire`. Each operation
cites evidence as `(memcell_id, message_id, quote)`. A decision's `topic_id`
and a dissent's `about` name a rendered item, or a topic the same reply
creates by its label, before or after the `topic.new`. Code mints every topic
id and maps the label to it.

**Verification.** Code checks every operation and drops (never repairs) one
that fails:

- the memcell id is one of the batch's segments and the message id belongs to it;
- the quote is a verbatim substring of that message's text, after whitespace normalisation;
- the speaker is taken from the message, never from the model;
- a `decision.*` operation has at least one evidence message from a human member;
- an id it updates, supersedes, resolves or retires is one of the rendered
  current items. A topic the reply creates is written as it stands at the end
  of the batch, so its label is no target: a `topic.update` of it in the same
  reply is dropped.

A link never drops its operation: a `topic_id` or `about` that names neither a
rendered current item nor a topic the reply creates is dropped, and the
operation is kept as if the field were empty (a superseding decision keeps the
replaced decision's topic). Dropped operations are counted in metrics by
reason (`everos_group_profile_ops_dropped_total`), dropped links likewise
(`everos_group_profile_links_dropped_total`: `unknown_reference`, or
`ambiguous_reference` for a label two new topics share).

**Reading the profile.** The current profile is the fold of the items: the
newest version of each topic, decisions not superseded, open dissent, active
roles. Superseded items stay readable through history and enter answers in
history mode.

### 7.6 Assistant-generated content

The assistant is a member with `kind = assistant`. Its messages are memory: a
summary it posted, a list of options it gave. They never count as something the
members established:

- in episodes, through the speaker label and the roster-block rule, plus `origin`;
- in member-in-group profiles, through the operator's own rule and the source check;
- in the group profile, through the decision-evidence check;
- in search results and constraints, through `origin`, so an answer can say
  "the assistant suggested" rather than "the group decided".

### 7.7 Scenes and vector state

2.0 scenes run per owner, so a group gets its own scenes under
`groups/<group_id>/scenes/`, built from its episodes exactly as for a user. The
group also gets its own state for the query encoder, updated from its episodes.
A group-scoped query is encoded with the group's state, never with the asker's
personal state. The state only affects ranking; it never returns content.

### 7.8 Cost

Per segment with `s` distinct human senders:

| Work | One-to-one path with several senders (today) | Group path |
|---|---|---|
| Episode extraction calls | 1 | 1 |
| Episode entries, index rows | `s` | 1 |
| Episode embeddings (body and subject) | `2s` | 2 |
| Picture memories per picture | `s` | 1 |
| Scene updates | `s` | 1 |
| Profile extraction calls | `s` | `s / member_profile_every`, amortised |
| Group-profile calls (Step 2) | none | `1 / group_profile_batch`, amortised |

The [GroupAssistBench](#131-groupassistbench) cost probe measures the real
numbers per message before any full build.

---

## 8. Read path

### 8.1 Scope resolution

A search first resolves its scope through the access policy
([Section 9.3](#93-the-check-and-where-it-runs)): the groups, the asker's
windows in each, and the status and version rules of 2.0. The result is one
predicate compiled into the `where` of every recall. The multi-round decider
never sees a candidate the asker cannot see: visibility is applied in the index
query, not after ranking.

### 8.2 Asker conditioning

The same question can have different answers for different askers, for three
reasons:

1. **Windows.** Each asker's windows select different records.
2. **The question names the asker.** The question the multi-round loop starts
   from is rendered as `"<display_name>: <question>"`, the same line shape the
   decider reads candidates in. "What do I still have to do?" asked by Betty
   recalls Betty's commitments, and the decider reads it as Betty's question.
3. **Profiles.** With `include_profile`, the asker's member-in-group profile is
   attached, plus those of members the query names (roster names and aliases,
   at most three).

Groups in a multi-group scope are searched as one candidate pool: the predicate
is a disjunction over the groups, each with its own windows.

### 8.3 Relational expansion (Step 2)

A request often does not name the memory it needs ("draft the reminder" needs
the revised date). After each round, the loop follows stored links from the
round's core items:

| Link | Followed to |
|---|---|
| scene membership | the scene's other episodes, and the scene itself |
| decision chain | the newest version of a superseded decision and its source segment's episode; open dissent about it |
| shared topic | the topic's current decisions |
| subjects | the most recent records about the same members (bounded) |

Linked records that pass the visibility predicate join the next round's
candidates in reserved slots (`[search] link_slots`, default 5), inside the
block of the sub-query that surfaced their anchor. They are rendered like any
other candidate, so the decider's input format does not change. Expansion uses
ids and links only, never the request's wording.

### 8.4 Constraint block (Step 2)

After the loop stops, the response carries a constraint block: what the answer
must respect.

```json
{
  "id": "c1",
  "statement": "Depart on April 30 instead of May 1.",
  "kind": "decision",
  "subject_ids": ["u_chen", "u_betty"],
  "scope": {"group_id": "trip-team", "applies_to": "group"},
  "status": "current",
  "supersedes": [{"statement": "Fly out on May 1.", "at": 1772359200000}],
  "decided_by": ["u_chen"],
  "agreed_by": ["u_betty"],
  "origin": "member",
  "source": {
    "memcell_id": "mc_7f3a",
    "message_id": "m-102",
    "speaker_id": "u_chen",
    "at": 1772442120000,
    "quote": "Let's leave on April 30 instead."
  },
  "audience": {"present": ["u_chen", "u_betty", "bot_ava"], "joined_after": ["u_evan"]}
}
```

It is built in two layers:

1. **Rules (default).** Group-profile items linked to the core (decisions,
   dissent, topic status) become constraints directly: their quotes, speakers and
   statuses were verified at write time. Core episodes and picture memories not
   covered by an item get a metadata constraint (subjects, status, source segment,
   speakers, times) whose quote is the source message with the highest token
   overlap with the record.
2. **Model (optional).** `[search] constraints = "model"` adds one call with the
   extractor role: given the question, the core and the rule constraints, it
   writes applicability and scope per constraint, with exact quotes. Code
   verifies every quote against the cited message and drops the ones that fail,
   then removes duplicates. Whether this becomes the default is decided on
   GroupAssistBench L2.

### 8.5 Partial memory (Step 2)

An open request ("write the reminder", "summarise where we are") is answerable
from partial memory. EverOS does not return an empty result because coverage is
incomplete:

- the response always carries the core items assembled so far (core first, as today);
- `coverage` says `complete` (the decider stopped with a core), `partial` (the
  round budget ran out, or the loop ended on a round the decider did not
  decide -- listed in `degraded`) or `none` (no core);
- for group scopes the current group-profile items are attached
  (`include_group_profile`), which grounds requests about the group itself
  (purpose, current plan, who does what);
- the documented answer prompt for applications tells the model to complete open
  requests from what is known, name what is missing, and decline only when
  neither memory nor general knowledge supports the request. Benchmarks keep their
  official answer prompts unchanged.

The answer prompt for applications (EverOS does not call it; an application
puts the search response where it says `{memory}`):

```
You answer for the group's assistant. {asker} asks: {question}

Memory from the group (JSON: "episodes" and "images" are the records found,
the decider's core first; "group_profile" is the group's current topics,
decisions, open disagreements and roles; "constraints" are what the answer
must respect, each with the message it rests on; "coverage" says how complete
the search was):
{memory}

Rules:
1. Respect every constraint. Never present a decision whose status is
   "superseded" as current; when a decision is "disputed", say who disagrees.
2. A record whose origin is "assistant" is something the assistant said or
   suggested, not something the group decided.
3. Complete an open request -- a reminder, a summary, a plan -- from what is
   known. When "coverage" is "partial" or "none", or memory covers only part
   of the request, do that part and say plainly what is missing.
4. Use general knowledge where memory is silent, and say which parts come
   from it.
5. Decline only when neither memory nor general knowledge supports the
   request.
6. People in a record's "audience.joined_after" joined after it was said:
   write for them as people who have not heard it.
```

### 8.6 Who knew what (Step 2)

From the roster and record times, every group record gets:

- `audience.present`: members whose windows cover all of its sources;
- `audience.joined_after`: current members who joined after its last source.

Both are computed at read time (nothing new is stored) and shown only on
records the asker can see. "Who still has not heard about the new date?" is
answered from `joined_after` on the current decision.

### 8.7 Project roll-up (Step 2)

- `project_scope: true` searches every group of the project the asker currently
  belongs to, as one pool.
- `GET /api/v2/projects/{project_id}/profile?asker_id=` returns the visible
  current group-profile items of those groups, each tagged with its group, in
  time order. Duplicates across groups are not merged.
- The project's knowledge base is readable by project members. A project without
  groups keeps today's knowledge behaviour.

---

## 9. Visibility

### 9.1 Rules

| Memory | Visible to |
|---|---|
| Personal memory (`users/<u>/`) | `u` only |
| Group memory (`groups/<g>/`: episodes, picture memories, scenes, member-in-group profiles, group-profile items, buffered messages) | current members of `g`, and only records all of whose sources fall inside one of the asker's membership windows in `g` |
| Roster and group meta | current members of `g` |
| Project knowledge base | project members, when the project has groups |
| Agent memory | the agent; cases extracted from a group segment carry `source_group_id` and are returned only to reads in that group's context |

Consequences, stated explicitly:

- A former member has no access to the group's memory. Their personal memory is unaffected.
- A member who joined later sees nothing from before their first window.
- A record with several sources (a scene, a profile item) is visible only if every source is.
- The window test for a record is `source_start >= from AND source_end < to` for one of the asker's windows (no upper bound for the open window).
- Invisible records behave as absent: a get by id returns 404, a search omits them, a picture hash referenced only by invisible group memories returns 404.
- An owner or admin role grants roster management and forgetting, not extra reading.

### 9.2 Identity

Every group request carries `asker_id`. EverOS has no authentication
([SECURITY.md](../SECURITY.md)): the identity is the caller's assertion. A
deployment that serves several people puts a gateway in front and sets
`[access] identity_header`; EverOS then reads the asker from that header and
rejects a request whose body `asker_id` differs (403). One-to-one requests
without `asker_id` are read as asked by their owner, which is today's behaviour.

The CLI runs in-process as the operator. Background strategies run with an
internal identity scoped to the owner they work on. Neither is reachable through
the HTTP API.

### 9.3 The check and where it runs

One module, `src/everos/memory/access/`, holds the policy:

- `read_scope(identity, request) -> ReadScope`: the owners, the per-group
  windows, and the compiled predicate (it replaces the owner part of
  `compile_filters`, `src/everos/memory/search/filters.py:51-72`);
- `visible(identity, record) -> bool` for reads that do not go through the index
  (markdown reads, picture bytes);
- `authorize(identity, action, target)` for roster changes, forget, undo and delete.

Every read path calls it:

| # | Read path | Today | With the policy |
|---|---|---|---|
| 1 | Search: every recaller, the multi-round loop, expansion lookups | owner predicate only | `read_scope` predicate in every `where` |
| 2 | Search: pictures attached to episodes | attached by reference | attached only if visible |
| 3 | Search: buffered messages of a session (`manager.py:334-351`) | all senders | the session's group window; one-to-one only for a sender of the session |
| 4 | Search: attached profiles | owner's profile | member-in-group profiles of the scope; personal profile only in one-to-one scope |
| 5 | Get (`src/everos/memory/get/`) | owner predicate | `read_scope` predicate |
| 6 | History | n/a | visible versions only |
| 7 | Picture bytes (`routes/images.py:233-264`) | no check | a hash referenced by any live group picture memory needs an asker who can see one of them; personal-only hashes behave as today |
| 8 | Forget, undo, revise, delete | owner given by caller | `authorize`, then the same scope for what is touched |
| 9 | Knowledge search and get | per (app, project) | project members when the project has groups |
| 10 | Group and project reads | n/a | current members |
| 11 | Background strategies | per owner | group strategies read only the group's own segments; member-in-group profiles never read `users/<m>/`; personal strategies never read group segments |
| 12 | Agent-memory reads | per agent | group-sourced cases filtered by the read's group context |
| 13 | MCP | through the HTTP API | same as the API; the MCP server forwards `asker_id` |

The test suite in [Section 13.5](#135-built-in-checks) targets every row.

### 9.4 Forget and delete propagation

Forgetting stays a reversible mark (2.0). Inside a group it travels from what
was forgotten to everything derived from it (Step 2):

| Forgotten | Effect |
|---|---|
| A message | Recorded in `message_forget` and the forget log. Each segment containing it is re-extracted without it: the new episode version supersedes the old; picture memories of pictures in it are forgotten; scenes containing the old episode get a new version. |
| A member's messages | As above for each of them; the member-in-group profile of that member is forgotten. |
| A session or segment | Its episodes and picture memories are forgotten; derived records follow the source rule below. |
| A record | That record is forgotten; derived records follow the source rule. |
| A picture | Its group picture memories are forgotten and referring episodes re-extracted without it; the file stays while another live memory (any owner) refers to it. |

**Source rule** for derived records (scenes, member-in-group profile items,
group-profile items): forgotten when all of their sources are forgotten;
otherwise marked `stale` and rewritten from the remaining sources by the next
update, which appends a new version. Re-extraction and quotes skip forgotten
messages; the memcell payload keeps them so undo can restore them.

Undo restores the marks and makes the previous versions current again. Delete
(owner/admin) runs the same propagation and removes instead of marking.

---

## 10. One-to-one conversations stay unchanged

- A batch without `group_id`, for a session that is not bound to a group, runs
  today's path: same extraction, same per-sender files under `users/<sender>/`,
  same index rows.
- A batch without `group_id` that carries several human senders keeps the
  per-sender behaviour and logs `memorize.multi_sender_without_group` once per
  session. Group chats must send `group_id`.
- Search, get and forget with `user_id` behave as today; `asker_id` is optional
  and defaults to the owner.
- Two changes reach this path, both visibility fixes. D17: profile items read
  from a picture go to its poster; with a single human sender this is the same
  sender, so the output is identical. And a session-filtered search returns
  buffered messages only when the owner is a sender of that session (today it
  returns them for any session id).
- Regression gate: with a fake model and fixed embeddings, one-to-one fixtures
  (LoCoMo and LongMemEval conversations) produce byte-identical markdown and
  identical index rows before and after the change, and the full LoCoMo and
  LongMemEval runs score the same within run-to-run noise.

---

## 11. Where the prompts live

The architecture places algorithm assets in everalgo, which EverOS consumes as
the `everalgo-*` packages. Two 2.0 prompts already live in EverOS: the
picture-memory prompt (`src/everos/memory/extract/pipeline/image_memory.py`),
because everalgo's message type is text-only, and the multi-round decider prompt
(`src/everos/memory/search/llm_multiround.py`), because the decider is a
pluggable hook of the EverOS retrieval loop. A new everalgo operator also needs an
everalgo release before EverOS can depend on it. Group memory therefore follows
the same path:

| Prompt | Step | Lives in 2.0 | After 2.0 |
|---|---|---|---|
| Group episode | 1 | No new prompt: everalgo `EpisodeExtractor` (text segments) and the EverOS picture-memory prompt (segments with pictures), unchanged. Group information enters as data: speaker labels and the roster block. | unchanged |
| Roster block | 1 | `chat.group_roster_block`, a rendering template in the EverOS prompt registry | moves with the operators below |
| Member-in-group profile | 1 | No new prompt: everalgo `ProfileExtractor` with `sender_id = member_id` | unchanged |
| Group-profile update | 2 | `chat.group_profile_update` in EverOS: `src/everos/memory/extract/group_profile/` (prompt, reply schema, verifier, fold) | an everalgo-user-memory operator (`GroupProfileUpdater`) under the same registry id |
| Constraint binding (optional model layer) | 2 | `search.constraint_bind` in EverOS: `src/everos/memory/search/constraints/` | an everalgo operator under the same registry id, if the model layer becomes the default |

Every prompt and template is registered with an id, a version and a fingerprint
and can be overridden like the other 2.0 prompts. Keeping the registry id when a
prompt moves to everalgo keeps data built from the registry valid. Step 1 needs
no everalgo release at all.

---

## 12. Reuse map from EverMemOS

Paths are relative to the EverMemOS repository root (`src/`). "Reused" means
the design or text is carried over; nothing is imported.

### Reused as is

| EverMemOS | What | Where it goes in 2.0 |
|---|---|---|
| `memory_layer/memory_extractor/group_profile_memory_extractor.py:79-85` | topic status `exploring` / `disagreement` / `consensus` / `implemented` | `topic.status` |
| `memory_layer/memory_extractor/group_profile_memory_extractor.py:67-76`; `memory_layer/prompts/en/group_profile_prompts.py:211-241` | the seven observed group roles with definitions, evidence and `strong`/`weak` confidence | `role` items |
| `memory_layer/prompts/en/group_profile_prompts.py:15-21`; `memory_layer/memory_extractor/group_profile/data_processor.py:269-301` | evidence cited by segment ids from `=== MEMCELL_ID ===` markers | the update call's segment markers |
| `memory_layer/memory_extractor/group_profile_memory_extractor.py:360-364`; `memory_layer/memory_extractor/group_profile/data_processor.py:23-110` | accept only evidence ids present in the input | first rule of the verifier |
| `memory_layer/prompts/en/group_profile_prompts.py:76-92`; `memory_layer/memory_extractor/group_profile/topic_processor.py:106-272` | incremental topic protocol: the model answers `new` or `update` with the old topic id; the system mints ids | `topic.new` / `topic.update` |
| `memory_layer/prompts/en/group_profile_prompts.py:71`, `:91-117` | what counts as a topic (substantive threads) and what does not (scheduling, greetings, bot notices, confirmations) | the update prompt's topic rules |
| `api_specs/dtos/conversation_meta.py:25-59` | participant details: full name, `user` / `assistant`, job role | roster `display_name`, `kind`, `title` |

### Adapted

| EverMemOS | What it did | 2.0 form |
|---|---|---|
| `api_specs/request_converter.py:227-272` | `group_id` / `group_name` on every message | `group_id` on the add request; the session is bound once (D18) |
| `api_specs/dtos/conversation_meta.py:93-127`, `:379-420` | conversation meta create and patch, with participants | group upsert and roster events with join and leave times (D2) |
| `biz_layer/mem_memorize.py:613-628` (the group task) | one group episode per memcell with `user_id = None` | a group-owned episode (D1, D3) |
| `biz_layer/mem_memorize.py:589-610` | `assistant` scene: only the group episode | no `group_id`: the unchanged one-to-one path (D14) |
| `api_specs/memory_models.py:152-171` | `(user, group)` profile | member-in-group profile under the group (D6) |
| `agentic_layer/memory_manager.py:150-214`; `infra_layer/adapters/out/search/repository/episodic_memory_es_repository.py:257-278` | search by `user_id` and/or `group_id`, with a participant filter | group scope plus asker with membership windows (D9, D10) |
| `agentic_layer/memory_manager.py:265-292` | pending (unconsumed) messages returned with search, filtered by user and group | buffered messages filtered by the asker's window (Section 9.3, row 3) |
| `memory_layer/memory_extractor/group_profile/role_processor.py:23-166` | role assignments merged with historical evidence, strong before weak | `role.add` / `role.retire` items with evidence |

### Dropped, or replaced because 2.0 overrides it

| EverMemOS | What | Why |
|---|---|---|
| `biz_layer/mem_memorize.py:619-626`; `memory_layer/memory_extractor/episode_memory_extractor.py:335-377` | one personal episode per participant, in addition to the group episode | **Overrides** (D3): `k` extra model calls; the English personal and group prompts are identical (`memory_layer/prompts/en/episode_mem_prompts.py:10` and `:75`), so the copies differ only by sampling. |
| `memory_layer/memory_extractor/group_profile/topic_processor.py:25-60`, `:251-262` | at the topic cap, delete the oldest topic (implemented topics older than 30 days first) | **Overrides** (D7): topics are never deleted; the same ordering decides only which topics are rendered into the update prompt. |
| `infra_layer/adapters/out/persistence/document/memory/group_profile.py:80-86`; `.../group_user_profile_memory.py:27-32` | profile documents rewritten with `version` / `is_latest` | **Overrides** (D7): append-only items with `supersedes`. |
| `memory_layer/memory_extractor/group_profile/llm_handler.py:27-111` | two parallel calls per extraction (content, behaviour) | **Overrides** (D8): one call per batch; decisions and dissent added. |
| `memory_layer/prompts/en/group_profile_prompts.py:267-312` | aggregation of daily profiles into longer periods | Append-only items need no aggregation. |
| `memory_layer/memory_extractor/group_profile_memory_extractor.py:271-294` | skip groups without a name unless an environment variable is set | Every group is processed. |
| `memory_layer/memory_extractor/group_profile_memory_extractor.py:124-127` | topic id from an MD5 of the topic name | Minted ids; renaming a topic keeps its id. |
| `infra_layer/adapters/out/search/repository/episodic_memory_es_repository.py:257-275` | `MAGIC_ALL` wildcard owner filters | **Overrides** (D9): incompatible with a single visibility check. |
| `api_specs/memory_models.py:192-205` | combined group and global profile in one response | **Overrides** (D6): personal memory never enters a group read. |
| `infra_layer/adapters/out/persistence/document/memory/group_user_profile_memory.py:40-95` | fixed work-profile fields (hard skills, soft skills, decision style, ...) | The 2.0 profile item format with categories. |
| MongoDB, Elasticsearch and Milvus repositories for group records | storage | Markdown, SQLite and LanceDB. |

Note: in that snapshot the group-profile extractor is complete
(`memory_layer/memory_manager.py:199-205`, `:362-380`) but no memorize path
requests it, and the API lists the group profile as not implemented
(`api_specs/memory_models.py:106`). Its design is reused; its behaviour has not
been exercised at scale, which is why the group profile is evaluated on its own
([Section 13](#13-evaluation)).

---

## 13. Evaluation

Each benchmark is run and reported on its own; scores from different benchmarks
are never added or averaged. Evaluation-side protocol (answer model, answer
prompt, judge, parameters) follows each benchmark's official setup; the memory
system's output goes into the official context slot.

### 13.1 GroupAssistBench

Real group chats with a deployed assistant ([Hu et al., 2026](#16-references)):
50 groups, 80,299 messages, 2,395 instances. L1 tests recall against reference
facts (Atom Recall, factual F1); L2 tests use with per-instance rubrics whose
penalties come from real user feedback (rubric score `S`). M1 / M2 / M3 split
single, multiple and conflicting memories. The paper reports the predecessor,
EverMemOS, above full context on recall (L1 0.621 against 0.570 with the same
task model) and far below it on use (L2 0.247 against 0.431); across the memory
systems it tested, 64% of the memories that went unused were already in the
model's input. Its two interventions, relational evidence refinement and
assistance constraint binding, raise EverMemOS's L2 by 4.43 points together.

Data facts that shape the adapter (computed on the released files): messages
carry a speaker label and a turn index, with no timestamps and no sessions;
15,096 messages are the assistant's; groups have 68 to 4,442 messages (median
1,202.5); every instance has a cutoff turn, 2,194 distinct cutoffs in all (3 to
106 per group); 1,159 instances are native (posed by a member at a known turn)
and 1,236 authored. The data is CC BY-NC 4.0 and the code is for academic use:
users download the data themselves, it is used for evaluation only, and nothing
from it is committed.

Adapter (`benchmarks/adapters/groupassistbench.py`):

- **Mapping.** One group per GroupAssistBench group, one session per group;
  every speaker label is a member from the group's start (the data has no join
  times), `bot` is the assistant member; `message_id` is the turn index;
  timestamps are synthetic and monotonic (`t0 + turn * 60 s`).
- **Cutoffs.** Streaming ingestion: ingest turns below the next cutoff, flush,
  wait until the index has no pending changes and the group's background queue is
  idle, ask every instance with that cutoff, continue. The harness asserts, from
  memcell timestamps, that nothing at or after a cutoff was ingested when its
  questions were asked.
- **Asker.** For native instances, the member who posted the source turn; for
  authored ones, an evaluator member present from the start.
- **Context.** The official answer prompt (Table 16 of the paper) verbatim,
  with EverOS's result in its `{context}` slot, rendered as `#<turn> [<speaker>]`
  lines so the prompt's own rule about indices applies.
- **Judging.** The official L1 and L2 judge prompts and model, verbatim and
  hash-pinned.
- **Cost probe first.** Build one group of median size (1,154 or 1,251 messages),
  record tokens per stage from traces, extrapolate by message count, and get the
  full run approved.
- **Report.** Overall and by M1 / M2 / M3 and native / authored; L2 also reports
  the share of responses that decline.
- **Baseline.** The same build models with every group ingested under one owner,
  the way benchmarks are ingested today.

Targets: L1 not below the published predecessor (0.621 overall; 0.679 / 0.540 /
0.561 on M1 / M2 / M3). L2 above the published predecessor with both
interventions (0.247 + 0.0443 = 0.291); stretch target: full context with the
same task model (0.431). Ablations in Step 2: without expansion, without
constraints, and `constraints = "rules"` against `"model"`.

### 13.2 EverMemBench, per member

5 topics, each with three concurrent group chats; 170 people; 2,400 questions.
The current adapter maps a whole topic to one owner
(`benchmarks/adapters/evermembench.py:142-143`) and sends every message under
that owner's id (`:186-193`). The per-member adapter:

- one project per topic, one group per group chat (from the raw release's
  per-date, per-group dialogues), one member per speaker with a stable id and the
  raw name as `display_name`;
- questions asked by an evaluator member of all three groups from the start (the
  benchmark models no access rules), searching with `group_ids` over the three
  groups;
- the asker a question names in its opening clause (434 of the 2,400) is matched
  as a named member, so that person's member-in-group profiles are attached;
- the existing answer and judge protocol; report the nine-category mean (the
  benchmark's convention) and the question-weighted total, plus accuracy by the
  number of groups a question involves.

Target: the nine-category mean not below the collapsed-owner baseline, with the
profile categories (style, skill, title) improved.

### 13.3 GateMem

Shared-memory governance across principals ([Ren et al., 2026](#16-references)):
91 episodes, 2,218 checkpoints (728 utility, 727 access control, 763
forgetting), scored as `MGS = U * (1 - A) * (1 - F)`. Its updates, permission
changes and deletion requests arrive as natural language inside the
conversation. The adapter:

- maps principals to members and their roles to access roles; each principal's
  protected records go to that principal's personal scope, shared records to the
  group;
- turns permission changes into roster events and deletion requests into forget
  calls, through a detector in the harness; reports both end-to-end and with the
  benchmark's labelled operations if the release has them;
- reports U, A, F, MGS and over-refusal, plus time and tokens per checkpoint.

Access rules that depend on relationships (a guardian reading a dependant's
record, delegated access) are outside the 2.0 model and reported separately.
First task: confirm the data release and its licence. Target: A and F close to
zero on the mapped part, U not below the long-context setting with the same model.

### 13.4 Other sets

H2HMem has five multi-party group chats, today ingested one owner per dialogue;
they are rebuilt with real members and reported next to the existing numbers.
LoCoMo and LongMemEval are the one-to-one regression gate of
[Section 10](#10-one-to-one-conversations-stay-unchanged).

### 13.5 Built-in checks

- **Leak suite.** Generated rosters and records cover: a late joiner, a member who
  left, a member who rejoined, a non-member, a personal record read by another
  user, a picture hash probe, a session-buffer read, an expansion link into an
  invisible record, a constraint or profile item whose source is invisible, an
  agent case from another group. Every case asserts the record is absent from
  search, get, history, attachments, constraints and picture bytes. Zero leaks is
  the pass mark.
- **Attribution sample.** 100 group segments labelled by hand for senders,
  subjects and origin; report precision and recall of `subject_ids` and the rate
  at which assistant statements are written as member decisions.
- **Cost.** Calls, tokens and embeddings per segment on the group and the
  one-to-one path, from traces.
- **Latency.** Search time with group scopes against one-to-one on the same
  store size.

---

## 14. Task list

Ordered; each task lists its acceptance tests.

### Step 1

1. **Owner type and roster store.** `OwnerType` gains `group`; models for group,
   interval and roster event; `group.md` writer and reader; SQLite
   `group_meta`, `group_member`, `session_binding`; cascade owner resolution for
   `groups/`.
   - Interval folding: join, leave, rejoin, backdated join; leave without join and a second open interval are 409.
   - `group.md` round-trips; a rebuild from a wiped `.index/` restores identical roster rows.
   - A file under `groups/<g>/episodes/` indexes with `owner_type = group` (fails on today's resolver).
2. **Group API and add fields.** Group upsert, roster events, group read; `group_id`, `roster`, `message_id` on add; implicit create and join; session binding.
   - The OpenAPI document lists the new routes and fields.
   - An unknown `group_id` creates the group; an unknown sender joins at its first message with `source = implicit`.
   - A session bound to one group answers 409 for another; a batch without `group_id` inherits the binding.
   - `data/team_chat_en.json`'s `session_meta.user_details` imports as a roster: five members with titles.
3. **Access policy.** `memory/access` with `read_scope`, `visible`, `authorize`; `[access] identity_header`.
   - Property test over random rosters and record times: a record is returned iff the asker is a current member and all its sources lie in one of the asker's windows.
   - A former member gets 403 on group reads; an invisible id and a missing id return the same 404 body.
   - A personal read with a different `asker_id` and a header mismatch both return 403.
4. **Group write path.** Group render (labels, roster block), attribution fields, writers and index columns, pictures once per group, poster-only picture profile items, roster-change cut.
   - With a fake model, a segment with three human senders yields one entry, one index row and two embedding calls on the group path, against three entries and six calls on the one-to-one path.
   - `sender_ids`, `subject_ids` and `origin` match fixtures, including an assistant-only segment and a former member named in the text.
   - A picture yields one memory with `sender_ids = [poster]`; its profile items reach only the poster (a test that reproduces today's leak on the multi-sender path now passes).
   - No segment spans a roster change sent while messages were buffered.
   - An agent case extracted from a group segment carries `source_group_id`.
5. **Member-in-group profile.**
   - One `members/<m>.md` per speaking member; every item cites source segments containing a message by `m`.
   - Two concurrent updates for one member lose no item; updates for two members run in parallel.
   - `member_profile_every` is honoured; the member's `users/<m>/user.md` is untouched by group segments.
6. **Group scenes and vector state** (on top of 2.0 scenes).
   - Scenes are written under `groups/<g>/scenes/`; a scene is withheld from an asker who cannot see all its episodes, while those episodes still return individually.
   - Group queries use the group's state.
7. **Group search, get, history, buffer, picture bytes.**
   - Two askers with different windows get different results for one query on a fixture.
   - The decider's question carries the asker's display name; `include_profile` returns the asker's and named members' in-group profiles.
   - A session-filtered search returns no buffered message outside the asker's window.
   - A picture hash referenced only by group memory returns 404 to a non-member.
   - `group_ids` over several groups returns one ranked pool.
   - An agent case sourced from one group is not returned to a read in another group's context or a one-to-one context.
8. **Leak suite and one-to-one gate.** The suite of [Section 13.5](#135-built-in-checks) passes with zero leaks; the [Section 10](#10-one-to-one-conversations-stay-unchanged) golden test and score gate pass.
9. **EverMemBench per member.** Adapter and run as in [Section 13.2](#132-evermembench-per-member).
   - Five projects with three groups each, real member ids; askers parsed for 434 questions.
   - The nine-category mean is not below the collapsed-owner baseline; per-category and by-group-count tables are reported.
10. **GroupAssistBench adapter and L1.**
    - The cost probe is reported and approved before the full build.
    - The cutoff assertion holds for every instance; official prompts are hash-pinned.
    - L1 Atom Recall and F1 are reported by M1 / M2 / M3 and native / authored.

### Step 2

11. **Group profile.** Update strategy, prompt, verifier, writer, index table, fold.
    - Crafted replies with an unknown segment id, a paraphrased quote, a wrong speaker, or assistant-only decision evidence are each dropped and counted.
    - A decision or dissent that names a topic created in the same reply, by its label and before or after the `topic.new`, links to the minted topic id; a link that names nothing is dropped and counted, the item kept; targets stay strict.
    - Superseding a decision appends the new item; the old one indexes as superseded and both read through history.
    - Dissent moves from open to resolved; no topic is ever deleted; the batch and idle triggers fire.
12. **Forget and delete in groups.**
    - A message forgotten by its sender re-extracts its segment, and its text appears nowhere afterwards: not in markdown, index rows, search results, profiles or constraints.
    - Another member's attempt is 403; an owner or admin may forget any message; undo restores the previous versions.
    - A member who leaves stays visible to the others; `dry_run` counts match the real run.
13. **Relational expansion.**
    - On a fixture where the query matches only an old decision's wording, the newest version enters the next round.
    - Expansion never adds an invisible record; the slot cap holds; a snapshot test shows the decider's input format unchanged.
14. **Constraint block.**
    - Every returned quote is a verbatim substring of its cited message; constraints that fail are absent.
    - Status is right on conflict fixtures (current, superseded, disputed); the model layer runs only when configured.
15. **Partial memory.**
    - `coverage` takes each of its three values on fixtures; group-profile items are attached for group scopes.
    - The application answer prompt is documented; the decline share on GroupAssistBench L2 is reported.
16. **Who knew what.** `audience.present` and `audience.joined_after` are right on roster fixtures and appear only on visible records.
17. **Project roll-up.**
    - `project_scope` covers exactly the asker's current groups in the project; the project profile is the union of visible items, tagged by group.
    - Knowledge reads require project membership when the project has groups.
18. **GroupAssistBench L2 and ablations** as in [Section 13.1](#131-groupassistbench).
19. **GateMem** as in [Section 13.3](#133-gatemem).

### Implementation notes (Step 1, tasks 1-3)

The first half of Step 1 differs from the text above in these places:

- **Entry ids.** Group logs use the prefixes `gep_` and `gim_`. A row id is
  `<owner>_<entry>`, so with `ep_` and `im_` a group whose id equals a user's
  id would share that user's row ids.
- **Creation time.** It is stored as `group_meta.group_created_at`, because
  `created_at` is the row's own timestamp.
- **Session bindings** are also logged in `group.md`, so a rebuild restores
  them.
- **Unknown groups.** Reading an unknown group answers 403, the same as a group
  the asker is not in, so a probe cannot tell which ids exist. Clients treat
  either answer as "not visible" and let the `PUT` decide.
- **Implicit joins.** Some messages are accepted without changing the roster:
  one older than a current member's join, and one from a former member that
  falls inside or before an earlier interval.
- **Inline roster events.** An event without `at` takes the batch's first
  message time. Like the roster route, a batch that carries roster events
  cuts the open buffer of every session of the group before the change
  applies (tasks 4-7 closed the gap where it cut only its own session).
- **Groups without an owner or admin** may be managed by any current member, so
  an implicitly created group can always be administered.
- **Route prefix.** The group routes are also mounted under `/api/v1`, like
  every other v2 route.
- **Picture bytes.** An asker who holds a personal memory of the same picture
  (the same hash) may also fetch it.
- **One search method.** Multi-round retrieval (`llm_multiround`) is the only
  search method in 2.0, for one-to-one and group scopes alike; any other
  method answers 422. Agent memory is listed by `/memory/get` (with
  `context_group_id`), not searched: a search with `agent_id` answers 422.
- **Client message ids** are kept only on group batches. They are not yet
  persisted in memcells.
- **Agent-scope buffer reads** follow the users' rule: the agent must be a
  sender of the session, or hold cases from it.
- **D17 also applies to one-to-one chats.** A picture the assistant posts adds
  no items to the user's profile.

### Implementation notes (Step 1, tasks 4-7)

The second half of Step 1 differs from the text above in these places:

- **No 2.0 scenes or query-encoder state on this line.** Task 6 builds on
  both, and neither exists here, nor yet on any other branch. Group scenes
  are therefore **off by default** (`[groups] scenes = false`) and are an
  opt-in until 2.0's per-owner scenes exist: until then a group would get a
  record type users do not have, and every group episode that joins a scene
  would cost ingestion one more reflector call. Switched on, group scenes
  are built from operators that do exist, with no new prompt text. Each new
  group episode is placed by everalgo's `cluster_by_geometry`, with the
  `[clustering]` threshold and time window that users' episode clusters use.
  The narrative comes from everalgo's `EpisodeReflector`: INIT over the
  members when the scene is new, then UPDATE with each new episode. The
  group's state for the query encoder is not built: no encoder on this line
  takes a state. So "group queries use the group's state" holds only in the
  weak sense that no query is encoded with a personal state.
- **Scene storage.** Scene versions are entries of daily logs,
  `groups/<g>/scenes/scene-<YYYY-MM-DD>.md` (entry ids `gsc_`), not
  `scenes/<scene_id>.md`. An entry records `scene_id`, `version` and
  `member_ids`. A new version marks the one it replaces in
  `deprecated_entries`, in the same locked write. The index rows are in the
  `episode` table with `parent_type = scene`; there is no `scene` table. So
  scenes compete with episodes in every recall, under the same window
  predicate. Search and get mark them `type = "Scene"`, and their
  `source.memcell_id` is null. A session-filtered search finds no scene,
  because a scene belongs to no session. `[groups] scenes = true` switches
  scenes on; they are off by default (see the note above).
- **Prompt registry.** There is no prompt registry on this line either. The
  roster block is a versioned constant next to its renderer
  (`memory/extract/pipeline/group_render.py`): id `chat.group_roster_block`,
  version 1, with a fingerprint the tests pin.
- **Member-in-group profile format.** There is no 2.0 profile format on this
  line. A member's profile uses `user.md`'s fields plus `group_id`,
  `member_id`, `pending_memcell_ids` (the throttle) and `history`. Each item
  carries `item_id`, `source_memcell_ids` and the span `source_start` /
  `source_end`. The profile operator does not say which segment an item came
  from, so an item's sources are all the segments of the update that added or
  rewrote it. The index row id is `<group_id>:<member_id>`. A new value of an
  item carries `supersedes` and `since`, its history entry `superseded_by`;
  the index row's items carry `previous` (section 7.4).
- **Member-profile visibility.** An item is shown when the span of its
  sources lies in one of the asker's windows. The summary is written over
  every item, so it is shown only when every item is.
- **Throttle on flush.** A flush of a session bound to a group publishes
  `GroupFlushed`, whether or not it cut a segment. Every member with pending
  segments is then updated.
- **Named members.** `include_profile` attaches the asker's profile and those
  of up to three members the query names, from each group of the scope.
- **Subjects.** Only human members are subjects; an assistant the text names
  is not. A picture memory's subjects are the members its description or OCR
  names, otherwise its human poster.
- **Picture time.** A group picture memory is dated by the message that
  carried the picture, not by the segment's end. Its window test uses that
  time.
- **Inline roster cut.** A batch that carries roster events holds the locks
  of every session of its group with its own, in sorted order, and cuts their
  buffers before the change applies. A busy session is skipped, as on the
  roster route.
- **Agent cases.** An agent read in a group's context is `agent_id` plus
  `context_group_id`. Its asker, the agent itself unless `asker_id` names
  another, must be a current member, or the read is 403. A group-sourced case
  carries no source span, so windows are not applied to it. Skills are not
  filtered by context.
- **Question rendering** applies to the whole multi-round loop, to the first
  round's recall as well as to the decider, and only to that method. It can
  be switched off with `[groups] asker_in_question`.
- **Get.** `image_memory` and `member_profile` are available on group scopes
  only: users have no picture listing on this line.
- **Table schema version 5.** Version 4 had already shipped with the first
  half. `user_profile.subject_id` and `agent_case.source_group_id` are
  therefore added by version 5. Milvus still has no migration step, so a
  Milvus store reports drift and needs `everos cascade rebuild`.
- **Episode reader.** Episodes read back from markdown now keep `sender_ids`
  and `language`. Scenes need the members' senders, and an undo used to drop
  both.
- **Conversation language.** Every group extraction call is told the
  conversation's language, resolved in code as on the one-to-one path: from
  the segment's human messages, and from its session's earlier segments when
  it has too little text of its own. The group episode on both routes, the
  poster's picture profile items, the member-in-group profile updates (on a
  flush, from the segments the update reads) and the scene narratives are
  written in it. The one-to-one golden of
  [Section 10](#10-one-to-one-conversations-stay-unchanged) was regenerated
  for that fix; it differs only where its fake model echoes a digest of the
  prompt.

### Implementation notes (Step 2, tasks 11-17)

Step 2 differs from the text above in these places:

- **Flush fan-out.** A flush updated every member with pending segments at
  once, each reading its pending segments one id at a time, which saturated
  the SQLite pool. Each member's segments are now read in one query
  (`MemcellStore.cells_by_id`), and at most `[sqlite] pool_size` members of
  one run read storage at a time; their model calls stay parallel.
- **Group-profile entries.** Evidence is stored one compact JSON object per
  line under `### Evidence`, so it round-trips exactly; entries also carry
  `sender_ids` and `observed_at`. The daily file is dated by the write. The
  pending batch is `profile/pending.md` (frontmatter only, matched by no
  cascade kind). A decision's member scope is `applies_to` (empty for the
  whole group). Resolving a dissent and retiring a role append a new version
  with `resolved` / `retired` set.
- **`group_profile_item` columns.** No `status`, `superseded_by` or `sources`
  column: a superseded or forgotten item is marked in its file's
  `deprecated_entries` (the mechanism the other versioned kinds use) and the
  index row's `deprecated_by` holds the mark; the status is read from it. The
  item travels as `payload_json`, its segments as `source_memcell_ids`.
- **Update triggers.** A new switch, `[groups] group_profile` (default
  `true`). The idle trigger is the scheduler's: it fires again on every scan
  while a group stays idle, and a tick with nothing queued returns at once.
- **An update whose call yields nothing usable** (a reply that never
  validates, a failed call) is not raised: raised, it was retried at once by
  the scheduler and called again by every idle scan, about six calls a minute
  for one idle group in the 2026-11-11 check. It is counted in
  `everos_group_profile_update_failures_total` (by `reason` and `outcome`),
  and the batch stays queued with a back-off kept in `profile/pending.md`
  (`failures`, `retry_at`): no trigger of the group reads it before
  `group_profile_idle_seconds` has passed, then twice that, while new
  segments still queue. The third failure in a row drops the batch: its
  operations are lost, its segments stay searchable. A forget that queues
  segments rewrites the batch and clears the record.
- **Reply parsing.** The update reply is lenient about form: a key left out
  or `null` reads as empty, and one id where a list is expected as a list of
  it (`""` as no one); in the check, 33 of 42 replies had left out the keys
  of unused fields or written `decided_by` as a string, and each failed the
  whole batch. The schema sent is unchanged (every field required, no
  default), and the verifier judges the values as before: an operation
  without evidence is dropped as `no_evidence`.
- **Verifier.** Beyond the listed checks it drops an operation that names a
  person who did not write its evidence (`decided_by`, `agreed_by`,
  `held_by`, a role's member), names someone outside the roster
  (`assignees`, `applies_to`), or repeats a target. A `topic_id` or `about`
  that names nothing drops only the link (see "Topics created in the same
  reply" below). A topic id is minted by code.
  The verbatim check reads curly quotation marks as straight ones on both
  sides (all four quote drops of the check were that alone), and a stored
  quote is the message's own words. The same check
  (`everos.memory.groups.verbatim_part`) serves the constraint binding and
  the read-time check of stored quotes.
- **Prompts, version 2.** Both prompts show each speaker's member id on
  the segment lines, `[<message_id>] <time> <name> (<member_id>): <text>`
  (the binding call's lines carry no time). `chat.group_profile_update`
  requires a cited message of everyone it names in `decided_by`,
  `agreed_by` and `held_by` (24 of 37 ungrounded drops in the check named
  an agreeing member without citing them). `search.constraint_bind` defines
  `segment`, lets an `applies: false` binding cite nothing, and asks for a
  non-empty member list.
- **Topics created in the same reply (`chat.group_profile_update` version
  3).** Version 2 let `topic_id` and `about` name listed ids only, so a reply
  had no way to name a topic it created. On a new group's first batch the
  model either made an id up, and the verifier dropped every decision and
  dissent naming it as `unknown_reference` (the 2026-11-11 integration
  check: 3 of 7 operations, no decision stored), or left the field empty, and
  the decisions were stored without their topic (four fresh groups: 0 of 8
  decisions and 0 of 2 dissent linked). In version 3 a `topic.new` puts a
  label of its own in `topic_id` (`new_1`, ...) for the reply's other
  operations to name the topic by. The verifier checks the topic operations
  first, so a reference may come before or after the `topic.new`, and maps
  the label to the id it mints; a label that repeats a rendered id stays
  that id's. A link that names nothing drops only the link
  (`everos_group_profile_links_dropped_total`); replayed, the integration
  check's reply keeps its 2 decisions and its dissent, unlinked. Targets stay
  strict, labels included: a topic created in a reply is not updated in it,
  since the update would have no stored version to supersede and the profile
  would carry two current versions of the topic. Only topics take labels: a
  dissent about a decision made in the same reply is kept without its link.
  The same four groups again (DeepSeek V4.1 Flash, thinking off): every reply
  used labels, no operation or link was dropped, and 9 of 9 decisions and 2
  of 2 dissent were linked to their topic. The call's prompt grew by 61
  tokens (about 4%); there is still one call per batch.
- **Message ids.** A group batch's `messages[].message_id` is the message's
  id in its segment; without one the generated id stays.
- **Forget (task 12).** Superseded and forgotten versions are taken out of
  markdown and the index, not marked, and kept byte for byte in the forget
  log (`forget_log.undo_json`) so that undo restores them under their own
  ids; they are therefore absent from history reads. Profile items that may
  quote a forgotten message are removed at once and rewritten from their
  remaining segments by the next update (absent until then, not marked
  stale). Targets: `message`, `member`, `session`, `record` (episode,
  picture memory, scene, group-profile or member-profile item ids), `image`;
  there is no separate segment target. A group without owner or admin lets
  any current member forget; managers may forget records outside their own
  windows. Undo is last in, first out (409 while a later forget holds what
  this one wrote). A group-scoped delete is an owner's or admin's forget of a
  session or of a member's messages. Not scrubbed: the memcell payload (by
  design), the scheduler's event log, and older LanceDB table versions until
  the cascade prunes them.
- **Expansion.** Only group scopes expand. A core scene leads to its
  episodes; an episode's scene is not looked up (scene membership is not
  indexed). A linked record scores 0 and reaches the final injection only if
  the decider makes it core. Nothing is expanded after the last round.
- **Constraints.** The core is the decider's (multi-round retrieval is the
  only search method in 2.0). A decision whose newer version the asker
  cannot see is returned as it is, `superseded`, without naming its
  successor. A metadata constraint quotes at most 300 characters of its
  message (a prefix). The model layer only refines: a failing binding is
  dropped (`everos_constraint_bindings_dropped_total`) and the constraint
  keeps its rules form; a bound one gains `applies`. A binding with
  `applies: false` may cite nothing and state no scope, and then keeps the
  constraint's source and scope; what it does fill in is checked in full.
  One member id where a list is expected reads as a list of it. A binding
  call that yields nothing usable leaves the rules layer's block as the
  answer, counted in `everos_constraint_bind_failures_total`; it no longer
  fails the search with a 503. Quotes are checked again at read time
  against the segment read without forgotten messages.
- **Partial memory.** `coverage` is reported for group scopes only; a loop
  ended by the no-new-core patience counts as `partial`. The group profile is attached by default, each kind capped at
  `[search] group_profile_cap` (default 20). The decline share on
  GroupAssistBench L2 is reported with task 18.
- **Who knew what.** `present` reads membership stays, not visibility
  windows, and includes assistants; `joined_after` leaves out members who
  were present.
- **Projects.** `GET /projects/{id}` lists the asker's groups and their
  members only: a roster is read by a group's current members. A knowledge
  read of a project with groups answers 422 without an asker and 403 for a
  non-member. Fetching a document or topic by id is not scoped to its
  project (as before).
- **One-to-one.** Search responses of user scopes and get responses of user
  and agent scopes carry the new fields with their empty defaults
  (`audience`, `group_profile`, `constraints`, `coverage`); the golden test
  lists them as added fields and its fixture is unchanged.
- **Real-model check (2026-11-11).** DeepSeek V4.1 Flash, JSON mode,
  thinking off; gab04 (all 68 messages), gab27 (first 300), `team_chat_en`
  (first 300), then 22 keyword searches with `[search] constraints =
  "model"`. Before these fixes (version 1 prompts): 33 of 42 group-profile
  replies failed the schema and their batches were called again and again;
  replayed offline, 71 of 339 proposed operations were dropped (37
  `ungrounded_member`, 13 `no_evidence`, 12 `unknown_message`, 4
  `quote_not_verbatim`, 3 `invalid_field`, 1 `unknown_target`, 1
  `duplicate`), and 58 of 151 bindings; $0.37. After (version 2): 0 of 9
  replies failed; 7 of 75 operations were dropped (4 `unknown_message`, 1
  each `ungrounded_member`, `unknown_reference`, `quote_not_verbatim`); 34
  of 331 bindings were dropped at run time, 32 of them `applies: false` with
  an empty scope, which the binding check has accepted since (1 of 331 when
  the same replies are checked again); $0.28. Boundary detection cut the
  three groups into 34 segments in the first run and 41 in the second, so
  the counts compare runs, not segments.
- **Decisions (2026-11-11).** Five questions the implementation raised,
  decided by the product owner; none changed the code.
  1. Group-profile items that may quote a forgotten message are removed at
     once and rewritten from their remaining segments by the next update,
     absent until then: showing them, even marked stale, would show the
     forgotten words.
  2. A forget does not scrub the scheduler's event log or prune older
     LanceDB table versions at once, so that undo keeps working (2.0 does no
     hard deletes). Until version cleanup, forgotten text can still exist in
     the scheduler's event log, in older LanceDB table versions (until the
     cascade prunes them), and in `forget_log.undo_json` (what undo
     restores from); the stored memcell payload keeps it too, by design,
     and every reader drops the forgotten messages from it.
  3. In a group with no owner or admin, any current member may forget;
     managers may forget records outside their own windows.
  4. Undo stays last in, first out: undoing a forget while a later forget
     holds what it wrote answers 409.
  5. The verifier stays strict: a message cited under the wrong segment,
     even one of the same session, is dropped (`unknown_message`).

---

## 15. Risks

| Risk | Mitigation |
|---|---|
| A visibility bug is a leak. | One policy module; the predicate goes into the index query, not into post-filtering; property tests; invisible equals absent; the leak suite runs in CI. |
| Strict windows hide everything before a member joined, including the group's purpose and current plan. | The rule is the design; its effect on utility is measured on fixtures and reported. |
| Subject tagging by name misses pronouns and unlisted nicknames. | Aliases in the roster; the attribution sample measures it; the group profile carries model-cited subjects with evidence. |
| The multi-round decider has not seen asker-prefixed questions, expansion slots or group candidates. | Each is measured with and without on GroupAssistBench and EverMemBench; each has a switch. |
| The group-profile prompt is new; the extraction model may follow it poorly. | Code verification drops bad operations; drop rates are monitored; the evidence rules keep a bad reply from writing a false decision. |
| Batching delays profiles (member profile every N segments, group profile per batch). | Episodes are searchable at once; flush runs pending batches; N is configurable. |
| Benchmarks without access rules (EverMemBench) cannot test visibility. | An evaluator member of all groups for the main number; visibility is tested by the leak suite and GateMem. |
| GroupAssistBench's official task and judge models are paid, and its licence is non-commercial. | Cost probe and approval before each full run; evaluation-only use; data never committed. |
| Streaming evaluation waits at every cutoff. | Cutoffs of a group are processed in order with one wait each; groups run in parallel. |
| Agent cases learned in a group could carry its content elsewhere. | Cases carry `source_group_id` and are filtered by read context; skills summarise across contexts and are told not to copy names or values. |
| Stores that collapsed groups into one user must be rebuilt. | Documented; no converter is offered. |
| Members with many windows make long predicates. | Adjacent intervals are merged; the number of windows per member is bounded in practice and measured in the latency check. |

### Open questions and current defaults

Each default below is what Step 1 and Step 2 implement until a product
decision replaces it.

| Question | Default |
|---|---|
| Does a member see history from before they joined? | No: strict membership windows. A per-group setting `history_visible_from` takes `join` (default) or `creation`; `creation` shows a new member the group's history from its start. |
| A reply posted to the whole group | `/memory/search` takes `audience`: `asker` (default, the asker's view) or `group` (only content that every current member can see). A client posting into the group uses `group`. |
| Hard delete inside a group | `/memory/delete` keeps its user and agent scopes. A group-scoped delete is owner/admin only and behaves as a forget (soft, propagated) in Step 2; until then group-scoped delete and forget answer `422`. |
| Step 2's two new prompts and the extraction model | They run zero-shot, guarded by the code checks of [Section 7.5](#75-group-profile-step-2) and [Section 8.4](#84-constraint-block-step-2); group-chat data is collected for a later fine-tune. |
| Paid evaluation runs | A one-group cost probe precedes every full GroupAssistBench run; GateMem is used only once its data licence is confirmed. |

---

## 16. References

- Hu, Zhang, Zhao, et al. *One Assistant, Many Memories: Benchmarking Real-World Group Assistance When Recall Is Not Enough* (GroupAssistBench). arXiv:2611.04867, 2026.
- Hu et al. *Evaluating Long-Horizon Memory for Multi-Party Collaborative Dialogues* (EverMemBench). arXiv:2602.01313, 2026.
- Ren et al. *GateMem: Benchmarking Memory Governance in Multi-Principal Shared-Memory Agents.* arXiv:2606.18829, 2026.
- Yang et al. *GroupMemBench: Benchmarking LLM Agent Memory in Multi-Party Conversations.* arXiv:2605.14498, 2026.
- Owolabi. *SocialMemBench: Are AI Memory Systems Ready for Social Group Settings?* arXiv:2605.17789, 2026.
- Zheng et al. *SpeakerMem-R1: Speaker-Centered Dual-Track Memory for Multi-Party Dialogue.* arXiv:2609.26780, 2026.
- Rezazadeh et al. *Collaborative Memory: Multi-User Memory Sharing in LLM Agents with Dynamic Access Control.* arXiv:2505.18279, 2025.
- Margalit et al. *Governed Shared Memory for Multi-Agent LLM Systems.* arXiv:2606.24535, 2026.
- Zhang et al. *G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems.* arXiv:2506.07398, 2025.
- EverOS docs: [architecture.md](architecture.md), [storage_layout.md](storage_layout.md), [api.md](api.md), [multimodal.md](multimodal.md).
