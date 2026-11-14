# Changelog

All notable changes to **EverOS** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

**Memory deletion.** Two endpoints aligned with the Cloud API, and the
machinery that keeps a deleted memory from coming back.

### Changed

- **Every chat cell is sent the original episode request again.** everalgo's
  rewritten episode rules (2026-10) told the model not to copy a message's
  time into the content: with GPT-4.1-mini extraction 1% of LoCoMo episodes
  carried a date, and temporal questions fell to 53% (90% on the stores the
  original built). Every cell -- one-to-one or group, with pictures or
  without -- is now sent everalgo-user-memory 0.3.1's request byte for byte,
  as EverOS 1.2.0 sent it: one user message holding the conversation start
  time, one pseudo-JSON block per message and the operator's custom
  instructions in their slot (the original's five default principles when
  none are set). The blocks EverOS adds follow it: a group's roster block
  (`<group_chat>`, its speakers labelled by roster name in the messages), a
  picture cell's `<pictures>` block (the picture rules, the `images` field,
  the labels in order; the pictures follow the text, each after its label),
  then the `<output_language>` block naming the conversation's language. The
  reply is `title` and `content` (and `images`), read as the original read
  it: any other top-level key is ignored (DeepSeek V4.1 Flash added keys of
  its own -- `type`, `_meta`, `_comment`, ... -- to 123 of 127 replies on a
  LoCoMo probe; the names are logged at debug level and counted in
  `everos_episode_reply_keys_ignored_total`). The summary is the content's
  first 200 characters, cut in code. `chat.episode` is version 4;
  everalgo-user-memory is the local build `0.9.0rc6.dev7+v031episode`, which
  carries the original prompt and its renderer
  ([vendor/wheels/README.md](vendor/wheels/README.md)).
- **A multi-round search lists each memory once per round.** A round after the
  first shows the decider one block per sub-query, and a memory that several
  sub-queries retrieved was listed in each of their blocks, under an index of
  its own: 16.7% of the candidate slots of recorded LoCoMo searches.
  `[decider] dedupe_within_round` (default `true`) lists it once, in the block
  where it ranks highest (a tie goes to the earlier block), and numbers the
  round's indices over what is listed, so a reply names the memories it was
  shown -- parsed, salvaged, as the fallback core or after a fit. Rounds are
  not compared with each other: a memory an earlier round showed is listed
  again, so one the decider passed over can still be chosen. The prompt
  template, the recall, the trace's recall pools and the result's ranking are
  unchanged; the round trace lists the copies left out (`decider_dedupe`).
  Replayed on recorded searches with the decider's tokenizer (no model call),
  decider input tokens per search fell by 15.3% / 17.6% / 12.7% / 9.8% on
  LoCoMo / LongMemEval / BEAM 1M / EverMemBench (LoCoMo p90 39.7k to 30.5k).
  Its effect on answer accuracy is not measured yet; `false` sends the round
  prompts from before ([docs/configuration.md](docs/configuration.md#decider)).
- **A user's profile is updated every five segments and at each flush.** A
  one-to-one conversation updated `user.md` after every new segment: the
  profile update was 70% of the write path's input tokens. Now a user's
  segments are queued on their profile (`pending_memcell_ids`) and read
  together after `[profile] update_every` (default 5) of them, and at every
  flush of one of their conversations (a new `SessionFlushed` event, which a
  flush of a session bound to no group publishes) -- the cadence a member's
  profile in a group already had. A user's first segment, while they have no
  profile, is still read at once, so a personal search never attaches an empty
  profile. A queued segment whose episode the owner forgot before the update
  is not read. An update that fails (and is dead-lettered after its retries)
  leaves every segment it was to read queued, the triggering one included,
  so a later update reads them: before, the next update read its own segment
  only and the failed window's facts never reached `user.md`. Measured on the LoCoMo excerpt (419 messages, two people,
  scripted model, tiktoken estimate), per 1,000 messages and the same for one
  message per add and for batched adds: profile update calls 224 -> 52, their
  input tokens 975k -> 240k, of which not a cached prefix 139k -> 121k (the
  queued segments are new text, so far less of each prompt repeats the
  previous one). `update_every = 1` restores the old cadence
  ([docs/configuration.md](docs/configuration.md#profile)).
- **Implicit groups start at three people; two people keep their own
  copies.** A session sent without `group_id` becomes the implicit group of
  its people (`implicit_<hash>`, the same for every session between the same
  people) once three or more people -- the role `user` senders of the batch
  and the open buffer -- talk in it. One or two people stay on the one-to-one
  path: one extraction call per segment, a copy of it (episode, picture
  memories, event) in each person's own memory and a personal profile update
  per person, read with a plain personal search or get (from 2026-11-12 to
  2026-11-13 two people were an implicit group). A third person who joins two
  people's session makes it the three's group from the batch in which the
  three meet in the open buffer; the earlier segments stay the two people's
  copies. A newcomer to an implicit group is refused (`409`), as before. In
  two people's conversation every forget acts on the asker's copies only,
  and a user's `message` forget answers `422` (both copies are read from one
  buffer and one set of segments). A session bound to two people's implicit
  group by a build of those two days stays that group's
  ([docs/group-memory.md](docs/group-memory.md#79-implicit-groups-three-or-more-people)).
- **One forget and undo engine for a user's memory and a group's.** Every
  forget is planned (a dry run answers the plan), applied, logged in one
  format and undone by its `forget_id`, whatever the owner or the target.
  A group forget now **marks** instead of taking records out of markdown: the
  versions that held what was forgotten stay, marked, their index rows keep
  `deprecated_by`, and a group's `/memory/history` leaves out every version a
  live forget marked or replaced (the reads stay as they were: no search,
  listing, decider input or history shows the forgotten text). A **picture**
  forget is a text edit everywhere -- the episodes and scenes that name the
  picture get a version without the reference -- with no model call: a group
  picture forget no longer re-extracts its segments (one episode call per
  segment, plus one scene call per changed scene with group scenes on, saved).
  A user may forget **messages** of their own conversation (`target.kind:
  "message"`, CLI `--message`): the segments that keep text are extracted
  again, a buffered message is never read. **Undo** clears the marks the
  forget wrote and writes the versions it replaced again as the newest (no
  record is reinstated under its old id any more), puts profile items back
  in their places by id (a forget logged before places were kept puts them
  back by description, as before), and is refused (`409`) while a later
  forget, undo or edit replaced a version the forget wrote: undo that one
  first. A text-edited version now keeps every field of the version it
  replaces (it used to lose its `language`). Forgets logged before still undo
  ([docs/api.md](docs/api.md#post-apiv2memoryforget),
  [docs/group-memory.md](docs/group-memory.md#94-forget-and-delete-propagation)).

### Removed

- **Hard-delete storage nothing called any more**: the stores'
  `remove_session`, `remove`, `remove_naming` and `undeprecate`, the writer's
  `remove_entries`, `OwnerDirStore.delete` (the start-up sweep of an earlier
  build's deletion trash stays), and the profile helpers the two old forget
  engines used (`forget_images`, `forget_segments`, `forget_items`).

### Added

- **`/memory/get` lists a user's picture memories.** `memory_type:
  "image_memory"` pairs with `user_id` as it does with a group scope: the
  user's current picture memories, each item naming its owner (`user_id`,
  or `group_id` for a group's; `group_id` is no longer required on the
  item), under the read rules of every personal listing. Before, a user's
  picture memory could be updated, forgotten and had a history, but a
  caller found it only through search.

- **`everos prompts` lists every prompt a chat store is built and read
  with.** Two were missing. The boundary detection that cuts every buffered
  conversation into segments is everalgo's (`everalgo.boundary.batch_detect`,
  package `everalgo-boundary`, its system and user templates fingerprinted).
  The multi-round decider's prompt is registered as `search.decider`
  (version 1) in two variants: `text`, the round prompt alone -- its
  fingerprint is the first 16 hex digits of the SHA-256 of
  `_DECIDER_PROMPT`, the digest the RL tooling records for the prompt a
  policy saw -- and `pictures`, which adds the `(picture)` mark of a picture's
  evidence line and the two text parts before a question's picture and a
  candidate's thumbnail. Its entry says `"used": "read"`: sent on every
  round of every search, as against `search`, an optional search step (the
  constraint binding). No prompt text changed; every model request is
  byte for byte what it was, and the other entries are unchanged.
- **The forgetting curve and asking about the past (both off by default).**
  `[search] forgetting_curve = true` ranks every search by a memory's
  retention: each candidate's fused score is multiplied by
  `1 - w + w * R` with `R = 1 / (1 + days / (9 * S))`, `w = retention_weight`
  (0.3) and `S` the memory's strength in days, and each block is sorted again
  before it is cut (nothing is dropped). `S` starts at `initial_strength_days`
  (30; episodes, scenes and picture memories alike) and grows by
  `1 + reinforce_gain * (1 - R)` (gain 2, at most `max_strength_days`, ten
  years) each time the decider keeps the memory as core or `get` returns it;
  the memory's versions share it. Strengths live in `system.db`
  (`memory_strength`, created at startup, empty until the curve reinforces
  something); a read only queues what it used and the server writes it after
  the response, one transaction per batch. A read-only server
  (`[cascade] enabled = false`, the evaluation harness's
  `EVEROS_DISABLE_CASCADE=1`) and `reinforce = false` never reinforce.
  `/memory/search` takes `asked_at` (epoch ms or ISO-8601), the time ages are
  measured to, and `include_past`: the user's own superseded versions and
  forgotten memories, or a group's superseded versions inside the asker's
  windows (never a forgotten record, nor a version a live forget retired),
  join the ranking at `past_state_factor` (0.3) of their score, and every
  returned episode and picture memory says its `status`, with the version
  that replaced it (`superseded_by`, `valid_until`) when the asker can see
  it. No model call is added; with both off, requests, results and traces are
  byte-identical to before. Whether the curve helps is not measured yet
  ([docs/configuration.md](docs/configuration.md#search),
  [docs/api.md](docs/api.md#post-apiv2memorysearch)).
- **Asking about the past by a question's wording (off by default).**
  `[search] past_triggers = true` answers a `/memory/search` request that
  does not send `include_past` as if it had sent `true` when its question --
  as asked, never a decider's sub-query -- holds past-time wording, and names
  what fired in `data.past_trigger` (absent otherwise, so every other
  response is byte for byte what it was). An `include_past` the request
  sends, `true` or `false`, always wins. Built-in lists in English and
  Chinese (simplified and traditional), in
  `src/everos/config/past_triggers_en.toml` and `past_triggers_zh.toml`:
  precise phrases ("used to", "back then", "no longer", "before I" with a
  past-tense verb, "the old / previous / former" ...; no bare "before",
  "earlier" or "old") and their Chinese counterparts, each language with
  exceptions for the longer words and phrases that hold a trigger without
  meaning the past ("is used to", "in the past week", "our previous
  conversation", a direction such as "walk over" or a deadline such as
  "before tomorrow" in Chinese). With
  `past_trigger_dates` (on with the switch), an explicit year, month or date
  whose whole period ended before the day the question is asked is a
  trigger too ("in 2019", "March 2024", "2024-03-05", "the 1990s" and the
  Chinese forms); relative words such as "yesterday" are not. The four lists
  (`past_trigger_words_en` / `_zh`, `past_trigger_exceptions_en` / `_zh`)
  are regular expressions; a list set in `everos.toml` replaces its built-in
  one and `[]` switches it off; an entry that is no regular expression or
  matches an empty question is refused at load. No model call; with the
  switch off, requests, results and model requests are byte-identical to
  before. On the evaluation sets' questions alone, the built-in lists fire
  on 25 of LongMemEval-S's 500 (9 of 78 knowledge-update questions) and on
  173 of LoCoMo's 1,540 scored ones, 168 of them by a date the question
  names; whether reading the past then helps an answer is not measured yet
  ([docs/configuration.md](docs/configuration.md#search),
  [docs/api.md](docs/api.md#post-apiv2memorysearch)).
- **One deliverer per group, and each message once.** A group names the
  identity its gateway asserts for the group's key as `deliverer_id`
  (`PUT /groups/{id}`, by owners, admins or the deliverer; `null` clears it;
  `GET` returns it): the one program that forwards the group's chat. Once
  set, `/memory/add` to the group's sessions and `/memory/flush` of them are
  taken from it alone (`403` otherwise; `/memory/flush` takes an optional
  `asker_id`), and it is the group's superadmin: every right of an owner,
  backdated roster events included, and the whole history to read, with a
  search that names no `audience` reading as `group`. Without one nothing
  changes. A group (explicit, or implicit of three or more people) also skips
  a message it already holds in any of its sessions, forgotten ones included,
  by `message_id` or by sender, time and content hash: no model call is made
  for it and the answer counts it in `duplicates`. A `/memory/delete` of a
  session releases its messages, so sending it again works as before.
  `system.db` gains `group_meta.deliverer_id` and the `group_message` ledger
  at startup; no rebuild ([docs/group-memory.md](docs/group-memory.md#51-groups-and-roster)).
- **`[decider] core_order` picks the order of the core in a search result.**
  `score` (default, unchanged): each core item's best RRF score over the
  sub-queries. `round`: the round in which the decider first kept the item,
  earliest first, then that score; ties keep the decider's own order. With
  more core than `top_k` the order also decides which core items are returned,
  since the cap keeps its head. Replayed offline on the 2026-11-10 searches,
  `round` raised hit@1 on HotpotQA / 2Wiki / MuSiQue / NQ by 0.22 / 0.17 /
  0.29 / 0.89 points for the RL-tuned 27B decider and by 0.03 / 0.63 / 0.66 /
  0.06 for its base model. Answer accuracy with the base decider, re-answered
  on every question whose returned core changes (2,781 questions, 12 memory
  sets, same answer and judge models): no set improved significantly
  (LoCoMo +0.13, LongMemEval +0.40, SubtleMemory +0.59 points) and
  EverMemBench fell (pooled -1.04 points, p 0.07; t4 -3.04, p 0.04), so
  `score` stays the default and `round` is opt-in for ranking-oriented use
  (`EVEROS_DECIDER__CORE_ORDER=round`). No model call, no
  change to the decider's input; with the default, requests, results and
  traces are byte-identical to before
  ([docs/configuration.md](docs/configuration.md#decider)).
- **A personal search can cover the user's groups (`include_groups`, off by
  default).** `/memory/search` with `user_id` and `include_groups: true`
  searches the user's memory and every group of the space the user currently
  belongs to as one candidate pool, each group read exactly as a group search
  by the user reads it: inside the user's membership windows, superseded and
  forgotten records left out, nothing from a group the user left
  (`AccessPolicy.member_groups` compiles the groups with the same code as a
  group scope). `include_group_ids` narrows the groups. The decider sees both
  kinds in its usual rendering, under a question that names the user; every
  returned record names its owner (`owner_type` `user`, or `group` with
  `group_id`), group records carry their `audience`, and `scope` lists the
  groups searched. Attached besides the personal profile, and only these: the
  user's own profile in each group (`include_profile`) and the groups' current
  decisions, capped at `[search] group_profile_cap`. No link expansion,
  constraint block or `coverage`. `/memory/get` takes the same fields for
  `episode` and `scene` listings, paged as one list. Cost: no model call;
  reads are one roster read per group of the space, the personal search's
  recall queries with a wider `where`, and one index read each for the
  decisions and per group for the profiles. Without the field a request, its
  response and the decider's input are unchanged; the one-to-one golden now
  pins the decider's input prompt by prompt
  ([docs/group-memory.md](docs/group-memory.md#88-a-personal-search-over-the-users-groups)).
- **Change, forget and the history of a memory, as endpoints (EverOS 2.0).**
  Nothing is edited or deleted in place; every change is a new version linked
  to the old one, and every forget a mark its undo takes away
  ([docs/api.md](docs/api.md#changing-and-forgetting-a-memory)):
  - **`POST /api/v2/memory/update`** writes a person's new version of an
    episode, a picture memory, a scene version, a profile item of `user.md`
    or of a member's profile in a group. The new version names the one it
    replaced (`supersedes`), who wrote it and when (`edited_by`,
    `edited_at`), and keeps what the old one rests on; the old one is marked
    superseded, and the cascade indexes the new one (both files are handed to
    it at once). `expected_version` (default: the id given) is the version
    the caller read: a newer one answers `409 VERSION_CONFLICT` naming it, a
    forgotten memory `409 MEMORY_FORGOTTEN`.
  - **Held.** A person's version is held unless `hold: false`: the next
    profile update (one-to-one and member) keeps a held item as it is -- the
    model still reads it, whatever its reply does to it is undone, and a new
    value it writes stays beside it as an item of its own, linked to nothing
    -- and a scene update writes no narrative over a held version (no call;
    the joining episode still joins the cluster). `hold: false` alone
    releases it.
  - **`GET /api/v2/memory/history`**: every version of a memory, current
    first, whichever of its ids is asked -- state (`current`, `superseded`,
    `merged`, `forgotten`), time, sources, who wrote it and the next
    version. Read from markdown and the forget log; a group asker sees only
    the versions inside their windows.
  - **`POST /api/v2/memory/forget` by target for a user**: `record` (ids of
    any kind), `session`, `owner` (everything, the whole profile included),
    `condition` (kinds and a time range) and `image`. It follows the
    sources (scenes, picture files, profile items, as a conversation's forget
    did), **is a dry run unless `dry_run: false`**, and a real run answers
    with its `forget_id`, which **`/memory/forget/undo`** takes: exactly what
    it marked comes back, except what another live forget still names.
  - **`everos memory history|update|forget|undo`**, the same service
    functions in-process ([docs/cli.md](docs/cli.md#everos-memory)). Each
    prints one JSON document on standard output and its log lines on
    standard error (`[logging] stream`, new, default `stdout`, which these
    commands set to `stderr`).
  - Cost: no model call in any of them; an update embeds its new version
    once (the embedding provider), a held scene saves the call its next join
    would have made.
- **`POST /api/v2/memory/delete`** ([#483](https://github.com/EverMind-AI/EverOS/issues/483)):
  delete by `user_id` / `agent_id`, by `session_id`, or one owner's session;
  answers `{"filters": [...], "count": N}`. A session still being memorized
  answers `503 SESSION_BUSY` and nothing is deleted. everos-mcp's
  `forget_session` now works against a self-hosted server. (In 2.0 it is a
  forget: see Changed; an `agent_id` is refused since agent memory was
  removed: see Removed.)
- **`POST /api/v2/memory/delete_by_ids`**: delete episodes by memory id, with
  what Reflection merged into them and their cluster membership. (Reflection
  is removed since: see Removed.)
- Configuration: `[memorize] delete_wait_seconds` (default `10`),
  `[cascade] delete_fence_warn_entries` (default `100000`).
- Metrics: `everos_lock_wait_seconds` / `everos_lock_hold_seconds` (by lock
  kind), `everos_lock_table_entries`, `everos_memory_delete_fence_entries` /
  `_runs`.
- Configuration: `[ome] enabled` (default `true`). `false` starts the server
  without the OME engine, for a server that only searches a store built
  elsewhere; a memorize there fails when it publishes its events. The root
  lock is still taken: one server per memory root.
- **A picture memory can carry a vector of the picture itself**
  (`[vision_embedding]`: a multimodal embedder such as Qwen3-VL-Embedding-8B,
  behind an endpoint that applies the model's own recipe). The cascade embeds
  each picture's bytes into `image_memory.visual_vector`, at its own width
  (`[vision_embedding] dim`), and multi-round search adds a visual leg to
  picture recall, fused with the description leg by RRF. An empty
  `[vision_embedding]` keeps the description-only behaviour. Table schema
  version 3 adds the column when a store is opened.
- **Group memory, first half** ([docs/group-memory.md](docs/group-memory.md),
  Step 1 tasks 1-3):
  - **A `group` owner** with its own tree (`groups/<group_id>/`) and a roster
    of membership intervals (join, leave, rejoin, backdated join). The roster
    is an append-only `group.md`, which the cascade folds into the
    `group_meta`, `group_member` and `session_binding` tables. A rebuild from
    the markdown restores identical rows.
  - **Routes:** `PUT /api/v2/groups/{group_id}`,
    `POST /api/v2/groups/{group_id}/roster` and `GET /api/v2/groups/{group_id}`.
  - **`/memory/add`** takes `group_id`, `roster` and `message_id`. A session
    binds to one group, and a batch for another group answers 409. An unknown
    group is created at its first message and an unknown sender joins there
    (`[groups] implicit_create` / `implicit_join`, both default `true`).
  - **Search and get** take `group_id` or `group_ids`, `asker_id` and
    `audience`.
  - **One access policy for every read path.** A group record is returned
    only to a current member whose membership windows cover all of the
    record's sources. With `[access] identity_header` set, the asker is read
    from the gateway's header.
  - **Table schema version 4** adds `subject_ids`, `origin`, `source_start`
    and `source_end` to `episode` and `image_memory`, plus `sender_ids` to
    `image_memory`. Existing rows read as one-to-one records.
- **Group memory, second half** ([docs/group-memory.md](docs/group-memory.md),
  Step 1 tasks 4-7):
  - **One group episode per segment.** A segment of a session bound to a
    group is written once, under `groups/<group_id>/episodes/`, never as each
    sender's personal memory. The extraction call is the one-to-one call: the
    speakers carry their roster names (an assistant member as
    `"<name> (assistant)"`) and a roster block (template
    `chat.group_roster_block` v1) follows the custom instructions. Code sets
    `sender_ids`, `subject_ids` (the members the episode names, else the human
    senders), `origin` (`member` / `assistant` / `mixed`) and the segment's
    span.
  - **Pictures once per group,** with `sender_ids` set to the poster. The
    profile items a picture yields reach only the poster's member-in-group
    profile.
  - **Member-in-group profiles** in `groups/<group_id>/members/<member_id>.md`
    (strategy `update_member_profiles`, everalgo's profile operator over the
    member's own segments of the group). A profile is updated after
    `[groups] member_profile_every` new segments (default `5`) and on every
    flush of the group. Every item records the segments it was read from.
  - **Group scenes** in `groups/<group_id>/scenes/` (strategy
    `update_group_scenes`), an opt-in: `[scenes] groups`, default `false`.
    A user's scenes, built the same way over the group's episodes (see "A
    user's scenes" below and Changed). Each change is a new version.
    Scenes compete with episodes in search. A scene is shown only to an
    asker who can see all of its episodes.
  - **Agent cases learned in a group** carry `source_group_id`. They are
    returned only to an agent read in that group's context:
    `context_group_id`, asked by a current member. (Removed since, with
    agent memory: see Removed.)
  - **Search:** a group episode carries `owner_type` and
    `source: {memcell_id, from, to}`. `include_profile` attaches the asker's
    and up to three named members' in-group profiles, cut to the asker's
    windows. The multi-round question starts with the asker's name
    (`[groups] asker_in_question`, default `true`).
  - **Get:** `memory_type` takes `scene`, `image_memory` and
    `member_profile` (with `member_id`) on group scopes.
  - **Table schema version 5** adds `user_profile.subject_id` and
    `agent_case.source_group_id`. Existing rows read as before. (The
    `agent_case` table is no longer opened: see Removed.)

- **Group memory, Step 2** ([docs/group-memory.md](docs/group-memory.md),
  tasks 11-17):
  - **The group profile.** Topics, decisions, dissent and observed roles as
    append-only items in `groups/<group_id>/profile/` and the new
    `group_profile_item` table (no vector). One model call per batch of
    segments (prompt `chat.group_profile_update` v3, in EverOS), after
    `[groups] group_profile_batch` segments (default `5`), after
    `[groups] group_profile_idle_seconds` without one (default `600`) and at
    every flush; `[groups] group_profile` switches it off. Code checks every
    operation the model returns -- the cited segment and message exist, the
    quote is verbatim, a decision rests on a member's message, the people it
    names wrote the evidence, the ids it replaces are current -- and drops a
    failing one, counted in `everos_group_profile_ops_dropped_total`; a
    topic or item link that names nothing drops only the link, counted in
    `everos_group_profile_links_dropped_total`. A call
    that yields nothing usable is counted in
    `everos_group_profile_update_failures_total` and retried after a
    back-off; the third failure in a row drops the batch.
    `/memory/get` lists the items (`memory_type: group_profile`, with
    `include_superseded`).
  - **Forget and delete inside a group.** `/memory/forget` and its undo take
    group targets (a message, a member's messages, a session, records, a
    picture) with `group_id` and `asker_id`, and `dry_run`; the forget travels
    to everything derived from it and the segment is re-extracted without the
    message. A group-scoped `/memory/delete` is an owner's or admin's forget.
    A group batch's own `message_id` is now kept as the message's id in its
    segment.
  - **Group search** adds relational expansion to the multi-round loop
    (`[search] expansion`, `[search] link_slots`, default `5`), a constraint
    block (`constraints`, `[search] constraints = "rules"` or `"model"`, prompt
    `search.constraint_bind` v2; a failed binding call leaves the rules
    block, counted in `everos_constraint_bind_failures_total`), the visible
    current group profile
    (`include_group_profile`, `[search] group_profile_cap`), `coverage`, and
    an `audience` on every group record. Requests and responses of user
    scopes are unchanged; the new response fields hold their empty defaults
    there.
  - **Projects.** `project_scope: true` searches every group of the project
    the asker belongs to; `GET /api/v2/projects/{project_id}` and
    `GET /api/v2/projects/{project_id}/profile` read them. A project's
    knowledge base is read by its members only once the project has groups
    (the knowledge reads take `asker_id`). (The knowledge base is removed
    since: see Removed.)
- **A member's fact that changes keeps its earlier value**
  ([docs/group-memory.md](docs/group-memory.md), section 7.4). When the
  profile operator's UPDATE rewrites a member-in-group profile fact
  (`explicit_info`) by its index, the new item names the item it replaced
  (`supersedes`) and when (`since`, the end of the update's span), and the
  old item's history entry names it back (`superseded_by`). Search
  (`include_profile`) and get (`memory_type: member_profile`) show such an
  item with `since` and `previous` (at most two replaced values, newest
  first, each with `description` and `until`), cut to the asker's windows. A
  replaced value a forget took out is never shown, and a forget by record id
  now reaches a replaced value. COMPACT, REGROUP and dedupe rewrites, and
  rewritten implicit traits, are not linked. No extra model call and no
  prompt change: the link is the `item_id` everalgo keeps on the item an
  UPDATE op patches. The one-to-one user profile is unchanged.
- **Two recoveries for a failed multi-round decider round**, both on by
  default. Each acts only on a round that would otherwise fall back, so a
  round that fits the window and parses is sent and recorded as before. Set
  either to `false` to reproduce a run from before 2026-11-11; off, requests
  and traces are byte-identical to before.
  - `[decider] fit_to_window`: when the decider endpoint rejects a round
    prompt as longer than its window, the prompt is re-rendered to fit
    `window - max_tokens` tokens, counted by the endpoint itself (vLLM's
    `/tokenize`, through the new `OpenAIProvider.acount_prompt_tokens`).
    CORE SO FAR texts are cut first (one shared cap, longest first), then
    candidate texts, then the last candidates are dropped. The round trace
    records what was cut (`decider_fit`). On BEAM 500K / 1M / 10M this
    rejection cost the RL decider up to 4.7% of its questions.
  - `[decider] salvage_truncated_core`: when every attempt fails to parse and
    a reply had already emitted its core list (`{"core": [0, 3],` and
    nothing after it), those indices become the round's core and the loop
    stops there. The round trace records it (`decider_salvaged`).
- **A search says when the decider did not decide a round, and why.** The
  response's `data.degraded` lists each such round (`component: "decider"`,
  `round`, `outcome` `fallback` or `salvaged`, `reason` = the last attempt's
  error with credential-shaped text masked, `attempts`); it is `[]` when every
  round was decided. Before, a fallback answered `200` like any other search
  and its reason was only in the server log. The round trace gains
  `decider_error`, and a group scope's `coverage` reads `partial` instead of
  `complete` when the loop ended on such a round. No extra model call.
- **`GET /health` lists the multi-round decider and checks it.** A `decider`
  block names its `model` and `base_url` (credentials and query stripped) and
  says whether it answered one request sent in the background at startup
  (`reachable`, `probe_error`, `probed_at`; `capabilities.decider` mirrors
  `reachable`). The searches keep it current at no cost: `rounds`,
  `failed_rounds`, `last_failure`, `last_failure_at`, and the metric
  `everos_decider_rounds_total{outcome}` (`decided`, `fallback`,
  `salvaged`). The probe is one request with `max_tokens = 1` and the
  decider's own `temperature` / `extra`, sent only through a client with a
  raw call (the production provider); its cost is about ten input tokens
  and one output token per server start.
- **`/add` returns what ingest dropped.** `data.rejected` counts pictures
  refused at ingest (over an `[images]` limit, not a picture, unreadable) and
  `data.failed` the non-text items the parser could not read; the rest of
  their messages is kept. Both are `0` on a clean batch. A queued batch
  (`202`) logs them as `memorize.ingest_dropped`.
- **A user's scenes** ([docs/scenes.md](docs/scenes.md)), off by default:
  `[scenes] users = false`. A scene is a user's episodes about one matter
  (placed with the `[clustering]` geometry) with one narrative over them,
  ending on where things stand, written to `users/<u>/scenes/` (`sc_` ids)
  and indexed as rows of the episode table (`parent_type = scene`), so it
  competes with the episodes in the same multi-round pool, rendered and fused
  like them; a search result marks it `type: "Scene"` and `get` lists it under
  `memory_type: "scene"` (never under `"episode"`).
  - Strategy `update_user_scenes`: one `chat.scene_update` call (prompt in
    EverOS, registered, v2) each time an episode joins a scene of
    `[scenes] min_members` (`2`) episodes or more; an episode that joins no
    scene costs no call. The narrative is capped (`narrative_max_words`,
    `300`): the prompt asks for it as a total, a longer reply loses its
    oldest part at a sentence start, the reply is held to a token cap with
    one retry, and one call reads at most `max_new_episodes` (`4`) episodes
    of at most `episode_max_chars` (`4000`), so its price does not grow with
    the scene.
  - Forgetting follows the sources: a session forget marks a scene forgotten
    once none of its episodes is alive (and erases its row), the undo brings
    it back, and a later version is written from the live episodes only. A
    picture forget rewrites a scene that names the picture without the
    reference (no model call). Deleting a session or episodes removes every
    scene version that names one of them.
  - A group's scenes are the same implementation (`[scenes] groups = false`;
    see Changed).
  - Measured cost (2026-11-12, DeepSeek V4.1 Flash, thinking off, product
    defaults, off-peak; everalgo-user-memory 0.9.0rc4): $0.026 per 1,000
    messages on LoCoMo conversations 0-1 (46 calls, +13% of the $0.200 text
    write cost) and $0.039 on MemEye's 8 conversations (151 calls, +21% of
    $0.186); peak prices double both. About $0.00044 a call; the prompt
    stays near 1,100-1,200 tokens over a run (largest 1,890). Read side,
    LoCoMo 0-1 with the 27B decider: a scene is in the core on 66% of
    questions, +5.6% decider prompt tokens, +2.3% returned context.
  - `ClusterStore.find_cluster_for_member` takes a `kind`: an episode now
    sits in its profile cluster and its scene's. (Profile clustering is
    removed since: see Removed.)
- **Search names the answering model's copy of every picture.**
  `SearchImageItem.model_path` -- on ranked pictures and on those attached to
  an episode -- is the `[images] model_edge` copy,
  `figures/.cache/<image_id>-1024.<jpeg|png>` by default: the size extraction
  read the picture at, and the copy to hand an answering model. Search renders
  it into the cache when it is missing, so the file is there when the response
  is read; it is empty only when the original cannot be read.
  `GET /api/v2/memory/images/{image_id}?variant=model` serves the same copy by
  id. `original_path` stays for a caller that wants the bytes as sent; the
  `SearchImageItem` docs no longer tell a caller to give the original to the
  answering model. `ImageStore.ensure_derived` names (and if needed renders) a
  scaled copy. Measured (2026-11-12, answers only, on the same 2.0 searches;
  the release's answer seats): on Mem-Gallery (Qwen2.5-VL-7B) the answer input
  fell from 7,269 to 4,289 tokens per call (a picture 1,316 -> 368 visual
  tokens), LLM-judge 84.60 -> 84.83 (p 0.33); on CUE-Mem, whose pictures are
  at most 1,024 px, tokens are unchanged and a picture is 159 KB instead of
  1.44 MB, accuracy 74.51 -> 74.98 (p 0.25). A seat that shrank a picture to
  768 px before (MemEye, gpt-5.4-mini) is projected to read more: about 907
  image tokens a picture instead of 533.


### Changed

- **A project's groups come from one port (EverOS 2.0).** Chat memory writes
  group memory only; a project read pools the memory of the project's groups,
  and the project's own memory is extracted by working memory. Which groups a
  project has is now the answer of `ProjectLinkStore`
  (`memory/ports/projects.py`), which the access policy asks for a
  `project_scope` search, `GET /api/v2/projects/{project_id}` and its
  `/profile`, and for `include_groups`; the rosters still decide what the
  asker sees. The relation is many-to-many and is maintained by a separate
  project-link feature, which plugs its adapter in there. The default,
  `PartitionProjectLinkStore`, is the rule these reads applied before: a
  project's groups are the groups stored in its `(app_id, project_id)`
  partition. Responses, the decider's input and costs are unchanged. A read
  scope covers one partition, so a link to a group stored in another one is
  refused (an error, not an empty answer) until the link feature reads across
  partitions; the reverse direction (the projects a group belongs to) is
  reserved, not yet a method
  ([docs/group-memory.md](docs/group-memory.md#44-project)).
- **One scene implementation for users and groups (EverOS 2.0, breaking
  config).** `update_user_scenes` and `update_group_scenes` run one core
  (`memory/strategies/_scenes.py`), and a forget inside a group writes a
  changed scene again through the same narrator: one placement, one capped
  `chat.scene_update` call, one `min_members`, members read through their
  marks. The switches are one section, one per owner: `[scenes] users`
  (was `[scenes] enabled`) and `[scenes] groups` (was `[groups] scenes`),
  both still `false`; a config naming the old keys is refused at startup.
  For users nothing changes. Group scenes change, deliberately:
  - the narrative is the capped call a user's scenes make
    (`narrative_max_words`, `max_new_episodes`, `episode_max_chars`, the
    reply's token cap), not everalgo's uncapped `EpisodeReflector`; a forget
    inside the group writes the scene from scratch with it as well;
  - a cluster becomes a scene at `[scenes] min_members` live episodes, not
    a hard-coded two (the forget's rewrite too);
  - members are read through their marks, as a user's are: an episode a
    person replaced (`/memory/update`) is read as its new version, and a
    version written from a member that is gone since is not carried
    forward;
  - the `group_scene_written` log line carries the fields `user_scene_written`
    does (`owner_id`, `joining`, `read`, `mode`, `narrative_chars`,
    `clipped`).
  - Cost: none for users. A group scene call is now bounded like a user's
    (about $0.00044 a call, prompt near 1,100-1,200 tokens, measured on the
    user side), where the reflector's UPDATE read the whole narrative and
    every new episode unclipped; read side unchanged (the same rows).
    Nothing billed was run for this change.
- **One profile for a person in a space (EverOS 2.0).** `users/<u>/user.md`
  and `groups/<g>/members/<m>.md` keep their paths and their bytes, and are
  one implementation: one model (`Profile`, which now carries
  `pending_memcell_ids`; `MemberProfile` is gone), one frontmatter schema
  (`ProfileFrontmatter`), one store (`MarkdownProfileStore`, the scope naming
  the space; `MarkdownMemberProfileStore`, `ProfileReader` and
  `ProfileWriter` are gone), one cascade handler (`ProfileHandler`, for both
  kinds), one update core (`memory/strategies/_profiles.py`: pictures,
  `aupdate`, stamping, held items, the bounded history) whose cadence is the
  space's (one-to-one: every segment; a group: `[groups]
  member_profile_every` and each flush), one fetch for search and get
  (`fetch_user_profile` / `fetch_member_profile`), and one remover and
  put-back for the items a forget takes (`take_out` / `put_back`, one
  picture rule). The forget logs keep their formats. Nothing changes for
  either space, nor its cost.
- **One lock family per owner.** `profile_lock(scope, person)`,
  `scenes_lock(scope)` and `forget_lock(scope)` replace the user/group pairs
  (`user_profile_lock` / `member_profile_lock`, `user_scenes_lock` /
  `group_scenes_lock`, `user_forget_lock` / `group_forget_lock`).

- **No hard delete (EverOS 2.0, breaking).** `POST /api/v2/memory/delete` and
  `/delete_by_ids` keep their requests and answer with a forget of the same
  scope, run at once: `user_id` is the user's `owner` target, `user_id` +
  `session_id` the user's `session`, `session_id` alone the conversation in
  every user's memory (one forget each), `delete_by_ids` the episodes'
  `record` target per owner (ids that match nothing are still skipped). The
  answers add `forget_ids`, the undo tokens. Nothing leaves markdown, and a
  conversation's segments and buffer stay in `system.db`: a conversation sent
  again after its delete writes new memories beside the forgotten ones (the
  evaluation harnesses' `--replace-sessions` keeps working; their stores keep
  the forgotten copy). `agent_id` is refused (`422`): an agent's memory has no
  forget yet; a session-only delete leaves an agent's cases of the
  conversation. (Agent memory is removed since: see Removed.) The
  `500 "deletion incomplete"` answer and
  `DeletionIncompleteError` are gone.
- **A forget by target is a dry run by default (breaking for group
  callers).** A group body on `/memory/forget` used to forget at once; it now
  needs `"dry_run": false`, as a user's target body does. The picture body
  (`user_id` + `image_id`) and `/forget/session` still run at once.
- **`/forget/session` and its undo, the picture body of `/forget` and
  `/images/revise` are aliases** of the general routes, with their own bodies
  and answers. `/images/revise` now refuses a replaced occurrence (`409
  VERSION_CONFLICT`) instead of forking it. All of them, and `/history`,
  refuse a personal memory whose `asker_id` (or identity header) names
  someone else (`403`).
- **The profile skips what is forgotten, not a whole conversation forever.**
  A one-to-one profile update skipped every segment of a conversation with a
  live forget, including segments the conversation got after the forget; it
  now skips a segment whose episode is marked forgotten (by any forget). A
  conversation sent again or continued after its forget reaches the profile
  like any new one.
- **A conversation's forget marks its current versions only.** It used to
  mark every version of the conversation, so its undo turned a version a
  newer one had replaced (a picture forget's rewrite, a person's edit) back
  into a current one beside it. The undo of any forget now also leaves an
  entry another live forget names forgotten.
- **Unknown configuration keys are rejected.** A key no section declares --
  in `everos.toml`, or an `EVEROS_<SECTION>__<KEY>` variable of an existing
  section -- fails the load with `UnknownConfigKeyError`, listing every
  unknown key with the file or variable it came from and the nearest valid
  key; the server and the CLI print it and exit with code `3`. Settings used
  `extra="ignore"`, so a misspelt key silently fell back to its default.
  Keys EverOS itself once wrote and that the configurations in use still
  carry are accepted, ignored and logged once as `config_key_ignored`
  (`[llm] search_timeout_seconds`, `[decider] full_text` /
  `invalid_reply_retries` / `code_pin`, `[knowledge.search]
  session_lock_timeout_seconds`, `[observability] recall_hit_threshold`,
  `[search] facts_per_episode` / `vector_strategy`, `[memory] root`,
  `[multimodal] max_tokens`, which every benchmark launcher exports; see
  [docs/configuration.md](docs/configuration.md#unknown-keys-are-rejected)).
  Variables that name no section (`EVEROS_ROOT`, `EVEROS_DISABLE_OME`, ...)
  are not checked. No model call.

- **EverOS 2 opens only the stores EverOS 2 built.** `.index/store_version`
  holds the store's major (`2`). The server, every `everos cascade` command
  and backfill check it before anything in the root changes: a new root is
  stamped; a root from before the marker is adopted only when something in it
  can have been written by EverOS 2 alone (index table schema v3 or later, the
  `image_memory` / `group_profile_item` index tables, a 2.0 system table,
  `figures/`); an EverOS 1.x store -- or a marker naming another major -- is
  refused, unchanged, with `StoreVersionError` (CLI exit code 3). A 1.x store
  is therefore no longer migrated in place (LanceDB v2 to v5): serve it with
  1.x, or rebuild it with 2.0 from the source conversations. A root with
  markdown but no store state (hand-written, or a wiped `.index`) is a new
  store and is indexed as before. No model call; one small file read per
  start.

- **Memory search runs without an embedding provider.** A deployment with no
  embedding model (`[embedding] model = ""`, or a section that cannot build)
  searches by keyword: every multi-round recall is BM25 alone, and the loop,
  the decider and the ranking are otherwise unchanged. Ingest, the cascade
  (rows with no vector) and the user profile (its direct path) work as
  before; reflection and skill extraction stay off. `GET /health` reports
  `vector_recall` among `disabled_features` instead of `memory_search`;
  `vector_search`, `hybrid_search` and `agentic_search` are no longer
  reported, and a missing rerank provider disables `knowledge` only. The
  startup hint about rows without a vector is logged only when an embedder
  is configured. Cost: no embedding calls at all on such a deployment; the
  decider prompt is the same size. (Reflection, skill extraction, rerank and
  the knowledge base are removed since: see Removed.)

- **The profile reads each segment once, with two prompts** (everalgo
  `ProfileExtractor.aupdate`; the 2.0 design approved on 2026-10-30). Every
  new segment updates the profile of each user who spoke in it from that
  segment only, with the pictures that user posted in it; the update replies
  `add` / `modify` / `deprecate` / `none`, each naming the segments and
  pictures it rests on, and a changed value is a `modify` of the item holding
  the old one. A consolidation call (drop, merge, split, re-file) runs only for
  the whole profile past 60 items, the traits past 10 or one name past 8, and
  runs again on the same part only once it has grown by 15 / 3 / 4 items, so
  the regroup loop is gone. It used to re-send every segment of every cluster newer than the
  profile on every update (67 lines for a 17-line segment, measured
  2026-11-11), re-adding what was on file and sending 63 of 82 updates on to
  REGROUP; the profile was 85% of the text write cost. Member-in-group
  profiles keep `[groups] member_profile_every` and use the same prompts.
  Measured on 2026-11-11 (DeepSeek V4.1 Flash, thinking off, temperature 0;
  cold estimate per 1,000 messages, off-peak / peak): LoCoMo conversations
  0-1 $0.200 / $0.400 before, $0.115 / $0.229 now, the profile 84.5% of it
  before and 73% now; MemEye's eight conversations $0.186 / $0.372 before,
  $0.146 / $0.293 now. The update prompt is bounded by the profile, not the
  history: at most 8.3K tokens on LoCoMo (17.3K before) and 9.9K on MemEye
  (32.7K). Reading the poster's pictures adds 10% to the update prompt on
  MemEye, $0.042 per 1,000 pictures off-peak. Needs everalgo-user-memory
  0.9.0rc6.dev6, vendored for the test phase (below).
  `extract_user_profile` now triggers on the pipeline's `EpisodeExtracted`
  with or without an embedding provider and skips a segment an item already
  rests on; the clusters no longer drive it (`trigger_profile_clustering`
  still runs, for Reflection). (Both are removed since: see Removed.)
- **everalgo-user-memory is a vendored local build for the test phase.**
  `pyproject.toml` pins `everalgo-user-memory==0.9.0rc6.dev6` and
  `[tool.uv.sources]` points it at
  `vendor/wheels/everalgo_user_memory-0.9.0rc6.dev6-py3-none-any.whl`, built
  from the everalgo branch `feat/profile-two-prompts` at `46b322c` (sha256
  `9e315019...`; provenance in `vendor/wheels/README.md`), so `uv sync`
  installs it with no manual step. The other everalgo pins are unchanged.
  It is replaced by the PyPI release once everalgo publishes; until then
  `pip install` of this tree cannot resolve it.
- **The profile reads traits again.** Reading one segment at a time, the
  two-prompt update first wrote almost no trait (9 on LoCoMo conversations
  0-1 where rc5 held 100, 12 on MemEye where it held 44): the rewrite had
  dropped rc5's rule to look for them. The update now asks for them,
  conduct towards others included, and the traits are consolidated together
  past 10, because a disposition on file comes back as a new trait, often
  under a new name, whenever a segment shows it again. Against rc5's final
  profiles (same judge): LoCoMo 93 of its 100 traits covered (49 before) with
  31 duplicates (rc5 83), MemEye 42 of 44 (20 before) with 12 (rc5 29). This
  costs $0.014 per 1,000 LoCoMo messages off-peak and nothing measurable on
  MemEye; no trait pass or re-read of older segments was needed.
- **One profile writer.** The picture call no longer writes profile items:
  its `profile` reply field and prompt block are gone (the picture prompt is
  shorter), and `profile_delta` with them. The profile update reads the
  segment's pictures itself, the poster's only (D17); `[extraction] vision`
  off attaches none.
- **One episode prompt for every chat cell, assembled by rule**
  (`chat.episode`, version 2). A text cell went through everalgo's
  `EpisodeExtractor`, a picture cell through an EverOS prompt that carried
  everalgo's whole system prompt -- its reply format and an empty language
  block included -- plus a second language block and a second reply format.
  Every cell, one-to-one or group, now makes one call with one registered
  prompt in two variants. A cell without pictures (`text`) is sent everalgo's
  episode prompt byte for byte, so it costs what it cost; a cell with pictures
  (`pictures`) is sent the same rules first -- the prefix both share in the
  provider's cache -- then the `<pictures>` block, one reply format with
  `images` and one language block (2,703 tokens with English named; 2,724
  before). Each variant's fingerprint is pinned. A cell is never given a
  picture it did not send: a text cell's reply loses any `images` it writes,
  and a reference `(img-N)` to a label the cell did not attach is taken out of
  the episode (version 1, which sent the picture block to every cell, measured
  both on 2 of 43 LoCoMo text cells). A text cell gains the picture call's
  reading: the lenient reply form, one warmer retry of a reply cut at the
  output cap, and the cap, which now grows with the conversation as well as
  the pictures (2,048 tokens, plus 1,024 per picture, plus one per four
  characters of conversation, at most 32,768; the longest of 78,597 text
  episodes on record, 2,613 tokens, came from a 24,120-character transcript
  cell). A picture cell gains everalgo's guards: a blank `content` / `summary`
  / `title` is unusable, and a summary wider than 400 units is cut at a
  sentence. A text cell's reply model is `EpisodeReply`; a picture cell's is
  still `ImageEpisodeReply`, its JSON schema unchanged. `everos prompts`
  prints the manifest of the prompts EverOS sends: every registered prompt
  with its version, fingerprint and variants, and the everalgo episode rules
  and profile prompts with their package version.
- **Every profile item carries its provenance.** A one-to-one item now
  carries `item_id`, `source_memcell_ids`, `source_start` / `source_end` and,
  when read off a picture, `source_image_ids`, as a member-profile item does;
  an item records the segments its own operation named, not every segment of
  the update. A member profile's history entry now also says what replaced
  the item: `superseded_by`, `deprecated_by` and `reason`, `merged_into`, or
  `reason` for a consolidation drop.
- **The one-to-one profile keeps what it replaced**, as a member profile
  does. A fact given a new value is a new item linked to the old one
  (`supersedes`, `since`), and every item an update or a consolidation takes
  out moves to a `history` list in `user.md` with what replaced it, at most
  200 entries (left out of the file while empty); an item changes in place
  only to gain evidence and sources. A read shows an item's replaced values
  under `previous`. Forgetting a picture also takes the replaced values read
  off it alone, and the undo puts them back.
- **Forgetting reaches the profile.** `POST /api/v2/memory/forget/session`
  now reaches `user.md` through the items' sources: an item, or a replaced
  value in its history, goes once every segment it rests on is forgotten
  (this conversation's, or one the owner forgot before), and the undo puts it
  back, with what a later forget took because of it; both responses carry
  `profiles` (items). The profile update never reads a segment of a forgotten
  conversation, nor a picture the owner (for a member profile, the group)
  forgot: a forget leaves the bytes on disk for its undo, so the update asks
  the forget log. The update, a forget and its undo hold one lock per user,
  so no update writes back what a forget took out. A one-to-one forget by
  message does not exist; a group forget by message already reached the
  member profiles.
- **A Reflection merge finishes on the merged episode itself.**
  `finish_reflection` runs on the merged episode's `EpisodeExtracted` event
  and deprecates the replaced members at once; it used to wait for the
  merged episode's atomic facts to be extracted first. (Reflection is
  removed since: see Removed.)
- `reflection_report.deprecated_fact_count` is kept in `system.db` for stores
  that have it and is always written `0`. (The table is no longer opened:
  see Removed.)
- **Extraction is sampled at temperature 0 again.** `[llm] temperature`
  (default `0.0`) is sent on every boundary, episode, profile, picture-route
  and group call that sets none of its own. Every request sent 0.0 until
  2026-10-30, when the provider's default became "leave it out"; DeepSeek
  then samples at 1.0, and two runs over the same 10K-word
  transcript (2026-11-11) cut 5 and 9 memcells and cost 50% apart. A retry
  after an unusable reply is sampled at `[llm] retry_temperature` (default
  `0.3`) when that is higher, since at 0.0 the identical request tends to
  reproduce the identical bad reply. `"none"` leaves either field out, for an
  endpoint that rejects it. `[decider]` and `[multimodal]` are unchanged.
- **Boundary requests put the window's count after the messages.** Every add
  re-sends the session's uncut buffer, and everalgo renders the count in
  front of it (`<messages count="N">`), so consecutive calls of a session
  shared a provider cache prefix only through the system prompt. The count
  now follows the records (`</messages>` then `<count>N</count>`), so each
  call repeats the previous call's records as a prefix; the records, their
  numbering and the system message are byte-identical.

- **The multi-round decider is greedy and reads the full episode body.**
  `[decider] temperature` (default `0.0`) is sent on every call. Left out, a
  vLLM endpoint applied the model's `generation_config` (Qwen3.8-27B:
  temperature 1.0), and byte-identical rounds repeated their picks only
  30-47% of the time. The decider always sees the longer of `episode` and
  `summary`, as the training and evaluation code does; the summary-only
  switch is removed (see *Removed*).
- **The multi-round decider asks and reads the way the exp code does.** Its
  requests carry no `response_format` by default: `[decider]
  response_format_mode` now defaults to `prompt_only` and is no longer
  inherited from `[llm]` (`json_object` / `json_schema` remain an opt-in).
  Its replies are parsed tolerantly -- JSON in prose or code fences, string
  indices, a missing or null `next_queries`, a reply cut at `max_tokens`
  after a complete object -- instead of being validated against a strict
  schema, which spent eight requests on each such reply and then fell back
  to the fixed top-3 core. One decider attempt is one request, and a failed
  round's trace keeps its last raw reply and token counts.
- **BM25 ties are ordered deterministically.** Episode, atomic-fact and
  picture keyword recall fetch past the limit until the tie group at the cut
  is complete, then order by score (to 5 decimals) and id, so a repeated
  search returns the same list; LanceDB returns tied rows in no fixed order.
- **`EVEROS_DISABLE_CASCADE`, `EVEROS_DISABLE_CASCADE_WATCHER` and
  `EVEROS_DISABLE_OME` are honoured again**, as the evaluation harness sets
  them (`1` / `true` / `yes`): each turns off `[cascade] enabled`,
  `[cascade] watcher_enabled` or `[ome] enabled`, ahead of `EVEROS_*__*`
  variables and the TOML. This reverses 1.6.0, which stopped reading them;
  `EVEROS_OME_MAX_CONCURRENT_RUNS` / `EVEROS_OME_RUN_TIMEOUT_SECONDS` stay
  unread.
- Every markdown write of a memory owner holds that owner's lock and first
  checks that what it was derived from still exists; a write whose source was
  deleted is skipped and logged as `memory_write_skipped_source_gone`.
- User clustering and the Reflection finish embed before taking their locks.
  (Both are removed since: see Removed.)
- A succeeded OME run no longer keeps the triggering event (the conversation
  text) in `ome.db`; failed and dead-letter runs keep it for diagnosis.
- `/get` lists an episode Reflection merged from several sessions with
  `"session_id": null` instead of failing the page.
- `/add`, `/flush` and `/memory/delete` refuse a `session_id` with leading or
  trailing whitespace or a control character (`422`). A daily-log file
  stores it as one trimmed line, so such an id was read back as another one
  and could not be matched by a session filter or deleted.
- A SQLite write that waited out `[sqlite] busy_timeout_ms` on another
  writer's lock answers `503 EXTERNAL_SERVICE_UNAVAILABLE` with the fixed
  message `"database is busy"` (retryable) instead of an unhandled `500`; the
  failed statement, without its values, is logged as
  `sqlite_lock_wait_timed_out`. A queued `/add` failing that way records
  `last_error` `"StorageError: database is busy"`.
- Server log tracebacks are plain text (innermost 15 frames, no locals)
  instead of rich's syntax-highlighted rendering, which blocked the event
  loop for seconds per exception on a loaded host. CLI and TUI logs share
  the same configuration.
- **A picture's profile items go only to the person who posted it.** Before,
  they went to every sender in the segment. In a one-to-one chat, a picture
  posted by the assistant therefore no longer adds items to the user's
  profile.

### Removed

- **Agent memory, the knowledge base, rerank and Reflection (EverOS 2.0,
  breaking).** EverOS 2.0 keeps chat memory -- one-to-one and group chats,
  text and pictures, profiles, scenes, group profiles, and update / forget /
  history -- and multi-round retrieval, with all of the infrastructure under
  them. Removed:
  - **Agent memory.** The agent pipeline; the `extract_agent_case`,
    `trigger_skill_clustering` and `extract_agent_skill` strategies; agent
    cases and skills (the markdown under `<app>/<project>/agents/<id>/`,
    `skills/skill_<name>/SKILL.md` included, and the `agent_case` /
    `agent_skill` index tables on LanceDB and Milvus); the case judge and
    `[agent_memory] case_gate_enabled`; the `agent` owner type; the
    `agent_case` / `agent_skill` memory types of `/memory/get` and
    `data.agent_cases` / `data.agent_skills` in its answer. A request to
    `/memory/search`, `/memory/get` or `/memory/delete` that names
    `agent_id` (or `context_group_id`, or one of the two memory types)
    answers `422 INVALID_INPUT`: "EverOS 2.0 has no agent memory: name a
    user (user_id) or a group; agent_id, context_group_id and the
    agent_case / agent_skill memory types are not accepted". `/memory/add`
    never took an `agent_id`; an unknown field there is ignored as before.
  - **`[memorize] mode`.** Memory is chat memory. A tool row
    (`role: "tool"`, or an assistant message with `tool_calls`) is still
    accepted by `/memory/add` and dropped before segmentation, as
    `mode = "chat"` did, and so is a tool row an older build left in a
    session buffer. Every memcell's `raw_type` is `Conversation`. The default
    was `"agent"`, which kept tool rows in the cells for the agent pipeline.
  - **The knowledge base.** Every `/api/v1|v2/knowledge/...` route (document
    upload, list, get, patch and delete, topics, categories, knowledge
    search) now answers `404`; `KnowledgeService`; the markdown under
    `<app>/<project>/knowledge/`; the SQLite tables `knowledge_document` /
    `knowledge_topic` and the LanceDB table `knowledge_topic`; `[knowledge]`
    and `[knowledge.search]`; the error code `EXTRACTION_EMPTY`. A project's
    scope stays: `project_scope` group search and the `/projects` routes
    are chat features.
  - **Rerank.** The rerank component and `[rerank]` / `EVEROS_RERANK__*`;
    knowledge search was its one user. `GET /health` loses
    `capabilities.rerank`, and `disabled_features` holds at most
    `vector_recall` (no embedder: keyword recall only) and
    `multimodal_upload` (no multimodal LLM or parser: a non-text chat item
    is not read into text); `knowledge`, `reflection` and
    `skill_extraction` are gone.
  - **Reflection.** The weekly `reflect_episodes` merge (off by default),
    `finish_reflection`, the reflection orchestrator, the
    `reflection_report` table and `docs/reflection.md`. Scenes replace it
    ([docs/scenes.md](docs/scenes.md)). `custom_instructions/reflect.yaml`
    stays: the scene narrative call (`chat.scene_update`) reads it.
  - **Profile clustering.** `trigger_profile_clustering`, which fed only
    Reflection (the profile update reads each new segment directly). Scenes
    keep their clusters, and `[clustering]` stays as their geometry.
  - **Backfill's `clusters` and `skills` phases.** `everos cascade backfill
    --phase` takes `vectors` or `all`.
  - **Events.** `EpisodeExtracted` loses `source` and `reflection` (a run an
    older build persisted still reads); `AgentPipelineStarted`,
    `AgentCaseExtracted`, `ProfileClusterUpdated` and `SkillClusterUpdated`
    are gone.
  - **Dependencies.** `everalgo-agent-memory` and `everalgo-knowledge`.
    `everalgo-parser` becomes a direct dependency: it came in through
    `everalgo-knowledge`, reads a chat message's non-text items into text,
    and brings Pillow, which the picture store needs. `everalgo-rank` stays:
    multi-round search uses its RRF.
  - **Docs.** `docs/knowledge.md` and `docs/reflection.md`.

  Compatibility:
  - **Configuration.** `[agent_memory]`, `[knowledge]` (with
    `[knowledge.search]`), `[memorize] mode` and `[rerank]` are accepted and
    ignored with a `config_key_ignored` warning, so an `everos.toml` an older
    `everos init` wrote, and exported `EVEROS_RERANK__*` /
    `EVEROS_MEMORIZE__MODE` variables, keep loading
    ([docs/configuration.md](docs/configuration.md#unknown-keys-are-rejected)).
    An `ome.toml` section naming a removed strategy only logs
    `config_override_unknown_strategy`, and a run of one that a store
    persisted is dropped with `ome_run_dropped_unknown_strategy`.
  - **Stores.** The store-version gate still refuses 1.x stores. A 2.0
    store written before this change keeps its leftovers untouched: the
    `agents/` and `knowledge/` markdown matches no cascade kind and is
    never scanned; the `agent_case`, `agent_skill` and `knowledge_topic`
    index tables and the `knowledge_document`, `knowledge_topic` and
    `reflection_report` SQLite tables are neither opened nor created; the
    `memcell.case_*` columns stay unused; `cluster` rows of kind
    `user_memory` / `agent_case` are never read. An episode an earlier
    Reflection merged (`parent_type: cluster`) is still read, listed and
    searched like any episode, and the history of the episodes it replaced
    still shows `merged`. `everos cascade sync --force-all` re-queues every
    ledger row; a row of a removed kind is then marked failed ("no handler
    registered"), which `cascade status` counts and `/health`'s `healthy`
    ignores ([docs/storage_layout.md](docs/storage_layout.md)).
  - **Cost.** Write side: model calls on a chat are unchanged (a
    differential run against the base sends the same requests and writes
    the same markdown, index rows and answers); two background runs less per
    one-to-one segment (`extract_agent_case`, which skipped a segment
    without tool calls before any model call, and
    `trigger_profile_clustering`, with its one embedding call per episode)
    and one less per group segment (`extract_agent_case`); a conversation
    with tool calls no longer pays agent-case and skill extraction calls (a
    chat with fewer than three tool-call rounds never paid them), and a
    deployment that had switched Reflection on no longer pays its weekly
    merge. Read side: unchanged; search never covered agent memory or the
    knowledge base.
- **Hard deletion (EverOS 2.0).** The `Deleter` behind `/memory/delete` and
  `/delete_by_ids`, its race tests and its real-server acceptance suite
  (`tests/acceptance/memory_delete`, `make acceptance`) are gone; the routes
  are forgets (see Changed).
- **Atomic facts and foresight (EverOS 2.0).** Chat memory keeps episodes,
  picture memories and profiles. The `extract_atomic_facts` and
  `extract_foresight` strategies, the `AtomicFactsExtracted` and
  `UserPipelineStarted` events, the `users/<id>/.atomic_facts/` and
  `.foresights/` daily logs, the `atomic_fact` and `foresight` index tables
  (LanceDB and Milvus) and their cascade handlers are gone; everalgo still
  ships the two extractors, EverOS no longer calls them. A store written
  before keeps its leftover files and tables untouched and ignored (deletes
  no longer reach them); an `ome.toml` naming the two strategies only logs a
  warning; a cascade change still queued for them is marked failed; a
  background run still pending for them (or for the removed events) is
  dropped with a warning instead of failing in the scheduler.
- **Every search method but multi-round retrieval.** `keyword`, `vector`,
  `hybrid` and `agentic` are removed (the vector and hybrid routes ranked
  atomic facts); `llm_multiround` is the default and the only method. A
  request naming another method answers `422 INVALID_INPUT` saying only
  `llm_multiround` is supported. Gone with them: the request fields `radius`,
  `min_score` and `enable_llm_rerank` (now unknown fields, `422`), the
  response's `episodes[].atomic_facts`, `agent_cases` and `agent_skills`, and
  `[observability] recall_hit_threshold` (no calibrated score is left).
- **Agent memory search.** Multi-round retrieval covers user and group memory;
  a search with `agent_id` (or `context_group_id`) answers `422` and points at
  `POST /api/v2/memory/get`, which still lists an agent's cases and skills,
  in a group's context too. Cases and skills are still extracted and indexed.
  (Agent memory itself is removed since: see the first entry above.)
- `[decider] full_text` and `EVEROS_LLMMR_DECIDER_FULL_TEXT`. Its default
  showed the decider only the stored `summary`, a 200-character prefix of the
  body on most stores, which cost 9.3 points of LoCoMo core-only accuracy; a
  value still set is ignored.
- `[decider] invalid_reply_retries`. The decider reads its replies itself, so
  an unusable reply is one failed attempt of `[decider] retries`; a value
  still set for it is ignored.

### Security

- **The delete aliases take the gateway's identity.** With `[access]
  identity_header` set, `POST /memory/delete` and `/memory/delete_by_ids`
  ignored the asserted asker: one identity could forget another user's whole
  memory. Now a personal delete reaches the asserted user's own memory only,
  as every other personal route does: a `user_id` naming someone else is 403
  before anything is forgotten, a `session_id` alone forgets the asker's copy
  of the conversation, and `/delete_by_ids` refuses (403, nothing forgotten)
  an id of another user's episode; the forget log records the asker. A group
  delete stays an owner's, an admin's or the deliverer's. Without a gateway
  nothing changes.
- **A backdated message no longer opens a group's history.** A sender who is
  not a member joined at their first message's timestamp, so one
  `/memory/add` message dated before the group existed made a stranger a
  member of its whole history. Now, in a group that already holds messages,
  an implicit join from a batch whose asker is not an owner, an admin or the
  group's deliverer opens no earlier than the group's newest message (one
  millisecond after it, when the message is older). Managers and the
  deliverer still backdate, and a time-ordered import -- new senders never
  speak before what the group holds -- joins every sender as before.
- **A group without an owner or admin is managed by its deliverer.** Any
  current member could change such a group's meta and roster, and so lift
  their own window: open the history to every member, name themselves
  deliverer, or promote themselves and backdate a join. Now only the
  deliverer changes its meta and roster (and runs its group deletes), a
  member may still leave, and an owner or admin the deliverer names manages
  it from then on. A group a batch without an asker creates has neither: give
  a group its owner or deliverer when it is created. Implicit groups take no
  roster change, as before; forgetting is unchanged.
- **A request without the configured identity header is refused.** With
  `[access] identity_header` set, a request that did not carry the header was
  still served on its body's `asker_id` (and a personal read without one as
  its owner). Now it is 403 on every memory, group and project route.

### Fixed

- **Two messages with one timestamp, each sent alone, are both kept.** A
  message's server id was `m_<session>_<timestamp>_<index in its batch>`, and
  the buffer skips an id it holds: a user's message and the assistant's reply
  sent one per request in the same millisecond (or the same second, with
  timestamps in seconds) both got index `000`, and the second was dropped
  while both answers said `accumulated`. The id is now
  `m_<session>_<timestamp ms>_<digest>`, the digest of the sender and the
  content as sent (and a copy number for equal messages inside one batch,
  which are both kept), so two different messages never share one, and the
  same message sent again -- in any batch, at any position -- has the id it
  had: it is skipped and counted in the answer's `duplicates`, which a
  one-to-one batch now reports too (a re-send of buffered messages no longer
  makes a boundary call). Messages with one timestamp keep the order they
  arrived in; before, the id's index ordered them. A one-to-one `message`
  forget names a message by this id ([docs/api.md](docs/api.md#messageitem)).
- **The prompt manifest covers every text sent to a model.** `everos prompts`
  listed EverOS's own prompts and four of everalgo's, but not the
  output-language rules every extraction prompt carries, the parser prompts
  (a picture read into text when `[extraction] vision` is off, a document, a
  recording) or the four custom-instruction files appended to the boundary,
  episode, profile and scene prompts: two stores with one manifest could have
  been built with different prompt text. The manifest (version 2) now lists
  `everalgo.user_memory.language_rules`, `everalgo.parser.prompts` and one
  `custom_instructions.<operator>` entry per file, with the prompt it is
  appended to, whether it is on, and the fingerprint of its text when it is
  ([docs/cli.md](docs/cli.md#everos-prompts)).
- **A picture's copy is labelled as what it is.** The extraction model and
  the decider were sent every picture copy as `image/jpeg`, though a greyscale
  picture's copy is PNG; a strict endpoint may refuse the mismatch. The label
  is now read off the bytes, as the image route already did.
- **A search result's `thumbnail_path` names a file that exists.** Only the
  answering model's copy was rendered on demand; the decider's copy was named
  but written only when a decider was shown the picture, which by default
  (`image_candidates_per_round = 0`) is never. Both copies are now rendered
  when a result names them, and a picture the store cannot copy names no file
  (both paths empty).
- **`[images] thumbnail_edge` is retired.** It was accepted and had no
  effect: the thumbnail is the decider's copy, `[decider]
  image_thumbnail_edge`. The key is now ignored with a `config_key_ignored`
  warning, and gone from the shipped `default.toml`.
- **A bad configuration value exits `3` with the key, not a traceback.** A
  value no setting accepts, an unknown timezone, a past-trigger pattern that
  is not a regular expression and an `everos.toml` that is not TOML raised a
  validation error with a traceback and exit code `1`. They now fail the load
  as an unknown key does: every bad value at once, with its key, where it was
  set and what is wrong (never the value), and exit code `3`.
- **`everos config show` prints every section.** It printed 12 of the 24
  sections (no `[decider]`, `[groups]`, `[images]`, `[access]`, ...); it now
  prints them all, sub-tables included, secrets masked, and
  [docs/configuration.md](docs/configuration.md) documents every key (60 were
  missing, `[decider]`, `[observability]` and `[vision_embedding]` among
  them; a unit test now fails on an undocumented key).
- **A `### ` line or an entry marker in a memory's text no longer cuts it
  short.** The entry reader split sections on any line starting with `### `
  and entries on any close marker, so an episode whose text held `### Day 2`
  (or a summary with a heading, or a quoted `<!-- /entry:... -->`) kept its
  full text in markdown while the index, `/memory/history` and every later
  rewrite (update, forget, undo) read only the part before it, with nothing
  logged. Such a line, and a marker anywhere in a value, are now written
  behind one more backslash and read back without it; existing entries read
  unchanged ([docs/storage_layout.md](docs/storage_layout.md#4-entry-id-encoding)).
- **A segment whose extraction did not finish is extracted again, once.** Its
  memcell was committed and its messages left the buffer before the episode
  call ran; when that call failed (a provider's 5xx or timeout), or the
  process died before the episode was written, nothing ever extracted the
  segment again: a flush found an empty buffer, a restart did nothing, and in
  a group the client's re-send was skipped as duplicates -- the segment was
  lost for good. Every memcell is now written marked (`extraction_retry`, a
  column added to `memcell`) until its extraction succeeds, and the session's
  next `/add` or `/flush` -- in the same process or after a restart --
  extracts a marked segment again: a flush at once, an add that sends any of
  its messages again at once (the re-send is the segment's retry: its
  messages are not taken again, nor counted as duplicates), any other add
  once 30 seconds have passed since the failure; up to three attempts, after
  which it is left and logged (`memorize_cell_extraction_abandoned`). A retry
  that fails again never fails the call that made it; one that succeeds makes
  it answer `extracted`. One call holds a segment at a time and an owner
  whose copy was written before a failure is not written again, so a segment
  is extracted once; a group's ledger records a batch's keys once it is
  stored. A segment is not extracted while a forget of its conversation (a
  user's: or of everything, or by condition; a group's: or of a member)
  stands, and its messages are then new again, so a conversation sent again
  after its delete is taken ([docs/api.md](docs/api.md#asynchronous-mode)).
  The fault suite's three strict expected failures for this (a failed episode
  call, a crash between the memcell and the episode, a failed profile
  update) now pass; its `episode_write` pause point wraps the write every
  extraction makes (`_append_many`), so the crash test stops where it says.
- **A millisecond timestamp before 2001-09-09 no longer answers `500`.** The
  add API read every value below `10^12` as seconds, so `978307200000`
  (2001-01-01 in ms) landed in year ~33,000 and raised. A value below `10^10`
  is now seconds (up to 2286-11-20), every larger value milliseconds (from
  1970-04-26 on), for `messages[].timestamp` and integer search filters
  alike; a timestamp past 9999-12-30 is `422`. Every timestamp of a batch is
  normalised to milliseconds before use, so a group's roster no longer
  records a seconds value as milliseconds.
- **A forget survives the server stopping in the middle of it.** A forget
  wrote its log entry last, so a server killed after its marks and before
  that entry left the entries forgotten under no forget id: sending the
  forget again found nothing to do, and no undo brought them back. The entry
  is now written first, in a `forgetting` stage no read counts, and the run
  saves into it, ahead of each step, the change the step is about to make; it
  is completed with the run. A run the server did not finish is reversed from
  that record when the server starts again (or by the next forget or undo of
  the owner): marks go back, versions it wrote leave markdown (one it was
  writing is found by its text), profile values come back once whether or
  not the document took the change, message marks are released, and the
  entry ends `rolled_back`. A run that fails in-process is reversed by the
  same steps (when even that fails it stays an incomplete forget, as
  before). An undo is marked `undoing` before its first change and finished
  the same way, a copy or a value it already wrote back taken as written,
  never written twice.
- **An index write that finds the disk full is retried until it lands.** It
  was filed as unrecoverable: the file stayed `failed_permanent` after the
  disk was freed, unsearchable until `cascade fix`, while `/health` stayed
  healthy. A full disk (lance's `No space left on device`, or `ENOSPC`) is now
  transient: the file waits out a back-off -- `[cascade]
  storage_full_retry_seconds` (30 s), doubling up to
  `storage_full_retry_max_seconds` (15 min), one attempt and one warning per
  wait, outside the retry budget -- and is indexed with no manual step once
  space is freed. `cascade status` and `/health` count the files
  (`waiting_for_space`), and the cascade reads unhealthy meanwhile.
- **A failed group-profile update says so.** Its run was recorded as a
  success, the failure visible only in a warning and a metric. It now fails
  its run -- dead-lettered by the engine with the reason and what became of
  the batch, and no retry at once (the back-off is unchanged) -- and
  `/health` counts it in a new `ome` block, with every other strategy run
  that failed with no retry left since start.
- **`/health` reads the embedder an application was given.** The startup
  embedding probe built its own client from the `[embedding]` settings, so an
  application given its providers read `capabilities.embed: false` with a
  working embedder; the probe now asks the one it was given.
- **An assistant's implicit join no longer hides a group's history from
  `audience = group`.** The intersection counted assistant members, so an
  assistant's first reply cut everything said before it from the deliverer's
  default search. It now intersects the windows of current people only (the
  deliverer was already left out).
- **A forgotten group picture is quoted by no constraint.** A constraint's
  quote is the raw message, which kept the forgotten picture's
  `[IMAGE: <hash>]` reference; quotes (and the segments the binding call is
  shown) are now read without the pictures the group forgot.
- **Reads over the asker's groups read only the asker's rosters.**
  `include_groups`, `project_scope` and `GET /projects/{id}` read the roster of
  every group in the space -- about 600 SQL statements and half a second per
  request at 200 groups. They now ask the roster mirror for the asker's
  groups first (one indexed query) and read only those.
- **A record stays forgotten while a wider forget covers it.** Forgetting
  an episode and then its conversation (or everything) logged nothing of
  the episode in the wider forget -- it was forgotten already -- so
  undoing the first forget brought the episode back, searchable, with its
  profile item back in `user.md` and its picture file back in
  circulation, while the wider forget still stood. A forget now records
  what its target reaches that another live forget already holds
  (`covered` in the forget log): an entry, a profile value or a picture
  file stays out while any live forget marked, took or covers it, and the
  wider forget's undo brings it back.
- **Undo the later forget first, then the earlier: it works now.** Two
  forgets that rewrite one record (two pictures of one episode, two
  messages of one segment) left the earlier forget un-undoable (`409`
  for good), even after the later one was undone as the docs instruct.
  The undo now follows a later forget undone since to the version current
  now (the log keeps what each undo wrote, `restored`).
- **An edit no longer locks a forget for good.** After `/memory/update`
  of the version a forget wrote, that forget answered `409` and no route
  undid the edit. The undo now writes the version the forget replaced
  over the edit as a three-way merge per field: what the person left as
  the forget wrote it comes back, the rest stays the person's, and a text
  both changed is merged word by word when the changes do not meet. It
  answers `409` -- naming the edited version -- only when the edit changed
  the very text the forget changed.
- **A forgotten message's facts leave the user's profile.** After a
  `message` forget the profile items resting on the segment stayed while
  the user wrote anything else in it, so the fact the message stated
  stayed in `user.md` and came back in search. The one-to-one profile now
  follows the stale rule a member's profile in a group does: such an item
  goes, its remaining segments are queued in `user.md`, and the user's
  next profile update writes it again from what they still say, without
  the message (no extra model call); the undo puts it back and clears the
  queue. A user's undo also marks the forget's messages readable again
  (only a group's did).
- **A buffered message forgotten before its segment was cut is never
  read.** It still went into the next boundary call and the profile
  update, and `user.md` got an item from it. Boundary detection now keeps
  a row a live forget names in the buffer, out of every cut and every
  model call, until the undo; the one-to-one profile update reads its
  segment without forgotten messages.
- **A forgotten picture's bytes answer `404`.** `GET
  /memory/images/{image_id}` kept serving the bytes of a picture a user's
  forget took out of circulation. While no live memory refers to the hash
  and a forget holds it, every variant now answers as a missing picture;
  the undo serves it again.
- **A forgotten group picture is quoted by no group-profile item.** An
  item's evidence quote is the raw message, `[IMAGE: <hash>]` included, and a
  group's picture forget left it as it was: `get group_profile` and a
  search's `group_profile` sources kept naming the forgotten picture. The
  forget now writes each such item again without the reference (no model
  call, as for an episode) and marks forgotten every version that quotes it;
  the undo writes the quote back.
- **A picture forget keeps the text's whitespace.** Cutting a picture's
  reference out of an episode or scene joined the whole body on single
  spaces (paragraphs and lists became one line, a literal `()` vanished).
  Only the reference and the spaces before it go now.

- **One broken markdown entry costs that entry only.** A hand edit that
  broke one entry -- a timestamp that no longer parses, a deleted close
  marker -- failed the projection of its whole file (no entry of it was
  indexed again, `failed (retryable=FALSE)`), and a missing close marker
  silently dropped every entry after it, from the index as from history.
  Now the readers skip that entry and read the rest: the cascade indexes
  every other entry of the file (a row it already held for the broken one
  keeps its content) and marks the file done; `read_all`, a group's profile
  items and a scene's newest version skip it; a read that asks for that
  entry by id, segment or picture -- a forget of a session, a message, a
  picture or a record does -- still fails on it, so such a forget never
  misses what it was asked to forget (a forget of a whole owner walks every
  entry and skips it, as it did before). Each skipped entry
  is logged as `markdown_entry_unreadable` with its file, entry id, line and
  reason (it replaces the group profile's own `group_profile_entry_unreadable`;
  a profile item is still `profile_entry_dropped`, now with the file, its
  bucket position and the reason), counted on the file's ledger row (new
  nullable column `md_change_state.unreadable_entries`, added at startup),
  and reported: `everos cascade status` prints `unreadable entries: N in M
  file(s)` and names the files, and `GET /health` carries
  `cascade.unreadable_entries`, informational like `failed_permanent`. In
  `group.md` an unreadable roster entry never widens who may read the group:
  the member it names is left out of the roster until it is fixed, an entry
  that names no member still fails the file, and a roster change is refused
  while the log holds an entry it cannot read. An entry left without its
  close marker keeps its id: the next append never reuses it. Well-formed
  files read, index and answer exactly as before.
- **A search over several groups attaches each group's own profile items.**
  An item id (`gp_<date>_<seq>`) is unique within its group only, and the
  group profile a `group_ids` or `project_scope` search attaches was folded
  by id alone: the first item two groups wrote on the same day was one item,
  so the response showed one group's decision twice and never the other's.
  Items are told apart by group and id now.
- **`everos memory` keeps a settings warning off its JSON.** A key the
  settings ignore is named while they load (`config_key_ignored`; benchmark
  launchers still export `EVEROS_LLM__SEARCH_TIMEOUT_SECONDS`), before the
  settings that send the command's logs to standard error exist, so that
  warning was printed on standard output in front of the answer and a script
  could not parse it. The command points its logs at standard error before it
  loads the settings.
- **A user and a group with the same id no longer share forgets.** Users and
  groups share one id space, but the forget log was read by owner id alone:
  a personal undo of a picture or a conversation could reverse the group's
  newer forget of it instead of the user's own, and the pictures one of them
  forgot were left out of the other's profile updates. Every lookup is keyed
  on the owner's id and type now, and a group's undo by `forget_id` also
  checks the app and project it was made in.
- **A member's profile reads a segment once.** A segment event delivered
  again after the member's profile update had read it (a replay) queued the
  segment again, so the next update read it twice and could add its facts a
  second time. The member path now skips a segment an item already rests on,
  as `user.md` did; a forget that needs a segment read again queues it
  itself, as before.
- **A forget drops the profile summary on both profiles.** The summary is
  written over every item, so it may repeat the one a forget took out. A
  member's profile in a group already cleared it; a user's `user.md` kept it.
  Now both clear it whenever a forget removes an item (a replaced value alone
  leaves it): the next profile update writes it again from the items that
  remain -- everalgo derives it on every update, so no flag is kept -- and an
  undo puts it back while no update has. A user's forget records the dropped
  summary beside the items it took (part `summary`), which also replaces the
  `profile_summary` the owner forget kept in its undo record. Cost: none.
- **The one-to-one picture reader no longer imports group code.**
  `without_pictures`, which takes a forgotten picture out of a segment's text,
  moved from `memory.groups` to the shared `extract.pipeline.pictures`, which
  the personal profile, the group pipeline and a group forget all read with;
  the behaviour is the same. A new import contract (`pictures-neutral`) keeps
  it that way.
- **No tzlocal warning from a crontab check.** Validating a strategy's
  `Cron` trigger or an `ome.toml` `cron` override asked APScheduler to build a
  trigger in the OS's timezone; in a container with none, tzlocal warned
  ("Can not find any timezone configuration") on every CLI command and at
  every import of the strategies. The check names UTC now: a syntax check does
  not depend on the timezone, and the running scheduler is unchanged.
- **Undoing a forgotten conversation re-indexes its episodes at once.** The
  undo handed the cascade the restored episodes' entry ids where it expects
  their files, so nothing was projected again and the episodes stayed out of
  search until the next full scan of the memory root.
- **A refused picture no longer reaches the parser.** With `[extraction]
  vision = true`, a picture over an `[images]` limit (or not a picture) was
  still handed to the `[multimodal]` parser: a paid call whose text was then
  dropped, or -- with no `[multimodal]` configured -- an
  `LLMNotConfiguredError` that failed the whole `/add`. It is now counted in
  `rejected` and never sent to a model.
- **`[extraction] vision = false` delivers pictures again.** Every picture was
  marked refused, parsed anyway, and its parsed text dropped, so no picture
  reached extraction. Pictures are parser input on that route, rendered as
  `[IMAGE: name]` plus the parsed text.
- **A failed parser call fails one item, not the request.** `aparse_file`
  reports a failed model call as `LLMServiceError`; ingest caught only its
  subclass, so the documented per-item degradation never happened. The item
  is now marked failed and counted in `failed`.
- A refused picture was counted twice in the ingest result's
  `unparsed_non_text_count`.
- **A new group keeps its first decisions, linked to their topics.** The
  group-profile reply could not name a topic it created itself: `topic_id`
  and `about` took listed ids only, and topic ids are minted by code. On a
  new group's first batch the model either made an id up, and the verifier
  dropped every decision and dissent naming it (`unknown_reference`; 3 of 7
  operations in the 2026-11-11 integration check, so no decision was
  stored), or left the field empty, and the decisions lost their topic. A
  `topic.new` now gives its topic a label in `topic_id` (`new_1`, ...),
  which the reply's other operations use before or after it, and the
  verifier maps the label to the id it mints (prompt
  `chat.group_profile_update` v3). A `topic_id` or `about` that still names
  nothing drops only the link: the decision or dissent is kept, and the link
  is counted in `everos_group_profile_links_dropped_total` (by `reason`) and
  in the `group_profile_updated` log line. Targets and the evidence, member
  and assistant checks stay as strict as before; a topic is not updated in
  the reply that creates it. Four fresh groups on DeepSeek V4.1 Flash: 9 of
  9 decisions linked to their topic (0 of 8 before); still one call per
  batch, 61 more prompt tokens per call.
- **A picture call that repeats itself is cut early and sampled again, instead
  of running to the output limit and losing its cell.** At temperature 0
  (the extraction default since this release), a picture call on a
  text-dense picture could transcribe it into `ocr` over and over: on
  2026-11-11 two or three of MemEye's 168 picture calls (comic pages) did so
  in each of three builds, each ran to the 32,768 output tokens the builds
  allowed (57 times an average reply, 12-14% of the whole build's cost), and
  the cut reply failed its cell, which kept neither an episode nor picture
  memories. A picture call now asks for at most 2,048 + 1,024 output tokens
  per picture, never above 32,768 (`image_memory.reply_token_cap`; the
  longest of 914 complete replies measured was 2,423 tokens, for 8
  pictures), and a reply cut there is retried once at
  `[llm] retry_temperature`, with the same cap, like a reply that fails
  validation; a second cut fails the cell as before. The first attempt keeps
  the configured temperature. `OpenAIProvider.achat` takes `retry_cut` for
  this, and a call's `max_tokens` now lowers a larger one set in `[llm]
  extra` instead of being overridden by it. Cut replies are counted in
  `everos_llm_cut_replies_total` (`outcome`: `retried` or `failed`) and each
  cut attempt is logged as `llm_attempt_failed` with `reason=cut_short`.
- **A picture call is no longer paid twice for a slip of form in its reply.**
  On the picture route, 7 of MemEye's 168 extraction replies (4.2%, DeepSeek
  V4.1 Flash in JSON mode at the then-default temperature 1.0; 3 of CUE-Mem's
  125 at temperature 0) failed the reply schema, and each failure resent the
  call with every picture attached, although none said anything a valid reply
  would not have: an unknown key left empty (`"profile": []` inside a
  picture's entry, `"ocr_note": ""`, `"action_note": ""`), a profile operation
  whose closing brace was left off before the next one or the array's `]`,
  and a reply split into two objects. The reply is now read leniently in
  form: an unknown key whose value is empty reads as absent, `null` reads as
  empty (`ocr`, `source_image_id`) or as no entries (`images`, `profile`),
  one entry given bare reads as a list of it, and a reply that is not valid
  JSON only because an array element was left open or the reply was split
  into objects with distinct keys is mended before it is validated
  (`everos.memory.extract.reply_form`). Meaning stays strict: an unknown key
  that carries anything, a missing description or episode, a wrong type and
  any other syntax error still fail and are retried as before. The prompt
  and the JSON schema the call sends are unchanged; on the measured replies
  the schema failures drop from 7 to 0 (MemEye) and 3 to 1 (CUE-Mem).
- **A group flush no longer saturates the SQLite pool.** Each member's
  pending segments are read in one query, and at most `[sqlite] pool_size`
  members read at a time; the profiles written are unchanged.
- **Three races between `/memory/delete` and concurrent writers** that made
  the delete jitter test fail or hang about one run in five: a cancelled
  request cut a session's group lookup mid-read; a by-id delete read an
  episode's cluster outside the cluster lock; a skill file moved by a
  whole-agent delete was read as an error.
- **An owner whose conversation is one memcell gets a profile.** The profile
  strategy resolved the memcells of the triggering cluster through the
  episode index, which the cascade fills on its own schedule, so the episode
  that triggered the run was usually not indexed yet: every run missed its
  newest memcell, and an owner with a single memcell logged
  `profile_extraction_below_min_memcells memcell_count=0` and got no profile.
  The event's own memcell is now always selected and read from the SQLite
  ledger, as the direct (no-embedding) path already did.
- **Extractions are written in the conversation's language.** No extraction
  prompt named a language, so the model was left to work it out, and the 2.0
  backbone answered English picture cells in Chinese -- episode, picture
  descriptions and profile items together (2.0 stores of English benchmarks:
  4.6% to 45% of picture descriptions in Chinese). The language read back off
  such a reply then pinned the cell's atomic facts and froze the owner's
  profile language. The language is now resolved in code from the session's
  human messages and named in every prompt of the cell (episode on both
  routes, picture descriptions, picture profile items, atomic facts, profile,
  foresight): one vote per message, picture references, code blocks and links
  ignored, and a cell with too little text of its own (a picture with no
  caption) reads its session's earlier cells (`MemcellStore.recent_in_session`).
  A group segment gets its language the same way, named to the group
  episode on both routes and carried on to the poster's member-in-group
  picture items, the member-in-group profile updates (an update on a flush
  resolves it from the segments it reads) and the scene narratives.
  Only a session without a letter anywhere still leaves the choice to the
  model. An explicit language is honoured as before: a profile's stored
  `primary_language` still wins. Language detection also stopped reading long
  English text as Spanish or German on two accented names: German and Spanish
  markers must now outnumber English function words too. The one-to-one
  golden (`tests/fixtures/one_to_one_golden/golden.json`) is regenerated: its
  fake model writes a digest of the prompt into the episode, so naming the
  language changes the episode text and what derives from it (content
  hashes, vector digests, search scores and order); its events, adds and
  picture memories are unchanged.
- `/add` and `/flush` no longer leave half-written state when a SQLite write
  lock times out: boundary detection records the memcells, the session's
  status and the remaining buffer in one transaction. Before, a timeout at
  the second or third write left memcells that were never extracted, or
  buffered messages behind a `500` the caller could not act on.
- A burst of searches no longer fails with `Too many open files`: LanceDB
  search queries are capped per process (`[lancedb] max_concurrent_queries`,
  default `32`; metrics `everos_lancedb_query_wait_seconds`,
  `everos_lancedb_queries_in_flight`), where one agentic search fanned out to
  about 28 queries that each opened every data file of a table. The server
  also raises its open-file soft limit to 8192 at start (capped at the hard
  limit, never lowered; logged), from Linux's 1024 and macOS's 256.
- Search results are no longer converted with their vector columns, which
  every search caller discarded; the conversion ran on the event loop and was
  most of its busy time under search load, holding other requests' SQLite
  write transactions open.
- A second server using the same memory root now refuses startup before
  opening shared storage or resetting active cascade claims (#545). The
  existing exclusive lock stays held through storage shutdown.

- **On Windows, reading a memory file no longer fails while another write
  replaces it.** An `open` that races `os.replace` of the same file can be
  refused for a moment (`PermissionError`, errno 13). Writes already waited
  that out; reads did not, so a deletion could fail with
  `DeletionIncompleteError` (`500`) and a cascade projection could fail
  permanently. Every markdown read now goes through
  `MarkdownReader.read_text`, which retries with the writer's budget
  (about 2.5 s). Other platforms are unaffected.

- **A cluster's `count`, centroid and `last_ts` now include an episode that
  joined it while Reflection was merging it**
  ([#563](https://github.com/EverMind-AI/EverOS/issues/563)). Reflection
  wrote them from the members it read before its LLM call, so the new
  member stayed in the cluster but was not counted, and the user-profile
  gate, which sums `count`, undercounted.

- **One empty section no longer fails a daily-log file's whole batch.** The
  four daily-log handlers send every changed entry's text in one
  `embed_batch` call, and embedding endpoints reject an empty string inside
  the input array — text-embeddings-inference answers
  `400 Input validation error: inputs cannot be empty`. An entry whose
  embedded section is blank (an `atomic_fact` scaffold the extractor left
  behind before its sentence was written, a `foresight` with an empty
  `Foresight`, …) therefore failed the call for the *entire file*, so its
  other entries — already extracted, already on disk — never reached the
  index, and the projection failed permanently over content the user never
  sees. Blank texts are now left out of the request and written with
  `vector = None`, the same soft-degrade a missing provider produces;
  `vector_backfill` re-embeds the row once the md carries content.

- **A conversation dated before 2001-09-09 can be memorized.** Memcell and
  entry timestamps are epoch milliseconds, and before 2001-09-09 such a value
  is under the `1e12` threshold `from_timestamp` uses to tell seconds from
  milliseconds: read through that heuristic it landed in year ~32,700 and
  boundary detection raised `ValueError: year … is out of range`. Values
  EverOS computes itself are now read back as milliseconds; the add API's own
  `timestamp` keeps accepting seconds.

- **Deleting a session deletes its picture memories.** They name their
  session like the other kinds but were left out, so a deleted session's
  pictures stayed searchable with no memcell behind them, and re-sending the
  session added a second copy. An owner delete inventories them too.

- **A phone's multi-frame `.jpg` (MPO) is accepted.** Pillow reports such a
  file as `MPO`; it is filed and served as the JPEG it starts with, which is
  what every JPEG decoder shows, and its bytes are kept as sent, so the hash
  still names the file.

- **Re-projecting a daily-log file no longer commits a LanceDB version per
  entry.** Deprecation marks were written for every entry of every file on
  every projection, one `update` and so one table version each, even when
  nothing changed (a rebuilt store reached about 173k versions over three
  tables, queued on the table's write lock until they timed out as
  `VectorStoreBusyError`). Only rows whose stored mark differs from the file
  are written, one update per distinct value; a kind without a
  `deprecated_by` column is skipped.
- **A session-filtered search no longer returns other people's unprocessed
  messages.** The session's buffer is returned only to a sender of that
  session.
- **Picture bytes referenced by group memory are served only to askers who
  can see them.** Anyone else gets the same 404 as for a missing picture.
- **A roster change sent inline on `/memory/add` cuts the open buffer of every
  session of the group,** as the roster route does. It used to cut only the
  batch's own session, so another session's next segment could span the
  change.
- **Episodes read back from markdown keep their `sender_ids` and `language`.**
  The reader dropped both, so an episode an undo restored lost them.
- **`GET /memory/images/{image_id}?variant=thumbnail` labels a greyscale
  picture's copy `image/png`.** The copy is PNG for a greyscale picture, and
  the route labelled every copy `image/jpeg`.
- **A picture's copies show transparency as white.** The decider's thumbnail,
  the extraction model's copy and the answering model's copy flattened a
  transparent region onto white only when the original was a JPEG, which has
  no transparency; a PNG, WEBP or GIF reached every model with whatever colour
  lay under its transparent pixels, usually black. Every alpha channel,
  transparent palette entry and colour key is now blended onto white. A
  picture with nothing transparent renders as before, byte for byte, and a
  copy's format still follows the picture alone (PNG when greyscale, JPEG
  otherwise). `scaled_copy` no longer takes `format_name`. Copies already in
  `figures/.cache/` keep their old pixels: delete that directory to have them
  re-rendered on demand (it is always safe to drop).
- **`thumbnail_path` names the decider's copy with its real extension.** A
  search result named it `.jpeg` whatever the picture; a greyscale picture's
  copy is a `.png`. It now takes the extension the picture's copies have (its
  model copy's), so a caller resolving it finds the file.

### Known issues

- A decider reply whose `core` holds a string such as `"--1"` raises instead
  of being skipped (it passes the digit check, then `int()` fails), and the
  search fails. The exp code behaves the same; it is kept for parity and will
  be fixed in both together.
- **Group scopes:** there is no history route.
- **A group's query-encoder state is not built.** The design encodes group
  queries with the group's own state. No query encoder on this line takes a
  state, so group and personal queries are encoded the same way.
- **Milvus stores** have no migration to table schema versions 4 and 5. They
  report schema drift and need `everos cascade rebuild`.

## [1.6.0] - 2026-10-28

**One place builds the application.** Providers, storage adapters, the OME
engine, the cascade and the domain objects are now assembled once, by
`everos.service.build_application(settings)`, and the HTTP server and the CLI
both use the resulting `Application` through three facades:
`MemoryService`, `KnowledgeService` and `OpsService`.

### Added

- **An optional correctness gate for agent cases** (`[agent_memory]
  case_gate_enabled`, default `false`). On, the extraction LLM judges
  whether the work in an agent conversation was completed correctly (one
  extra call per conversation up to 50k tokens; longer ones are admitted
  without a call), and only admitted conversations become cases, instead of
  the complexity and value filters deciding. An admitted case drives skill
  extraction whatever its quality score. The judge is
  `everos.component.llm.LLMCaseJudge`.
- **The log level is configuration** (`[logging] level`, default `"INFO"`).
  `everos.service.configure_process(settings)` sets up logging, tracing and
  metrics once per process; `build_application` calls it, and the CLI no
  longer configures logging itself. `server start --log-level` and
  `EVEROS_LOG_LEVEL` still override the level for the server, and
  interactive commands still log `WARNING` (`INFO` with `--verbose`).
- **One language decision per cell, pinned downstream.** The episode is
  still written in whatever language the model picks from the conversation;
  that language is then detected from the episode text by Unicode script
  (`everos.core.utils.language.detect_output_language`, one of everalgo's
  `OutputLanguage` values) and carried on the episode and its
  `EpisodeExtracted` / `ProfileClusterUpdated` events. Atomic facts are
  extracted in it, and a profile is written in it and then frozen: the
  first profile extraction that knows the language stores it as
  `primary_language` in `user.md`, and later extractions keep that value
  even when an episode arrives in another language (changing it is a manual
  edit). Episode entries gain an inline `language` field. Both are left out
  while unknown, so episode logs and `user.md` written before this release
  read back with no language and extraction falls back to everalgo's own
  inference. The search index, the API and `/get` are unchanged.

### Changed

- **Internal module layout: one convention across `infra` and `service`.**
  Every `infra` subpackage now keeps its implementation in private
  directories behind its `__init__` facade (`cascade/_projections`,
  `cascade/_worker`, `markdown/_base|_mds|_readers|_writers`,
  `sqlite/_base|_repos|_tables`, `lancedb/_base|_repos`, matching `ome`);
  the backfill steps live in `service/ops/backfill/`; the index port
  package no longer re-exports the record tables, which are imported from
  `everos.infra.persistence.records`. Public facades are unchanged, and
  `import-linter` now forbids the outer layers from reaching into
  `everos.infra.cascade.**` as it already did for `persistence`. Logger
  names follow module paths, so a log filter on a renamed module (for
  example `everos.infra.persistence.sqlite.repos.*`) needs the `_repos`
  spelling.
- **`IndexRepository.find_by_owner` is removed.** It read rows by
  `owner_id` across every `(app_id, project_id)` scope; nothing called it
  (#480). Scoped reads go through `find_where` with an explicit filter or
  `find_by_owner_entries`.
- **The cascade and OME engine switches are settings.** `[cascade]
  enabled` and `[cascade] watcher_enabled` replace the
  `EVEROS_DISABLE_CASCADE` and `EVEROS_DISABLE_CASCADE_WATCHER` environment
  variables, and the new `[ome]` section's `max_concurrent_runs` and
  `run_timeout_seconds` (`0` disables the ceiling) replace
  `EVEROS_OME_MAX_CONCURRENT_RUNS` and `EVEROS_OME_RUN_TIMEOUT_SECONDS`.
  The old variables are no longer read; set the keys in `everos.toml`
  instead. *(Reversed after 1.6.0 for the two `EVEROS_DISABLE_*` switches:
  they are read again, as is the new `EVEROS_DISABLE_OME`; see
  [Unreleased].)*
- **Building the application requires `[llm]`.** `build_providers` (and so
  `build_application`) raises `LLMNotConfiguredError` when `[llm]` has no
  `api_key` or `base_url`, instead of building without an extraction LLM
  and leaving the server's start to refuse. The server still refuses to
  start, now at assembly; every `everos cascade …` command builds the same
  application, so those commands also need a configured `[llm]` and exit
  with code 3 and the toml section to edit when it is missing. Python
  callers that passed `Providers(llm=None, …)` must pass a client.
- **`GET /health` no longer reports `capabilities.llm`.** It was always
  `true` -- a server without an extraction LLM does not start -- so it is
  removed; clients that read it should drop the check. `Capabilities.llm` /
  `.decider` and `ProviderStatus.llm` are removed from `everos.service` for
  the same reason.
- **`everos.component.config` is renamed to `everos.component.yaml_loader`.**
  The package name no longer collides with `everos.config`. Update imports to
  `from everos.component.yaml_loader import YamlConfigLoader`; the class name
  is unchanged.
- **`MemoryService` write use cases are `add` and `flush`.**
  `MemoryService.submit_add` / `submit_flush` are now `add` / `flush`, the
  names the HTTP API uses; the inline `memorize()` / `flush()` steps are
  private. Python callers of the facade update the two names; HTTP clients
  are unaffected.
- **Local parser input handling lives in the parser component.** Application
  assembly supplies the raw-file builder to ingestion, and ingestion no longer
  falls back to a built-in one. `everos.memory.extract.parser.mapping` is
  removed; import `build_raw_file`, `read_file_uri` and `to_raw_file` from
  `everos.component.parser`; `build_raw_file` and `read_file_uri` now require
  a `settings_source` instead of falling back to `load_settings`. URI checks,
  allowlist rules, size checks and error behavior are preserved. The `file_uri_hydrated` log event is now emitted by
  the `everos.component.parser.input` logger; update log filters that matched
  the old module name.
- **Multi-round search receives explicit configuration readers.** Application
  assembly supplies tuning and trace-path readers, and `SearchManager`,
  `LLMRoundDecider` and `search_episodes_llm_multiround` now require them
  instead of falling back to a built-in reader. The existing settings cache,
  legacy environment overrides and per-call read timing are preserved. Retry
  defaults and execution rules are unchanged by this refactor. The
  `llm_multiround_bad_env_override` log event is now emitted by the
  `everos.config.search_runtime` logger; update log filters that matched the
  old module name.
- **Reflection delegates file grouping to the Markdown stores.** Retirement
  keeps the existing file order, partial-write failure behavior and report
  counts. The unreleased `EpisodeStore.deprecate` and
  `AtomicFactStore.deprecate` contracts now accept ordered `entries` and a
  `replaced_by` ID instead of a per-file replacement mapping.
- **Index synchronization and vector backfill share embedding rules.**
  Episode, atomic fact, foresight, agent case, agent skill and knowledge topic
  projections now use the same source-text and vector-column declarations.
  Backfill keeps its existing partial-failure recovery and historical text
  normalization behavior; no data migration is required.
- **An agent memcell that produces no case now records why.** everalgo
  0.6 extracts an agent case only from a closed trajectory (a user request
  through to the assistant's final answer); when boundary detection leaves
  an unclosed slice -- typically by cutting just before the final answer --
  that slice yields no case. The extraction outcome is now persisted on the
  `memcell` row: `case_extracted_at`, `case_skip_reason` (`NULL` exactly
  when a case was produced; otherwise everalgo's skip reason, or
  `no_agent_sender`) and `case_skip_detail_json`. Existing `system.db`
  files gain the three columns automatically at startup.
- **`everos.service.memorize_queue` is gone.** The `async_mode` queue now
  lives in the `MemoryService` package; `TaskItem`, `TaskPage`,
  `QueueHealth` and `TASK_STATUSES` are imported from `everos.service` (or
  `everos.service.memory`).
- **Entry points import only the service.** A new import-linter contract
  (`entrypoints-forbidden`) keeps `everos.entrypoints` off `everos.memory`,
  `everos.infra`, `everos.component` and `everalgo`; request / response
  DTOs (`GetRequest`, `SearchRequest`, ...) are re-exported by
  `everos.service`.
- **`everos.memory.ports.ClusterStore` speaks in cluster operations.** The
  port shrinks from eleven methods to seven: `add_member`, `remove_members`
  and `update_metadata` become one `replace_members(cluster_id, *, removed,
  added, member_type, centroid, count, last_ts, preview)`, which swaps the
  members and restamps the cluster in a single SQLite transaction (before,
  a crash between the three calls could leave the membership and the
  snapshot disagreeing). `list_ids_and_member_counts` and `count` are gone:
  read `len(cluster.members)` from `list_for_owner`. Custom `ClusterStore`
  implementations need the same change.
- **`everos.entrypoints.api.utils` is now `everos.entrypoints.api.dependencies`**
  (internal): it holds the FastAPI request dependencies (`application`,
  `extract_request_id`), named as FastAPI's project layout names them.
- **`session_lock_timeout_seconds` sits under `[memorize]` in the shipped
  `default.toml`** (it had landed under `[knowledge.search]`, where it was
  ignored). The value, and so the behaviour, is unchanged.
- **Timings are constructor arguments instead of module constants to patch.**
  `LanceRepoBase` takes `timeouts=LanceTimeouts(...)` (and `sweep_husks=`),
  `LanceIndexBackend` takes `schemas=`, `CascadeWorker` takes
  `maintenance_task_timeout_seconds` / `rebuild_conflict_backoffs_seconds` /
  `loop_restart_backoffs_seconds` / `on_unrecoverable`, `CascadeConfig` gains
  `drain_idle_limit_seconds`, `memory_root_lock` takes `poll_interval_seconds`,
  and `MemorizeQueueWorker` takes `stop_grace_seconds` / `mark_retry_seconds`.
  Every default is the previous constant. The lance husk sweep is public as
  `remove_empty_index_dirs`.
- **`everos.service.memory` is a package** (the facade stays importable from
  the same path). The process-wide session-lock registry
  (`everos.service._session_lock`) is gone: each `MemoryService` owns its
  per-session locks.
- **The operator, upload and queueing rules moved from the HTTP routes and
  the CLI into the service facades.** `OpsService.health()` assembles the
  `/health` payload (a failed probe still reads as an unhealthy block, never
  a 500); `OpsService.list_tasks` validates `status` and caps `page_size` at
  100; `MemoryService.submit_add` / `submit_flush` decide between running
  inline and queueing (`Queued` when queued); `KnowledgeService.create_from_upload`
  / `replace_from_upload` check the size, pick UTF-8 or the parser and
  extract, and `KnowledgeService.require_search_providers` is the
  embedding + rerank gate. HTTP status codes and bodies are unchanged.
  `everos.component.capabilities` is removed; `compute_disabled_features` is
  exported from `everos.service`.
- **Backfill is part of `OpsService`.** `everos.service.backfill` is removed.
  `OpsService.backfill(phase, auto_yes=, presenter=, summary=)` runs one
  phase or `"all"`, stops on a missing provider, a running server or a
  declined prompt, and returns a `BackfillOutcome`; each phase's result
  lands in the caller's `BackfillSummary`. Its collaborators are injected as
  one `BackfillDeps`, so the service layer no longer imports `infra`.
  `BackfillPresenter` gains `phase_started(phase)`; `BackfillPhase` is gone
  (the CLI keeps its own phase copy). The public result types are
  `VectorsPhaseResult`, `ClustersPhaseResult` and `SkillsPhaseResult`.
- **`Application` closes its storage whether or not it was started**, and
  `aclose()` is idempotent; `async with build_application(settings) as app:`
  closes on exit. The CLI uses it for every command.
- **`MemoryRootLockedError` (`everos.core.errors`) is what callers catch
  when another process holds the memory root.** The OME engine's
  `EngineLockHeldError` is a subclass; `everos.infra.ome.OmeLock` takes the
  same lock from outside a running engine.
- **The datetime helpers moved to `everos.core.utils.datetime`.**
  `everos.component.utils` is removed; import `get_utc_now`,
  `to_display_tz`, `UtcDatetime` and the other two-zone helpers from
  `everos.core.utils.datetime` instead. They are stateless tools every layer
  uses, the HTTP and CLI entry points included, so they live with the other
  cross-cutting runtime code.
- **The service layer's module-level functions are gone.** `everos.service`
  no longer exports `memorize`, `search`, `get`, `get_task`, `list_tasks` or
  the knowledge functions (`create_document`, `search_knowledge`, ...); build
  an `Application` and call `application.memory.*`, `application.knowledge.*`
  or `application.ops.*`. `everos.service.memorize`, `.search`, `.get` and
  the `everos.core.lifespan` / `everos.entrypoints.api.lifespans` packages
  are removed. `create_app()` takes `settings` or a pre-built `application`
  instead of `lifespan_providers`.
- **The process-wide provider accessors are gone.**
  `everos.component.llm.get_llm_client` / `get_decider_llm_client` /
  `get_multimodal_llm_client`, `everos.component.embedding.get_embedding_capability`
  and `EmbeddingCapability`, `everos.component.rerank.get_rerank_capability` and
  `RerankCapability`, and the `everos.component.multimodal` package are removed.
  Take providers from `everos.service.build_providers(settings)` (a provider
  that is not configured is `None`) or build one with the component factories;
  `resolve_decider_config` now requires its `settings` argument.
  `everos.component.parser.aparse_file` takes the multimodal LLM as `llm=`.
  `missing_config_error` moved from `everos.component.utils.config_hints` to
  `everos.config`. `set_metrics_registry` / `reset_metrics_registry` are
  replaced by `configure_metrics(registry)`.
- **The storage singletons are gone.** The module-level repository
  instances (`episode_repo`, `atomic_fact_repo`, ..., `md_change_state_repo`,
  `cluster_repo`, `knowledge_document_repo`, ...), the engine / connection
  accessors (`everos.infra.persistence.sqlite.get_engine` /
  `get_session_factory` / `dispose_engine`,
  `everos.infra.persistence.lancedb.get_table` / `get_connection` /
  `dispose_connection`, the Milvus `get_client` / `dispose_connection`) and the
  index facade functions (`startup`, `shutdown`, `ensure_business_indexes`,
  `drop_business_tables`, `read_embedding_model`, ...) are removed, together
  with `everos.infra.persistence.index.router` and the `lancedb.indexes` /
  `milvus.indexes` modules. Storage is now a set of objects built over one
  memory root: `SqliteDatabase` + `SqliteRepos.over(database)`,
  `LanceDbConnection` / `MilvusConnection`, and the `DerivedIndex` returned by
  `everos.infra.persistence.factories.build_derived_index(settings, root)`,
  whose `repositories` replace the old index instances. The SQLite repository
  classes are public (`ClusterRepo`, `MdChangeStateRepo`, ...; the knowledge
  topic row repository is `KnowledgeTopicRowRepo`) and take their session
  factory in the constructor. `CascadeOrchestrator`, `CascadeAdmin`,
  `CascadeScanner`, `CascadeWatcher`, `CascadeWorker`, `HandlerDeps` and
  `KnowledgeService` take the storage they use as constructor arguments, and
  `KindSpec.index_repo` is removed (a handler's repository comes from
  `HandlerDeps.index`).
- **One embedding provider per process.** The cascade handlers and the startup
  provenance check use the application's embedding provider instead of
  building a second one, so `[embedding] max_concurrent` is no longer doubled
  in practice.
- **`OMEConfig.jobstore_path` is required.** The engine no longer resolves the
  memory root on its own; the application passes `<root>/.index/sqlite/ome.db`.
- **Backfill logs under `everos.service.backfill`** (was
  `everos.memory.cascade._backfill`); update log filters keyed on the old name.
- **A merged episode with an empty subject no longer writes an empty
  `### Subject` section** to its markdown.
- **A profile JSON column that does not decode to a list reads as empty**
  instead of being passed through as-is.
- **Reflection runs in two steps.** `reflect_episodes` writes the merged
  episode; the new `finish_reflection` strategy deprecates the source episodes
  once the merged episode's atomic facts exist. If fact extraction for the
  merged episode dead-letters, the sources now stay searchable (they used to
  be deprecated anyway), and the old 120 s wait is gone.
- **`UserPipelineStarted` is published after its cell's episode is extracted**
  (it used to be published before any cell ran). Its only subscriber,
  `extract_foresight`, ships disabled.
- **Prompt slots are replaced by custom instructions.**
  `config/prompt_slots/*.yaml` (`enabled` + `template`, a whole replacement
  prompt) and `everos.memory.prompt_slots.PromptLoader` are removed. The
  user-memory operators now take `custom_instructions=`, appended after
  their bundled rules: configure them in
  `config/custom_instructions/<operator>.yaml` (`enabled` + `instructions`)
  for `boundary_detection`, `episode_extract`, `profile_extract` and
  `reflect`, read by `everos.memory.custom_instructions.CustomInstructions`.
  See [docs/custom_instructions.md](docs/custom_instructions.md).
- **Boundary detection is a domain object.** `everos.service._boundary` is
  replaced by `everos.memory.extract.boundary.BoundaryDetection`, which
  `MemoryService` receives instead of `llm` / `prompt_loader`. Both modes now
  run `everalgo.boundary.adetect_boundaries` (agent mode keeps tool rows in
  the cells and now honours `[boundary_detection]` hard limits too). When
  the model judges the trailing segment of a non-final call complete
  (`tail_complete`), that segment is closed into a last cell and extracted
  at once; a segment it judges open, or one no model judged, waits in the
  buffer for the next batch or `/flush`.
- **An unusable model reply is `everalgo.llm.LLMOutputError`**, no longer a
  `ValueError`. Boundary detection and episode extraction try it three times
  (`everos.memory.extract.llm_retry`, the one host-side retry), with no
  backoff (episode extraction used to sleep 1 s and 2 s between attempts).
  The session quarantine counts calls that end in `LLMOutputError` (after
  its retries) or in a `ValueError` -- a message everalgo refuses
  (`InvalidInputError`) or a row that maps to no conversation item, such as
  an orphan tool result -- which is never retried; other errors propagate
  uncounted.
- **`Episode.summary` is required** (everalgo 0.9 writes one for every
  episode and every merge) and `Episode.subject` defaults to `""`; episodes
  indexed before either existed read back as `""`. `AtomicFact.fact` is
  renamed `AtomicFact.content`. `Profile.explicit_info` /
  `implicit_traits` hold `ExplicitInfoItem` / `ImplicitTraitItem` (same
  shape as everalgo's); a profile item stored as a bare sentence reads back
  as an item with only its `description`. A stored item with no
  `description` is dropped when the profile is read (`/get`, `/search`'s
  profile, the next profile update), with a `profile_entry_dropped`
  warning; writing a profile stays strict. `/get` and `/search` return every
  profile item with all four keys (`description`, `category` / `trait`,
  `evidence` / `basis`, `observed_at`), the ones a stored item lacks as
  `null` -- an additive change.
- **Agent memory says why it stayed empty.** A memcell that yields no agent
  case logs `agent_case_skipped` with everalgo's `reason` (for example
  `too_few_tool_rounds`, `filter_rejected`) and its `detail`; a skill run
  logs `agent_skill_skipped` for a case below the quality floor and
  `agent_skill_op_dropped` for each operation the model proposed that was
  dropped. The case quality floor before skill clustering is everalgo's
  `should_extract_skill` instead of a copy of its threshold.
  `AgentCase.from_algo` moved to `everos.memory.mappers.agent_case`.
- **The multi-round retrieval decider speaks the everalgo LLM contract.**
  `method="llm_multiround"` asks its decider with
  `achat(..., response_format=...)`, so a reply must be the
  `{"core": [<int>...], "next_queries": [<str>...]}` object exactly (the
  provider retries an invalid one); a reply with prose around the JSON or
  string indices is no longer accepted, and fails the attempt instead. The
  trace field `decider.tokens` is `null` for the prompted decider (token
  usage is reported by the LLM call observer).
- **Index searches return domain records.** `memory.ports.IndexHit` (a row
  read by column name) is replaced by `SearchHit[T]` (its `stored` record and `score`), and
  knowledge-topic searches return `ScoredKnowledgeTopic`. Recall hands the
  rank algorithms everalgo `Candidate`s whose `metadata` holds only the
  fields an algorithm reads, with the stored records alongside
  (`RecallResult.stored_by_id`); responses are built from those records.
  An LLM rerank (`enable_llm_rerank=true`) no longer sees index-only columns
  such as `md_path`, `content_sha256` or the `*_tokens` columns in its
  candidate JSON. `everos.memory` no longer re-exports the `Algo*` aliases
  of everalgo's types.
- **The knowledge service works through two ports.** `KnowledgeService`
  takes a `KnowledgeStore` (markdown documents, taxonomy, uploaded
  originals; `MarkdownKnowledgeStore`) and a `KnowledgeCatalog` (the
  document / topic records; `SqliteKnowledgeCatalog`) instead of a memory
  root and two repositories. Replacing a document now extracts its topics
  before touching the old files. `LLMNotConfiguredError` lives in
  `everos.core.errors` (still importable from `everos.component.llm`).
- **`everos.service.knowledge` is a package** (the facade and its result
  types stay importable from the same path). The unused
  `KnowledgeService.parse_file` method is removed. Error types are no longer
  reachable through the module: import them from `everos.core.errors`.
- **`KnowledgeDocument` carries no storage path.** The `md_path` field is
  removed; the knowledge store finds a document by its id.
  `KnowledgeStore.delete_document` no longer takes `md_path`,
  `KnowledgeStore.update_document` returns a `WrittenDocument` (which now
  also carries `md_path`, the document's `index.md` relative to the memory
  root), and `KnowledgeCatalog.record_document` takes that `md_path` as a
  keyword. API responses are unchanged.
- **A request that needs a model while `[llm]` (or `[multimodal]`) is not
  configured answers 422 `PROVIDER_NOT_CONFIGURED`** instead of 500
  `INTERNAL_ERROR`.
- **Vector backfill's row work moved into the index package.** The vectors
  phase's row-level events (`cascade_backfill_row_*`,
  `cascade_backfill_batch_*`, `cascade_backfill_table_*`, ...) are now emitted
  by the `everos.infra.persistence.index.vector_backfill` logger instead of
  `everos.service.backfill`; update log filters that matched the old name.
  `BackfillDeps` takes a `vectors` backfill object in place of its
  `tokenizer` and `prune_retention` fields.
- **The vector width is read from `[embedding] dim` once, by the
  application, and nowhere else.** The index record classes no longer read
  the configuration while they are imported; they mark their embedding
  columns (`DenseVector`) without a width, and each backend builds its tables
  at the width the application hands it (`schema_for(model, dim=...)`,
  `LanceIndexBackend(connection, dim=...)`, `MilvusIndexBackend(connection,
  dim=...)`). The startup error "the index record classes were built for
  dim=... but configuration reports dim=..." can no longer occur, and the CLI
  no longer pre-scans `--root` before parsing its arguments. A table built at
  another width is still refused at startup, as schema drift. For code that
  imports the persistence package: `everos.infra.persistence.records` no
  longer exports `LanceModel` / `Vector`, `BaseLanceTable` is now
  `IndexRecord` (a plain pydantic model with no `to_arrow_schema`), and the
  LanceDB Arrow schema comes from
  `everos.infra.persistence.lancedb.base.arrow_schema_for`.

### Removed

- **`everos.infra.ome.testing` is no longer shipped.** `StrategyTestHarness`
  was a test fixture; it now lives in the repository's own test helpers.

### Fixed

- **The agentic episode search reranks on the episode text.** Its
  cross-encoder read `metadata["episode"]`, which the bridge to everalgo's
  prompt format had already turned into a `{"subject", "content"}` dict, so
  every passage was that dict's Python repr; it now reads the plain body.
  `method="agentic"` scores change accordingly (a fix, not a regression).
- **Agent skill clustering failed on the 2026-10-13 everalgo line.**
  `trigger_skill_clustering` awaited `cluster_by_llm`, which is now a
  blocking bridge that refuses to run inside the event loop, so no case was
  clustered and no skill extracted; it now awaits `acluster_by_llm`.

### Upgrade

One command, then a restart:

```bash
uv pip install -U everos      # or: pip install -U everos
```

(`'everos[multimodal]'` if that extra was installed.) The everalgo packages
are pinned exactly, so they move to the new contract line with it, with no
pre-release flag. The first start after upgrading migrates `system.db` on
its own (see below); Markdown, the index and the OME jobstore are untouched.

Four things a running deployment may have to change by hand; each is also
described under *Changed* above:

| If you | Then |
|---|---|
| set `EVEROS_DISABLE_CASCADE`, `EVEROS_DISABLE_CASCADE_WATCHER`, `EVEROS_OME_MAX_CONCURRENT_RUNS` or `EVEROS_OME_RUN_TIMEOUT_SECONDS` | move the values to `[cascade] enabled` / `watcher_enabled` and `[ome] max_concurrent_runs` / `run_timeout_seconds` in `everos.toml`; the variables are no longer read, so an unmoved `EVEROS_DISABLE_CASCADE` silently turns the cascade back on (the `EVEROS_DISABLE_*` switches are read again after 1.6.0; see [Unreleased]) |
| run `everos cascade …` on a root with no `[llm]` section | add `[llm]`; building the application now requires it, and the command exits 3 with the file to edit |
| read `capabilities.llm` from `GET /health` | read `status` or `cascade.healthy` instead; the field is gone |
| replaced a prompt template through the old prompt slots | move the requirements your template added to `config/custom_instructions/<operator>.yaml` (details below); the template itself is no longer loaded |

Extraction output changes with the everalgo line: episodes, facts and
profiles are produced by the new prompts, and a profile's language is
frozen from its first extraction after the upgrade. No action is needed;
a different style in new entries is expected, not a fault.

Code that imports `everos.service.memorize` / `search` / `get` or calls
`create_app(lifespan_providers=...)` must switch to `build_application` and
the facades.

A deployment that customised a prompt slot must move it: keep only the
requirements the custom template added on top of the default prompt and put
them in `config/custom_instructions/<operator>.yaml` as `instructions:`
(the bundled prompt itself can no longer be replaced). Markdown and the
index schema are unchanged. The SQLite `memcell` table gains three nullable
columns (`case_extracted_at`, `case_skip_reason`, `case_skip_detail_json`);
the first start after upgrading adds them to an existing `system.db` on its
own (logged as `sqlite_column_added`), so there is nothing to run by hand.

This release pins the everalgo packages to one contract line, six of them
release candidates: `everalgo-core 0.6.0`, `everalgo-boundary 0.4.1`,
`everalgo-user-memory 0.9.0rc4`, `everalgo-agent-memory 0.6.0rc3`,
`everalgo-rank 0.5.0rc3`, `everalgo-knowledge 0.3.0rc2`,
`everalgo-clustering 0.3.0rc2` and, with the `multimodal` extra,
`everalgo-parser 0.3.0rc2`. The pins are exact (`==`), so `pip install
everos` resolves them without any pre-release flag; upgrade them together,
never one at a time.

### Fixed

- **The case→skill bridge matches again.** On HYBRID + LLM-rerank agent
  searches the bridge looked up skills by the matched cases' index row ids
  (`<agent_id>_ac_…`), while a skill's `source_case_ids` holds the md-side
  entry ids (`ac_…`), so no skill was ever bridged in (#478). The bridge is
  now keyed by entry id, and `SearchAgentCaseItem` exposes `entry_id` so a
  client can join cases to skills the same way.
- **The first agent `/add` no longer downloads tiktoken's encoding.** The
  algorithm library counts tokens with `o200k_base`, which tiktoken fetches
  on first use; a fresh deployment paid for that download inside the first
  agent write and answered a generic 500 when it failed (#277). The server
  now warms the encoding at startup, before any storage is opened, and a
  host that cannot load it fails to start with `TokenizerAssetError` naming
  the cause and the `TIKTOKEN_CACHE_DIR` pre-cache option.
- **OME crash recovery resumes every run of the previous process.** It
  used to skip RUNNING rows younger than 30 minutes on the theory that
  APScheduler might still own them; it never does (a one-shot job leaves
  the jobstore when it is submitted), so a run cut short by a restart
  inside that window was lost for good (#540). The engine holds the
  jobstore lock before it recovers, so every RUNNING row is a crash:
  all of them are marked CRASHED and re-enqueued.
  `OMEConfig.crash_recovery_timeout_seconds` is gone; it was not a
  user-facing setting.
- **`radius` is a pre-fusion cosine floor on every vector route.** It was
  applied to the dense pool of `vector` and `hybrid` searches but also
  passed into the ranker's input, which never read it, and the `agentic`
  routes ignored it entirely (#394). The dead ranker field is gone and
  the agentic dense recalls (cases, skills, episode facts) now apply the
  same floor; the API reference states the semantics.
- **A file restored with its old mtime is indexed again.** When the
  cascade worker found an enqueued file gone it folded the row into a
  deletion but left it saying `done / added`, so a scan that later saw
  the file back with the same mtime took it for indexed (#541). The
  worker now records the applied change (`deleted`), and the reconciler
  re-emits a present file whose row says `deleted` whatever its mtime.

## [1.5.1] - 2026-10-22

**Hybrid and vector search keep up with 10 requests a second out of the box.**
Each of those searches embeds its query first, and the slots reserved for
those calls were the limit: with 4 of them and a remote embedding endpoint
answering in about 0.45 s, search topped out near 9 requests a second and a
steady 10 queued without bound. The default is now 16, which held the same
load at a p50 of 0.42 s and a p95 of 1.45 s.

### Changed

- **`[embedding].query_max_concurrent` defaults to 16 instead of 4.** Every
  hybrid or vector search makes one query embedding call, and the budget
  reserved for those calls bounded search throughput: with a remote
  endpoint answering in about 0.45 s, 4 slots capped hybrid search near 9
  requests/s, so a steady 10 requests/s queued without bound (10,020
  indexed entries, 5 minutes: p50 64 s, p95 101 s). At 16 the same load
  held p50 0.42 s, p95 1.45 s, p99 2.11 s, all 3,000 requests served at
  9.99/s. In-flight requests to the endpoint can now reach
  `max_concurrent + 16`; set it lower if the endpoint is rate-limited.

### Upgrade

Nothing to migrate. One thing to check: requests in flight to the embedding
endpoint can now reach `max_concurrent + 16` (26 with the shipped values)
instead of 14. If your endpoint is rate-limited or a small self-hosted model,
set `[embedding] query_max_concurrent` (or `EVEROS_EMBEDDING__QUERY_MAX_CONCURRENT`)
back to what it can take. Roughly, the searches per second it can serve is
this value divided by the endpoint's per-call latency.

## [1.5.0] - 2026-10-21

**`/add` no longer has to wait for extraction, and an index can no longer be
wrong without saying so.** A client can hand a batch to `/add` with
`async_mode` and get a `202` back in milliseconds, while a durable in-process
queue runs extraction in session order; the field, the response body and the
task-status paths match EverOS Cloud, so one client works against both. Vector
columns now follow `[embedding] dim` instead of a hardcoded 1024, startup
refuses an index the configured model cannot have produced instead of
accepting writes that are truncated or searches that quietly return nothing,
and an emptied index is reported instead of looking healthy. The rest is
reliability: OME writes no longer starve one another on Windows, a session
lock timeout no longer strands a memcell without its episode, a message the
model can never segment no longer blocks its session, resolving a cluster's
members no longer reads two embedding vectors per member, and a dozen smaller
faults found by property-based tests are fixed.

### Added

- **`async_mode` on `/add`: accept now, extract in the background.** A request with `async_mode: true` is written to a durable
  SQLite queue (`memorize_inbox`) and answered with `202 Accepted`; an
  in-process worker then runs the unchanged memorize pipeline for each
  batch, per session in arrival order and several sessions at a time
  (`max_concurrency`, default 4), so a session's batches and its flush
  never overtake each other and one slow session does not hold up the
  others. A synchronous call on a session that still has
  queued batches is queued too, for the same reason; `/flush` has no
  `async_mode` of its own and queues only behind such a backlog. The queue is bounded
  (`413 PAYLOAD_TOO_LARGE`, `429 QUEUE_SATURATED`), a batch whose delivery
  fails is marked `failed` and skipped (no automatic retry by default: the
  pipeline is not atomic, so a re-run is not a clean retry), and every
  row is deleted seven days after it was accepted. `GET /api/v2/tasks/{task_id}`
  reports one request's outcome and `GET /api/v2/tasks` lists them (filter by
  `session_id` / `status`, paged). The field name, the `202` body, the task
  paths and the status vocabulary match the EverOS Cloud gateway. Replays
  after a crash are bounded (`max_replays`, default 3), the delivery loop
  relaunches itself if it ever dies, and `GET /health` reports the queue
  (`memorize_queue` block: loop alive, backlog, failed, restarts). Configured under
  `[memorize.async_queue]`; `enabled = false` stops accepting new async
  work while rows accepted earlier are still delivered. Synchronous calls
  without a backlog are unchanged.

- **`[embedding] dim` sets the width of every vector column.** It used to be
  hardcoded at 1024, so a model of any other width could not be used: a
  narrower one failed every write, a wider one was truncated without saying
  so. `dim` now drives the schema, and startup checks the configured model
  against it. (#434)

- **`cascade status` and `/health` tell an emptied index apart from a healthy
  one.** Deleting `.index/lancedb` used to leave a store that reported perfect
  health: the ledger lives in `.index/sqlite` and survives the delete, so every
  file stayed `done`, the scanner enqueued nothing, `cascade status` showed
  `done: 106` with `lag: 0`, and `/health` reported zero failures — while
  search returned nothing. Startup now counts rows per business table and
  compares them against the ledger. `cascade status` prints a live count per
  table and warns when every one of them is empty; `/health` carries the
  startup snapshot in `cascade.index_rows` and `cascade.index_emptied`.
  Reported, not enforced: `healthy` does not flip, because these are startup
  counts and a signal that cannot clear until the next restart is worse than a
  number an operator can read. A fresh root, a queue still draining, and a root
  whose markdown was all deleted are each quiet. Recovery is `everos cascade
  rebuild`; markdown is the source of truth and is never touched. (#501, #522)

### Changed

- **Startup refuses an index the configured embedding cannot have produced.**
  Two cases stop a start that used to proceed and fail later: the model
  returns fewer dimensions than `dim` (nothing could ever be stored), and the
  index records a *different* model than the one configured (distances
  between vectors from two models are meaningless, and searches just return
  nothing useful). Both name the values involved and point at `everos cascade
  rebuild`; the CLI exits 3 without a traceback. Deployments with no embedding
  configured (keyword-only, an empty `api_key`) are unaffected — the check is
  skipped whole.

- **An index records which embedding model built it.** LanceDB keeps it in the
  `episode` table's `vector` column metadata, Milvus as a collection property.
  An index that predates this release has no such record: the first start
  after upgrading writes one in place, without touching a row or a column
  type, and logs that it did. `everos cascade rebuild` records the model it
  rebuilt with.

- **`/health` `capabilities.embed` reports the startup probe, not whether a
  provider can be constructed.** It used to be true whenever `model`,
  `api_key` and `base_url` were set, including against a dead endpoint. It is
  now the result of the one embedding call made at startup, and does not flip
  back without a restart.

- **Every `everos` command makes one embedding call at startup** to measure
  the model's real output width, with a 5-second budget and no retries. If it
  fails, the start continues with a warning and the width check is skipped —
  the model-change check does not depend on it.

### Fixed

- **OME: concurrent run-record writes could starve one another until the
  busy timeout.** Every store write opened its own connection and left the
  waiting to SQLite's busy handler, which sleeps and retries, so under
  several writers a late arrival could keep taking the write lock ahead of
  an early one. On Windows, where a commit costs 30-80 ms, a writer starved
  that way hit the 5 s timeout and failed with `database is locked`
  (measured on CI: 16 writers, p99 3.8 s, one failure in 30 s), which
  silently drops the run when it lands on the RUNNING insert and leaves the
  row RUNNING until a restart when it lands on a status update. Writers now
  queue on an in-process lock that nobody overtakes before touching the
  file. The engine also keeps one connection open for its lifetime: with
  a connection per statement, every close used to be the last close, which
  checkpoints and deletes the WAL files under an exclusive lock and on
  Windows held a fresh connection's first statement past the same timeout. Found by the OME run state-machine test, which failed on Windows CI.
- **A message the model can never segment no longer blocks its session.**
  Boundary detection retries an unusable reply three times per call; a
  message that always produced one stayed in the session's buffer, every
  later `/add` or `/flush` merged it back in and failed the same way, and the
  session was stuck until someone cleared the buffer by hand (the async queue
  marked that session's batches `failed` one after another). The third
  consecutive call that ends this way now moves the session's buffer and the
  call's own messages to the `quarantine` track of `unprocessed_buffer`,
  answers `422 INVALID_INPUT` naming the session and the count, and the next
  call starts from an empty buffer. Infrastructure failures (no quota,
  network) are not counted and still propagate as `EXTERNAL_SERVICE_UNAVAILABLE`,
  so a buffer waits out an outage.
- **A session-lock timeout could cancel extraction after the memcell was
  written.** `memorize()` wrapped the whole call in
  `session_lock_timeout_seconds`; a timeout that landed after the boundary
  had written the memcell row and consumed the buffer cancelled the
  extraction pipelines, and nothing retried them: the cell stayed without an
  episode for good. The timeout now bounds the two places a call can hang
  while another waits on the lock, the lock wait and the boundary LLM call,
  both of which run before any write; once a cell is committed its
  extraction runs to completion, after the lock is released and shielded
  from a cancelled request. Found by a property test that lands random
  concurrent calls under random timeouts.
- **A replayed clustering run inflated the cluster's count.** OME delivers a
  strategy run at least once; when `trigger_profile_clustering` or
  `trigger_skill_clustering` ran again for an episode or case it had already
  folded in (a crash after the upsert, a retry after the emit failed), it
  folded it in a second time: the member row was deduplicated but `count`
  rose and the centroid was re-weighted, and `extract_user_profile` gates on
  that count. An entry already among the owner's members is now left where it
  is; the downstream event is still emitted.
- **OME `inspect` ignored the Counter gate's cooldown.** The dry run compared
  the bucket's counter with the threshold only, so during a cooldown window
  it reported `will_run: true` for an event `dispatch` would hold back. The
  store now answers the dry run with the same rule `dispatch` applies
  (threshold and cooldown, one clock).
- **Knowledge: a failed `PUT /knowledge/{doc_id}` could leave two
  directories for one document, or bury the old one inside the new.** The
  replace path backs the document's directory up, writes the new one and
  restores the backup on failure; when the failure came *after* the writer
  had created the new directory (the original-file step, for one), the
  restore moved the backup inside it if the name was unchanged, or left it
  next to the restored one if the title or category had changed. The
  half-written directory is now removed before the backup comes back.
- **A hand-edited `entry_count` could make a daily-log writer reuse an entry
  id.** The episode, atomic-fact, foresight and agent-case writers allocated
  the next id from the frontmatter's `entry_count` alone, so lowering it by
  hand made the next append produce an id the file already held (the index
  keeps one of the two), and a non-numeric value raised a `TypeError`. The
  next id is now allocated from the higher of `entry_count` and the highest
  sequence number already in the file, and a frontmatter that disagrees is
  logged.
- **OME `wait_for_event` returned while a retry was still pending.** The
  runner records a failed attempt as `FAILED` before its backoff sleep and
  gives the retry a fresh run id, so during the backoff the event's only row
  is `FAILED`; `wait_for_event` counted that as terminal and returned.
  Reflection waits on it before deprecating what the extraction replaced, so
  it could act on an outcome that was not final. `FAILED` now means "retry
  pending"; a failure with no retry left is `DEAD_LETTER`, as before.
- **`to_timestamp_ms` is now the exact inverse of `from_timestamp_ms`.** It
  computed `int(timestamp() * 1000)`, and the float product sat just below
  the integer for about 0.6 % of instants (1095954883735 came back as
  1095954883734); integer arithmetic on the timedelta since the epoch
  replaces it.
- **The watcher classified md files the scanner never visits.** Its kind
  matcher was anchored on the right only, so a path with extra leading
  components (`x/app/project/users/u/episodes/episode-2026-01-01.md`) was
  accepted and queued while the root-anchored scanner glob ignored it; the
  match now requires the same number of path components.
- **`default_app` / `default_project` are rejected as ids.** They are the
  on-disk directory names of the default scope, so an app with that literal
  id shared the default app's directory and its index rows were attributed
  to `default`. The API now answers `422` for them.
- **Search: a duplicate id inside the dense pool kept its last score, not
  its highest**, and the multi-round decider's JSON parser gave up on a reply
  with a stray `{` in the prose before the object (its docstring promised
  otherwise); it now also decodes from every brace.
- **Cascade worker: a bounded wait between optimize and rebuild cancelled the
  task it waited on.** The optimize runner waits up to 180 s for an in-flight
  index rebuild of the same kind, and the rebuild loop waits for an in-flight
  optimize, so the two never commit on one LanceDB manifest at once. Both
  waits used `asyncio.timeout` around `await task`, which cancels the awaited
  task when the deadline fires: a rebuild longer than the wait, with a write
  landing on that table meanwhile, was killed mid-flight, and the rebuild loop
  ended with it, silently, so no periodic rebuild ran again until restart and
  that table's index UUID / FTS segment count grew unbounded. The mirror wait
  cancelled the optimize beat it was meant to stay clear of. Both waits now
  only stop waiting. Present since 1.2.2.
- **Cascade: the handler limit held per drain pass, not per worker.** A worker
  runs at most four handlers of a batch at once, but the limit was created
  inside each drain pass, so two passes running together each got their own:
  `POST /cascade/quiesce` drains while the background loop is still draining,
  and that overlap peaked at eight concurrent handlers. The limit is now
  shared by the worker. (#492)
- **Resolving a cluster's members read each member's full row, vectors
  included.** The profile path maps a fresh cluster's episodes back to their
  parents and needed two fields per member, yet fetched every member's whole
  episode row with both embedding vectors; a cluster can hold thousands of
  members, so that read dominated the path's memory. LanceDB now reads only
  `parent_type` and `parent_id` (Milvus still loads full rows), and the
  reflection path's bulk read skips the vector columns. (#490)
- **`method="vector"` finds episodes when no atomic fact is in scope.**
  Vector search over episodes recalls through atomic facts, and it returned
  an empty list (HTTP 200, no log line) whenever none were in scope — fact
  extraction disabled, not caught up yet, or failing — while keyword search
  kept answering. It now falls back to each episode's own vector, with
  `radius` applied, and logs `vector_recall_fact_pool_empty` at info.
  (#502, #509)

### Upgrade

`pip install --upgrade everos` is enough for most roots, and nothing in this
release rewrites or re-embeds anything on its own. The first start records
which model built the index and logs one line saying so.

The first start also creates the `memorize_inbox` table in `system.db` for the
async queue. `async_mode` is opt-in: clients that never send it get the same
synchronous responses as before.

Two deployments need attention:

- **A model narrower than `dim`** — the server now refuses to start instead of
  accepting writes that could never be stored (this is #434). The error names
  both numbers and the two steps: set `[embedding] dim` (or
  `EVEROS_EMBEDDING__DIM`) to what the model returns, **then** run `everos
  cascade rebuild` — the tables were already created at the old width, so
  fixing the configuration alone is not enough. Your markdown is untouched
  throughout.
- **A model wider than `dim`** — unchanged behaviour, but the start now warns
  that the extra dimensions are truncated. The shipped default is this case
  (`Qwen/Qwen3-Embedding-4B` returns 2560, `dim` is 1024) and nothing is
  wrong: that model is Matryoshka-trained, 1024 is one of its published
  truncation points, and the index compares with cosine distance. Set
  `dimensions = 1024` to silence it, if your endpoint accepts that parameter.

**Changing `dim` or `model` later** stops the next start and requires `everos
cascade rebuild`, which re-embeds every entry from markdown — budget for it.

**Downgrading** to an earlier EverOS is safe as long as `dim` is still 1024:
the older schema check compares column types only and ignores the new record.
If you changed `dim`, set it back and rebuild before downgrading.

## [1.4.2] - 2026-10-14

**A server that stays healthy for days, not hours.** Three faults in this
release compounded on exactly the deployments that ran longest: a long-lived
index leaked file descriptors until searches took seconds, interactive searches
queued behind the cascade's background embedding, and a memory root holding more
than one project had each project's index rows overwrite the other's. All three
are fixed, and a LanceDB root that the last fault already damaged repairs itself
on the first start after upgrading.

### Changed

- **`[embedding].max_concurrent` defaults to 10 instead of 5.** The
  `EmbeddingSettings` class default (50) is aligned to the same value; the
  shipped `default.toml` always overrode it, so the effective default moves
  from 5 to 10. (#486)

- **Milvus collections must be rebuilt once after upgrading.** They gain the
  `__pk` primary-key column; the server refuses to start on an old
  collection and names `everos cascade rebuild`, which recreates them from
  markdown.

- Response `id`s are unique within `(app_id, project_id)`, not globally; the
  shape of `id` is unchanged.

- **The cascade optimizes a LanceDB table at most once a minute, not once
  every ten seconds.** Each optimize pass leaves an `_indices/<uuid>/`
  directory behind until cleanup reclaims it, so the old 10 s floor set the
  ceiling on how many could pile up: about 126k directories a day and ~1.8M
  at the retention horizon. The 60 s default costs a sixth of that (~21k a
  day, ~300k at the horizon, roughly 1.2 GB) and keeps a full index sweep
  inside its 60 s deadline. The interval is a built-in floor on how often a
  table may be optimized, not a setting; a write-heavy deployment sees an
  index that trails its markdown by up to a minute longer than before.

### Fixed

- **A long-running server no longer leaks file descriptors on LanceDB index
  readers.** Every `optimize()` mints a new index version whose reader is
  kept in a session cache, and that reader holds the descriptors of its
  `_indices/<uuid>/` files. Lance keys those cache entries by dataset
  version and never evicts them when cleanup deletes the files underneath
  (`lance-format/lance#7718`), so the count grew for the life of the
  process: a 65-minute soak went from 74 to 558 index descriptors with no
  plateau, and a deployment on a 1024-descriptor limit exhausts it in
  hours. EverOS capped only one of the session's two caches and left the
  metadata cache unbounded, which is the one that decides whether the count
  converges. Both are now capped (`[lancedb].metadata_cache_size_bytes`,
  default 1 MB): the same soak holds between 6 and 82 descriptors, falling
  as often as it rises. Reported upstream as `lance-format/lance#9934`.

- **An interactive search no longer queues behind the cascade's embedding
  work.** Query embeddings and background write embeddings shared one
  process-wide semaphore, so a burst of writes put the query a user was
  waiting on at the back of the queue. Under three concurrent writers,
  vector search p50 measured 46 s and p95 112 s, while the keyword leg —
  which needs no embedding — answered in 0.35 s. Queries now draw on their
  own budget (`[embedding].query_max_concurrent`, default 4): vector p50
  drops to 0.60 s and p95 to 3.70 s under the identical load. Total
  in-flight requests to the endpoint become `max_concurrent +
  query_max_concurrent` rather than a silently shared bound. (#473)

- **The same owner's memories in two projects no longer overwrite each other
  in the derived index.** Index rows were merged on `id` alone, and `id` is
  unique only within one `(app_id, project_id)`: a user's episodes, atomic
  facts, agent cases, foresights, skills and profile written under a second
  project replaced the first project's rows (markdown was never affected).
  Every table now declares `PRIMARY_KEY = ("app_id", "project_id", "id")`;
  LanceDB merges on it and Milvus keys its collections on the composed
  `__pk` column. Profile search and get, the handlers' change detection and
  the vector backfill address rows by the full key; the backfill also keeps
  each row's own vector within a batch, and reflection deprecates atomic
  facts only inside the project it merged.

- **A multi-project LanceDB root indexed before this release gets its
  overwritten rows back by itself.** The first full pass over the root
  after upgrading (server start, or `everos cascade sync`) re-reconciles
  every indexed markdown file against the index once and records that in
  `.index/.index_reconcile_version`. Entries whose rows are present and
  unchanged are skipped by digest, so a healthy root pays one read pass
  and no embedding; only the rows that were lost are re-embedded.
  `everos cascade sync --force-all` runs the same pass on demand. Two
  roots need that command instead: a server started with
  `EVEROS_DISABLE_CASCADE=1` never runs the pass (stop it and run
  `everos cascade sync --force-all` once), and a multi-project root where
  reflection ran before this release needs `everos cascade rebuild` once,
  because the pass does not clear the `deprecated_by` stamps one project's
  reflection left on another project's rows.

- **`POST /cascade/quiesce` freezes a complete index.** Its drain stopped
  at the worker's per-drain bound, and inside a server it could also stop
  while the background worker still held the last batch, which the stop
  that follows then orphaned in `processing`; the ledger an upgrade
  re-queues made both routine. The drain now runs until the ledger holds
  nothing pending, waiting for the worker's in-flight batch, and
  `everos cascade sync` / `rebuild` use the same loop.

- **A cascade pass embeds a file's changed entries in one batched request,
  not one request per entry.** The four daily-log handlers (episode, atomic
  fact, foresight, agent case) each called the embedding endpoint once per
  entry, so a turn that wrote ~20 atomic facts queued ~20 single-input
  requests on the process-wide embedding semaphore, and the next search's
  query embedding waited behind all of them: about five times the
  endpoint's latency at the previous default `max_concurrent = 5`. The
  handlers now
  send every text of a file through `embed_batch`, which
  `[embedding].batch_size` splits into requests (23 entries become 3
  requests at the default 10), so after a single turn's writes the query
  embedding waits for one request; a large backlog still queues ahead of
  it (#473). (#486)

- **Changing a knowledge document's category keeps it inside `knowledge/`.**
  `PATCH /knowledge/documents/{doc_id}` with `category_id` set to `..` moved
  the document's directory up into the project directory, where the markdown
  scan no longer finds it. The category now goes through the same rule as
  document creation, so `.` and `..` fall back to `Others`, and the move is
  refused if its target would leave `knowledge/`. A document an earlier move
  left outside is moved back on its next category change. Reported by
  White0xdi3.

### Upgrade

`pip install --upgrade everos` is enough for most roots: the first full pass
after the upgrade (server start, or `everos cascade sync`) re-reconciles every
indexed markdown file against the index once, re-embeds only the rows that were
lost, and records that it ran in `.index/.index_reconcile_version`. A healthy
root pays one read pass and no embedding.

Three roots need a command instead:

- **A root where reflection already ran across projects** — `everos cascade
  rebuild` once. The automatic pass does not clear the `deprecated_by` stamps
  one project's reflection left on another project's rows.
- **A server started with `EVEROS_DISABLE_CASCADE=1`** — it never runs the
  pass. Stop it and run `everos cascade sync --force-all` once.
- **A Milvus derived index** — `everos cascade rebuild` once. Collections gain
  the `__pk` primary-key column and the server refuses to start on an old
  collection.
## [1.4.1] - 2026-09-24

**A fresh install works again.** The `openai` SDK released its 3.x line on
2026-09-24; an unconstrained install picked it up together with an httpx
pre-release, and every LLM and embedding call failed before reaching the
network. This release pins the SDK to the 2.x line the project is tested
against. Nothing else changed since 1.4.0.

### Fixed

- **`openai` is pinned below 3.** Fresh installs from PyPI resolved
  `openai 3.19.2`, whose client raises `AttributeError: module 'httpx' has no
  attribute 'Timeout'` on every request, so memorize and hybrid / vector
  search returned 500. Existing environments built from `uv.lock` were never
  affected.

### Upgrade

- `pip install --upgrade everos` brings `openai` back to 2.x. If a fresh
  install of 1.4.0 (or any earlier version) today shows the `httpx` error
  above, upgrade to 1.4.1 or run `pip install "openai<3"`.

## [1.4.0] - 2026-09-24

**EverOS runs natively on Windows, and dense search stops scanning the whole
table.** `pip install everos` on a stock Windows 11 machine now works with no
manual steps beyond an LLM key; the unit suite runs on Windows in CI. Every
vector column gets an IVF_FLAT index once it holds enough rows, built and
maintained by the cascade worker, so hybrid search no longer reads the entire
column per query. A ten-hour Windows soak drove the rest of this release: a
second process can no longer write the index behind a running server, watcher
events for one file commit in the order they arrived, and a transient Lance
spill failure is retried instead of filing the markdown file as unrecoverable.

### Added

- **Native Windows support.** The package ships the MSVC runtime
  (`msvc-runtime`, Windows only) and registers its directory before any
  native extension loads, so a clean machine needs no Visual C++
  Redistributable. Text fixtures are read as UTF-8 regardless of the system
  locale, and a `unit tests (Windows)` CI job covers the platform.
  [docs/windows.md](docs/windows.md) leads with the native install and keeps
  the WSL2 walkthrough.
- **Vector (ANN) indexes.** Once a vector column holds
  `[lancedb] vector_index_min_rows` (default 2000) non-null vectors, the
  cascade worker builds an IVF_FLAT cosine index on it — on its first rebuild
  sweep after server start and on the heavy maintenance beat — and retrains it
  once light-beat delta indices pile past 16. Partition size (4096 rows) and
  `nprobes` (32) are pinned together, so the search is exact up to ~130k rows.
  Measured on 27k rows of 1024-dim vectors: 245 ms → 84 ms per query with
  recall@10 of 0.98. The CLI never builds indexes; only the server's worker
  does.
- **`[cascade] scan_interval_seconds`** is a setting (default 30 s) instead of
  a constant.

### Changed

- **`everos cascade sync`, `cascade fix --apply` and `cascade rebuild` refuse
  to run next to a running server** (exit code 3) and hold the root's lock for
  their whole run. Two processes writing the same index inserted every row
  twice; a running server projects markdown changes on its own.
- **pyarrow floor raised to 25.0.1.** Materialising timestamp columns no
  longer imports `pytz` per value on the query hot path (a 312 ms → 13 ms
  difference per query on Windows).
- **Watcher paths are relativised without a `resolve()` per event**, which was
  ~40 % of the worker threads' CPU under write load on Windows.
- **A late `modified` event for a path that is already gone is recorded as a
  deletion**, so a create-then-unlink inside the event latency window cannot
  resurrect a file.
- **`POST /memory/add` rejects a `role = "tool"` message without
  `tool_call_id` with 422** instead of failing with 500 inside extraction.
- **OME strategies only receive event classes they declared.** A bare manual
  tick no longer reaches business-event strategies (which crashed on it), and a
  manual trigger reaches a Cron strategy again.
- **API docs** describe the response envelope's exceptions and the OME latency
  after a flush.
- **Milvus CI** pulls MinIO from quay.io (the Docker Hub image was withdrawn).

### Fixed

- **Watcher upserts for one file commit in delivery order.** Each event used to
  schedule its own upsert and the last committer won; Windows synthesises a
  `created` per freshly created parent directory, so a stale duplicate could
  put the first write's mtime back on the row after an atomic save. Sequence
  numbers are now unique as well.
- **Transient Lance spill failures are retried.** `LanceError(IO): Execution
  error: Spill has sent an error` (DataFusion spilling to the temp dir under
  heavy concurrent writes) surfaced as a bare `RuntimeError` on `merge_insert`
  and marked the markdown file permanently failed; it is now a retryable error
  and the same row projects on the next attempt.
- **Markdown staging swap survives a Windows sharing violation.** `os.replace`
  over a file another process has open (the worker, an antivirus scan) is
  retried with backoff instead of failing the write.
- **Deleting a knowledge document before it was indexed removes its directory**
  and reports the real topic count instead of leaving the files to be re-indexed.

### Upgrade

- **First server start builds the vector indexes** on every table past the row
  threshold: seconds per 10k rows of 1024-dim vectors (0.5 s at 27k, ~1.4 s at
  100k per column), during which searches keep working with a flat scan. Set
  `[lancedb] vector_index_min_rows` higher to defer it.
- **Stores created before the `subject_vector` column existed** fail the
  startup schema check with `LanceDBSchemaMismatchError`. Stop the server and
  run `EVEROS_ROOT=<memory root> everos cascade rebuild` — it re-indexes from
  markdown, which stays the source of truth. Note that `cascade rebuild` reads
  the root from `EVEROS_ROOT`, not from `--root`.
- **Scripts that ran `everos cascade sync` next to a running server** now get
  exit code 3. Drop the call (the server projects changes itself) or stop the
  server first.
- **pyarrow ≥ 25.0.1** is required; `pip install --upgrade everos` resolves it.

## [1.3.1] - 2026-09-08

**One reproducible runner for four long-term-memory benchmarks, plus an
LLM-guided multi-round retrieval method.** LoCoMo, LongMemEval, EverMemBench,
and SubtleMemory now share the same staged ADD → SEARCH → ANSWER → JUDGE
workflow, resume model, metrics, and run manifest. The new
`llm_multiround` search method iteratively recalls episode blocks, asks a
separately configurable decider to retain core evidence and issue follow-up
queries, and returns a bounded final context without a cross-encoder.

### Added

- **LLM-guided multi-round episode retrieval.**
  `POST /api/v2/memory/search` accepts `method = "llm_multiround"` for user
  memory. Each round independently fuses BM25 and vector candidates for the
  current sub-queries with RRF, then uses the decider to select core evidence
  and identify remaining gaps. Existing search response fields are unchanged;
  decider failures are recorded in structured logs and optional trace dumps.
- **Independent decider configuration.** The new `[decider]` section selects
  the model, endpoint, timeout, request extras, retry policy, and loop tuning.
  Empty connection fields inherit `[llm]`, preserving single-model setups.
- **Unified benchmark harness.** One runner and four adapters cover LoCoMo,
  LongMemEval, EverMemBench, and SubtleMemory with explicit dataset configs,
  resumable stage artifacts, deterministic run identities, shared IR metrics,
  decider preflight checks, and trace validation that rejects degraded
  multi-round runs before reporting scores.
- **Cascade snapshot control.** `POST /api/v1/cascade/quiesce` and
  `POST /api/v2/cascade/quiesce` drain the Markdown-to-index queue and stop the
  cascade subsystem until restart. Startup switches can disable all cascade
  work or only the filesystem watcher for managed read-only workloads.

### Changed

- **OME attempts are bounded.** One strategy attempt now has a 1,800-second
  default wall-clock timeout and follows the existing retry/dead-letter path on
  timeout. Environment settings can tune concurrency or disable the timeout.
- **SQLite pool saturation fails visibly.** Pool size, overflow, checkout
  timeout, recycle, and pre-ping are explicit settings; exhausted pools now
  raise after a bounded wait and emit saturation diagnostics instead of waiting
  indefinitely.
- **Extraction LLM transport is configurable.** `[llm]` now exposes request
  timeout and provider-specific SDK arguments while retaining the previous
  60-second default.

### Fixed

- **Permanent embedding request errors no longer retry.** HTTP 400, 401, 403,
  404, 413, 414, and 422 responses are classified as rejected input or
  configuration and leave the retry loop; timeouts, 429 responses, and server
  errors remain retryable.
- **EverMemBench owner mapping no longer changes profile extraction.** The
  adapter uses one stable synthetic owner per topic for both ingestion and
  search while preserving each real participant's name in message metadata.
  The production Profile cluster/direct extraction paths are unchanged.

### Upgrade

- No storage migration or index rebuild is required. Existing search methods
  keep their prior behavior unless callers explicitly select
  `llm_multiround`.
- Deployments with a healthy OME strategy that legitimately runs for more than
  1,800 seconds must raise `EVEROS_OME_RUN_TIMEOUT_SECONDS` or set it to `0` /
  `off` before upgrading.

## [1.3.0] - 2026-09-07

**Milvus as an optional index backend, behind a port that hides which one you
run.** The rebuildable BM25/vector index can now live in a remote Milvus Server
or Zilliz Cloud instead of the embedded LanceDB, selected by one setting and
verified at startup. Nothing about the default installation changes: LanceDB
stays the backend, `pymilvus` stays an optional extra, and markdown stays the
source of truth — the index is derived either way, so switching backends is a
rebuild, not a migration. Both backends implement the same typed ports, so
cascade, search and `/get` never branch on physical storage; a filter is a
backend-neutral predicate tree rather than a rendered SQL string, which is what
lets one set of contract tests hold both adapters to the same behaviour. The
vector metric is now cosine everywhere, which corrects an inconsistency
described under Changed.

### Added

- **Optional Milvus derived-index backend.** Install with
  `pip install everos[milvus]` and select it with:

  ```toml
  [index]
  backend = "milvus"

  [milvus]
  uri = "http://127.0.0.1:19530"   # or a Zilliz Cloud endpoint
  token = ""                        # Zilliz Cloud API key
  db_name = ""
  consistency_level = "Session"     # Strong | Bounded | Session | Eventually
  collection_prefix = "everos"
  ```

  Every field binds to an environment variable
  (`EVEROS_INDEX__BACKEND`, `EVEROS_MILVUS__URI`, ...). The seven business
  tables are created as `<collection_prefix>_<kind>`.

  **Remote Milvus Server or Zilliz Cloud only.** A local database path is
  rejected at startup rather than silently accepted: embedded Milvus Lite is
  not supported, because its pure-Python rewrite degrades a single query to a
  full scan and drops search-time parameters.

  Startup verifies the physical collection field by field — datatype, primary
  key, nullability, vector dimension, and the index metric (dense `COSINE`,
  BM25 output fields) — so a collection that disagrees with the declared schema
  fails immediately instead of at the first write or search.

### Changed

- **Vector search now uses cosine distance on LanceDB too.** The generic
  vector path called `nearest_to()` without a distance type, so it ran on
  LanceDB's L2 default while `agent_skill` recall and every Milvus query
  already used cosine. All of them are cosine now. **Ranking can shift for
  existing deployments** — the same vectors are scored by a different metric.
  No action is required and no rebuild is involved; the index is unchanged.
- **everalgo's transitive layer is pinned.** `everalgo-boundary`,
  `everalgo-core` and `everalgo-clustering` are now direct `==` dependencies
  matching the versions the test suite resolves. The consumer packages declare
  them as `>=0.2.0,<2.0.0`, which treats a 0.x line as if only a major bump
  could break compatibility, so a fresh resolution could pick releases the
  suite had never run against. Raise them deliberately, together with a smoke
  run against a PyPI-resolved install — `make package` installs the built
  wheel with `--no-deps`, so it cannot catch a resolution break on its own.

### Fixed

- **`POST /api/v1/memory/add` no longer fails on a freshly resolved install.**
  `everalgo-boundary` 0.3.0 added a third required field to the public
  `DetectionResult` NamedTuple while `everalgo-agent-memory` 0.4.0 still built
  it with two, so a default (`memorize.mode = "agent"`) install answered
  `500 TypeError: DetectionResult.__new__() missing 1 required positional
  argument: 'should_wait'`. Chat mode was unaffected. The pin above resolves
  the compatible pair.
- **Milvus datetime round-trips are exact below the millisecond threshold.**
  Timestamps were written in milliseconds but read back through a
  seconds-or-milliseconds heuristic, so any instant before 2001-09-09 either
  landed in the year 30000 or raised `ValueError`. Physical column reads now
  use an exact inverse.
- **Concurrent `update()` on the Milvus backend no longer loses writes.** The
  read half of the read-modify-write cycle sat outside the write lock, so a
  backfill writing `vector` and a reflection writing `deprecated_by` could
  overwrite each other on the same row. Both halves share one lock, and the
  silent 10,000-row truncation on that path is gone.
- **Milvus queries stay inside the engine's result window.** Row scans and
  search `topK` are bounded by the 16,384-row ceiling instead of requesting
  more and failing.

## [1.2.3] - 2026-08-07

**Background maintenance that fails loudly instead of quietly.** A soak run on
1.2.2 found a table that had stopped reclaiming disk for 100 minutes while
`/health` stayed green — nothing had *failed*, a call had simply never returned,
and every signal was built from failure counters. Auditing for that shape turned
up six more places it could happen: reads with no deadline (which stop the whole
md to LanceDB projection, not just one table), background loops that die
permanently on one exception with no log at all, an alert counter reset by the
remediation it triggers. All of them are now bounded, and a stall that does
happen names the table it happened to. Alongside that, agent-skill extraction is
rescued from a retry-then-dead-letter loop, keyword search no longer returns 500
during an index rebuild, and the maintenance cadences moved into settings.

### Fixed

- **Agent skill extraction is no longer stuck in a retry-then-dead-letter
  loop.** Target case data now travels on `SkillClusterUpdated` and existing
  skills for the cluster are read from markdown (strong-consistency), so the
  strategy never races cascade indexing. Prior to this fix, running a fresh
  agent trajectory produced zero `SKILL.md` files — `.skills/` did not exist.
  **The related stale-index clobber is fully closed only for clusters at or
  below `MAX_SKILLS_IN_PROMPT` (10).** Above it, markdown still supplies the
  candidate set but LanceDB orders it, and the skill a lagging index omits is
  by definition the one written most recently — the one most likely to need
  `update` — so it can be ranked out of the prompt and re-added instead. The
  window is narrow (it needs a cluster over 10 skills *and* an index that has
  not caught up) and the consequence is the pre-existing full-replace, not a
  new failure mode.
- **`POST /api/v2/ome/trigger` no longer masks strategy state.** The `status`
  field now distinguishes `not_dispatched` (all dispatch gates rejected the
  strategy — usually a missing `"force": true`) from `ok` (dispatched and
  settled). The new `runs` field surfaces dead-lettered strategy runs that
  were previously invisible to the caller. **If your client matches
  `status` exhaustively (Python `Literal`, TypeScript union), add a
  `not_dispatched` branch.**
- **Agentic search on agent memory now uses the skill-shaped rerank
  passage.** The cross-encoder previously saw only the raw `description`
  field instead of the `name + description + skill instruction` triple that
  the HYBRID lane uses. A skill with empty `description` (a legal everalgo
  output — see `everalgo/agent_memory/skill_ops.py:294`) no longer causes
  HTTP 500 during the LLM sufficiency check.
- **OME strategy retries now back off between attempts.** A retry-class error
  (e.g. waiting on eventually-consistent state) previously exhausted its
  `max_retries` budget in milliseconds; the loop now sleeps
  `min(base * 2**(attempt-1), cap)` plus up to `jitter` seconds
  (defaults: `1s` base / `10s` cap / `0.5s` jitter — code-only defaults,
  not currently exposed via `everos.toml` or `ome.toml`). **`engine_sem` is
  now held per attempt rather than across the whole retry chain**, so the
  backoff sleep does not occupy a concurrency slot. The cap bounds
  concurrent strategy *work* — LLM calls, embeddings, storage IO — and a
  coroutine waiting to retry consumes none of it; holding the slot would
  have turned a partial outage into a total stall, since enough
  simultaneously-failing runs park every one of the
  `max_concurrent_runs` slots in `asyncio.sleep` and starve strategies that
  would have succeeded. Backpressure on failing work is intended;
  backpressure on everything else is not.
- **Path-traversal hardening for LLM-generated agent-skill names (CWE-22).**
  `AgentSkillFrontmatter.name` comes straight from LLM output
  (`extract_agent_skill`) and was concatenated unsanitized into the
  `skills/skill_<name>/` directory segment on both the write and read
  paths; given a sufficiently long `../` prefix, the write target could
  escape the memory root. This is the same class of defect previously
  fixed for knowledge-upload titles/categories (see `knowledge_writer.py`
  in an earlier 1.2.x). The sanitizer is now a single shared helper
  (`everos.core.persistence.markdown.sanitize_dirname`) used by both
  `KnowledgeWriter` and the new `SkillPathMixin.skill_dir_name()` /
  `sanitize_skill_name()`, instead of two independently maintained copies.
  `extract_agent_skill` now sanitizes the LLM-emitted name *before*
  constructing `AgentSkillFrontmatter`, so **`AgentSkillFrontmatter.name`
  and the LanceDB `agent_skill` primary key now hold the sanitized name**
  (spaces become `_`, characters outside `[\w\-.]` are dropped, capped at
  50 chars), not the raw LLM output — a user-visible change for anything
  that reads a skill's `name` field expecting the verbatim LLM string.
  `AgentSkillFrontmatter.name` also gained a validator rejecting a name
  containing a path separator, or being exactly `..`, so a hand-edited
  `SKILL.md` that bypasses the writer's sanitization is caught on read
  rather than silently relocated (the substring form, e.g. a name that
  merely *contains* `..`, is deliberately allowed — sanitized output can
  legitimately contain runs of literal dots). `sanitize_dirname` itself
  falls back (not just on an empty result, but also on `.` or `..`) so a
  short input that is itself a sanitizer fixpoint — e.g. `"../"` sanitizes
  to `".."` verbatim without this fallback — cannot resolve to the same
  directory or its parent; this closes both the agent-skill case and an
  equivalent one-level escape on the knowledge-upload path, which has no
  `skill_`-style prefix protecting its sanitized segment. **No data
  migration is needed for agent skills**: extraction has never
  successfully produced a `SKILL.md` before this release (see the
  cascade-lag fix above), so there is no legacy skill corpus whose
  directory names would change. **Knowledge documents do have a
  pre-existing corpus**, and two inputs resolve to a different directory
  than before: a decomposed (NFD) topic or category now keeps its
  combining marks (`"Résumé"` no longer degrades to `"Resume"`) because
  the shared helper NFC-normalizes first, and a topic or category of
  exactly `.` or `..` now falls back instead of resolving onto the
  parent directory. Precomposed input — including CJK — is unaffected;
  the character class is unchanged from the previous private copy.
  Sanitizing is lossy: skills whose raw names
  differ only in characters the sanitizer drops or replaces (e.g.
  `"fix django"` vs. `"fix_django"`) now share one `SKILL.md`, and so do
  names differing only in a combining mark regardless of script (e.g.
  Devanagari `"किताब"` vs. `"कताब"` — a combining mark alone is not `\w`
  and is stripped either way; same for Thai tone marks, Hebrew niqqud,
  Arabic harakat). The later write wins — the earlier skill's
  `source_case_ids`, `maturity_score`, and body are silently lost, not
  merged. Case is *not* folded, so `"Fix Django"` and `"fix django"` stay
  two distinct sanitized names — two LanceDB rows, but one directory on a
  case-insensitive filesystem (macOS APFS and Windows NTFS defaults),
  where the index then advertises a name whose content was overwritten.
  This is accepted for now rather than mitigated: detecting a collision
  and raising would reintroduce the dead-letter DoS the sanitizer was
  built to avoid, and a disambiguating suffix — the workable option —
  needs a collision probe plus a case-folding rule, so it is deferred to
  a deliberate pass rather than added here.
- **A renamed skill no longer leaves an orphan directory that pollutes the
  next extraction.** everalgo treats a name change as a first-class update
  (`skill_ops._apply_update` preserves `prior.id` while swapping the name),
  so the emitted skill was written to a new `skill_<new_name>/` while the
  old directory survived carrying the same `cluster_id`. Because existing
  skills are now read from markdown rather than LanceDB, that orphan did
  not merely sit on disk — it came back in the next run's
  `existing_relevant_skills` as a duplicate of a skill the LLM had already
  renamed, feeding exactly the `add`-instead-of-`update` full-replace
  clobber this release set out to close, once more per rename. The old
  directory is now reaped after the new one is written, keyed on the
  skill's `id` (the only thing that survives a rename; a fresh `add` mints
  a uuid and can never match). A prior name that another skill in the same
  batch just claimed is never deleted.
- **`extract_agent_skill` retire ops are documented as unimplemented rather
  than silently mispersisted.** `AgentSkillExtractor.aextract` returns a
  flat list with no op discriminator, so a retirement arrives as an
  ordinary skill with `confidence < retire_confidence` and was written back
  like any other — staying in markdown, in the next prompt, and in search.
  The behaviour is unchanged; the module docstring no longer claims retire
  is handled. Honouring it is a design decision (delete the directory, or
  add a `retired` flag that the enumeration, cascade, and search all
  filter on) deferred to its own change.
- **`reference_name` and `script_filename` are sanitized.** Both are
  appended *after* the `skill_<name>` segment, so `skill_dir_name` never
  covered them; they now go through the same `sanitize_dirname` primitive
  on both the reader and the writer. No caller in `src/` reaches them
  today, so nothing was exploitable — this closes the gap before
  progressive disclosure wires them up.
- **A single unparseable `SKILL.md` no longer disables skill extraction
  for its whole cluster.** `AgentSkillReader.list_by_cluster` propagated
  any frontmatter `ValidationError`, which aborted the enumeration that
  feeds `extract_agent_skill` its existing skills — so one hand-edited
  file (or, after a future schema revision adds a required field, every
  existing file at once) dead-lettered that cluster's extraction on every
  run. Offending files are now logged and skipped. `read_main` still
  raises, since a caller naming one specific skill needs to hear about
  corruption rather than receive the `None` that already means "not
  created yet".
  merged. This is accepted, not mitigated, on two grounds: a
  disambiguating suffix would break the `name` ≡ directory-suffix
  identity the reader/writer relies on, and detecting a collision and
  raising would reintroduce the dead-letter DoS the sanitizer was built
  to avoid.
  `"fix django"` vs. `"fix_django"`) now share one `SKILL.md`, and the
  later write wins — the earlier skill's `source_case_ids`,
  `maturity_score`, and body are silently lost, not merged. This is
  accepted, not mitigated: the LLM's add/update decision is keyed on the
  name it sees, so a collision usually reads as an intended update
  anyway.
  under the new sanitizer.
  `KnowledgeWriter` and the new `SkillPathMixin.skill_dir_name()`, instead
  of two independently maintained copies. `AgentSkillFrontmatter.name`
  also gained a validator rejecting path separators / `..` so a
  hand-edited `SKILL.md` is caught on read rather than silently
  relocated. No data migration: agent-skill extraction has never
  successfully produced a `SKILL.md` before this release (see the
  cascade-lag fix above), so there is no legacy skill corpus whose
  directory names would change under the new sanitizer.
- **Reads now carry a deadline** (`count` / `get_by_id` / `find_where` /
  `find_where_paginated` / `search`). The write-side deadline work skipped them
  on the reasoning that a read takes no lock and so blocks no writer — true, but
  the cascade drain loop reads on every batch and advances strictly one batch at
  a time, so a read that never returns stops the **whole md → LanceDB
  projection**: claimed rows stay `processing` forever, nothing new is indexed,
  and `/health` still reports healthy because a hang raises nothing. Budget 60s
  (~1000x the measured 62ms flat scan over 117k rows); expiry raises the
  retryable `VectorStoreBusyError`.
- **Background loops are supervised.** The drain / heartbeat / rebuild loops were
  plain `create_task` coroutines: one uncaught exception ended that loop
  permanently, and because the worker holds a strong reference to the task the
  interpreter never printed "Task exception was never retrieved" either — the
  loop's job simply stopped happening with zero output. Each now runs under a
  supervisor that logs and restarts with escalating backoff (5s / 15s / 45s),
  then asks the process to exit via `SIGTERM` so a restarting supervisor
  (systemd, Docker, k8s) can recover it. The restart budget counts consecutive
  *quick* crashes, not crashes over the process lifetime — a body that ran 60s+
  before raising starts a fresh incident, so independent transients days apart
  cannot pool into a process exit (same windowed counting as systemd's
  `StartLimitIntervalSec`). A done-callback covers the case the supervisor
  itself ends unexpectedly.
- **The optimize-failure alert is reachable again.** The fallback rebuild reset
  the same counter the health verdict reads, so a table failing 100% of the time
  cycled `1..5 → 0 → 1..` and the threshold value existed only during the
  sub-second rebuild — roughly 1% observable against a 30s scrape, so
  `cascade.healthy` stayed green while the table never reclaimed a version. The
  rate limiter now lives in its own counter (`failures_since_fallback`); only a
  successful optimize clears the alert streak. Same shape as the cross-kind
  `max()` masking bug: a remediation path refreshing the signal meant to report
  it.
- **A rebuild no longer leaves the column without an FTS index.** `rebuild_indexes`
  dropped every index and recreated it, on the assumption that LanceDB falls
  back to a brute-force scan meanwhile. That holds for vector search and **not**
  for FTS: with no inverted index a BM25 query raises `Cannot perform full text
  search unless an INVERTED index has been created`, and since the recall legs
  are gathered without `return_exceptions`, one failing leg 500s the whole
  search request. Now uses `create_index(replace=True)`, which swaps atomically
  — measured 0 failures across 49 queries spanning 3 replaces, versus 55
  failures for the same test against drop-then-create — and collapses the live
  index fragment set exactly as before (7 index files back to 4).
- **The empty-index-dir sweep is bounded by lance's own threshold.** lance's
  `cleanup.rs` unlinks a superseded index's files but never its directory — it
  contains no `rmdir` at all, which is structural rather than an oversight: it
  targets object stores, where paths are flat keys and an empty directory does
  not exist. Only a local filesystem materialises them, and a soak run reached
  13061 dirs, 98% empty. everos sweeps them, now with three independent
  guarantees instead of a self-chosen age: `rmdir` cannot delete a non-empty
  directory (the kernel refuses it, so no file can be lost and there is no
  check-then-act window), live index UUIDs are excluded via `list_indices()`,
  and anything else must outlive `UNVERIFIED_THRESHOLD_DAYS = 7` — lance's own
  bound for deciding an unreferenced index UUID is dead rather than mid-build.
  The previous 300s was our invention, which is what made it indefensible.
  Two consequences worth knowing: the age gate reads the dir's mtime, which
  POSIX bumps when lance's cleanup empties it, so the effective reclaim horizon
  is up to ~14 days (file wait + age gate) and the ceiling-load steady state is
  ~1.8M dirs / ~7GB; and a sweep that blows its 60s deadline is swallowed
  inside `prune()` — the cleanup commit already succeeded, so escaping would
  bill a prune "failure" (feeding the fallback-rebuild threshold) and stall the
  prune-staleness clock for a cleanup stall that did not happen.
- **The optimize runner's wait on an in-flight rebuild is bounded too.** The two
  maintenance jobs park on each other — whichever arrives second waits — so an
  unbounded wait on this side is the same hazard as the one already fixed on
  the rebuild side, just seen from the other end: the kind's task slot stays
  occupied, `_schedule_optimize` keeps short-circuiting on it, and that table
  quietly stops being pruned. It was left open on the argument that
  `rebuild_indexes` carries its own 300s deadline, which covers its critical
  section but not the task's dispatch and teardown, so the transitive bound was
  never real. Now bounded at 180s, logging
  `cascade_lancedb_optimize_skipped_rebuild_unfinished` and skipping the beat
  rather than compacting under a live rebuild — the two commit on the same
  manifest, which is what the wait exists to prevent.
- **A rebuild that loses a commit race is retried** instead of waiting out the
  full 12h cadence. Lance labels the conflict `Retryable` and it is: a
  concurrent writer in another process won the manifest, nothing is wrong with
  the table. Retries are scheduled on the kind (10min / 30min / 3h) rather than
  slept through, so the other kinds in the sweep are not parked behind the
  backoff. A soak run at a 600s cadence hit 3 conflicts in 119 attempts, all
  while a concurrent CLI storm was running.

- **The index-rebuild sweep can no longer park forever** waiting on the
  optimize runner. That wait had no deadline, and the runner's loop condition is
  "keep going while there is unindexed data" — which under sustained writes is
  never, since the drain loop re-raises the flag every second against a 10s
  cooldown. Now bounded at 180s, logging
  `cascade_lancedb_rebuild_skipped_optimize_unfinished` and skipping the sweep
  rather than dropping indices under a live optimize. **This makes the stall
  visible, not absent**: under sustained ingest every sweep still times out, so
  active index-UUID / FTS `part_N` growth stays unbounded there. The functional
  fix requires the optimize runner to yield when a rebuild is pending, which
  changes the optimize/rebuild mutual-exclusion contract and needs its own
  validation — the rebuild cadence is 12h, longer than any soak run so far, so
  the periodic sweep has never been exercised under load.
- **The memory-root lock wait is bounded and visible.** Acquisition polls with
  `LOCK_NB` instead of blocking inside a worker thread: a blocking `flock` could
  not be bounded or cancelled — cancelling the awaiting coroutine left the
  thread to acquire the lock later with nobody to release it. The wait itself is
  by design (the second process is supposed to wait, then find the migration
  already done), but it now logs `memory_root_lock_waiting` and gives up after
  `timeout_seconds` (default 30min) instead of leaving a server startup looking
  like a hang whose last message is `lifespan_provider_startup name=lancedb`.
  The default sits an order of magnitude above the worst legitimate hold (a
  large migration is minutes) on purpose: the wait is already visible from the
  first poll, and against the one case the bound exists for — a holder that is
  alive but wedged — giving up at 5 minutes buys nothing over 30, while a bound
  near the legitimate hold turns a slow migration into startup crashes for
  every waiting process.

### Added

- `OfflineEngine.trigger_manual` now returns
  `tuple[BaseEvent, list[tuple[StrategyMeta, str]]]` instead of `None`,
  enabling the `dispatched`/`runs` fields below.
- `TriggerResponse` gains `dispatched: int` and `runs: list[RunSummary]`.
- `OMEConfig` gains `retry_backoff_base_seconds`, `retry_backoff_cap_seconds`,
  and `retry_jitter_seconds` for the retry-loop sleep.
- `AgentSkillReader.list_by_cluster()` enumerates the cluster's SKILL.md
  files from markdown (strong-consistency existence check).
- **`[cascade]` settings section** — the four maintenance cadences
  (`optimize_heartbeat_seconds`, `optimize_prune_interval_seconds`,
  `optimize_prune_retention_seconds`, `optimize_rebuild_interval_seconds`) are
  now configurable. They were already constructor arguments on `CascadeWorker`,
  but `CascadeConfig` did not carry them and no production path passed one, so
  the defaults were unreachable — which is why the 12h rebuild sweep could not
  be exercised by any soak run shorter than half a day. The deadlines that
  bound a hung call are deliberately **not** exposed: they are hang-catchers
  sized from measured durations, where too low manufactures failures on a
  healthy table and too high leaves a wedged one invisible for longer. Note
  `optimize_prune_retention_seconds` has a second effect worth reading before
  tuning — it also decides how long index files keep a manifest naming them,
  and below LanceDB's 7-day unverified window they then wait out the full 7
  days.

### Changed

- **`extract_foresight` now ships disabled** (`enabled=False`). Not because
  it is broken — the crash below is fixed — but because it is one LLM call
  per sender per memcell whose output nothing in EverOS reads today: no
  search route surfaces foresights and no prompt slot consumes them. Until
  something does, running it by default spends tokens on write-only data.
  **Re-enable per install** in `ome.toml` (hot-reloaded, no restart):

  ```toml
  [strategies.extract_foresight]
  enabled = true
  ```

  Editing `default_ome.toml` alone would not have reached existing installs
  — `everos init` does not overwrite an existing `~/.everos/ome.toml` — so
  the code default is what changed.
- **`extract_foresight` no longer crashes on a memcell containing tool
  calls.** The sender scan read `m.role` off every item, but only
  `ChatMessage` carries it (`ToolCallRequest` has `sender_id` without it,
  `ToolCallResult` has neither), so the first tool call raised
  `AttributeError` — before any sender was resolved. The strategy was
  correct on plain user chat and dead-lettered every time on agent
  trajectories. everalgo explicitly contracts for the mixed case
  (`user_memory/_render.chat_messages`: the caller need not pre-filter),
  and every other user-memory extractor gets that for free by delegating;
  this was the one place the filter was hand-rolled. The scan now tests
  `isinstance(m, ChatMessage)`, so a pure agent trajectory yields no senders
  and returns without an LLM call. Matters even with the strategy off by
  default: it is what makes the opt-in above actually usable.
- **`SkillClusterUpdated` carries the case's 1024-dim embedding, growing the
  OME `run_record` table.** The event payload is persisted verbatim in
  `run_record.event_payload` (and in the APScheduler jobstore while a job is
  queued), so a `skill_cluster_updated` record goes from roughly 0.8 KB to
  14 KB. At the default `max_records_per_strategy = 1000` ring buffer that is
  ~14 MB for this one strategy instead of ~0.8 MB. **Operators sizing
  `~/.everos/.index/sqlite/ome.db` should expect this.** The vector is only
  read when a cluster holds more skills than `MAX_SKILLS_IN_PROMPT`, so it
  usually rides along unused; trimming it from the persisted copy is not a
  local change, because crash recovery replays `event_payload` to rebuild the
  event and a trimmed payload would silently take the recovered run down a
  different branch than the original. Tracked as a follow-up.
- **`cascade_lancedb_optimize_conflict` now records `pruned`** — which
  maintenance beat lost the commit race. Lance labels both beats' commit the
  same way (`This Rewrite transaction was preempted by concurrent transaction
  …`), so the message alone cannot separate a free loss from a costly one: a
  lost **light** beat retries ~10s later, while a lost **heavy** beat means
  that table skipped a whole prune cadence and its superseded files stay on
  disk. Attributing index-dir growth previously meant back-inferring which
  beats were heavy from the 300s cadence. Log level (`debug`) and the
  benign-conflict semantics are unchanged — this adds one field.

### Upgrade

Two behaviour changes to know about before upgrading.

`extract_foresight` now ships disabled — a deployment relying on foresight
entries must set `enabled = true` for it in `ome.toml`.

`SkillClusterUpdated` now carries the case's 1024-dim embedding, so a
`skill_cluster_updated` row in the OME `run_record` table grows from ~0.8 KB to
~14 KB — about 14 MB for that strategy's default 1000-record ring buffer,
against ~0.8 MB before. Anyone sizing `~/.everos/.index/sqlite/ome.db` should
account for it.

Nothing else needs action: the new `[cascade]` section is optional and a config
written by an earlier version falls back to the same defaults (verified on a
clean install).

One deployment note. When a supervised background loop crashes repeatedly and
exhausts its restart budget, the worker now sends itself `SIGTERM` rather than
serving on with a dead projection pipeline. That assumes something restarts the
process — systemd `Restart=always`, Docker `restart: unless-stopped`, a k8s
Deployment. Without one the process simply stops, which is still preferable to a
server answering searches from a silently frozen index, but it is worth knowing
before the first time it happens.

## [1.2.2] - 2026-08-04

### Added

- **`GET /health` now carries a `cascade` readiness block** — `healthy`,
  human-readable `reasons`, and the counters behind them (`pending`,
  `failed_permanent`, `failed_retryable`, `drain_consecutive_failures`,
  `unrecoverable_total`, `optimize_failure_streak`, `prune_stale_seconds`).
  `null` when the app runs without the cascade lifespan. **Alert on
  `cascade.healthy`**: it flips false only on operational faults — drain loop
  failing (≥3 in a row), index maintenance wedged (≥5), or version cleanup
  stalled on some table (≥3 missed 300s beats, and `reasons` names the table).
  `failed_permanent` is a data-quality backlog awaiting `cascade fix` and
  deliberately does **not** flip `healthy`, otherwise the signal sits red until
  a human edits markdown. The HTTP status stays 200 even when the block says
  unhealthy — it is a liveness signal, and a degraded projection must not
  trigger a container restart. If the probe itself fails (locked / full
  SQLite), the block returns `healthy=false` with a `cascade health probe
  failed: …` reason and zeroed counters — read zeros next to that reason as
  "unknown", not "clean".
- **`everos cascade rebuild` CLI command** — drops every business LanceDB table,
  clears the cascade queue, and re-indexes all markdown from scratch. The
  supported recovery from a drifted or corrupt index: unlike deleting the index
  directory, it re-enqueues every file (a bare `rm -rf` leaves the queue marked
  `done`, so nothing re-indexes and the index comes back empty), and unlike
  deleting `.index/` it preserves SQLite state that markdown cannot rebuild —
  notably `unprocessed_buffer`. **Requires the server to be stopped**: it
  refuses to start (exit code `3`) while a server holds the OME lock, because a
  live daemon keeps writing through cached table handles to the dropped
  dataset. `--yes/-y` for non-interactive use; `Ctrl-C` exits `130` and the
  re-index resumes on the next run or server start.
- **Startup schema verification now detects column *type* drift**, not just
  missing / extra columns. Catches the class of corruption behind #337 — an
  `episode.subject_vector` left as `string` by an older build while the schema
  declares a 1024-d `fixed_size_list` — which a name-only check waved through
  and which then failed deep inside `merge_insert` with an opaque
  `LanceError(IO)`. The error now points at `everos cascade rebuild`.

### Changed

- **LanceDB maintenance is split into compaction and reclamation.**
  `optimize()` is lock-free compaction; the new `prune()` runs
  `cleanup_older_than` under the per-table write lock. Fixes unbounded index
  growth: the previous bundled call issued a Rewrite that concurrent writes
  kept preempting, so version cleanup lost the race indefinitely (a soak run
  measured 16 successes against 547 conflicts over 21h, with the index
  directory growing to the disk guardrail). Reclamation now completes on every
  beat, at the cost of a brief same-table write stall (measured ~40ms).
  Retention is decoupled from cadence: files older than 60s are eligible,
  reclaimed on a 300s beat.
- **Every write-lock critical section is now bounded.** All seven operations
  (`add` / `upsert` / `update` / `delete` / `delete_by_md_path` / `prune` /
  `rebuild_indexes`) run under a deadline that covers **lock acquisition as
  well as the body**, so no code path can wait for the lock — or hold it —
  indefinitely. Budgets are sized from measured durations (row writes are
  2–25ms, worst observed 63ms → 15s; index rebuild → 300s; prune → 60s).
  Expiry raises the retryable `VectorStoreBusyError`, so the cascade worker
  retries the row instead of marking it permanently failed. Without this, one
  operation stuck outside the old narrow timeout wedged a table permanently:
  every writer blocked on acquire, and the maintenance scheduler skipped a kind
  whose task never finished, so that table stopped reclaiming versions
  altogether (observed: 150 versions retained, disk 11x live size, with nothing
  logged because nothing failed).
- **Benign LanceDB commit conflicts no longer count as failures.** A lost
  optimistic-concurrency race logs at `debug` on either maintenance beat. The
  heavy beat needs this too: its write lock is in-process only, so a second
  process (`cascade sync`, `cascade backfill`) can preempt its commit. Counting
  those triggered spurious fallback index rebuilds, which drop every index
  before recreating them — and if the rebuild also lost the race, the table sat
  without an FTS index and every search on that kind returned 500 until the
  next 12h sweep.
- **A query vector whose width disagrees with the embedding provider's declared
  `dim` now fails immediately** with `CONFIGURATION_ERROR` instead of reaching
  LanceDB. It previously surfaced as an opaque `ValueError` after the query was
  built — 13–14s per request, as an unhandled 500.
- **Exception logging no longer renders frame locals.** structlog's default
  traceback formatter (`show_locals=True`, up to 100 frames) rendered one
  unhandled exception on an async stack into 6423 log lines — 85MB of logs
  across 11 exceptions in one soak run — at ~290ms of synchronous CPU each, and
  risked printing request payloads into logs. Now locals-off and capped at 15
  frames: the same traceback is 103 lines.
- **`cascade backfill` reclaims through the daemon's retention window** rather
  than at zero age, so it cannot delete files out from under an in-flight
  `/search` in the server process.
- **`lancedb` pinned to `>=0.34.0,<0.35.0`.** 0.35 embeds lance-rust v9; 0.34.0
  is the version validated under sustained churn. Environments installing from
  `uv.lock` are unaffected (already 0.34.0). Never widen the floor below 0.34 —
  older lance cannot read v8 data.

### Fixed

- **Maintenance deadlines now cover the whole call, not just the critical
  section.** Resolving a table handle sat outside the timeout, so a hang there
  never returned — and because the scheduler runs one maintenance task per kind
  and skips a kind whose task is in flight, that table stopped being maintained
  permanently and silently (a soak run caught one table 13 minutes without a
  reclaim, retained versions climbing, while its siblings reclaimed normally and
  nothing was logged because nothing failed). Handle resolution moved inside the
  deadline for all seven locked operations, the lock-free compaction beat got
  its own deadline, and the scheduler adds a last-resort 180s bound on the whole
  call. The per-kind staleness alert added in this release is what surfaced it.
- **AGENTIC search crashed on agent memory** (`agent_case` / `agent_skill`) —
  candidate metadata now satisfies the everalgo `_format_docs` contract,
  removing a `TypeError` in the sufficiency / multi-query steps.
- **Per-kind version-cleanup staleness is no longer masked.** The health signal
  reported time since the newest successful prune across all kinds, so on a
  multi-kind deployment one kind whose cleanup died was hidden by the others
  pruning on schedule. It now reports the worst kind and names it.
- **`cascade backfill` silently skipped compaction and reclamation** — it still
  called the removed `optimize(cleanup_older_than=…)` signature, and the
  resulting `TypeError` was swallowed by a best-effort `except`, so the disk
  growth this release fixes came back after every backfill.
- **Empty `_indices/<uuid>/` husks are removed after cleanup** (a soak run
  accumulated 13061 directories, 98% of them empty), which bloated inode usage
  and slowed directory scans.

### Docs

- Rewrote the cascade runbook's recovery paths: the `/health` cascade block and
  its alert thresholds, `cascade rebuild` (including the stop-the-server
  requirement), why `rm -rf .index/lancedb` yields an empty index, and why
  `rm -rf .index` loses un-extracted buffered messages.

## [1.2.1] - 2026-07-29

### Added

- **`[embedding]` and `[rerank]` are now soft runtime dependencies** — EverOS
  boots and serves requests with only `[llm]` configured. Missing or
  misconfigured embedding / rerank / multimodal providers no longer abort
  startup; the corresponding accessor logs `<provider>_capability_build_failed`
  and reports `available=False`. Features degrade gracefully into three tiers:
  Tier 1 (`[llm]` only) → KEYWORD search + add/flush + md writes + cascade sync;
  Tier 2 (`+ [embedding]`) → adds VECTOR/HYBRID search + reflection + skill
  extraction + backfill; Tier 3 (`+ [rerank]`) → adds AGENTIC search + knowledge
  write/search. Tier upgrades require a server restart. Downgrades are
  read-safe: knowledge documents stay readable/renamable/deletable after
  a Tier-3 → Tier-2 downgrade; only write / search endpoints return 422.
- **`everos cascade backfill` CLI command** — three-phase interactive
  backfill (`vectors` → `clusters` → `skills`, or `--phase all`) for
  upgrading Tier-1 rows to Tier-2 after `[embedding]` is configured. Each
  phase prints row/token estimates and blocks on `y/N`; `--yes`/`-y` for
  CI. Exit codes: `0` success, `1` user declined, `2` phase preconditions
  unmet, `3` server running, `4` completed-with-failures, `130` SIGINT.
- **LanceDB schema v2** — the six business tables (`episode`, `atomic_fact`,
  `foresight`, `agent_case`, `agent_skill`, `knowledge_topic`) now allow
  `vector NULL`. Cascade writes rows without vectors when `[embedding]` is
  unavailable; a later backfill fills them in. Migration runs once on first
  startup under a cross-process `memory_root_lock` (`fcntl.flock`),
  followed by `optimize(cleanup_older_than=timedelta(0))` per table to
  physically prune older manifest versions.
- **Startup unbackfilled-rows banner** — after LanceDB lifespan, an
  unconditional sweep emits `unbackfilled_memory_rows` when rows with
  `vector IS NULL` exist, pointing at `everos cascade backfill` in the
  hint text.
- **PyPI Trusted Publishing workflow** — tag-triggered `.github/workflows/release.yml`
  builds, smoke-tests, and uploads via OIDC (no stored token) behind the
  `release` environment's manual-approval gate. Version-tag mismatch
  aborts publish. Companion `/release` skill lives under
  `.claude/skills/release/`.

### Changed

- **`ProviderNotConfiguredError` → HTTP 422 `CAPABILITY_UNAVAILABLE`** —
  write / search endpoints that need embed or rerank now return 422 with
  a section-aware hint (points at `everos.toml` section, never at
  `EVEROS_*` env vars) instead of erroring at startup or 500-ing at
  request time.
- **`GET /health` returns a Pydantic `HealthResponse` model** — with typed
  `capabilities` and `disabled_features` fields, so OpenAPI codegen
  produces real shapes instead of `additionalProperties: true`.
- **`MemoryRoot.default()` → `MemoryRoot.resolve()`** — the classmethod
  that resolves the memory root from `--root` / `EVEROS_ROOT` / default was
  renamed to make its behavior explicit (`resolve` walks the precedence
  chain; `default` was ambiguous with "default location"). `MemoryRoot`
  is publicly exported from `everos.core.persistence`; callers outside
  the repo may have used the old name. **A `default()` alias is kept**
  as a backward-compatibility shim that forwards to `resolve()` and
  emits a `DeprecationWarning`. The alias will be removed in a future
  major release — update call sites when convenient.
- **Uncalibrated recall scores moved to their own name** — `KEYWORD` and
  single-route `VECTOR` searches now report their top score as
  `recall_top_score_raw`; `recall_top_score` is reserved for the calibrated
  methods (`HYBRID` LR sigmoid, `AGENTIC` cross-encoder), whose values share a
  comparable `[0, 1]` scale. Langfuse aggregates scores by name, so the previous
  single name meant a chart could average an unbounded BM25 score together with
  a probability. Every recall score also carries
  `metadata = {"method": ..., "calibrated": ...}` now, a structured field that
  can be split on, alongside the existing human-readable comment. Dashboards
  built on `recall_top_score` for keyword search need to switch to the new name.
- **Docs and examples now use `/api/v2`** — README, QUICKSTART, the `docs/`
  reference set, the Langfuse example, and `everos demo --live` all call the
  canonical `/api/v2` prefix instead of `/api/v1`. `/api/v1` keeps resolving
  to the same handlers, so nothing breaks; it is now described as a **legacy
  compatibility alias that may be removed in a future major release** rather
  than a permanent one. New integrations should target `/api/v2`.
- **`cluster_repo.find_cluster_id_for_member` now requires
  `(app_id, project_id, owner_id)`** — reverse-index lookups JOIN `Cluster`
  for scope filtering. `entry_id` is per-owner unique by design; the
  reverse index alone could collide across owners writing on the same day.
- **All six cascade handlers register unconditionally** — Tier-3 → Tier-2/1
  downgrade no longer strands DELETE / PATCH events. Embed-requiring
  branches inside each handler body-guard on capability availability at
  execution time.
- **Interactive TTY log level defaults to WARNING** — avoids INFO log lines
  drowning out backfill y/N prompts; non-interactive / CI stays at INFO.
  `--verbose` / `-v` forces INFO.
- **`click>=8.1` promoted to first-class dependency** — was previously
  transitive via typer. `typer.Abort` and `click.exceptions.Abort` are
  distinct classes under typer 0.15+ (typer vendored click); the interrupt
  catch in `cascade backfill` covers both.
- **Test harness pins `EVEROS_ROOT` to a temp path** — `conftest.py` scrubs
  every `EVEROS_*` env var so a developer's `~/.everos/everos.toml` cannot
  make tests accidentally green against a real provider.

### Fixed

- **Fixes present in `1.1.4` but missing from `1.2.0`** — cascade retry
  classification + total-retry budget, the delete/modify race, the
  embedding empty-data guard, and episode extraction retries (each
  detailed under 1.1.4) were absent from the branch `1.2.0` was built
  from, so `1.2.0` regressed to pre-1.1.4 behaviour on all of them. See
  Security below for the path traversal.
- **Filename validation on knowledge upload** — NUL byte and > 255-byte
  UTF-8 filenames now fail fast with `InvalidInputError → HTTP 400`
  instead of surfacing OS errors as 500 with a half-written md file.
- **HTML upload no longer takes the UTF-8 fast-path** — knowledge
  upload's plaintext short-circuit uses an explicit allowlist
  (`text/plain`, `text/markdown`, `text/x-rst`, `text/x-markdown`) plus
  known extensions; `text/html` is deliberately excluded so HTML still
  goes through everalgo's `clean_html_for_llm`. Prevents 503 when a
  Tier-3 user without `[multimodal]` uploads a markdown doc.
- **Broken table-of-contents links in `docs/api.md`** — the endpoint anchors
  still pointed at the pre-1.2.0 `#post-apiv1…` slugs after the headings moved
  to `/api/v2`, so all five endpoint links in the TOC were dead.

### Removed

- **README "Star us" section** — cleanup.

### Security

- **Knowledge upload path traversal (CWE-22)** — see the 1.1.4 entry for
  the fix description. **Affected:** every release before `1.1.4`, and
  `1.2.0`. **Not affected:** `1.1.4`. **Fixed in:** `1.2.1`. The fix
  shipped in `1.1.4` but was not present on the branch `1.2.0` was built
  from, so upgrading `1.1.4` → `1.2.0` reintroduced it.

## [1.2.0] - 2026-07-24

> This release is missing fixes that shipped in `1.1.4`, including a
> knowledge-upload path traversal (CWE-22). See the Security section
> under 1.2.1 for the affected-version range.

### Added

- **`/api/v2` API prefix** — every business endpoint (`memory/*`, `ome/*`,
  `knowledge/*`) is now served under `/api/v2`, aligning the open-source API
  with the EverOS Cloud contract. `/api/v1` is retained as a legacy
  compatibility alias: both prefixes resolve to the same handlers with
  identical request/response contracts, so existing integrations keep working
  unchanged. Infrastructure endpoints (`/health`, `/metrics`) stay unversioned.
- **Native OpenTelemetry tracing** — memory operations (add / flush, memcell
  boundary, episode extraction, search, and OME reflection) export to any
  OTLP backend (e.g. Langfuse) as nested traces carrying LLM/embedding token
  usage, per-request correlation, and recall-quality scores. Off by default;
  enabled via the `[observability]` config with the optional `otel` extra.
  Content capture (query / extracted memory) is opt-in and redaction-aware.

## [1.1.4] - 2026-07-20

> The entries below reflect the code shipped as `everos==1.1.4` on PyPI.
> The 1.1.4 sdist was built from the internal release lane and contains
> fixes that were not represented in this file when 1.1.4 was tagged;
> this section restores them so the changelog matches the wheel.

### Fixed

- **Knowledge upload path traversal (CWE-22)** — the original-file write
  path is now contained to the document directory; adversarial filenames
  (`..`, absolute paths, symlink games) are rejected on
  `POST /api/v1/knowledge/documents` and `POST /api/v2/knowledge/documents`.
- **Cascade reliability — retry classification, budget, and races** — the
  worker catches `ExternalServiceError` (embedding / LLM / rerank transient
  failures) and retries inline up to 3 times before marking
  `retryable=True`; a total retry budget of 12 attempts across scanner
  cycles bounds retries on prolonged outages. The reconciler no longer
  re-enqueues `pending` / `processing` rows on stable mtime (previously
  overwrote the worker's `mark_done`); `failed` rows with
  `retryable=False` on stable mtime skip auto-retry so users can edit and
  re-save. SQLite `REAL` float precision loss in mtime comparisons is
  now absorbed via a 10 ms tolerance. LanceDB `optimize()` failures
  escalate to a `drop_index + create_index` rebuild after 5 consecutive
  misses (workaround for `lance-format/lance#7653` panic path).
- **Cascade delete/modify race** — when a file disappears after its
  modified event is queued, the worker processes it as a deletion
  instead of leaving a stale indexed row and permanently failed queue
  item.
- **Embedding provider raises on empty API data** — the provider now
  raises `EmbeddingServiceError` when the API returns HTTP 200 with an
  empty `data` array (previously silently returned zero-length vectors,
  corrupting search).
- **Episode extraction retries on malformed LLM output** — the `/flush`
  synchronous path retries everalgo `ValueError` (typically OpenRouter
  truncated responses) twice with 1 s / 2 s backoff before surfacing a
  500.
- **Langfuse live-server traces use only real telemetry** — synthetic
  child spans are now limited to responses that provide stage details,
  while real servers emit accurate top-level latency, output, and
  recall-quality scores.

### Added

- **Optional `dimensions` parameter for MRL-capable embedding models** —
  opt-in via `[embedding] dimensions = N` in `everos.toml`; forwarded to
  the API for server-side truncation with re-normalization (OpenAI
  text-embedding-3-\*, Qwen3-Embedding).
- **LLM `finish_reason` diagnostic warnings** — logs `content_len` /
  `content_tail` / `model` when the provider returns a non-`stop`
  `finish_reason`, aiding OpenRouter truncation triage.
- **Langfuse integration example** — added an OpenTelemetry-based
  wrapper for tracing EverOS add, flush/extract, search, and reflection
  operations, with a built-in mock and support for connecting to a real
  EverOS server.

### Changed

- **`everalgo-user-memory` bumped 0.3.1 → 0.3.2**.

## [1.1.3] - 2026-07-10

### Fixed

- **LanceDB FTS optimize crash and disk growth** — disabled unused positional
  data in OR-mode BM25 indexes, automatically rebuilds affected indexes, and
  escalates repeated optimize failures so cleanup cannot fail silently.

## [1.1.2] - 2026-07-07

### Fixed

- **Agent-track search broken by `deprecated_by IS NULL` filter** —
  `compile_filters()` unconditionally appended a `deprecated_by IS NULL`
  clause to every LanceDB query, but only `episode` and `atomic_fact`
  tables have this column. Agent-track search (`agent_case`,
  `agent_skill`) failed on any method. The clause is now conditional on
  `owner_type == "user"`.

## [1.1.1] - 2026-07-06

### Added

- **DashScope rerank provider** — Aliyun Bailian `gte-rerank-v2` adapter;
  configure with `rerank.provider = "dashscope"` in `everos.toml`.
- **`everos demo` TUI command** — Textual-based interactive CLI demo for
  showcasing EverOS core features.
- **Benchmark runner** — full LoCoMo benchmark suite: `benchmarks/run.py`
  with TOML configuration, automated ingestion, search evaluation, and
  scoring.
- **Hybrid search: heap-expand algorithm** — rewrote `hierarchy.py` to
  heap-driven lazy expansion with global top-N competition, replacing the
  serial four-layer pipeline.

### Fixed

- **Knowledge: atomic upsert prevents StaleDataError** — cascade handler
  switched from get→update to `INSERT ... ON CONFLICT DO UPDATE`, fixing
  concurrent cascade race conditions.
- **API: OpenAPI version read from `__version__`** — no longer hardcoded to
  `0.1.0`; version now stays in sync with `pyproject.toml`.
- **Profile middleware no longer swallows exceptions** — inner handler
  errors now re-raise correctly instead of silently returning HTTP 200.

### Performance

- **Cascade optimize throttle 1s → 10s** — reduced unnecessary LanceDB
  `optimize()` I/O by raising the minimum interval between calls.

### CI / Build

- **CI Python version matrix** — test and integration jobs now run on both
  Python 3.12 and 3.13.
- **pyproject.toml improvements** — added `project.urls`, `Typing :: Typed`
  classifier, relaxed `jieba` version constraint, removed unused
  `python-dotenv` dependency, cleaned up sdist include list, added `RUF`
  lint rules and coverage configuration.
- **`make ci` includes coverage** — `ci` target now runs
  `lint + test + integration + cov`.

### Documentation

- Fixed stale references across 13 files (v1.1.0 freshness sweep).
- Added GitHub sync guide (`docs/github-sync.md`).
- Added v1.1.0 release notes and v1.0.0 migration guide as standalone docs.
- Added `README.zh-CN.md` (Chinese README).
- Expanded `QUICKSTART.md` with source install instructions and `uv run`
  usage notes.
- Clarified cascade `optimize()` semantics in docstrings and runbook.

## [1.1.0] - 2026-06-24

### Added

- **Knowledge base subsystem** — full-stack document management exposed via
  `/api/v1/knowledge/*`. Upload documents (PDF / HTML / DOCX via multimodal
  parser), CRUD operations, and hybrid search (BM25 + vector + rerank +
  category boost). Ships with a 20-category default taxonomy
  (`.taxonomy.md`, auto-generated on first use). Original uploaded files are
  preserved alongside extracted Markdown. New settings group:
  `knowledge.*` (search tuning, `max_upload_bytes`, etc.).
- **Reflection V1** — offline memory self-improvement engine.
  Select → Merge → Re-extract → Deprecate: clusters related episodes within
  existing 7-day windows, merges them via LLM, re-extracts consolidated
  episodes, and deprecates the originals. Runs as an OME strategy
  (`reflect_episodes`); configure via `ome.toml`
  (`[strategies.reflect_episodes]`, cron `0 2 * * 1`), changes are
  hot-reloaded within ~2 s, no restart needed; **disabled by default**.
  Requires `everalgo-user-memory>=0.3.1`.
- **Standardized error response contract.** All API errors now return a
  canonical envelope with a semantic `ErrorCode` (10 codes: `NOT_FOUND`,
  `CONFLICT`, `INVALID_INPUT`, `EXTRACTION_EMPTY`, `UNSUPPORTED_FORMAT`,
  `EXTERNAL_SERVICE_UNAVAILABLE`, `CAPABILITY_UNAVAILABLE`,
  `CONFIGURATION_ERROR`, `INTERNAL_ERROR`, `BAD_REQUEST`), per-type
  exception handlers with MRO dispatch, and an `ErrorResponse` Pydantic
  model visible in OpenAPI docs. Replaces the v1.0 two-code scheme
  (`HTTP_ERROR` / `SYSTEM_ERROR`).
- **Search: hierarchical fact eviction** (Layer-4) with `min_score` floor —
  low-confidence atomic facts are evicted before fusion, improving
  precision.
- **Knowledge search degradation guidance** — when the embedding or rerank
  provider fails at call time, the knowledge search route enriches the
  error message with actionable guidance (e.g. retry with `method=keyword`,
  which needs no embedding) before returning `503`.
- **Knowledge topic recaller** — dual-column BM25 recall for knowledge
  topics, integrated into the search manager alongside existing recall
  types.

### Changed

- **`everos init` now generates `gpt-4.1-mini`** as the default LLM model
  (was `gpt-4o-mini`). Existing user configurations are not affected.
- **API error `code` values have changed.** v1.0 returned only `HTTP_ERROR`
  (all 4xx) and `SYSTEM_ERROR` (all 5xx). v1.1 returns fine-grained
  semantic codes (see Added above). Clients that match on `error.code`
  string values need to update. The envelope structure
  (`request_id` + `error.{code, message, timestamp, path}`) is unchanged.
- **DDD-aligned exception hierarchy** — domain errors reorganized:
  `ValidationError` → `InvalidInputError`;
  `DocumentAlreadyExistsError` → `DuplicateDocumentError`;
  `EmbeddingError` → `EmbeddingServiceError`;
  `RerankError` → `RerankServiceError`;
  `LLMError` → `LLMServiceError` (at the boundary);
  `MultimodalError` split into `UnsupportedModalityError` (domain) +
  `MultimodalNotEnabledError` (infrastructure).
  New base classes: `CapabilityError`, `ConfigurationError`.
- **`infra/` restructured** — storage adapters moved under
  `infra/persistence/{markdown,sqlite,lancedb}`; each sub-package's
  `__init__.py` is the sole public API (enforced by import-linter).
- **Parser capability extracted** to `component/parser` (shared by memorize
  and knowledge upload paths).

### Fixed

- **Knowledge search no longer returns a bare `500 INTERNAL_ERROR` when the
  embedding or rerank provider is unconfigured.** `_require_search_providers`
  now raises `ConfigurationError` → `500 CONFIGURATION_ERROR`. A provider
  that is configured but fails at call time still surfaces as
  `503 EXTERNAL_SERVICE_UNAVAILABLE`.
- **Knowledge document uploads are capped** at `knowledge.max_upload_bytes`
  (default 50 MiB); oversized uploads are rejected with `422` before parsing.
- **Knowledge search `query` is bounded** to 2000 chars.
- **`GET /knowledge/documents?sort_by=updated_at`** is now accepted.
- **`POST /knowledge/documents` returns `original_file_path`** so callers no
  longer need a follow-up `GET` to locate the preserved upload.
- **Rerank providers no longer echo the upstream HTTP response body** into the
  client-facing `503` message (vLLM / DeepInfra); the body is logged instead.
- **Knowledge FK cascade race** — removed the foreign key on
  `knowledge_topics.doc_id` that caused delete-order race conditions;
  cascade cleanup handled at application level.
- **Knowledge `replace_document`** — atomic PUT: backup old Markdown before
  re-extraction; removed explicit SQLite delete for atomicity.
- **Knowledge duplicate `doc_id`** rejected on create; title collision
  resolved by appending `doc_id` to directory name.
- **Knowledge `md_path` resolution** fixed in `delete_document` (was not
  resolved against `memory_root`).
- **OME file-handle leak** — portalocker file handle is now closed on lock
  contention instead of being left open.
- **jieba / Python 3.12 compatibility** — deferred jieba import to avoid
  `SyntaxError` from invalid escape sequences; suppressed
  `DeprecationWarning` in tests.
- **Test isolation** — tests no longer leak `.env` state or depend on module
  import ordering.

### Documentation

- Added knowledge base technical documentation.
- Corrected the onboarding flow: `everos init` writes `everos.toml` +
  `ome.toml` (TOML), not a `.env` file; removed the nonexistent
  `--xdg` / `--env-file` options and the false `0600`-permissions claim
  from `README.md` / `QUICKSTART.md`; fixed the stable-version line
  (`v1.0.1`) and completed the `docs/cli.md` command tree.
- Updated error handling docs to match the new DDD exception hierarchy.

## [1.0.1] - 2026-06-16

### Security

- **Path-traversal hardening for caller-supplied identifiers.** `sender_id`
  (which flows through to `owner_id` and becomes a directory segment on the
  episode write path) now carries the same path-safety guard as `app_id` /
  `project_id`: a character whitelist plus rejection of the `.` / `..` tokens.
  The whitelist admits `@` and `+` so real-world ids (email-style,
  plus-addressing) still pass.
- **Defense-in-depth write containment.** `MarkdownWriter` now rejects any
  write target that resolves outside the configured memory root, before any
  filesystem touch (both the write `mkdir` and the append read-modify-write
  read). This backstop holds even if an identifier reaches the writer
  unsanitised (e.g. an `owner_id` set in the extract pipeline rather than from
  the DTO). The API layer maps the resulting error to HTTP 400.

### Documentation

- Add a multimodal usage guide and correct the multimodal error semantics
  after end-to-end verification.
- Rename the algorithm library to `everalgo` across docs and
  code comments (no code identifiers changed).
- Fix accuracy drift found in an adversarial doc audit; reflect the
  `everalgo` packages being published and the v1.0.0 stable status.

## [1.0.0] - 2026-06-03

First public release of EverOS — a Markdown-first memory extraction framework
for AI agents.

### Added

- **Markdown as source of truth** — all memory persists as plain `.md` files you
  can open, edit, grep, and version with Git.
- **Lightweight three-piece storage** — Markdown (truth) + SQLite (state / queue
  / audit) + LanceDB (vector + BM25 + scalar index). No external services
  required.
- **Hybrid retrieval** — BM25, vector, and scalar filtering in a single LanceDB
  query.
- **Cascade index sync** — editing a `.md` file triggers a file watcher →
  entry-level diff → sub-second LanceDB sync.
- **Dual-track memory** — user-track (Episodes / Profiles) and agent-track
  (Cases / Skills).
- **Multi-source extraction** — conversations, workflows, agent traces, and file
  knowledge.
- **CLI + HTTP API** — the `everos` command-line tool and a FastAPI server,
  async-first throughout.
- **Pluggable providers** — LLM / embedding / rerank via the OpenAI-compatible
  protocol (works with OpenAI, OpenRouter, vLLM, Ollama, …).
- **Decoupled algorithms** — memory extraction algorithms live in the standalone
  `everalgo-*` libraries published on PyPI.

[Unreleased]: https://github.com/EverMind-AI/everos/compare/v1.3.1...HEAD
[1.3.1]: https://github.com/EverMind-AI/everos/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/EverMind-AI/everos/compare/v1.2.3...v1.3.0
[1.1.4]: https://github.com/EverMind-AI/everos/compare/v1.1.3...v1.1.4
[1.1.3]: https://github.com/EverMind-AI/everos/compare/v1.1.2...v1.1.3
[1.1.2]: https://github.com/EverMind-AI/everos/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/EverMind-AI/everos/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/EverMind-AI/everos/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/EverMind-AI/everos/releases/tag/v1.0.1
[1.0.0]: https://github.com/EverMind-AI/everos/releases/tag/v1.0.0
