# Scenes

A scene is the layer of chat memory above the episodes: an owner's episodes
about one matter (a plan, a trip, a project, a relationship) with one
narrative written over them, in time order, ending on where things stand. A
user's scenes and a group's are one implementation; this page describes a
user's, and a group's differ only where
[group-memory.md](group-memory.md) 7.7 says so.
Detail questions are answered by episodes; "how did it go, where does it
stand" questions by scenes. Both sit in the same search pool, and the
multi-round decider picks between them.

Scenes are **off by default** (`[scenes] users = false`, `[scenes] groups =
false`) until their cost is accepted; see [Cost](#cost).

## What is stored

| Where | What |
|---|---|
| `users/<u>/scenes/scene-<YYYY-MM-DD>.md` | one entry per scene version (`sc_<date>_<seq>` ids); a superseded version is listed in the file's `deprecated_entries` against its successor, a forgotten one against `forgotten` |
| LanceDB `episode` table | each version is a row, `parent_type = scene`, `parent_id` = the scene id; only the newest current version is searchable |
| SQLite `cluster` | the user's scene clusters (kind `user_scene`, members = episode entry ids), singletons included |

A version carries `scene_id`, `version`, `member_ids` (the episode entry ids
it was written from), `timestamp` (its newest member's), `source_start` (its
oldest member's), and the sections `Subject` (title), `Summary` and
`Content` (the narrative). Episodes are never changed by a scene.

A group's scenes (`[scenes] groups`, [group-memory.md](group-memory.md) 7.7)
are written by the same core (`memory/strategies/_scenes.py`): the same
placement (kind `group_scene`), the same capped call and caps, the same
version rules and forget-aware member reads, under `groups/<g>/scenes/`
(`gsc_` ids). A group's version also spans its members, so a read shows it
only to an asker who can see every one of them.

## How a scene is written

The `update_user_scenes` strategy runs on every `EpisodeExtracted` the
one-to-one pipeline publishes (not on a Reflection merge), and
`update_group_scenes` on every `GroupEpisodeExtracted`; both run the scene
core under the owner's scene lock:

1. **Place.** The episode is embedded and folded into the user's scene
   clusters with everalgo's `cluster_by_geometry` (`[clustering] threshold`
   and `time_window_days`), or starts a cluster of its own.
2. **Skip.** A cluster with fewer than `[scenes] min_members` live episodes
   is not a scene yet: no call. Most episodes that join nothing cost nothing.
3. **Write.** Otherwise one `chat.scene_update` call writes the narrative
   again over the scene's current narrative (empty for a new scene) and the
   joining episodes, and the result is appended as the next version,
   superseding the previous one.

The prompt lives in EverOS (`memory/extract/scene/prompt.py`, registered
with id, version and fingerprint). Everything it reads and writes is
bounded, so a call's price does not grow with the scene:

- the narrative is asked for in at most `narrative_max_words` words (300),
  however long the input; a reply past 8 characters a word loses its oldest
  part at a sentence start, so the newest events and the closing state stay;
- the prompt carries that bounded narrative plus at most `max_new_episodes`
  joining episodes (the newest), each clipped to `episode_max_chars`;
- the reply is held to `3 * narrative_max_words + 512` tokens, and a reply
  cut there is retried once at `[llm] retry_temperature` (a runaway guard).

## Search

A scene version is a row of the episode table, so it is recalled by the same
BM25 and vector legs, fused by the same RRF, and shown to the decider in the
same `subject - body` line as an episode. A search result marks it
`type: "Scene"`. `get` lists a user's scenes under `memory_type = "scene"`
and never under `"episode"`.

Measured on the LoCoMo conversations 0-1 store of the cost probe (233
questions, search only, Qwen3.8-27B base as the decider, top 20), against
the same store with its scenes removed: a scene reached the decider's core
in 154 questions (66%) and made up 22% of the core items; the core grew
from 4.5 to 5.3 items; the decider read 845 more prompt tokens a question
(+5.6%) and the returned context grew by about 200 tokens (+2.3%). Scenes
took 3.6 of the 20 returned places, so the returned episodes covered all
gold sessions on 98.7% of the questions instead of 100% (a scene has no
session of its own). Whether answers improve is not measured yet.

## A person's version

`POST /memory/update` with a scene version's id writes the scene's next
version with the person's subject, summary and narrative, the same members,
and `supersedes` / `edited_by` / `edited_at` / `held` (docs/api.md). While
the newest version is **held**, an episode joining the scene joins its
cluster and no narrative is written over the person's: no call. Released
(`hold: false`), the next join writes the narrative over the person's, with
every episode that joined meanwhile (at most `max_new_episodes` read).
`GET /memory/history` lists every version of the scene, newest first.

## Forgetting

Forgetting follows the sources (EverOS 2.0 has no hard delete):

- **A forget of episodes** -- a session, episodes by id, a time range, the
  whole user -- marks a scene's current version forgotten once none of its
  episodes is alive any more, and erases its row. While one is alive the
  scene stays. The undo clears the mark again when an episode comes back.
- **A forget of the scene itself** (by id, by a time range, the whole user)
  marks its current version forgotten and removes its cluster, so the next
  episode does not write it again; the undo puts both back. Its episodes stay.
- **A later version** is written from the live episodes only. When a member
  was forgotten after the previous version, that narrative is not carried
  forward: the next version starts over from the live members. A forgotten
  version is never superseded, so its mark stays for the undo.
- **A picture forget** writes a scene whose narrative names the picture
  again without the reference (a text edit, no model call), like the
  episodes that named it -- a held version too. A member episode that was
  rewritten counts as its new version.
- **`/memory/delete`** is a forget of the same scope.

## Cost

Measured on 2026-11-12 with DeepSeek V4.1 Flash (thinking off), product
defaults plus `[scenes] users = true`, prompt `chat.scene_update` v2,
everalgo-user-memory 0.9.0rc4, off-peak prices (peak doubles them):

| | LoCoMo conv 0-1 (788 messages) | MemEye, 8 conversations (1,696 messages) |
|---|---|---|
| Scene calls | 46 (20 new scenes, 26 updates) | 151 (22 new scenes, 129 updates) |
| Episodes that joined no scene (no call) | 20 of 86 | 27 of 200 |
| Scenes; episodes per scene (mean, max) | 20; 3.3, 8 | 22; 7.9, 39 |
| Scenes, $ per 1,000 messages | $0.026 | $0.039 |
| Same, against the text write cost without scenes | +13% of $0.200 | +21% of $0.186 |
| One call: $, prompt tokens (mean, max), output tokens (mean) | $0.00044, 1,181 / 1,461, 495 | $0.00044, 1,078 / 1,890, 527 |

The prompt does not grow over a run: on LoCoMo the mean scene prompt was
1,171 tokens over the first third of the calls and 1,205 over the last; on
MemEye 1,150 and 1,039, with a 39-episode scene among them. A reply past the
cap was clipped in 13 of 46 and 24 of 151 calls.

Cheaper triggers, counted on the same joins (not shipped; `min_members` is a
setting):

| Trigger | LoCoMo calls | MemEye calls |
|---|---|---|
| Every join once a scene has 2 episodes (shipped) | 46 | 151 |
| A scene from 3 episodes (`min_members = 3`) | 26 | 129 |
| Every second join | 32 | 82 |
| One call per scene per session | 44 | 151 |

In a conversation with two human speakers each speaker owns a copy of every
episode and gets their own scenes: 10 of LoCoMo's 46 calls repeated another
speaker's request exactly.

What grows with history:

- nothing in a call: the prompt and the reply are capped;
- per update, the strategy reads the scene's member files and the owner's
  scene files (file I/O, no model call), and lists the owner's scene
  clusters;
- one more embedding per episode (the placement), on the local embedder.
