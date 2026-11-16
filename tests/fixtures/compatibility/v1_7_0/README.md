# Frozen compatibility baseline: EverOS 1.7.0

`episode_extracted.json` is an `EpisodeExtracted` payload in the per-copy
shape EverOS 1.7.0 persisted (one event per owner copy: `owner_id`,
`episode_entry_id` and `owner_name` at the top level). It was produced on
2026-11-16 by that release's own model: `src/everos/memory/events.py` at
commit `489ac7e9` (v1.7.0), loaded on its own and serialized with
`model_dump_json`. The field values are synthetic.

The current `EpisodeExtracted` is one event per memcell with a `copies`
list; this payload must keep validating as a one-copy event, because OME
replays the persisted payload of every run that was pending or running when
the previous release stopped.

Keep the file unchanged; add a new versioned directory for later formats.
