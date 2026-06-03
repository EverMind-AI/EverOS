# Test Fixtures

The fixtures in this directory are checked-in artifacts. The two largest
of them were originally produced by converter scripts that have been
removed from the repository — the fixtures themselves are the source of
truth and should rarely need to be regenerated.

## `long_conversation_locomo_caroline_melanie.json`

Derived from the LoCoMo dataset (`snap-research/locomo`, file
`data/locomo10.json`).

**License: CC BY-NC 4.0** — non-commercial use only. This file is
carved out of the project-wide Apache-2.0 license; see `NOTICE` at the
repository root for the explicit declaration.

If you need to regenerate it: fetch `locomo10.json` from the upstream
LoCoMo project, then adapt it to the fields the tests consume
(`everos_session_id`, `everos_user_id_for_test`, `speakers`,
`dialog`, ...). The original conversion script is not part of the
public repository.

## `agent_trajectories/*.json`

Hand-curated subset of agent tool-call traces. The selection that the
e2e tests load is enumerated in
`tests/e2e/test_add_flush_agent_pipeline_e2e.py` — that file is the
source of truth.

If you need to add or replace a trajectory, author the fixture by hand
(or with your own one-off script). The previous converter pulled from
internal paths and is not maintained as part of this repository.

## `search_seed/`

LanceDB / SQLite seed bundle for `/search` e2e. Produced by
`_dump_search_seed.py` (kept in-tree); regenerate by running that
script after a successful `add → flush` pipeline against the LoCoMo
fixture above.
