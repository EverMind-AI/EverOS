# Test Fixtures

The fixtures in this directory are checked-in artifacts. The largest of
them was originally produced by a converter script that has been removed
from the repository — the fixtures themselves are the source of truth and
should rarely need to be regenerated.

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

## `search_seed/`

LanceDB / SQLite seed bundle for `/search` e2e. Produced by
`_dump_search_seed.py` (kept in-tree); regenerate by running that
script after a successful `add → flush` pipeline against the LoCoMo
fixture above.

## `member_profiles_per_member_ea7d2570.json`

Every model request and every written member document of one scripted group
(three members and an assistant, five segments, a stored profile with a
frozen language, `[groups] member_profile_every = 2`, then a flush), recorded
through `update_member_profiles` at ea7d2570, the commit before
`[groups] member_profile_mode` existed. `test_update_member_profiles_parity.py`
replays the same scenario in `per_member` mode with `profile_on_flush = true`
and must reproduce it exactly. Recorded with everalgo's `FakeLLMClient`; no
model was called. Regenerate only from ea7d2570 (or a commit whose
member-profile requests are meant to match it).
