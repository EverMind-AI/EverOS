# Vendored wheels (test phase only)

`everalgo_user_memory-0.9.0rc6.dev9+v031episode-py3-none-any.whl` is a local
build of everalgo-user-memory carrying the 2.0 profile (the two profile
prompts, the item lineage, the changes-only consolidation reply and the trait
fix), the update of several people's profiles in one call
(`ProfileExtractor.aupdate_many`, the batched member-in-group profiles), and
the original episode request as its only episode prompt:
everalgo-user-memory 0.3.1's whole-memcell episode prompt, its renderer and
its summary rule, on which the episode call builds every chat cell's request.
The rewritten episode prompt of 2026-10 and `EpisodeSummarizer` are not in it.
EverOS depends on it through `[tool.uv.sources]` in `pyproject.toml`, so
`uv sync` installs it from this file with no manual step.

- **Provenance.** Built with `uv build --wheel` in
  `packages/everalgo-user-memory` of the everalgo branch
  `feat/episode-prompt-original` at `396583b` (branched from
  `feat/profile-two-prompts` at `46b322c`); a rebuild there is
  byte-identical. sha256
  `2569073964e7304e6557f0807669a5253e65dffda412ce0d4ebfc8284f9815ef`,
  121,687 bytes. The branch is not tagged or published.
- **Version.** `0.9.0rc6.dev9+v031episode`: the local label (`+v031episode`)
  keeps it off every package index and apart from every published version,
  everalgo's own `0.9.0rc5`-`0.9.0rc8` included, which this build does not
  contain. It replaces `0.9.0rc6.dev8+v031episode` (everalgo `e038fd3`),
  which had no `aupdate_many`; everything else the two builds send is the
  same request (the one-person profile prompts, the episode request and the
  language rules have the same text).
- **Test phase only.** When everalgo publishes the release that carries this
  work, pin that version from PyPI in `pyproject.toml`, drop the
  `[tool.uv.sources]` entry and this directory, and re-lock.
- **Until then** `pip install` of this tree cannot resolve
  `everalgo-user-memory==0.9.0rc6.dev8+v031episode`: install with `uv sync`,
  or install this wheel first.
