# Vendored wheels (test phase only)

`everalgo_user_memory-0.9.0rc6.dev8+v031episode-py3-none-any.whl` is a local
build of everalgo-user-memory carrying the 2.0 profile (the two profile
prompts, the item lineage, the changes-only consolidation reply and the trait
fix) and the original episode request as its only episode prompt:
everalgo-user-memory 0.3.1's whole-memcell episode prompt, its renderer and
its summary rule, on which the episode call builds every chat cell's request.
The rewritten episode prompt of 2026-10 and `EpisodeSummarizer` are not in it.
EverOS depends on it through `[tool.uv.sources]` in `pyproject.toml`, so
`uv sync` installs it from this file with no manual step.

- **Provenance.** Built with `uv build --wheel` in
  `packages/everalgo-user-memory` of the everalgo branch
  `feat/episode-prompt-original` at `e038fd3` (branched from
  `feat/profile-two-prompts` at `46b322c`); a rebuild there is
  byte-identical. sha256
  `c17f0074cf4db1c2c5104d3cf530d7a97bc7268bb3b59723f2cec6327d23def4`,
  111,047 bytes. The branch is not tagged or published.
- **Version.** `0.9.0rc6.dev8+v031episode`: the local label (`+v031episode`)
  keeps it off every package index and apart from every published version,
  everalgo's own `0.9.0rc5`-`0.9.0rc8` included, which this build does not
  contain. It replaces `0.9.0rc6.dev7+v031episode` (everalgo `2ce2cdd`),
  which still carried the rewritten prompt beside the original; the two
  builds render the same requests for everything EverOS sends.
- **Test phase only.** When everalgo publishes the release that carries this
  work, pin that version from PyPI in `pyproject.toml`, drop the
  `[tool.uv.sources]` entry and this directory, and re-lock.
- **Until then** `pip install` of this tree cannot resolve
  `everalgo-user-memory==0.9.0rc6.dev8+v031episode`: install with `uv sync`,
  or install this wheel first.
