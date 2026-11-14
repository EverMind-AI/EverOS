# Vendored wheels (test phase only)

`everalgo_user_memory-0.9.0rc6.dev7+v031episode-py3-none-any.whl` is a local
build of everalgo-user-memory carrying the 2.0 profile (the two profile
prompts, the item lineage, the changes-only consolidation reply and the trait
fix) and the original episode request: everalgo-user-memory 0.3.1's
whole-memcell episode prompt, its renderer and its summary rule, which the
episode call sends one-to-one text cells. EverOS depends on it through
`[tool.uv.sources]` in `pyproject.toml`, so `uv sync` installs it from this
file with no manual step.

- **Provenance.** Built with `uv build --wheel` in
  `packages/everalgo-user-memory` of the everalgo branch
  `feat/episode-prompt-original` at `2ce2cdd` (branched from
  `feat/profile-two-prompts` at `46b322c`); a rebuild there is
  byte-identical. sha256
  `6acffed3568053cf99b292680f3f723547db48f365c00a927ff139c88f6130f2`,
  123,486 bytes. The branch is not tagged or published.
- **Version.** `0.9.0rc6.dev7+v031episode`: the local label (`+v031episode`)
  keeps it off every package index and apart from every published version,
  everalgo's own `0.9.0rc5`-`0.9.0rc8` included, which this build does not
  contain.
- **Test phase only.** When everalgo publishes the release that carries this
  work, pin that version from PyPI in `pyproject.toml`, drop the
  `[tool.uv.sources]` entry and this directory, and re-lock.
- **Until then** `pip install` of this tree cannot resolve
  `everalgo-user-memory==0.9.0rc6.dev7+v031episode`: install with `uv sync`,
  or install this wheel first.
