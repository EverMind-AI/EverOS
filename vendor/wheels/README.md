# Vendored wheels (test phase only)

`everalgo_user_memory-0.9.0rc6.dev6-py3-none-any.whl` is a local build of
everalgo-user-memory carrying the 2.0 profile: the two profile prompts, the
item lineage, the changes-only consolidation reply and the trait fix. EverOS
depends on it through `[tool.uv.sources]` in `pyproject.toml`, so `uv sync`
installs it from this file with no manual step.

- **Provenance.** Built with `uv build --wheel` in
  `packages/everalgo-user-memory` of the everalgo branch
  `feat/profile-two-prompts` at `46b322c`; a rebuild there is byte-identical.
  sha256 `9e3150194751e3ecc8f695dd3e2de66bb21074d85ffa8bb336c6c0f751518d34`,
  118,399 bytes. The branch is local: not pushed, tagged or published.
- **Test phase only.** When everalgo publishes the release that carries this
  work, pin that version from PyPI in `pyproject.toml`, drop the
  `[tool.uv.sources]` entry and this directory, and re-lock.
- **Until then** `pip install` of this tree cannot resolve
  `everalgo-user-memory==0.9.0rc6.dev6`: install with `uv sync`, or install
  this wheel first.
