# Language policy rule (always loaded)

The project targets a global audience and is **English-first**.

- **Code, comments, docstrings, docs, commit messages, identifiers**: English only.
- **CJK characters are allowed only in**:
  - test fixtures under `tests/` and `data/` (multilingual input is the point), and
  - locale-suffixed mirror files (e.g. `*_zh.json`),
  - locale directories such as `prompts/zh/`,
  - prompt assets under `src/everos/**/prompts/` (the extraction prompts quote
    multilingual examples because handling non-English input is the point,
    like test fixtures),
  - third-party integration samples under `use-cases/` (their own demo
    content), and
  - a single line marked `cjk-allow` in a comment (a language switcher such as
    README's `[中文]` link, a community badge). <!-- cjk-allow: examples -->
- Do **not** introduce CJK into `src/` (outside prompt assets), `docs/`, or config.

Enforcement: `make check-cjk` scans for stray CJK outside the allowlist (advisory).
Keep user-facing strings and error messages in English.
