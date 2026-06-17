# Acknowledgments

[Home](README.md) > [Docs](docs/index.md) > Acknowledgments

Thanks to the following projects, communities, and individuals for their
inspiration and support.

---

## Inspiration & references

### [memsearch](https://github.com/zilliztech/memsearch)

Inspired our markdown-as-source-of-truth design and the SHA-256 +
file-watcher incremental sync model. memsearch is the closest project in
spirit to EverOS.

### [mem0](https://github.com/mem0ai/mem0)

Inspired the "one provider per file" flat adapter layout that EverOS uses
for `component/llm/` and `component/embedding/`.

### [Letta (MemGPT)](https://github.com/letta-ai/letta)

Inspired the multi-tier memory mapping (Core / Recall / Archival) that maps
naturally onto our MemCell / Episode / Archival pipeline.

### [MemOS](https://github.com/MemTensor/MemOS)

Provided a reference for memory taxonomy decisions (textual / parametric /
activation) and helped sharpen our scope choice to focus on textual memory.

### [Memos](https://github.com/usememos/memos)

A comprehensive open-source note-taking service whose plain-text-first
design philosophy reinforced our decision to keep markdown files as the
single source of truth.

### [Nemori](https://github.com/nemori-ai/nemori)

A self-organising long-term memory substrate for agentic LLM workflows that
provided valuable inspiration for our extraction pipeline.

---

## Open-source libraries

EverOS is built on top of excellent open-source libraries and frameworks:

### Core

- **[Python](https://www.python.org/)** — Programming language (3.12+)
- **[uv](https://github.com/astral-sh/uv)** — Fast Python package manager
- **[FastAPI](https://fastapi.tiangolo.com/)** — Modern async web framework (HTTP API)
- **[Pydantic](https://docs.pydantic.dev/)** — Data validation and settings

### Storage

- **[LanceDB](https://lancedb.com/)** — Embedded vector + BM25 + scalar database
- **[SQLite](https://sqlite.org/)** — Embedded relational database (state + audit log)

### Tooling

- **[Ruff](https://docs.astral.sh/ruff/)** — Lint + format
- **[import-linter](https://import-linter.readthedocs.io/)** — Layered architecture enforcement
- **[Hatchling](https://hatch.pypa.io/)** — Wheel build backend
- **[pytest](https://pytest.org/)** — Testing framework
- **[pre-commit](https://pre-commit.com/)** — Git hooks framework

### LLM & embedding providers

EverOS is provider-agnostic by design. Tested provider integrations include
OpenAI, Anthropic, Ollama, and SBERT. See [`component/llm/`](src/everos/component/llm/)
and [`component/embedding/`](src/everos/component/embedding/) for the
adapter layouts.

---

## Contributors

Thanks to all the developers who have contributed to this project.

See the full list of contributors on
[GitHub](https://github.com/EverMind-AI/EverOS/graphs/contributors).

<!-- Future: contributor image grid
<a href="https://github.com/EverMind-AI/EverOS/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=EverMind-AI/EverOS" />
</a>
-->

---

## Community

Thanks to our community for valuable feedback, bug reports, and feature
suggestions:

- **GitHub Issues & Discussions** — bug reports and feature requests
- **Discord** — [Join our Discord server](https://discord.gg/gYep5nQRZJ)
- **X / Twitter** — [@EverMindAI](https://x.com/EverMindAI)

---

## Supporting organizations

- **Shanda Group** — for supporting the development of EverOS

---

## Special thanks

- To everyone who starred the repository
- To those who shared EverOS with others
- To researchers and developers using EverOS in their work

---

## Want to contribute?

Contributions are welcome! See the [Contributing Guide](CONTRIBUTING.md)
to get started.

---

## See also

- [Citation](CITATION.md)
- [Changelog](CHANGELOG.md)
- [Contributing Guide](CONTRIBUTING.md)
