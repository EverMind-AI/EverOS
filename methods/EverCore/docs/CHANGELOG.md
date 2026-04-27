# Changelog

[Home](../README.md) > [Docs](README.md) > Changelog

All notable changes to EverCore will be documented in this file.

---

## [1.2.0] - 2025-01-20

### Changed
- 🔌 **API Enhancement**: Added `role` field to `POST /memories` endpoint to identify message source (`user` or `assistant`)
- 🔧 **Settings API**: Global settings are now managed via `PUT /api/v1/settings` (singleton, no group_id)

### Improved
- 🚀 **Database Efficiency**: Major performance improvements to database operations

### Breaking Changes
- ⚠️ **Data Migration Required**: Database schema changes may cause incompatibility with data created in previous versions. Please backup your data before upgrading.

---

## [1.1.0] - 2025-11-27

**🎉 🎉 🎉 EverCore v1.1.0 Released!**

### Added
- 🔧 **vLLM Support**: Support vLLM deployment for Embedding and Reranker models (currently tailored for Qwen3 series)
- 📊 **Evaluation Resources**: Full results & code for LoCoMo, LongMemEval, PersonaMem released

### Links
- [Release Notes](https://github.com/EverMind-AI/EverOS/releases/tag/v1.1.0)
- [Evaluation Guide](../evaluation/README.md)

---

## [1.0.0] - 2025-11-02

**🎉 🎉 🎉 EverCore v1.0.0 Released!**

### Added
- ✨ **Stable Version**: AI Memory System officially open sourced
- 📚 **Complete Documentation**: Quick start guide and comprehensive API documentation
- 📈 **Benchmark Testing**: LoCoMo dataset benchmark evaluation pipeline
- 🖥️ **Demo Tools**: Get started quickly with easy-to-use demos

### Links
- [Release Notes](https://github.com/EverMind-AI/EverOS/releases/tag/v1.0.0)
- [Getting Started Guide](dev_docs/getting_started.md)
- [Demo Guide](../demo/README.md)

---

## Future Plans

Stay tuned for upcoming releases! Follow our progress:
- [GitHub Releases](https://github.com/EverMind-AI/EverOS/releases)
- [GitHub Discussions](https://github.com/EverMind-AI/EverOS/discussions)
- [Reddit](https://www.reddit.com/r/EverMindAI/)

---

## See Also

- [Overview](OVERVIEW.md)
- [Contributing Guide](../CONTRIBUTING.md)
- [GitHub Issues](https://github.com/EverMind-AI/EverOS/issues)
