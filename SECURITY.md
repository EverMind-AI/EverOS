# Security Policy

## Supported Versions

EverOS is released and in the v1.x stable line. Security fixes are applied to the
latest release line only.

| Version | Supported |
|---------|-----------|
| 1.x     | ✅        |
| < 1.0   | ❌        |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues,
discussions, or pull requests.**

Instead, email **evermind@shanda.com** with:

- A description of the vulnerability and its potential impact
- Steps to reproduce, or a proof-of-concept
- The affected version / commit
- Any suggested mitigation, if you have one

We will acknowledge your report within **5 business days**, keep you informed of
progress, and aim to ship a fix or mitigation before any public disclosure.
Reporters are credited in the release notes unless you prefer to remain
anonymous.

## Scope & Threat Model

EverOS runs as a **local-first service** for single users or small teams
(Markdown + SQLite + LanceDB on the local filesystem). Please keep the
following in mind:

- Exposing the HTTP API (`everos server`) to an untrusted network is **outside
  the supported threat model** — it assumes a trusted local caller. The server
  binds to `127.0.0.1` by default (env `EVEROS_API__HOST`) so a fresh install
  is loopback-only. Only set the bind to `0.0.0.0` (or any routable interface)
  after you have placed your own gateway / auth layer in front;
  `everos server start` will log a warning when you bind to `0.0.0.0`.
- Secrets (LLM / embedding / rerank API keys) normally live in your local
  `<root>/everos.toml`, or in `EVEROS_*` environment variables for container
  deployments. Protect those values as you would any credential. EverOS never
  transmits them anywhere except the providers you configure.
- Memory content is stored as plaintext `.md` files; apply OS-level file
  permissions or disk encryption if your data is sensitive.
