# Security Policy

## Supported Versions

Security fixes are applied to the latest release line only; older minor lines do
not receive backports.

| Version | Supported |
|---------|-----------|
| `1.2.x` (current) | ✅ |
| `1.1.x` and older | ❌ — upgrade to the current line |

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
Reporters are credited in the advisory and the release notes unless you prefer
to remain anonymous.

## Published Advisories

Confirmed issues are published as GitHub Security Advisories, each carrying the
affected version ranges and the release that fixes them:

<https://github.com/EverMind-AI/EverOS/security/advisories>

To check whether an installation is affected by a past issue, read the version
range on the advisory itself — the range is authoritative.

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
- **Documents you ingest are untrusted input.** Filenames, metadata, and
  content that originate outside your control are parsed, indexed, and written
  to disk. An instance that ingests third-party documents is processing
  untrusted data even when the API itself is reachable only from loopback.
- Secrets (LLM / embedding API keys) live in your local `.env`; protect that
  file as you would any credential. EverOS never transmits them anywhere except
  the providers you configure.
- Memory content is stored as plaintext `.md` files; apply OS-level file
  permissions or disk encryption if your data is sensitive.
