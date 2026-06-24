# EverOS Demo

`everos demo` is an interactive TUI that lets new users feel the memory
lifecycle — type a memory, ask for it back, watch EverOS recall it — before they
configure their own API keys.

## Run It

```bash
everos demo
```

This opens a full-screen terminal UI with an input box. You type a memory and a
recall question directly in the UI, and each round runs the **real** memory
pipeline (`add -> flush -> search`) against EverMind's hosted demo server. The
panels follow your own input: conversation -> memory sphere -> recall -> source
proof -> confetti.

The hosted server holds the LLM and embedding keys server-side, so you need no
local keys. Each run uses a fresh, isolated `(session_id, user_id)` pair, so
concurrent visitors never see each other's memories.

After a few rounds the demo points you at configuring your own keys (`everos
init`, then `everos demo --live`).

The hosted endpoint can be overridden with the `EVEROS_CLOUD_DEMO_URL`
environment variable (or `--server-url <url>`). If the server is unreachable or
the free quota is exhausted, the UI says so rather than faking a result.

## Run It Against Your Own Server

After `everos init` and `everos server start`, run the same interactive TUI
against your own server (your own keys):

```bash
everos demo --live
```

Each round performs `GET /health` -> `POST /api/v1/memory/add` ->
`POST /api/v1/memory/flush` -> `POST /api/v1/memory/search`. If your server is
not on `http://127.0.0.1:8000`, pass `--server-url <url>`.

> Before the hosted demo server (and its DNS) is deployed, you can point the
> default demo at a local server with
> `EVEROS_CLOUD_DEMO_URL=http://127.0.0.1:8000 everos demo`.

## Static Previews

For non-interactive shells or a copyable preview (no input box, no network):

```bash
everos demo --plain
```

For the looping showroom view used by README media:

```bash
everos demo --cinematic
```

## Source Layout

The CLI command adapter stays under `src/everos/entrypoints/cli/commands/demo.py`
because the public command is still `everos demo`.

The TUI implementation lives under `src/everos/entrypoints/tui/demo/`:

- `app.py` renders the Textual app and drives the interactive rounds.
- `cloud.py` is the hosted-demo HTTP client (`add -> flush -> search`).
- `data.py` holds the static showcase story for `--plain` / `--cinematic`.
- `widgets/sphere.py` builds the memory sphere frames.
- `readme_media.py` renders README media.

To regenerate README media locally:

```bash
uv run python -m everos.entrypoints.tui.demo.readme_media --out-dir /tmp/everos-demo-media
```
