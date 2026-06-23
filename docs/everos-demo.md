# EverOS Demo

`everos demo` is a local educational TUI. It helps new users feel the memory
lifecycle before they configure API keys, start the server, or write real
memory through the API.

## Run It

```bash
everos demo
```

The command asks for one memory and one recall question, then opens a
full-screen terminal UI. The visual flow is deterministic and local to the CLI:
conversation -> memory sphere -> recall -> source proof -> confetti.

For non-interactive shells or a copyable preview, use:

```bash
everos demo --plain
```

For the looping showroom view used by README media, use:

```bash
everos demo --cinematic
```

## Run It Against A Server

After `everos init` and `everos server start`, run:

```bash
everos demo --live
```

Live mode keeps the same TUI, but the memory lifecycle is backed by real
server calls:

1. `GET /health`
2. `POST /api/v1/memory/add`
3. `POST /api/v1/memory/flush`
4. `POST /api/v1/memory/search`

If your server is not running on `http://127.0.0.1:8000`, pass
`--server-url <url>`.

## What It Does Not Do

By default, `everos demo` does not connect to the EverOS server, call LLM
providers, or write production memory files. It is intentionally hardcoded so
users can try the experience before configuring the full runtime. Use
`everos demo --live` when you want the same visual flow backed by a running
server.

## Source Layout

The CLI command adapter stays under `src/everos/entrypoints/cli/commands/demo.py`
because the public command is still `everos demo`.

The TUI implementation lives under `src/everos/entrypoints/tui/demo/`:

- `app.py` renders the Textual app.
- `data.py` builds the deterministic demo story.
- `widgets/sphere.py` builds the memory sphere frames.
- `readme_media.py` renders README media.

To regenerate README media locally:

```bash
uv run python -m everos.entrypoints.tui.demo.readme_media --out-dir /tmp/everos-demo-media
```
