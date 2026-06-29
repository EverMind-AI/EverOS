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
pipeline against [EverOS Cloud](https://everos.evermind.ai) (`https://api.evermind.ai`):
`POST /api/v1/memories` -> `POST /api/v1/memories/flush` ->
`POST /api/v1/memories/search`. The panels follow your own input: conversation
-> memory sphere -> recall -> source proof -> confetti.

EverOS Cloud holds all model keys and manages storage, so the demo needs only a
single platform API key — no server to deploy, no model keys locally. Each run
uses a fresh, isolated `(session_id, user_id)` pair, so demo visitors never see
each other's memories.

The demo key is read from the `EVEROS_CLOUD_DEMO_KEY` environment variable (a
restricted, shippable key can be baked in later). If no key is set, the key is
rejected, or the quota is exhausted, the UI says so and points you at
configuring your own key — it never fakes a result.

The endpoint can be overridden with `EVEROS_CLOUD_DEMO_URL` (or `--server-url`).

## Run It With Your Own Cloud Key

Get a key from <https://everos.evermind.ai/api-keys>, then:

```bash
export EVEROS_CLOUD_API_KEY=<your-key>
everos demo --live
```

`--live` runs the same flow against the same platform, just authenticated with
your own key instead of the restricted demo key.

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
- `cloud.py` is the EverOS Cloud HTTP client (`add -> flush -> search`).
- `data.py` holds the static showcase story for `--plain` / `--cinematic`.
- `widgets/sphere.py` builds the memory sphere frames.
- `readme_media.py` renders README media.

To regenerate README media locally:

```bash
uv run python -m everos.entrypoints.tui.demo.readme_media --out-dir /tmp/everos-demo-media
```
