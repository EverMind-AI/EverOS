# EverOS Demo

`everos demo` is an interactive TUI that lets new users feel the memory
lifecycle — type a memory, ask for it back, watch EverOS recall it — before they
configure their own API keys.

## Run It

```bash
everos demo
```

This opens a full-screen terminal UI with an input box. Type something EverOS
should remember, then ask a question that recalls it. No API key or server setup
is needed for the default demo.

Each round runs the memory lifecycle and visualizes its four stages:

1. **Ingest** receives what you want EverOS to remember.
2. **Extract** identifies the useful memory inside the conversation.
3. **Index** prepares that memory for retrieval.
4. **Recall** finds it again when you ask a related question.

The same particle sphere flows continuously across all four stages. At the end,
the particles burst across the memory field and fade away. A small core of
yellow and white particles keeps moving at the center, then expands smoothly as
the next ingest cycle begins.

If the demo service is temporarily unavailable or the trial limit is reached,
the UI explains what happened and points you toward running with your own key.
It never fabricates a memory result.

## Run It With Your Own Cloud Key

Get a key from <https://everos.evermind.ai/api-keys>, then:

```bash
export EVEROS_CLOUD_API_KEY=<your-key>
everos demo --live
```

`--live` bypasses the relay and runs the same flow directly against the platform
with your own key.

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
- `cloud.py` runs the demo memory requests (`add -> flush -> search`).
- `data.py` holds the static showcase story for `--plain` / `--cinematic`.
- `widgets/sphere.py` builds the memory sphere frames.
- `readme_media.py` renders README media.

To regenerate README media locally:

```bash
uv run python -m everos.entrypoints.tui.demo.readme_media --out-dir /tmp/everos-demo-media
```
