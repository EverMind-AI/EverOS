// Node bridge for the OpenClaw benchmark adapter.
//
// Dispatches BridgeCommand JSON to either:
//   * the real OpenClaw CLI at $OPENCLAW_REPO_PATH/openclaw.mjs (when the
//     env var points at a valid repo), or
//   * built-in stub handlers that still honor the BridgeResponse shape
//     defined in openclaw_types.py.
//
// The stub path is always safe to use in CI - contract tests lock the
// response shape - and the native path keeps the wire protocol identical
// so swapping OPENCLAW_REPO_PATH on/off should be transparent to Python
// callers. Smoke validation against the native path is documented in
// docs/plans/2026-04-13-openclaw-benchmark-a.md Task 8 Step 4.

import { readFileSync, existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import path from "node:path";

function respond(obj) {
  process.stdout.write(JSON.stringify(obj));
}

function fail(message, command) {
  respond({ ok: false, command, error: message });
  process.exit(0);
}

function epochSeconds() {
  return Math.floor(Date.now() / 1000);
}

function readStdin() {
  return readFileSync(0, "utf8");
}

function resolveLauncher() {
  const repo = process.env.OPENCLAW_REPO_PATH;
  if (!repo) return null;
  const launcher = path.join(repo, "openclaw.mjs");
  return existsSync(launcher) ? launcher : null;
}

function envForSandbox(input) {
  // Minimal env - mirrors v0.1/v0.2 isolation. Inheriting the full parent
  // env would leak OPENAI_API_KEY etc into OpenClaw's auto-provider
  // selection, which we explicitly do NOT want: the resolved config file
  // already carries the sophnet credentials for embedding, and the bench
  // LLM key is for our own post-retrieval answer prompt, not for OpenClaw's
  // internal flush agent.
  const env = {
    PATH: process.env.PATH || "",
    HOME: input.home_dir || input.workspace_dir || "",
    NODE_OPTIONS: "",
    NPM_CONFIG_USERCONFIG: "/dev/null",
    NPM_CONFIG_GLOBALCONFIG: "/dev/null",
  };
  if (input.config_path) env.OPENCLAW_CONFIG_PATH = input.config_path;
  if (input.state_dir) env.OPENCLAW_STATE_DIR = input.state_dir;
  return env;
}

function cwdForSandbox(input) {
  return input.cwd_dir || input.workspace_dir || undefined;
}

function runLauncher(launcher, args, env, cwd) {
  return new Promise((resolve, reject) => {
    const proc = spawn("node", [launcher, ...args], {
      env,
      cwd,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    proc.stdout.on("data", (b) => (stdout += b.toString()));
    proc.stderr.on("data", (b) => (stderr += b.toString()));
    proc.on("close", (code) => resolve({ code, stdout, stderr }));
    proc.on("error", reject);
  });
}

function extractJsonTail(stdout) {
  const trimmed = stdout.trim();
  try {
    return JSON.parse(trimmed);
  } catch (_) {
    const lines = trimmed.split("\n").reverse();
    for (const line of lines) {
      if (line.startsWith("{")) {
        try {
          return JSON.parse(line);
        } catch (_) {
          // continue
        }
      }
    }
    return null;
  }
}

async function handleIndex(input, launcher) {
  if (!launcher) {
    return {
      ok: true,
      command: "index",
      flush_epoch: 0,
      index_epoch: 0,
      input_artifacts: [],
      output_artifacts: [],
    };
  }
  const env = envForSandbox(input);
  const cwd = cwdForSandbox(input);
  const { code, stdout, stderr } = await runLauncher(
    launcher,
    ["memory", "index", "--force"],
    env,
    cwd
  );
  if (code !== 0) {
    const tail = [stderr, stdout].filter((s) => s && s.trim()).join("\n---\n");
    return { ok: false, command: "index", error: tail || `exit ${code}` };
  }
  return {
    ok: true,
    command: "index",
    flush_epoch: epochSeconds(),
    index_epoch: epochSeconds(),
    input_artifacts: [],
    output_artifacts: [],
  };
}

async function handleFlush(input, launcher) {
  // OpenClaw has no standalone flush; re-run index and report the epochs.
  const result = await handleIndex(input, launcher);
  if (!result.ok) return { ...result, command: "flush" };
  return { ...result, command: "flush" };
}

async function handleStatus(input, launcher) {
  if (!launcher) {
    return {
      ok: true,
      command: "status",
      settled: true,
      flush_epoch: 0,
      index_epoch: 0,
      active_artifacts: [],
    };
  }
  const env = envForSandbox(input);
  const cwd = cwdForSandbox(input);
  const { code, stdout, stderr } = await runLauncher(
    launcher,
    ["memory", "status", "--json"],
    env,
    cwd
  );
  if (code !== 0) {
    const tail = [stderr, stdout].filter((s) => s && s.trim()).join("\n---\n");
    return { ok: false, command: "status", error: tail || `exit ${code}` };
  }
  const parsed = extractJsonTail(stdout);
  if (!parsed) {
    return { ok: false, command: "status", error: "stdout not JSON" };
  }
  // OpenClaw's `memory status --json` returns an array:
  //   [{ agentId: "main", status: { backend, files, chunks, dirty, dbPath, ... }}]
  // Map to our BridgeResponse shape with a best-effort `settled` flag.
  const agentStatus = Array.isArray(parsed) ? (parsed[0] || {}).status : parsed.status;
  const s = agentStatus || {};
  const settled = s.dirty === false;
  return {
    ok: true,
    command: "status",
    settled,
    files: Number(s.files || 0),
    chunks: Number(s.chunks || 0),
    backend: s.backend || null,
    provider: s.provider || null,
    flush_epoch: Number(s.lastFlushEpoch || 0),
    index_epoch: Number(s.lastIndexEpoch || 0),
    active_artifacts: [],
    native: true,
  };
}

async function handleSearch(input, launcher) {
  if (!launcher) {
    return { ok: true, command: "search", hits: [] };
  }
  const env = envForSandbox(input);
  const cwd = cwdForSandbox(input);
  const args = [
    "memory",
    "search",
    "--query",
    String(input.query ?? ""),
    "--max-results",
    String(input.top_k ?? 30),
    "--json",
  ];
  const { code, stdout, stderr } = await runLauncher(launcher, args, env, cwd);
  if (code !== 0) {
    const tail = [stderr, stdout].filter((s) => s && s.trim()).join("\n---\n");
    return { ok: false, command: "search", error: tail || `exit ${code}` };
  }
  const parsed = extractJsonTail(stdout);
  if (!parsed) {
    return { ok: false, command: "search", error: "stdout not JSON" };
  }
  const rawResults = parsed.results || [];
  const hits = rawResults.map((r) => ({
    score: Number(r.score ?? 0),
    snippet: r.snippet ?? "",
    artifact_locator: {
      kind: "memory_file_range",
      path_rel: r.path ?? "",
      line_start: Number(r.startLine ?? 0),
      line_end: Number(r.endLine ?? 0),
    },
    metadata: {
      source: r.source ?? "memory",
    },
  }));
  return { ok: true, command: "search", hits };
}

async function handleGet(input) {
  // OpenClaw has no get command; read the markdown file range directly.
  const locator = input.artifact_locator || {};
  if (!input.workspace_dir || !locator.path_rel) {
    return { ok: true, command: "get", artifact_locator: locator, snippet: "" };
  }
  try {
    const absPath = path.join(input.workspace_dir, locator.path_rel);
    const content = await readFile(absPath, "utf8");
    const lines = content.split("\n");
    const start = Math.max(0, (locator.line_start ?? 1) - 1);
    const end = Math.max(start, locator.line_end ?? lines.length);
    const snippet = lines.slice(start, end).join("\n");
    return { ok: true, command: "get", artifact_locator: locator, snippet };
  } catch (err) {
    return {
      ok: true,
      command: "get",
      artifact_locator: locator,
      snippet: "",
    };
  }
}

const raw = readStdin();
let input;
try {
  input = JSON.parse(raw);
} catch (err) {
  fail(`invalid input json: ${err.message}`, undefined);
}

const launcher = resolveLauncher();
const command = input.command;

(async () => {
  try {
    let resp;
    switch (command) {
      case "index":
        resp = await handleIndex(input, launcher);
        break;
      case "flush":
        resp = await handleFlush(input, launcher);
        break;
      case "status":
        resp = await handleStatus(input, launcher);
        break;
      case "search":
        resp = await handleSearch(input, launcher);
        break;
      case "get":
        resp = await handleGet(input);
        break;
      default:
        return fail(`unknown command: ${command}`, command);
    }
    respond(resp);
  } catch (err) {
    fail(err.stack || err.message || String(err), command);
  }
})();
