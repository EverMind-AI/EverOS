// Node bridge stub for the OpenClaw benchmark adapter.
//
// Reads a BridgeCommand JSON object from stdin and writes a BridgeResponse
// object to stdout. This stub intentionally echoes the command back without
// touching the OpenClaw runtime - Task 8 replaces these branches with real
// index / flush / search / get / status calls.
//
// The response shape must already match openclaw_types.BridgeResponse so the
// Python side's contract tests keep working once native commands land.

import { readFileSync } from "node:fs";

function readStdin() {
  return readFileSync(0, "utf8");
}

function respond(obj) {
  process.stdout.write(JSON.stringify(obj));
}

function fail(message, command) {
  respond({ ok: false, command, error: message });
  process.exitCode = 0; // error is carried in JSON, not the exit code
}

const raw = readStdin();
let input;
try {
  input = JSON.parse(raw);
} catch (err) {
  fail(`invalid input json: ${err.message}`, undefined);
  process.exit(0);
}

const command = input.command;

switch (command) {
  case "index":
  case "flush":
    respond({
      ok: true,
      command,
      flush_epoch: 0,
      index_epoch: 0,
      input_artifacts: [],
      output_artifacts: [],
    });
    break;
  case "status":
    respond({
      ok: true,
      command,
      settled: true,
      flush_epoch: 0,
      index_epoch: 0,
      active_artifacts: [],
    });
    break;
  case "search":
    respond({ ok: true, command, hits: [] });
    break;
  case "get":
    respond({ ok: true, command, artifact_locator: input.artifact_locator ?? {} });
    break;
  default:
    fail(`unknown command: ${command}`, command);
}
