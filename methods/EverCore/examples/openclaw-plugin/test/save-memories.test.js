import assert from "node:assert/strict";
import test from "node:test";

import { saveMemories } from "../src/api.js";

// Regression for #237: POST /api/v1/memories was 422-ing with
// "Field required: user_id" because saveMemories sent flat per-message bodies.
// PersonalAddRequest is an envelope: top-level user_id + a messages[] array,
// each MessageItem carrying a required unix-milliseconds integer timestamp.
test("saveMemories posts a PersonalAddRequest envelope with top-level user_id (fix #237)", async () => {
  const cfg = { serverUrl: "http://localhost:1995" };
  const captured = [];

  const realFetch = globalThis.fetch;
  globalThis.fetch = async (url, opts) => {
    captured.push({ url: String(url), opts });
    return {
      ok: true,
      status: 200,
      async json() {
        return { status: "ok" };
      },
      async text() {
        return "";
      },
    };
  };

  try {
    await saveMemories(cfg, {
      userId: "alice",
      sessionId: "sess-1",
      messages: [
        { role: "user", content: "hi" },
        { role: "assistant", content: "hello" },
      ],
      idSeed: "k:1",
    });
  } finally {
    globalThis.fetch = realFetch;
  }

  // A single batched POST to /api/v1/memories (not one request per message).
  assert.equal(captured.length, 1, "expected exactly one batched POST");
  assert.equal(captured[0].opts.method, "POST");
  assert.match(captured[0].url, /\/api\/v1\/memories$/);

  const body = JSON.parse(captured[0].opts.body);

  // The 422 fix: top-level user_id must be present.
  assert.equal(body.user_id, "alice", "body must carry a top-level user_id");
  assert.equal(body.session_id, "sess-1");
  assert.ok(Array.isArray(body.messages), "body must carry a messages array");
  assert.equal(body.messages.length, 2);

  // MessageItem.timestamp is a REQUIRED unix-milliseconds integer.
  for (const m of body.messages) {
    assert.equal(typeof m.timestamp, "number", "each message needs a unix-ms timestamp");
    assert.equal(Number.isInteger(m.timestamp), true);
    assert.equal("create_time" in m, false, "legacy create_time field must be gone");
  }

  // Personal-scene sender_id rules: user turn carries sender_id=user_id,
  // assistant turn omits sender_id so the backend generates a distinct one.
  const userMsg = body.messages.find((m) => m.role === "user");
  const asstMsg = body.messages.find((m) => m.role === "assistant");
  assert.equal(userMsg.sender_id, "alice");
  assert.equal("sender_id" in asstMsg, false, "assistant turn must not pin sender_id to user_id");
});

test("saveMemories sends nothing when there are no messages", async () => {
  const cfg = { serverUrl: "http://localhost:1995" };
  let called = false;
  const realFetch = globalThis.fetch;
  globalThis.fetch = async () => {
    called = true;
    return { ok: true, status: 200, async json() { return {}; }, async text() { return ""; } };
  };
  try {
    await saveMemories(cfg, { userId: "alice", messages: [] });
  } finally {
    globalThis.fetch = realFetch;
  }
  assert.equal(called, false, "no POST should be sent for an empty message batch");
});
