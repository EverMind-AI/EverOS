import assert from "node:assert/strict";
import test from "node:test";

import { searchMemories } from "../src/api.js";
import { parseSearchResponse } from "../src/prompt.js";

// A canned v1 search 200 body: { data: { episodes, raw_messages, ... } }.
const V1_RESPONSE = {
  data: {
    episodes: [
      {
        id: "ep1",
        user_id: "alice",
        summary: "Alice likes dark roast coffee",
        subject: "coffee preference",
        episode: "Alice mentioned she likes dark roast coffee.",
        score: 1.7,
        timestamp: "2026-06-03T05:00:00Z",
      },
    ],
    raw_messages: [
      {
        sender_name: "alice",
        content_items: [{ type: "text", text: "and oat milk" }],
        created_at: "2026-06-03T05:01:00Z",
      },
    ],
    profiles: [],
    agent_memory: null,
    query: { text: "coffee", method: "keyword", filters_applied: { user_id: "alice" } },
  },
};

function stubFetch(captured) {
  const real = globalThis.fetch;
  globalThis.fetch = async (url, opts) => {
    captured.push({ url: String(url), opts });
    return {
      ok: true,
      status: 200,
      async json() {
        return V1_RESPONSE;
      },
      async text() {
        return "";
      },
    };
  };
  return () => {
    globalThis.fetch = real;
  };
}

// Regression for the openclaw plugin's memory recall: the v0 form
// (GET + flat top-level user_id + `retrieve_method`) 405s then 422s against the
// v1 search route. This asserts the request is the v1 envelope.
test("searchMemories POSTs a v1 SearchMemoriesRequest with a Filters DSL", async () => {
  const cfg = { serverUrl: "http://localhost:1995" };
  const captured = [];
  const restore = stubFetch(captured);
  try {
    await searchMemories(cfg, {
      query: "coffee",
      user_id: "alice",
      group_id: undefined,
      memory_types: ["episodic_memory", "profile"],
      retrieve_method: "hybrid",
      top_k: 10,
    });
  } finally {
    restore();
  }

  assert.equal(captured.length, 1);
  assert.equal(captured[0].opts.method, "POST", "search must POST (GET 405s on the v1 route)");
  assert.match(captured[0].url, /\/api\/v1\/memories\/search$/);

  const body = JSON.parse(captured[0].opts.body);
  // user_id must live inside the Filters DSL, not at the top level.
  assert.deepEqual(body.filters, { user_id: "alice" }, "user_id must live in the Filters DSL");
  assert.equal("user_id" in body, false, "no top-level user_id");
  // retrieve_method must be renamed to the DTO's `method` field.
  assert.equal(body.method, "hybrid");
  assert.equal("retrieve_method" in body, false);
  assert.equal(body.query, "coffee");
  assert.equal(body.top_k, 10);
  // Only backend-searchable types are sent.
  assert.deepEqual(body.memory_types, ["episodic_memory"]);
});

test("searchMemories maps the v1 { data: { episodes, raw_messages } } response into the caller contract", async () => {
  const cfg = { serverUrl: "http://localhost:1995" };
  const restore = stubFetch([]);
  let out;
  try {
    out = await searchMemories(cfg, {
      query: "coffee",
      user_id: "alice",
      memory_types: ["episodic_memory"],
      top_k: 5,
    });
  } finally {
    restore();
  }

  assert.equal(out.status, "ok");
  // episodes -> memories, tagged so the downstream episodic filter matches.
  assert.equal(out.result.memories.length, 1);
  const m = out.result.memories[0];
  assert.equal(m.memory_type, "episodic_memory");
  assert.equal(m.summary, "Alice likes dark roast coffee");
  assert.equal(m.score, 1.7);
  // raw_messages -> pending_messages, content_items flattened to text.
  assert.equal(out.result.pending_messages.length, 1);
  assert.equal(out.result.pending_messages[0].content, "and oat milk");

  // End-to-end: the existing parseSearchResponse must consume the mapped shape.
  const parsed = parseSearchResponse(out);
  assert.ok(parsed, "parseSearchResponse accepts the mapped result");
  assert.equal(parsed.episodic.length, 1);
  assert.match(parsed.episodic[0].text, /coffee preference: Alice likes dark roast coffee/);
  assert.equal(parsed.pending.length, 1);
  assert.match(parsed.pending[0].text, /alice: and oat milk/);
});
