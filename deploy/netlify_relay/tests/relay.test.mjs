import assert from "node:assert/strict";
import test from "node:test";

import relay, { isAllowed } from "../netlify/functions/relay.mjs";

const ORIGINAL_FETCH = globalThis.fetch;
const ENV_NAMES = [
  "EVEROS_CLOUD_API_KEY",
  "EVEROS_CLOUD_UPSTREAM",
  "UPSTASH_REDIS_REST_URL",
  "UPSTASH_REDIS_REST_TOKEN",
  "RELAY_RATE_PER_MIN",
  "RELAY_DAILY_ROUNDS",
  "RELAY_UPSTREAM_TIMEOUT_MS",
  "RELAY_MAX_BODY_BYTES",
];
const ORIGINAL_ENV = Object.fromEntries(
  ENV_NAMES.map((name) => [name, process.env[name]]),
);

function configureEnvironment() {
  process.env.EVEROS_CLOUD_API_KEY = "server-secret";
  process.env.EVEROS_CLOUD_UPSTREAM = "https://api.test";
  process.env.UPSTASH_REDIS_REST_URL = "https://redis.test";
  process.env.UPSTASH_REDIS_REST_TOKEN = "redis-secret";
  process.env.RELAY_RATE_PER_MIN = "30";
  process.env.RELAY_DAILY_ROUNDS = "3";
}

function restoreEnvironment() {
  globalThis.fetch = ORIGINAL_FETCH;
  for (const name of ENV_NAMES) {
    if (ORIGINAL_ENV[name] === undefined) {
      delete process.env[name];
    } else {
      process.env[name] = ORIGINAL_ENV[name];
    }
  }
}

function quotaResponse(minuteCount = 1, dailyCount = 1) {
  return new Response(
    JSON.stringify([
      { result: minuteCount },
      { result: 1 },
      { result: dailyCount },
      { result: 1 },
    ]),
    { status: 200, headers: { "content-type": "application/json" } },
  );
}

test.afterEach(restoreEnvironment);

test("allows only the demo API surface", () => {
  assert.equal(isAllowed("POST", "/api/v2/memory/add"), true);
  assert.equal(isAllowed("POST", "/api/v2/memory/flush"), true);
  assert.equal(isAllowed("POST", "/api/v2/memory/search"), true);
  assert.equal(isAllowed("GET", "/api/v1/tasks/task-123"), false);
  assert.equal(isAllowed("DELETE", "/api/v2/memory/add"), false);
  assert.equal(isAllowed("GET", "/api/v2/memory/get"), false);
});

test("reports deployment readiness without exposing secrets", async () => {
  configureEnvironment();
  const response = await relay(
    new Request("https://demo.test/healthz"),
    { ip: "192.0.2.1" },
  );
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), {
    ok: true,
    upstream: "https://api.test",
    key_configured: true,
    quota_configured: true,
  });
});

test("rejects non-demo endpoints", async () => {
  const response = await relay(
    new Request("https://demo.test/api/v2/memory/get"),
    { ip: "192.0.2.1" },
  );
  assert.equal(response.status, 403);
});

test("injects the server key and never forwards client authorization", async () => {
  configureEnvironment();
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url: String(url), options });
    if (String(url).startsWith("https://redis.test")) {
      return quotaResponse();
    }
    return new Response(JSON.stringify({ data: { status: "queued" } }), {
      status: 202,
      headers: { "content-type": "application/json" },
    });
  };

  const response = await relay(
    new Request("https://demo.test/api/v2/memory/add", {
      method: "POST",
      headers: {
        authorization: "Bearer client-secret",
        "content-type": "application/json",
      },
      body: JSON.stringify({ messages: [] }),
    }),
    { ip: "192.0.2.1" },
  );

  assert.equal(response.status, 202);
  assert.equal(calls.length, 2);
  assert.equal(calls[1].url, "https://api.test/api/v2/memory/add");
  assert.equal(calls[1].options.headers.authorization, "Bearer server-secret");
  assert.notEqual(
    calls[1].options.headers.authorization,
    "Bearer client-secret",
  );
});

test("enforces the distributed per-minute quota", async () => {
  configureEnvironment();
  globalThis.fetch = async () => quotaResponse(31, 31);

  const response = await relay(
    new Request("https://demo.test/api/v2/memory/add", { method: "POST" }),
    { ip: "192.0.2.1" },
  );
  assert.equal(response.status, 429);
  assert.match((await response.json()).error, /rate limit/);
});

test("enforces three demo rounds per day", async () => {
  configureEnvironment();
  globalThis.fetch = async () => quotaResponse(1, 4);

  const response = await relay(
    new Request("https://demo.test/api/v2/memory/add", { method: "POST" }),
    { ip: "192.0.2.1" },
  );
  assert.equal(response.status, 429);
  assert.match((await response.json()).error, /daily demo round limit/);
});

test("flush does not spend another demo round", async () => {
  configureEnvironment();
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url: String(url), options });
    if (String(url).startsWith("https://redis.test")) {
      const commands = JSON.parse(options.body);
      assert.equal(commands.length, 2);
      return new Response(JSON.stringify([{ result: 1 }, { result: 1 }]), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }
    return new Response(JSON.stringify({ data: { status: "success" } }));
  };

  const response = await relay(
    new Request("https://demo.test/api/v2/memory/flush", { method: "POST" }),
    { ip: "192.0.2.1" },
  );
  assert.equal(response.status, 200);
  assert.equal(calls.length, 2);
});

test("fails closed when the server API key is missing", async () => {
  configureEnvironment();
  delete process.env.EVEROS_CLOUD_API_KEY;
  globalThis.fetch = async () => {
    throw new Error("fetch must not be called without a server key");
  };

  const response = await relay(
    new Request("https://demo.test/api/v2/memory/flush", { method: "POST" }),
    { ip: "192.0.2.1" },
  );
  assert.equal(response.status, 500);
  assert.match((await response.json()).error, /API key/);
});

test("fails closed when the quota store is unavailable", async () => {
  configureEnvironment();
  globalThis.fetch = async () => {
    throw new Error("offline");
  };

  const response = await relay(
    new Request("https://demo.test/api/v2/memory/flush", { method: "POST" }),
    { ip: "192.0.2.1" },
  );
  assert.equal(response.status, 503);
});
