import assert from "node:assert/strict";
import test from "node:test";

import register from "../index.js";

/**
 * Regression test for #150: register() must forward plugin config to the
 * context engine, preferring `api.pluginConfig` (current OpenClaw host contract)
 * and falling back to the factory-callback argument (legacy hosts).
 */

function makeApi({ pluginConfig } = {}) {
  let factory = null;
  const api = {
    logger: { info: () => {}, warn: () => {} },
    pluginConfig,
    registerContextEngine(id, fn) {
      factory = fn;
    },
    // Test helper: invoke the registered factory the way a host would.
    _invokeFactory(callbackConfig) {
      assert.ok(factory, "registerContextEngine was not called");
      return factory(callbackConfig);
    },
  };
  return api;
}

/** Run bootstrap() with a stubbed fetch and return the URL it called.
 *  bootstrap() hits `${cfg.serverUrl}/health`, so the host the engine resolved
 *  is directly observable — this is the concrete proof of which config won. */
async function observeBootstrapUrl(engine) {
  const original = global.fetch;
  let calledUrl = null;
  global.fetch = async (url) => {
    calledUrl = String(url);
    return { ok: true, status: 200, async json() { return {}; }, async text() { return ""; } };
  };
  try {
    await engine.bootstrap({ sessionId: "s", sessionKey: "k" });
  } finally {
    global.fetch = original;
  }
  return calledUrl;
}

test("register() prefers api.pluginConfig over the callback arg", async () => {
  const api = makeApi({ pluginConfig: { userId: "from-host", baseUrl: "http://host:1995" } });
  register(api);
  const engine = api._invokeFactory({ userId: "from-callback", baseUrl: "http://callback:1995" });

  // The host-provided baseUrl must win over the callback arg's baseUrl.
  const url = await observeBootstrapUrl(engine);
  assert.match(url, /^http:\/\/host:1995\/health/, "engine must resolve config from api.pluginConfig");
});

test("register() falls back to the callback arg when api.pluginConfig is absent", async () => {
  const api = makeApi({ pluginConfig: undefined });
  register(api);
  const engine = api._invokeFactory({ userId: "from-callback", baseUrl: "http://callback:1995" });

  const url = await observeBootstrapUrl(engine);
  assert.match(url, /^http:\/\/callback:1995\/health/, "engine must fall back to the callback config");
});

test("register() tolerates both config sources being empty", () => {
  const api = makeApi({ pluginConfig: undefined });
  register(api);
  const engine = api._invokeFactory(undefined);

  // resolveConfig() must apply defaults rather than throw on undefined config.
  assert.equal(typeof engine.assemble, "function");
});
