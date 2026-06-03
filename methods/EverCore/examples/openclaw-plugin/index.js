/**
 * EverOS OpenClaw Plugin — entry point.
 * Registers the EverOS backend as a ContextEngine for memory management.
 */

import { createRequire } from "node:module";
import { createContextEngine } from "./src/engine.js";

const require = createRequire(import.meta.url);
const pluginMeta = require("./openclaw.plugin.json");

export default function register(api) {
  const log = api.logger || { info: (...a) => console.log(...a), warn: (...a) => console.warn(...a) };
  log.info(`[${pluginMeta.id}] Registering EverOS OpenClaw Plugin`);

  api.registerContextEngine(pluginMeta.id, (pluginConfig) => {
    // The OpenClaw host may deliver plugin config in two ways depending on
    // host version: as `api.pluginConfig` (current contract) or via the
    // factory-callback argument (legacy). Prefer the host-provided config and
    // fall back to the callback arg so config forwarding works on both. #150
    const resolvedConfig = api.pluginConfig ?? pluginConfig ?? {};
    return createContextEngine(pluginMeta, resolvedConfig, api.logger);
  });
}
