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
    return createContextEngine(pluginMeta, pluginConfig, api.logger);
  });
}
