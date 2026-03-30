#!/usr/bin/env node

/**
 * EverOS OpenClaw Plugin Installer
 * Configures OpenClaw to use the EverOS backend through a ContextEngine.
 */

import fs from "node:fs";
import path from "node:path";
import { exec } from "node:child_process";
import { fileURLToPath } from "node:url";
import readline from "node:readline";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const HOME_DIR = process.env.HOME || process.env.USERPROFILE;
const CONFIG_PATH = path.join(HOME_DIR, ".openclaw", "openclaw.json");
const PLUGIN_ID = "evermind-ai-everos";
const PLUGIN_DIR = path.join(__dirname, "..");
const STABLE_PLUGIN_DIR = path.join(HOME_DIR, ".openclaw", "plugins", "evermind-ai-everos");
const DEFAULT_CONFIG = {
  baseUrl: "http://localhost:1995",
  userId: "everos-user",
  groupId: "everos-group",
  topK: 5,
  memoryTypes: ["episodic_memory", "profile", "agent_skill", "agent_case"],
  retrieveMethod: "hybrid",
};

const colors = {
  reset: "\x1b[0m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  blue: "\x1b[34m",
  red: "\x1b[31m",
  cyan: "\x1b[36m",
};

function log(message, color = "reset") {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

function success(message) {
  log(`✓ ${message}`, "green");
}

function error(message) {
  log(`✗ ${message}`, "red");
}

function info(message) {
  log(`ℹ ${message}`, "cyan");
}

function warn(message) {
  log(`⚠ ${message}`, "yellow");
}

function hr() {
  log("-".repeat(60), "blue");
}

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

async function prompt(question) {
  return new Promise((resolve) => {
    rl.question(question, (answer) => {
      resolve(answer.trim());
    });
  });
}

async function promptWithDefault(label, defaultValue) {
  const answer = await prompt(`${label} (default: ${defaultValue}): `);
  return answer || defaultValue;
}

function closeAndExit(code) {
  rl.close();
  process.exit(code);
}

async function checkBackendHealth(baseUrl) {
  try {
    const response = await fetch(`${baseUrl.replace(/\/*$/, "")}/health`, {
      signal: AbortSignal.timeout(5000),
    });
    if (!response.ok) {
      return { ok: false, reason: `HTTP ${response.status}` };
    }
    const data = await response.json().catch(() => null);
    return {
      ok: data?.status === "healthy" || data?.status === "ok" || response.ok,
      status: data?.status,
    };
  } catch (err) {
    return { ok: false, reason: err.message };
  }
}

function isDevelopmentCheckout(dir) {
  return fs.existsSync(path.join(dir, ".git"));
}

function ensureStablePluginPath() {
  if (isDevelopmentCheckout(PLUGIN_DIR)) {
    info(`Using local development checkout: ${PLUGIN_DIR}`);
    return { pluginDir: PLUGIN_DIR, loadPath: PLUGIN_DIR };
  }

  if (path.resolve(PLUGIN_DIR) !== path.resolve(STABLE_PLUGIN_DIR)) {
    fs.mkdirSync(path.dirname(STABLE_PLUGIN_DIR), { recursive: true });
    fs.cpSync(PLUGIN_DIR, STABLE_PLUGIN_DIR, {
      recursive: true,
      force: true,
    });

    const installerPath = path.join(STABLE_PLUGIN_DIR, "bin", "install.js");
    if (fs.existsSync(installerPath)) {
      fs.chmodSync(installerPath, 0o755);
    }

    info(`Installed plugin files to ${STABLE_PLUGIN_DIR}`);
  }

  return { pluginDir: STABLE_PLUGIN_DIR, loadPath: STABLE_PLUGIN_DIR };
}

function isLegacyPluginPath(entry, currentLoadPath) {
  if (typeof entry !== "string") return false;
  if (path.resolve(entry) === path.resolve(currentLoadPath)) return false;

  const normalized = entry.replace(/\\/g, "/");
  return normalized.includes("evermemos-openclaw-plugin") ||
    normalized.includes("evermind-ai-openclaw-plugin") ||
    normalized.includes("@evermind-ai/openclaw-plugin");
}

function loadConfig() {
  if (!fs.existsSync(CONFIG_PATH)) {
    return { exists: false, data: null };
  }

  try {
    return { exists: true, data: JSON.parse(fs.readFileSync(CONFIG_PATH, "utf-8")) };
  } catch (err) {
    error(`Failed to parse ${CONFIG_PATH}: ${err.message}`);
    error("Please fix the JSON syntax before continuing.");
    error("Installation aborted to avoid corrupting your configuration.");
    return { exists: true, error: err.message };
  }
}

function saveConfig(config) {
  try {
    if (fs.existsSync(CONFIG_PATH)) {
      fs.copyFileSync(CONFIG_PATH, `${CONFIG_PATH}.bak`);
    }
    fs.writeFileSync(CONFIG_PATH, `${JSON.stringify(config, null, 2)}\n`, "utf-8");
    return true;
  } catch (err) {
    error(`Failed to save config: ${err.message}`);
    return false;
  }
}

function ensureConfigShape(config) {
  config.plugins = config.plugins || {};
  config.plugins.load = config.plugins.load || {};
  config.plugins.load.paths = Array.isArray(config.plugins.load.paths) ? config.plugins.load.paths : [];
  config.plugins.allow = Array.isArray(config.plugins.allow) ? config.plugins.allow : [];
  config.plugins.slots = config.plugins.slots || {};
  config.plugins.entries = config.plugins.entries || {};
}

function mergePluginConfig(existingConfig, overrides = {}) {
  return {
    ...DEFAULT_CONFIG,
    ...(existingConfig || {}),
    ...overrides,
  };
}

function printSummary(pluginPath, entry) {
  hr();
  log("Installation Summary", "blue");
  log(`  Plugin ID:   ${PLUGIN_ID}`);
  log(`  Plugin path: ${pluginPath}`);
  log(`  Config file: ${CONFIG_PATH}`);
  log(`  Backend URL: ${entry.config.baseUrl}`);
  log(`  User ID:     ${entry.config.userId}`);
  log(`  Group ID:    ${entry.config.groupId}`);
  log("  Mode:        context-engine (natural language auto memory)");
  hr();
}

function printNextSteps(entry) {
  log("Next Steps", "green");
  log("  1. Make sure your EverOS backend is running.");
  log(`     curl ${entry.config.baseUrl}/health`, "cyan");
  log("");
  log("  2. Restart OpenClaw so the context engine can load.");
  log("     openclaw gateway restart", "cyan");
  log("");
  log("  3. Verify with natural language.");
  log('     Say: "Remember: I like espresso."', "cyan");
  log('     Then ask: "What coffee do I like?"', "cyan");
  hr();
}

function restartGateway() {
  return new Promise((resolve) => {
    exec("openclaw gateway restart", (err) => {
      if (err) {
        warn(`Could not restart OpenClaw automatically: ${err.message}`);
        info("Please restart manually: openclaw gateway restart");
      } else {
        success("OpenClaw gateway restarted.");
        info('After about 1 minute, send a natural language test such as "Remember: I like espresso." to verify recall.');
      }
      resolve();
    });
  });
}

async function install() {
  hr();
  log("EverOS OpenClaw Plugin Installer", "blue");
  log("This keeps your current ContextEngine flow and enables automatic memory recall/save.");
  hr();

  const configResult = loadConfig();
  if (configResult.error) {
    closeAndExit(1);
  }

  let config;
  if (!configResult.exists) {
    warn(`OpenClaw config not found at ${CONFIG_PATH}`);
    const answer = await prompt("Create a new OpenClaw config now? (Y/n): ");
    if (answer.toLowerCase() === "n") {
      info("Installation cancelled.");
      closeAndExit(0);
    }

    fs.mkdirSync(path.dirname(CONFIG_PATH), { recursive: true });
    config = {};
  } else {
    config = configResult.data;
  }

  ensureConfigShape(config);

  const { pluginDir, loadPath } = ensureStablePluginPath();
  config.plugins.load.paths = config.plugins.load.paths.filter((entry) => {
    if (path.resolve(entry) === path.resolve(loadPath)) return true;
    if (isLegacyPluginPath(entry, loadPath)) {
      warn(`Removing old plugin path: ${entry}`);
      return false;
    }
    return true;
  });

  if (!config.plugins.load.paths.includes(loadPath)) {
    config.plugins.load.paths.push(loadPath);
    success(`Added plugin path: ${loadPath}`);
  } else {
    info(`Plugin path already configured: ${loadPath}`);
  }

  if (!config.plugins.allow.includes(PLUGIN_ID)) {
    config.plugins.allow.push(PLUGIN_ID);
    success(`Added to allow list: ${PLUGIN_ID}`);
  } else {
    info(`Plugin already allowed: ${PLUGIN_ID}`);
  }

  if (config.plugins.slots.contextEngine !== PLUGIN_ID) {
    const previous = config.plugins.slots.contextEngine || "none";
    config.plugins.slots.contextEngine = PLUGIN_ID;
    success(`Set contextEngine slot: ${previous} -> ${PLUGIN_ID}`);
  } else {
    info(`contextEngine slot already points to ${PLUGIN_ID}`);
  }

  if (config.plugins.slots.memory !== "none") {
    const previous = config.plugins.slots.memory || "unset";
    config.plugins.slots.memory = "none";
    warn(`Set memory slot to none to avoid conflicts (was: ${previous})`);
  } else {
    info("memory slot already set to none");
  }

  // Migrate config from legacy plugin IDs if present
  const LEGACY_IDS = ["@evermind-ai/openclaw-plugin", "evermind-ai-openclaw-plugin", "everos"];
  for (const oldId of LEGACY_IDS) {
    const old = config.plugins.entries[oldId];
    if (old?.config && !config.plugins.entries[PLUGIN_ID]?.config) {
      info(`Migrating config from legacy plugin "${oldId}"`);
      config.plugins.entries[PLUGIN_ID] = { ...old, enabled: true };
    }
    if (config.plugins.entries[oldId]) {
      delete config.plugins.entries[oldId];
      // Also remove from allow list
      config.plugins.allow = config.plugins.allow.filter((id) => id !== oldId);
      info(`Removed legacy entry: ${oldId}`);
    }
  }

  const entry = config.plugins.entries[PLUGIN_ID] || {};
  const hadExistingConfig = !!entry.config;
  entry.enabled = true;

  if (hadExistingConfig) {
    entry.config = mergePluginConfig(entry.config);
    info("Reusing existing plugin config and filling any missing defaults.");
  } else {
    hr();
    info("Enter the minimum settings needed for the EverOS backend.");
    const baseUrl = await promptWithDefault("EverOS backend URL", DEFAULT_CONFIG.baseUrl);
    const health = await checkBackendHealth(baseUrl);

    if (health.ok) {
      success(`EverOS backend is reachable (${health.status || "ok"}).`);
    } else {
      warn(`EverOS backend is not reachable yet (${health.reason || "unknown reason"}).`);
      warn("You can continue, but memory recall/save will not work until the backend is running.");
      info("Typical start command: cd EverMemOS && uv run python src/run.py");
    }

    const userId = await promptWithDefault("User ID", DEFAULT_CONFIG.userId);
    const groupId = await promptWithDefault("Group ID", DEFAULT_CONFIG.groupId);
    entry.config = mergePluginConfig(null, { baseUrl, userId, groupId });
    success("Plugin config created.");
  }

  config.plugins.entries[PLUGIN_ID] = entry;

  hr();
  info("Saving OpenClaw configuration...");
  if (!saveConfig(config)) {
    closeAndExit(1);
  }
  success("Configuration saved.");

  printSummary(pluginDir, entry);
  printNextSteps(entry);

  const shouldRestart = await prompt("Restart OpenClaw gateway now? (Y/n): ");
  if (shouldRestart.toLowerCase() !== "n") {
    info("Restarting OpenClaw gateway...");
    info('After restart, wait about 1 minute and test with a natural language memory prompt.');
    await restartGateway();
  } else {
    info("When ready, run: openclaw gateway restart");
    info('Then test with: "Remember: I like espresso."');
  }

  closeAndExit(0);
}

install().catch((err) => {
  error(`Installation failed: ${err.message}`);
  console.error(err);
  closeAndExit(1);
});
