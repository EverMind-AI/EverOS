/**
 * EverOS ContextEngine — all lifecycle hooks.
 */

import { resolveConfig } from "./config.js";
import { saveMemories } from "./api.js";
import { toText } from "./messages.js";
import { ContextAssembler } from "./subagent-assembler.js";
import { convertMessage } from "./convert.js";
import { isSessionResetPrompt } from "./messages.js";

/** Internal/ephemeral sessions that should not be persisted to the memory backend. */
function isEphemeralSession(sessionKey) {
  return sessionKey?.startsWith("temp:") || sessionKey?.startsWith("internal:");
}

/** Slice messages from the last user message onward (raw, no conversion). */
function collectLastUserTurn(messages) {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]?.role === "user") return messages.slice(i);
  }
  return [];
}

/**
 * @param {Object} pluginMeta - Plugin manifest (id, name, version)
 * @param {Object} pluginConfig - Runtime plugin configuration
 * @param {Object} logger - Logger instance
 * @returns {Object} ContextEngine implementation
 */
export function createContextEngine(pluginMeta, pluginConfig, logger) {
  const cfg = resolveConfig(pluginConfig);
  const log = logger || { info: (...a) => console.log(...a), warn: (...a) => console.warn(...a) };
  const L = `[${pluginMeta.id}]`;
  const assembler = new ContextAssembler(cfg, log);

  log.info(`${L} ContextEngine config: baseUrl=${cfg.serverUrl}, userId=${cfg.userId}`);

  // Session state - shared across all sessions for this engine instance
  const sessionState = new Map();
  const SESSION_TTL_MS = 2 * 60 * 60 * 1000; // 2 hours

  function pruneStaleSessionState() {
    const now = Date.now();
    for (const [key, state] of sessionState) {
      if (now - (state.lastActiveTime || 0) > SESSION_TTL_MS) {
        sessionState.delete(key);
        log.info(`${L} pruned stale session state: ${key}`);
      }
    }
  }

  function initState() {
    return {
      turnCount: 0,
      lastActiveTime: Date.now(),
      savedUpTo: 0,
    };
  }

  function ensureState(sessionKey) {
    if (!sessionState.has(sessionKey)) {
      sessionState.set(sessionKey, initState());
    }
    return sessionState.get(sessionKey);
  }

  return {
    info: {
      id: pluginMeta.id,
      name: pluginMeta.name,
      version: pluginMeta.version,
      ownsCompaction: false,
    },

    async bootstrap({ sessionId, sessionKey }) {
      pruneStaleSessionState();

      // Verify EverOS backend health
      try {
        const response = await fetch(`${cfg.serverUrl}/health`, {
          signal: AbortSignal.timeout(5000),
        });
        if (!response.ok) {
          log.warn(`${L} bootstrap: backend unhealthy, status=${response.status}`);
        }
      } catch (err) {
        log.warn(`${L} bootstrap: health check failed: ${err.message}`);
      }

      ensureState(sessionKey);
      return { bootstrapped: true };
    },

    async ingest({ sessionId, sessionKey, message }) {
      if (message.isHeartbeat) {
        return { ingested: false };
      }
      ensureState(sessionKey);
      return { ingested: true };
    },

    async ingestBatch({ sessionId, sessionKey, messages, isHeartbeat }) {
      if (isHeartbeat) {
        return { ingestedCount: 0 };
      }
      ensureState(sessionKey);
      return { ingestedCount: messages?.length || 0 };
    },

    async afterTurn({ sessionId, sessionKey, messages, prePromptMessageCount }) {
      pruneStaleSessionState();

      if (isEphemeralSession(sessionKey)) return;

      const state = ensureState(sessionKey);

      state.turnCount++;
      state.lastActiveTime = Date.now();
      if (state.savedUpTo > messages.length) {
        state.savedUpTo = 0;
      }

      const sliceStart = prePromptMessageCount !== undefined
        ? Math.max(prePromptMessageCount, state.savedUpTo)
        : state.savedUpTo || 0;

      const newMessages = sliceStart > 0
        ? messages.slice(sliceStart)
        : collectLastUserTurn(messages);

      if (newMessages.length === 0) return;

      try {
        const converted = newMessages
          .filter((m) => m.role !== "toolResult" && m.role !== "tool")
          .map(convertMessage)
          .filter((m) => m.content);
        if (converted.length === 0) return;

        await saveMemories(cfg, {
          userId: cfg.userId,
          groupId: cfg.groupId,
          messages: converted,
          idSeed: `${sessionKey}:${state.turnCount}`,
        });
        state.savedUpTo = messages.length;
        log.info(`${L} afterTurn: saved ${converted.length} messages, turn=${state.turnCount}`);
      } catch (err) {
        log.warn(`${L} afterTurn: save failed: ${err.message}`);
      }
    },

    async assemble({ sessionId, sessionKey, messages, tokenBudget, prompt }) {
      pruneStaleSessionState();
      const state = ensureState(sessionKey);
      state.lastActiveTime = Date.now();

      const query = toText(prompt) || toText([...messages].reverse().find((m) => m.role === "user")?.content);

      if (!query || query.length < 3) {
        return { messages, estimatedTokens: 0 };
      }

      if (isSessionResetPrompt(query)) {
        return { messages, estimatedTokens: 0 };
      }

      try {
        const { context, memoryCount } = await assembler.assemble(query, messages, state.turnCount);
        if (memoryCount === 0) {
          return { messages, estimatedTokens: 0 };
        }

        log.info(`${L} assemble: retrieved ${memoryCount} memories`);
        return {
          messages,
          estimatedTokens: Math.floor(context.length / 4),
          systemPromptAddition: context,
        };
      } catch (err) {
        log.warn(`${L} assemble: failed: ${err.message}`);
        return { messages, estimatedTokens: 0 };
      }
    },

    async compact({ sessionId, sessionKey, tokenBudget, currentTokenCount }) {
      const state = sessionState.get(sessionKey);
      if (!state) {
        return { ok: true, compacted: false, reason: "no session state" };
      }

      state.savedUpTo = 0;

      const threshold = tokenBudget ? tokenBudget * 0.8 : 8000;
      const overBudget = currentTokenCount && currentTokenCount > threshold;

      return {
        ok: true,
        compacted: false,
        reason: overBudget
          ? `token count (${currentTokenCount}) exceeds 80% of budget (${tokenBudget}), host should compact`
          : "within threshold",
      };
    },

    async dispose({ sessionKey } = {}) {
      if (sessionKey) {
        sessionState.delete(sessionKey);
      } else {
        sessionState.clear();
      }
    },
  };
}
