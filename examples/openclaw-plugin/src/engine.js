/**
 * EverOS ContextEngine — all lifecycle hooks.
 */

import { resolveConfig } from "./config.js";
import { saveMemories } from "./api.js";
import { toText, isSessionResetPrompt } from "./messages.js";
import { ContextAssembler } from "./subagent-assembler.js";
import { SubagentTracker } from "./subagent-tracker.js";
import { convertMessage } from "./convert.js";

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
  const subagentTracker = new SubagentTracker(cfg, log);

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
      lastAssembleTime: 0,
      lastActiveTime: Date.now(),
      pendingFlush: false,
      savedUpTo: 0,
    };
  }

  return {
    info: {
      id: pluginMeta.id,
      name: pluginMeta.name,
      version: pluginMeta.version,
      ownsCompaction: false,
    },

    async bootstrap({ sessionId, sessionKey }) {
      log.info(`${L} bootstrap: session=${sessionId}, key=${sessionKey}`);

      // Verify EverOS backend health
      try {
        const response = await fetch(`${cfg.serverUrl}/health`, {
          signal: AbortSignal.timeout(5000),
        });
        if (response.ok) {
          const result = await response.json();
          log.info(`${L} bootstrap: backend healthy, status=${result.status}`);
        } else {
          log.warn(`${L} bootstrap: backend unhealthy, status=${response.status}`);
        }
      } catch (err) {
        log.warn(`${L} bootstrap: health check failed: ${err.message}`);
      }

      // Initialize or get session state
      if (!sessionState.has(sessionKey)) {
        sessionState.set(sessionKey, initState());
        log.info(`${L} bootstrap: initialized state for ${sessionKey}`);
      } else {
        log.info(`${L} bootstrap: reusing existing state for ${sessionKey}, turn=${sessionState.get(sessionKey).turnCount}`);
      }

      return { bootstrapped: true };
    },

    async ingest({ sessionId, sessionKey, message }) {
      log.info(`${L} ingest: session=${sessionKey}, role=${message?.role}, isHeartbeat=${message?.isHeartbeat}`);

      if (message.isHeartbeat) {
        return { ingested: false };
      }

      if (!sessionState.has(sessionKey)) {
        log.warn(`${L} ingest: no state for session=${sessionKey}`);
        return { ingested: false };
      }

      // Messages are not buffered here; actual saving happens in afterTurn()
      return { ingested: true };
    },

    async ingestBatch({ sessionId, sessionKey, messages, isHeartbeat }) {
      log.info(`${L} ingestBatch: session=${sessionKey}, count=${messages?.length}, isHeartbeat=${isHeartbeat}`);

      if (isHeartbeat) {
        return { ingestedCount: 0 };
      }

      if (!sessionState.has(sessionKey)) {
        log.warn(`${L} ingestBatch: no state for session=${sessionKey}`);
        return { ingestedCount: 0 };
      }

      // Messages are accepted but not buffered here; actual saving happens in afterTurn()
      return { ingestedCount: messages?.length || 0 };
    },

    async afterTurn({ sessionId, sessionKey, messages, prePromptMessageCount }) {
      const state = sessionState.get(sessionKey);
      if (!state) {
        log.warn(`${L} afterTurn: no state for session=${sessionKey}`);
        return;
      }

      state.turnCount++;
      state.lastActiveTime = Date.now();

      // Reset savedUpTo first if the array shrunk (compact/truncation happened)
      if (state.savedUpTo > messages.length) {
        log.info(`${L} afterTurn: savedUpTo (${state.savedUpTo}) > messages.length (${messages.length}), resetting`);
        state.savedUpTo = 0;
      }

      // Determine the slice start: use savedUpTo if available, otherwise prePromptMessageCount,
      // otherwise fall back to collectLastUserTurn (last user message onward).
      const sliceStart = prePromptMessageCount !== undefined
        ? Math.max(prePromptMessageCount, state.savedUpTo)
        : state.savedUpTo || 0;

      const newMessages = sliceStart > 0
        ? messages.slice(sliceStart)
        : collectLastUserTurn(messages);

      log.info(`${L} afterTurn: session=${sessionKey}, turn=${state.turnCount}, totalMessages=${messages.length}, sliceStart=${sliceStart}, newMessages=${newMessages.length}`);

      if (newMessages.length === 0) {
        log.info(`${L} afterTurn: session=${sessionKey}, turn=${state.turnCount}, no new messages to save`);
        return;
      }

      try {
        // Pre-filter tool messages before convertMessage: EverMemOS only accepts
        // user/assistant roles; filtering here avoids unnecessary conversion work.
        const converted = newMessages
          .filter((m) => m.role !== "toolResult" && m.role !== "tool")
          .map(convertMessage)
          .filter((m) => m.content);
        if (converted.length === 0) {
          log.info(`${L} afterTurn: session=${sessionKey}, turn=${state.turnCount}, no valid messages to save`);
          return;
        }
        await saveMemories(cfg, {
          userId: cfg.userId,
          groupId: cfg.groupId,
          messages: converted,
          flush: state.pendingFlush || false,
          idSeed: `${sessionKey}:${state.turnCount}`,
        }, log);
        state.savedUpTo = messages.length;
        log.info(`${L} afterTurn: session=${sessionKey}, turn=${state.turnCount}, saved ${converted.length} messages, savedUpTo=${state.savedUpTo}`);

        if (state.pendingFlush) {
          state.pendingFlush = false;
          log.info(`${L} afterTurn: flush flag consumed`);
        }
      } catch (err) {
        log.warn(`${L} afterTurn: save failed: ${err.message}`);
      }
    },

    async assemble({ sessionId, sessionKey, messages, tokenBudget }) {
      pruneStaleSessionState();

      if (!sessionState.has(sessionKey)) {
        sessionState.set(sessionKey, initState());
        log.info(`${L} assemble: initialized state for ${sessionKey}`);
      }

      const state = sessionState.get(sessionKey);
      state.lastActiveTime = Date.now();

      const lastUserMsg = [...messages].reverse().find((m) => m.role === "user");
      const query = lastUserMsg ? toText(lastUserMsg.content) : "";

      if (!query || query.length < 3) {
        return { messages, estimatedTokens: 0 };
      }

      if (isSessionResetPrompt(query)) {
        log.info(`${L} assemble: session reset detected, keeping current messages`);
        state.pendingFlush = true;
        return { messages, estimatedTokens: 0 };
      }

      try {
        state.lastAssembleTime = Date.now();

        const { context, memoryCount } = await assembler.assemble(query, messages, state.turnCount);

        if (memoryCount === 0) {
          return { messages, estimatedTokens: 0 };
        }

        const memoryMessage = {
          role: "system",
          content: `[Relevant Memory]\n${context}`,
          _memory: true,
        };

        log.info(`${L} assemble: session=${sessionKey}, retrieved ${memoryCount} memories`);

        return {
          messages: [memoryMessage, ...messages],
          estimatedTokens: Math.floor(context.length / 4),
        };
      } catch (err) {
        log.warn(`${L} assemble: ${err.message}`);
        return { messages, estimatedTokens: 0 };
      }
    },

    async prepareSubagentSpawn({ sessionKey, subagentId, prompt, subagentType }) {
      if (!sessionState.has(sessionKey)) {
        sessionState.set(sessionKey, initState());
      }

      const state = sessionState.get(sessionKey);
      subagentTracker.register(subagentId, {
        subagentType,
        parentTurnCount: state.turnCount,
      });

      try {
        const query = toText(prompt);
        const prependContext = query ? await assembler.assembleForSubagent(query) : "";
        return {
          prependContext,
          metadata: {
            subagentId,
            parentTurnCount: state.turnCount,
          },
        };
      } catch (err) {
        log.warn(`${L} prepareSubagentSpawn: ${err.message}`);
        return {
          prependContext: "",
          metadata: {
            subagentId,
            parentTurnCount: state.turnCount,
          },
        };
      }
    },

    async onSubagentEnded({ subagentId, messages, success }) {
      subagentTracker.unregister(subagentId);

      // Subagent messages are saved by the parent session's afterTurn(),
      // which is the single write path. Do not save here to avoid duplicates
      // (afterTurn and onSubagentEnded cannot share a stable idSeed).
      log.info(`${L} onSubagentEnded: subagent=${subagentId}, success=${success}, messages=${messages?.length || 0}`);
    },

    async compact({ sessionId, sessionKey, tokenBudget, currentTokenCount }) {
      const state = sessionState.get(sessionKey);
      if (!state) {
        return { ok: true, compacted: false, reason: "no session state" };
      }

      log.info(`${L} compact: session=${sessionKey}, tokens=${currentTokenCount}, budget=${tokenBudget}`);

      // Reset savedUpTo since compact may truncate the messages array
      state.savedUpTo = 0;

      // ownsCompaction is false — we don't perform compaction ourselves.
      // We report status so the host can decide whether to compact.
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
      if (sessionKey && sessionState.has(sessionKey)) {
        sessionState.delete(sessionKey);
        log.info(`${L} dispose: cleared state for ${sessionKey}`);
      } else if (!sessionKey) {
        sessionState.clear();
        log.info(`${L} dispose: cleared all session states`);
      }
    },
  };
}
