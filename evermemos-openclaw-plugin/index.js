import { resolveConfig } from "./src/config.js";
import { searchMemories, saveMemories } from "./src/memory-api.js";
import { parseSearchResponse, buildMemoryPrompt } from "./src/formatter.js";
import { collectMessages, toText, isSessionResetPrompt } from "./src/message-utils.js";

export default {
  id: "evermemos-openclaw-plugin",
  name: "EverMemOS OpenClaw Plugin",
  description: "Long-term memory search and capture for EverMemOS via lifecycle hooks",
  kind: "memory",

  register(api) {
    const cfg = resolveConfig(api.pluginConfig);
    const log = api.logger ?? { warn: (...a) => console.warn(...a) };
    // Track sessions that need flush (triggered by /new or /reset)
    const pendingFlushSessions = new Set();

    api.on("before_agent_start", async (event, ctx) => {
      const query = toText(event?.prompt);
      if (!query || query.length < 3) return;
      // Detect /new or /reset: skip search and mark session for flush
      if (isSessionResetPrompt(query)) {
        log.warn?.("[evermemos] session reset detected, skipping search");
        if (ctx?.sessionKey) pendingFlushSessions.add(ctx.sessionKey);
        return;
      }
      try {
        const params = {
          query,
          user_id: cfg.userId,
          group_id: cfg.groupId || undefined,
          memory_types: cfg.memoryTypes,
          retrieve_method: cfg.retrieveMethod,
          top_k: cfg.topK,
        };
        const block = buildMemoryPrompt(parseSearchResponse(await searchMemories(cfg, params)), {
          wrapInCodeBlock: true,
        });
        if (block) return { prependContext: block };
      } catch (err) {
        log.warn?.(`[evermemos] search: ${err}`);
      }
    });

    api.on("agent_end", async (event, ctx) => {
      const msgs = event?.messages;
      if (!event?.success || !msgs?.length) return;
      // Consume the flush flag for this session
      const flush = ctx?.sessionKey ? pendingFlushSessions.delete(ctx.sessionKey) : false;
      try {
        const messages = collectMessages(msgs);
        if (!messages.length) return;
        await saveMemories(cfg, { userId: cfg.userId, groupId: cfg.groupId, messages, flush });
      } catch (err) {
        log.warn?.(`[evermemos] capture: ${err}`);
      }
    });
  },
};
