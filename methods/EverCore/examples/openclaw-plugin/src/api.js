import { createHash } from "node:crypto";
import { request } from "./http.js";

const noop = { info() {}, warn() {} };
const TAG = "[evermind-ai-everos]";

/** Generate a deterministic message ID scoped by idSeed.
 *  Same seed + role + content always produces the same ID.
 *  Different seeds (different turns/sessions) produce different IDs,
 *  so repeated short messages like "ok" won't collide across turns. */
function messageId(idSeed, role, content) {
  const hash = createHash("sha256").update(`${idSeed}:${role}:${content}`).digest("hex").slice(0, 24);
  return `em_${hash}`;
}

export async function searchMemories(cfg, params, log = noop) {
  const { memory_types, ...baseParams } = params;

  const SEARCHABLE = new Set(["episodic_memory"]);
  const searchTypes = (memory_types ?? []).filter((t) => SEARCHABLE.has(t));

  if (!searchTypes.length) {
    return { status: "ok", result: { memories: [], pending_messages: [] } };
  }

  const p = { ...baseParams, memory_types: searchTypes };
  log.info(`${TAG} GET /api/v1/memories/search`);
  const r = await request(cfg, "GET", "/api/v1/memories/search", p);
  log.info(`${TAG} GET response`);

  return {
    status: "ok",
    result: {
      memories: r?.result?.memories ?? [],
      pending_messages: r?.result?.pending_messages ?? [],
    },
  };
}

export async function saveMemories(cfg, { userId, groupId, messages = [], flush = false, idSeed = "" }) {
  if (!messages.length) return;
  const stamp = Date.now();

  const payloads = messages.map((msg, i) => {
    const { role = "user", content = "" } = msg;
    // Always use userId as sender so the backend stores a consistent user_id
    // for both user and assistant messages. The `role` field distinguishes who spoke.
    const sender = userId;
    const senderName = role === "assistant" ? "assistant" : userId;
    const isLast = i === messages.length - 1;

    return {
      message_id: messageId(idSeed, role, content),
      create_time: new Date(stamp + i).toISOString(),
      role,
      sender,
      sender_name: senderName,
      content,
      group_id: groupId,
      group_name: groupId,
      scene: "assistant",
      raw_data_type: "AgentConversation",
      ...(flush && isLast && { flush: true }),
    };
  });

  for (const payload of payloads) {
    await request(cfg, "POST", "/api/v1/memories", payload);
  }
}
