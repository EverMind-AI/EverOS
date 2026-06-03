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

export async function saveMemories(cfg, { userId, sessionId, messages = [], idSeed = "" }) {
  if (!messages.length) return;
  const stamp = Date.now();

  // PersonalAddRequest (POST /api/v1/memories) is an ENVELOPE: a top-level
  // user_id plus a `messages` array of MessageItem. Posting flat per-message
  // bodies without a top-level user_id triggers HTTP 422
  // ("Field required: user_id") — see issue #237.
  const items = messages.map((msg, i) => {
    const { role = "user", content = "" } = msg;
    const senderName = role === "assistant" ? "assistant" : userId;

    const item = {
      message_id: messageId(idSeed, role, content),
      // MessageItem.timestamp is a REQUIRED unix-milliseconds integer.
      // The previous create_time(ISO) field was ignored by the converter and
      // left timestamp missing, failing validation.
      timestamp: stamp + i,
      role,
      sender_name: senderName,
      content,
    };

    // Personal-scene sender_id rules (request_converter.py):
    //   role=user      -> sender_id must equal user_id (or be omitted)
    //   role=assistant -> sender_id must NOT equal user_id (backend generates one)
    // So only set sender_id for user turns; let the backend derive it for
    // assistant turns to avoid a sender_id conflict.
    if (role === "user") {
      item.sender_id = userId;
    }

    return item;
  });

  const body = {
    user_id: userId,
    messages: items,
    ...(sessionId ? { session_id: sessionId } : {}),
  };

  await request(cfg, "POST", "/api/v1/memories", body);
}
