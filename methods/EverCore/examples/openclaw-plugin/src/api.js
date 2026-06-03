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
  const { memory_types, user_id, group_id, retrieve_method, ...baseParams } = params;

  const SEARCHABLE = new Set(["episodic_memory"]);
  const searchTypes = (memory_types ?? []).filter((t) => SEARCHABLE.has(t));

  if (!searchTypes.length) {
    return { status: "ok", result: { memories: [], pending_messages: [] } };
  }

  // SearchMemoriesRequest (v1) is a POST JSON body. A GET request hits the
  // POST-only route and 405s; a flat top-level user_id then 422s ("Field
  // required: filters") because user_id/group_id must live inside a Filters DSL
  // object, and the retrieval-method field is `method`, not `retrieve_method`.
  // Build the v1 request envelope so the search reaches and is accepted.
  const filters = {};
  if (user_id) filters.user_id = user_id;
  if (group_id) filters.group_id = group_id;

  const body = {
    ...baseParams, // query, top_k
    memory_types: searchTypes,
    filters,
    ...(retrieve_method ? { method: retrieve_method } : {}),
  };

  log.info(`${TAG} POST /api/v1/memories/search`);
  const r = await request(cfg, "POST", "/api/v1/memories/search", body);
  log.info(`${TAG} POST response`);

  // The v1 response is { data: { episodes, profiles, raw_messages, ... } }, not
  // the v0 { result: { memories, pending_messages } } the caller consumes. Map
  // episodes -> memories (tagging memory_type so the episodic filter matches)
  // and raw_messages -> pending_messages (flattening content_items to text).
  const data = r?.data ?? {};
  const memories = (data.episodes ?? []).map((e) => ({
    memory_type: "episodic_memory",
    score: e.score ?? 0,
    summary: e.summary,
    episode: e.episode,
    subject: e.subject,
    timestamp: e.timestamp,
  }));
  const pending_messages = (data.raw_messages ?? []).map((m) => ({
    content: (m.content_items ?? []).map((c) => c?.text ?? "").join(" ").trim(),
    sender_name: m.sender_name,
    created_at: m.created_at ?? m.timestamp,
  }));

  return { status: "ok", result: { memories, pending_messages } };
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
