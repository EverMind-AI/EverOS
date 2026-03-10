import { request } from "./http-client.js";

export async function searchMemories(cfg, params) {
  const { memory_types, ...baseParams } = params;

  const episodicTypes = (memory_types ?? []).filter((t) => t === "episodic_memory" || t === "profile");
  const caseTypes = (memory_types ?? []).filter((t) => t === "agent_case" || t === "agent_skill");

  const searches = [];
  if (episodicTypes.length) searches.push({ label: "episodic+profile", types: episodicTypes });
  if (caseTypes.length) searches.push({ label: "case+skill", types: caseTypes });

  const results = await Promise.all(
    searches.map(async ({ label, types }) => {
      const p = { ...baseParams, memory_types: types.join(",") };
      console.log("[memory-api] GET /api/v1/memories/search", label, JSON.stringify(p));
      const r = await request(cfg, "GET", "/api/v1/memories/search", p);
      console.log("[memory-api] GET response", label, JSON.stringify(r));
      return r;
    }),
  );

  // merge into a single response in order: episodic/profile first, then case/skill
  const merged = {
    status: "ok",
    result: {
      profiles: [],
      memories: [],
    },
  };
  for (const r of results) {
    if (r?.result?.profiles?.length) merged.result.profiles.push(...r.result.profiles);
    if (r?.result?.memories?.length) merged.result.memories.push(...r.result.memories);
  }
  return merged;
}

export async function saveMemories(cfg, { userId, groupId, messages = [], flush = false }) {
  if (!messages.length) return;
  const stamp = Date.now();
  for (let i = 0; i < messages.length; i++) {
    const { role: rawRole = "user", content = "", tool_calls, tool_call_id } = messages[i];
    const role = rawRole === "user" ? "user" : "assistant";
    const sender = role === "assistant" ? role : userId;
    const isLast = i === messages.length - 1;

    const payload = {
      message_id: `em_${stamp}_${i}`,
      create_time: new Date().toISOString(),
      role,
      sender,
      sender_name: sender,
      content,
      group_id: groupId,
      group_name: groupId,
      scene: "assistant",
      raw_data_type: "AgentConversation",
      ...(tool_calls && { tool_calls }),
      ...(tool_call_id && { tool_call_id }),
      ...(flush && isLast && { flush: true }),
    };
    console.log("[memory-api] POST /api/v1/memories", JSON.stringify(payload));
    const result = await request(cfg, "POST", "/api/v1/memories", payload);
    console.log("[memory-api] POST response", JSON.stringify(result));
  }
}
