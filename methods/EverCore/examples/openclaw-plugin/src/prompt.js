export const CONTEXT_BOUNDARY = "user\u200boriginal\u200bquery\u200b:\u200b\u200b\u200b\u200b";

function timestampToLabel(ts) {
  if (ts == null || ts === "") return "";

  if (typeof ts === "number") {
    const d = new Date(ts);
    if (Number.isNaN(d.getTime())) return "";
    const p = (n) => `${n}`.padStart(2, "0");
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
  }

  if (typeof ts === "string") {
    const s = ts.trim();
    if (!s) return "";
    // Unix epoch as string
    if (/^\d{10,13}$/.test(s)) return timestampToLabel(Number(s));
    // ISO 8601: extract date and HH:MM
    const dateEnd = s.indexOf("T");
    if (dateEnd === 10 && s.length > 15) return `${s.slice(0, 10)} ${s.slice(11, 16)}`;
    return s;
  }

  return "";
}

export function parseSearchResponse(raw) {
  if (raw?.status !== "ok" || !raw?.result) return null;

  const allMemories = raw.result.memories ?? [];

  const episodic = allMemories
    .filter((m) => m.memory_type === "episodic_memory" && (m.score ?? 0) >= 0.1)
    .map((m) => {
      const body = m.summary || m.episode || m.content || "";
      const subject = m.subject || "";
      return {
        text: subject ? `${subject}: ${body}` : body,
        timestamp: m.timestamp ?? null,
      };
    });

  const pending = (raw.result.pending_messages ?? [])
    .filter((m) => m.content)
    .map((m) => {
      const who = m.sender_name || m.sender || m.user_id || "";
      const body = m.content || "";
      return {
        text: who ? `${who}: ${body}` : body,
        timestamp: m.message_create_time ?? m.created_at ?? null,
      };
    });

  return { episodic, pending };
}

function oneLiner(text) {
  return text == null ? "" : String(text).replace(/[\r\n]+/g, " ").trim();
}

function factLine(fact) {
  const t = oneLiner(fact.text);
  if (!t) return "";
  const when = timestampToLabel(fact.timestamp);
  return when ? `  - [${when}] ${t}` : `  - ${t}`;
}

export function buildMemoryPrompt(parsed, opts = {}) {
  if (!parsed) return "";

  const episodicLines = parsed.episodic.map(factLine).filter(Boolean);
  const pendingLines = (parsed.pending ?? []).map(factLine).filter(Boolean);

  if (!episodicLines.length && !pendingLines.length) return "";

  const xmlBlock = [
    "<memory>",
    ...(episodicLines.length ? ["  <episodic>", ...episodicLines, "  </episodic>"] : []),
    ...(pendingLines.length ? ["  <recent_context>", "  <!-- Recent conversation not yet consolidated into memory. -->", ...pendingLines, "  </recent_context>"] : []),
    "</memory>",
  ];

  const memSection = opts.wrapInCodeBlock ? ["```text", ...xmlBlock, "```"] : xmlBlock;
  const nowLabel = timestampToLabel(Date.now());

  return [
    "Note: Reference memory below. Build on past successes; avoid repeating failed approaches.",
    ...(nowLabel ? [`- Time: ${nowLabel}`] : []),
    "",
    ...memSection,
    "",
    "**Note**: for memory, please not read from or write to local `MEMORY.md` or `memory/*` files as they are provided above.",
    "",
    CONTEXT_BOUNDARY,
  ].join("\n");
}
