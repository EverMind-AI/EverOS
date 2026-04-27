import { CONTEXT_BOUNDARY } from "./prompt.js";
import { stripChannelMetadata } from "./messages.js";

const MAX_CHARS = 20000;

/** Strip injected memory context from user message text */
function stripContext(text) {
  if (!text) return text;
  const cut = text.lastIndexOf(CONTEXT_BOUNDARY);
  return cut < 0 ? text : text.slice(cut + CONTEXT_BOUNDARY.length).replace(/^\s+/, "");
}

/** Truncate to MAX_CHARS */
function cap(s) {
  return s && s.length > MAX_CHARS ? `${s.slice(0, MAX_CHARS)}…` : (s || "");
}

/**
 * Convert OpenClaw AgentMessage to EverOS message format.
 * Strips injected memory context from user messages to avoid memory pollution.
 * @param {Object} msg - OpenClaw AgentMessage
 * @returns {{ role: string, content: string }}
 */
export function convertMessage(msg) {
  const content = msg.content;
  let role = msg.role;
  let textContent = "";

  // Only user and assistant are accepted by EverOS; drop system/unknown roles
  if (role !== "user" && role !== "assistant") {
    return { role: "user", content: "" };
  }

  // Handle text content (simple string)
  if (typeof content === "string") {
    const clean = role === "user" ? cap(stripChannelMetadata(stripContext(content))) : cap(content);
    return { role, content: clean };
  }

  // Handle content blocks (array of {type, ...})
  if (Array.isArray(content)) {
    for (const block of content) {
      if (!block || !block.type) continue;

      if (block.type === "text") {
        const text = block.text ?? "";
        textContent += (textContent ? "\n" : "") + text;
      }
      // Tool call blocks - keep a text summary, skip structured data
      else if (block.type === "toolCall" || block.type === "tool_use") {
        textContent += (textContent ? "\n" : "") + `[Tool: ${block.name || "unknown"}]`;
      }
      // Tool result blocks - skip
      else if (block.type === "tool_result") {
        continue;
      }
    }

    // Assistant messages containing only [Tool: ...] tags have no semantic value for memory
    if (role === "assistant" && /^\s*(\[Tool:\s*[^\]]*\]\s*)+$/.test(textContent)) {
      return { role, content: "" };
    }

    const finalText = role === "user" ? cap(stripChannelMetadata(stripContext(textContent))) : cap(textContent);
    return { role, content: finalText || "" };
  }

  // Fallback for unexpected content types
  return { role, content: cap(content == null ? "" : String(content)) };
}
