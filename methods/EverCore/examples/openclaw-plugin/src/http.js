import { setTimeout as sleep } from "node:timers/promises";
import { TIMEOUT_MS } from "./config.js";

function buildUrl(base, path, params) {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params ?? {})) {
    if (v == null || v === "") continue;
    if (Array.isArray(v)) {
      v.filter((x) => x != null && x !== "").forEach((x) => qs.append(k, String(x)));
    } else {
      qs.set(k, String(v));
    }
  }
  const q = qs.toString();
  return q ? `${base}${path}?${q}` : `${base}${path}`;
}

async function send(url, method, headers, body, timeoutMs) {
  const ac = new AbortController();
  const t = setTimeout(() => ac.abort(), timeoutMs);
  try {
    const res = await fetch(url, { method, headers, body, signal: ac.signal });
    if (!res.ok) {
      const msg = await res.text().catch(() => "");
      throw new Error(`HTTP ${res.status}${msg ? ` – ${msg.slice(0, 200)}` : ""}`);
    }
    return res.json();
  } catch (err) {
    if (ac.signal.aborted) throw new Error(`timed out after ${timeoutMs}ms`);
    throw err;
  } finally {
    clearTimeout(t);
  }
}

export async function request(cfg, method, path, params) {
  const headers = { "Content-Type": "application/json" };
  // GET and DELETE use query string, POST uses body
  const url = (method === "GET" || method === "DELETE") ? buildUrl(cfg.serverUrl, path, params) : `${cfg.serverUrl}${path}`;
  const body = method === "POST" && params ? JSON.stringify(params) : undefined;
  const ms = TIMEOUT_MS;

  try {
    return await send(url, method, headers, body, ms);
  } catch (err) {
    console.warn(`[evermind-ai-everos] request failed, retrying: ${err.message}`);
    await sleep(150);
    return send(url, method, headers, body, ms);
  }
}
