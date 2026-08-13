const DEFAULT_UPSTREAM = "https://api.evermind.ai";
const DEFAULT_RATE_PER_MINUTE = 30;
const DEFAULT_DAILY_ROUNDS = 3;
const DEFAULT_TIMEOUT_MS = 20_000;
const DEFAULT_MAX_BODY_BYTES = 1_000_000;

const ALLOWED_EXACT = new Set([
  "POST /api/v2/memory/add",
  "POST /api/v2/memory/flush",
  "POST /api/v2/memory/search",
]);

function readPositiveInteger(name, fallback) {
  const value = Number.parseInt(process.env[name] ?? "", 10);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

export function isAllowed(method, path) {
  return ALLOWED_EXACT.has(`${method} ${path}`);
}

function clientIp(request, context) {
  return context?.ip || request.headers.get("x-nf-client-connection-ip") || "unknown";
}

async function hashClientIp(ip) {
  const bytes = new TextEncoder().encode(ip);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0"),
  )
    .join("")
    .slice(0, 24);
}

async function checkQuota(ip, countRound, now = Date.now()) {
  const redisUrl = (process.env.UPSTASH_REDIS_REST_URL ?? "").replace(/\/$/, "");
  const redisToken = process.env.UPSTASH_REDIS_REST_TOKEN ?? "";
  if (!redisUrl || !redisToken) {
    throw new Error("quota store is not configured");
  }

  const fingerprint = await hashClientIp(ip);
  const minute = Math.floor(now / 60_000);
  const day = new Date(now).toISOString().slice(0, 10);
  const minuteKey = `everos-demo:minute:${fingerprint}:${minute}`;
  const commands = [
    ["INCR", minuteKey],
    ["EXPIRE", minuteKey, 120],
  ];
  if (countRound) {
    const dailyKey = `everos-demo:rounds:${fingerprint}:${day}`;
    commands.push(["INCR", dailyKey], ["EXPIRE", dailyKey, 172_800]);
  }

  const response = await fetch(`${redisUrl}/pipeline`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${redisToken}`,
      "content-type": "application/json",
    },
    body: JSON.stringify(commands),
  });
  if (!response.ok) {
    throw new Error("quota store request failed");
  }

  const results = await response.json();
  if (!Array.isArray(results) || results.length !== commands.length) {
    throw new Error("quota store returned an invalid response");
  }
  if (results.some((result) => result?.error)) {
    throw new Error("quota store command failed");
  }

  const minuteCount = Number(results[0]?.result);
  const dailyCount = countRound ? Number(results[2]?.result) : 0;
  if (!Number.isFinite(minuteCount) || !Number.isFinite(dailyCount)) {
    throw new Error("quota store returned invalid counters");
  }

  return {
    minuteCount,
    dailyCount,
    minuteLimit: readPositiveInteger(
      "RELAY_RATE_PER_MIN",
      DEFAULT_RATE_PER_MINUTE,
    ),
    dailyLimit: readPositiveInteger(
      "RELAY_DAILY_ROUNDS",
      DEFAULT_DAILY_ROUNDS,
    ),
  };
}

async function forwardRequest(request, path, context) {
  const apiKey = process.env.EVEROS_CLOUD_API_KEY ?? "";
  if (!apiKey) {
    return jsonResponse({ error: "relay API key is not configured" }, 500);
  }

  let quota;
  try {
    const countRound =
      request.method === "POST" && path === "/api/v2/memory/add";
    quota = await checkQuota(clientIp(request, context), countRound);
  } catch {
    return jsonResponse({ error: "relay quota service is unavailable" }, 503);
  }
  if (quota.minuteCount > quota.minuteLimit) {
    return jsonResponse({ error: "rate limit exceeded, slow down" }, 429);
  }
  if (quota.dailyCount > quota.dailyLimit) {
    return jsonResponse(
      { error: "daily demo round limit reached, configure your own key" },
      429,
    );
  }

  const maxBodyBytes = readPositiveInteger(
    "RELAY_MAX_BODY_BYTES",
    DEFAULT_MAX_BODY_BYTES,
  );
  const body = request.method === "GET" ? null : await request.arrayBuffer();
  if (body && body.byteLength > maxBodyBytes) {
    return jsonResponse({ error: "request body is too large" }, 413);
  }

  const upstream = (process.env.EVEROS_CLOUD_UPSTREAM ?? DEFAULT_UPSTREAM).replace(
    /\/$/,
    "",
  );
  const incomingUrl = new URL(request.url);
  const upstreamUrl = `${upstream}${path}${incomingUrl.search}`;
  const timeoutMs = readPositiveInteger(
    "RELAY_UPSTREAM_TIMEOUT_MS",
    DEFAULT_TIMEOUT_MS,
  );

  let response;
  try {
    response = await fetch(upstreamUrl, {
      method: request.method,
      headers: {
        accept: "application/json",
        authorization: `Bearer ${apiKey}`,
        "content-type": "application/json",
      },
      body,
      signal: AbortSignal.timeout(timeoutMs),
    });
  } catch {
    return jsonResponse({ error: "upstream unreachable" }, 502);
  }

  const headers = new Headers();
  headers.set(
    "content-type",
    response.headers.get("content-type") ?? "application/json",
  );
  const retryAfter = response.headers.get("retry-after");
  if (retryAfter) {
    headers.set("retry-after", retryAfter);
  }
  return new Response(await response.arrayBuffer(), {
    status: response.status,
    headers,
  });
}

export default async function relay(request, context) {
  const path = new URL(request.url).pathname;
  if (path === "/healthz") {
    return jsonResponse({
      ok: true,
      upstream: process.env.EVEROS_CLOUD_UPSTREAM ?? DEFAULT_UPSTREAM,
      key_configured: Boolean(process.env.EVEROS_CLOUD_API_KEY),
      quota_configured: Boolean(
        process.env.UPSTASH_REDIS_REST_URL &&
          process.env.UPSTASH_REDIS_REST_TOKEN,
      ),
    });
  }
  if (!isAllowed(request.method, path)) {
    return jsonResponse({ error: "endpoint not allowed" }, 403);
  }
  return forwardRequest(request, path, context);
}

export const config = {
  path: ["/healthz", "/api/v2/*"],
};
