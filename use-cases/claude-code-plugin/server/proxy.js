#!/usr/bin/env node
/**
 * EverMem Dashboard Proxy Server
 *
 * Serves the dashboard and proxies API requests to EverMind,
 * working around the browser limitation of not supporting GET requests with body.
 *
 * Usage: node proxy.js
 * Or: EVERMEM_API_KEY=xxx node proxy.js
 */

import http from 'http';
import { readFileSync, existsSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { createHash } from 'crypto';

const __dirname = dirname(fileURLToPath(import.meta.url));

const PORT = process.env.EVERMEM_PROXY_PORT || 3456;
const API_BASE = 'https://api.evermind.ai';
const GROUPS_FILE = join(__dirname, '..', 'data', 'groups.jsonl');

/**
 * Compute keyId from API key (SHA-256 hash, first 12 chars)
 */
function computeKeyId(apiKey) {
  if (!apiKey) return null;
  const hash = createHash('sha256').update(apiKey).digest('hex');
  return hash.substring(0, 12);
}

/**
 * Read groups from JSONL file and filter by keyId
 */
function getGroupsForKey(keyId) {
  if (!existsSync(GROUPS_FILE)) {
    return [];
  }

  try {
    const content = readFileSync(GROUPS_FILE, 'utf8');
    const lines = content.trim().split('\n').filter(Boolean);

    // Aggregate by groupId for matching keyId
    const groupMap = new Map();

    for (const line of lines) {
      try {
        const entry = JSON.parse(line);
        // Only include entries matching this keyId
        if (entry.keyId !== keyId) continue;

        const existing = groupMap.get(entry.groupId);
        if (existing) {
          existing.sessionCount += 1;
          if (entry.timestamp > existing.lastSeen) {
            existing.lastSeen = entry.timestamp;
          }
          if (entry.timestamp < existing.firstSeen) {
            existing.firstSeen = entry.timestamp;
          }
        } else {
          groupMap.set(entry.groupId, {
            id: entry.groupId,
            name: entry.name,
            path: entry.path,
            firstSeen: entry.timestamp,
            lastSeen: entry.timestamp,
            sessionCount: 1
          });
        }
      } catch {}
    }

    // Sort by lastSeen (most recent first)
    return Array.from(groupMap.values()).sort((a, b) =>
      new Date(b.lastSeen).getTime() - new Date(a.lastSeen).getTime()
    );
  } catch {
    return [];
  }
}

function sendCorsHeaders(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
}

function sendJson(res, status, data) {
  sendCorsHeaders(res);
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(data));
}

const server = http.createServer((req, res) => {
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    sendCorsHeaders(res);
    res.writeHead(204);
    res.end();
    return;
  }

  // Forward POST /api/v1/memories/{search,get} to the EverMind API
  if (req.method === 'POST' && (req.url === '/api/v1/memories/search' || req.url === '/api/v1/memories/get')) {
    let body = '';
    req.on('data', chunk => { body += chunk; });

    req.on('end', async () => {
      const authHeader = req.headers['authorization'];
      if (!authHeader) {
        sendJson(res, 401, { error: 'Missing Authorization header' });
        return;
      }

      try {
        const upstream = await fetch(`${API_BASE}${req.url}`, {
          method: 'POST',
          headers: {
            'Authorization': authHeader,
            'Content-Type': 'application/json'
          },
          body
        });

        const text = await upstream.text();
        sendCorsHeaders(res);
        res.writeHead(upstream.status, {
          'Content-Type': upstream.headers.get('content-type') || 'application/json'
        });
        res.end(text);
      } catch (error) {
        console.error('Proxy error:', error.message);
        sendJson(res, 502, {
          error: 'Upstream request failed',
          message: error.message
        });
      }
    });
    return;
  }

  // Health check
  if (req.method === 'GET' && req.url === '/health') {
    sendJson(res, 200, { status: 'ok', port: PORT });
    return;
  }

  // Get groups for the current API key
  if (req.method === 'GET' && req.url === '/api/groups') {
    const authHeader = req.headers['authorization'];
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      sendJson(res, 401, { error: 'Missing or invalid Authorization header' });
      return;
    }

    const apiKey = authHeader.replace('Bearer ', '');
    const keyId = computeKeyId(apiKey);
    const groups = getGroupsForKey(keyId);

    sendJson(res, 200, {
      status: 'ok',
      keyId,
      groups,
      totalGroups: groups.length
    });
    return;
  }

  // Serve dashboard HTML
  if (req.method === 'GET' && (req.url === '/' || req.url.startsWith('/?') || req.url === '/dashboard' || req.url.startsWith('/dashboard?'))) {
    try {
      const dashboardPath = join(__dirname, '..', 'assets', 'dashboard.html');
      const html = readFileSync(dashboardPath, 'utf8');
      sendCorsHeaders(res);
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      res.end(html);
    } catch (error) {
      sendJson(res, 500, { error: 'Failed to load dashboard', message: error.message });
    }
    return;
  }

  // 404 for everything else
  sendJson(res, 404, { error: 'Not found' });
});

server.listen(PORT, () => {
  console.log(`EverMem Dashboard Proxy running on http://localhost:${PORT}`);
  console.log('');
  console.log('The dashboard can now connect to this proxy to fetch memories.');
  console.log('Press Ctrl+C to stop.');
});
