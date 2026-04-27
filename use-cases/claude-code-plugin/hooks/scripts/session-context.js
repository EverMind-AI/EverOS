#!/usr/bin/env node

/**
 * EverMem SessionStart Hook
 * Retrieves recent memories and displays last session summary
 * No AI summarization - uses local data only
 */

// Check Node.js version early
const nodeVersion = process.versions?.node;
if (!nodeVersion) {
  console.error(JSON.stringify({
    continue: true,
    systemMessage: '⚠️ EverMem: Node.js environment not detected. Please install Node.js 18+ to use EverMem.'
  }));
  process.exit(0);
}

const [major] = nodeVersion.split('.').map(Number);
if (major < 18) {
  console.error(JSON.stringify({
    continue: true,
    systemMessage: `⚠️ EverMem: Node.js ${nodeVersion} is too old. Please upgrade to Node.js 18+.`
  }));
  process.exit(0);
}

import { readFileSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { getMemories, transformGetMemoriesResults } from './utils/evermem-api.js';
import { getConfig, getGroupId } from './utils/config.js';
import { saveGroup } from './utils/groups-store.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const SESSIONS_FILE = resolve(__dirname, '../../data/sessions.jsonl');

const RECENT_MEMORY_COUNT = 5;  // Number of recent memories to load
const PAGE_SIZE = 100;          // Fetch more to get the latest (API returns old to new)

/**
 * Get the most recent session summary for current group
 * @param {string} groupId - The group ID to filter by
 * @returns {Object|null} Most recent session summary or null
 */
function getLastSessionSummary(groupId) {
  try {
    if (!existsSync(SESSIONS_FILE)) {
      return null;
    }

    const content = readFileSync(SESSIONS_FILE, 'utf8');
    const lines = content.trim().split('\n').filter(Boolean);

    // Search from end (most recent first)
    for (let i = lines.length - 1; i >= 0; i--) {
      try {
        const entry = JSON.parse(lines[i]);
        if (entry.groupId === groupId) {
          return entry;
        }
      } catch {}
    }
    return null;
  } catch {
    return null;
  }
}

/**
 * Format relative time (e.g., "2h ago", "1d ago")
 */
function formatRelativeTime(isoTime) {
  const now = Date.now();
  const then = new Date(isoTime).getTime();
  const diffMs = now - then;

  const minutes = Math.floor(diffMs / 60000);
  const hours = Math.floor(diffMs / 3600000);
  const days = Math.floor(diffMs / 86400000);

  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days < 30) return `${days}d ago`;
  return `${Math.floor(days / 30)}mo ago`;
}

async function main() {
  // Read hook input to get cwd
  let hookInput = {};
  try {
    let input = '';
    for await (const chunk of process.stdin) {
      input += chunk;
    }
    if (input) {
      hookInput = JSON.parse(input);
    }
  } catch (parseError) {
    console.log(JSON.stringify({
      continue: true,
      systemMessage: `⚠️ EverMem: Failed to parse hook input - ${parseError.message}`
    }));
    return;
  }

  // Set cwd from hook input for config.getGroupId()
  if (hookInput.cwd) {
    process.env.EVERMEM_CWD = hookInput.cwd;
  }

  const config = getConfig();

  // Save group to local storage (track which projects use EverMem)
  if (hookInput.cwd) {
    try {
      saveGroup(getGroupId(), hookInput.cwd);
    } catch (groupError) {
      // Non-blocking, but log for debugging
      console.error(`EverMem groups-store error: ${groupError.message}`);
    }
  }

  if (!config.isConfigured) {
    // Silently skip if not configured
    console.log(JSON.stringify({ continue: true }));
    return;
  }

  try {
    const groupId = getGroupId();

    // Fetch memories (API returns old to new, we'll reverse and take latest)
    const response = await getMemories({ pageSize: PAGE_SIZE });
    const memories = transformGetMemoriesResults(response);

    // Get last session summary from local storage
    const lastSession = getLastSessionSummary(groupId);

    if (memories.length === 0 && !lastSession) {
      // No memories and no last session, skip
      console.log(JSON.stringify({ continue: true }));
      return;
    }

    // Take the most recent memories
    const recentMemories = memories.slice(0, RECENT_MEMORY_COUNT);

    // Build context message for Claude (no AI summarization)
    let contextParts = [];

    // Add last session info if available
    if (lastSession) {
      const timeAgo = formatRelativeTime(lastSession.timestamp);
      contextParts.push(`Last session (${timeAgo}, ${lastSession.turnCount} turns): ${lastSession.summary}`);
    }

    // Add recent memories if available
    if (recentMemories.length > 0) {
      const memoriesText = recentMemories.map((m, i) => {
        const date = new Date(m.timestamp).toLocaleDateString();
        return `[${i + 1}] (${date}) ${m.subject}\n${m.text}`;
      }).join('\n\n---\n\n');
      contextParts.push(`Recent memories (${recentMemories.length}):\n\n${memoriesText}`);
    }

    const contextMessage = `<session-context>\n${contextParts.join('\n\n')}\n</session-context>`;

    // Build display output - show meaningful content, concise but informative
    let displayOutput;
    if (lastSession) {
      // Show last session: time, turns, summary
      const truncatedSummary = lastSession.summary.length > 40
        ? lastSession.summary.substring(0, 40) + '...'
        : lastSession.summary;
      const timeAgo = formatRelativeTime(lastSession.timestamp);
      displayOutput = `💡 EverMem: Last (${timeAgo}, ${lastSession.turnCount} turns): "${truncatedSummary}"`;

      // Add memory preview if available
      if (recentMemories.length > 0) {
        const memorySubjects = recentMemories.slice(0, 2).map(m => {
          const subj = m.subject || '';
          return subj.length > 15 ? subj.substring(0, 15) + '..' : subj;
        }).join(', ');
        displayOutput += ` | ${recentMemories.length} memories: ${memorySubjects}`;
      }
    } else if (recentMemories.length > 0) {
      // No last session, show recent memories with subjects
      const memorySubjects = recentMemories.slice(0, 3).map(m => {
        const subj = m.subject || '';
        return subj.length > 20 ? subj.substring(0, 20) + '..' : subj;
      }).join(', ');
      displayOutput = `💡 EverMem: ${recentMemories.length} memories: ${memorySubjects}`;
    } else {
      displayOutput = `💡 EverMem: Ready`;
    }

    // Output: display to user and add to context
    console.log(JSON.stringify({
      continue: true,
      systemMessage: displayOutput,
      systemPrompt: contextMessage
    }));

  } catch (error) {
    // Don't block session start on errors, but provide detailed error info
    const errorDetails = {
      message: error.message,
      code: error.code,
      name: error.name
    };

    // Provide user-friendly error messages
    let userMessage = '⚠️ EverMem: ';
    if (error.code === 'ENOTFOUND' || error.code === 'ECONNREFUSED') {
      userMessage += `Network error - cannot reach EverMem server. Check your internet connection.`;
    } else if (error.code === 'ETIMEDOUT') {
      userMessage += `Request timeout - EverMem server is slow or unreachable.`;
    } else if (error.message?.includes('401') || error.message?.includes('Unauthorized')) {
      userMessage += `Authentication failed. Check your EVERMEM_API_KEY in .env file.`;
    } else if (error.message?.includes('404')) {
      userMessage += `API endpoint not found. Check EVERMEM_BASE_URL in .env file.`;
    } else if (error.message?.includes('ENOENT')) {
      userMessage += `File not found: ${error.path || 'unknown'}`;
    } else {
      userMessage += `${error.name}: ${error.message}`;
    }

    console.log(JSON.stringify({
      continue: true,
      systemMessage: userMessage
    }));
  }
}

// Top-level error handler for uncaught exceptions during module load
process.on('uncaughtException', (error) => {
  let userMessage = '⚠️ EverMem SessionStart failed: ';

  if (error.code === 'ERR_MODULE_NOT_FOUND') {
    const moduleName = error.message.match(/Cannot find package '([^']+)'/)?.[1] || 'unknown';
    userMessage += `Missing dependency '${moduleName}'. Run: cd ${process.cwd()} && npm install`;
  } else if (error.code === 'ERR_REQUIRE_ESM') {
    userMessage += `Module format error. Ensure package.json has "type": "module"`;
  } else {
    userMessage += `${error.name}: ${error.message}`;
  }

  console.log(JSON.stringify({
    continue: true,
    systemMessage: userMessage
  }));
  process.exit(0);
});

main();
