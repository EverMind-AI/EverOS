#!/usr/bin/env node

/**
 * EverMem SessionEnd Hook
 * Saves session summary (first user prompt + stats) to local storage
 * No AI summarization - just extracts key info from transcript
 */

import { readFileSync, appendFileSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { getGroupId, getConfig } from './utils/config.js';
import { debug, setDebugPrefix } from './utils/debug.js';

setDebugPrefix('session-end');

const __dirname = dirname(fileURLToPath(import.meta.url));
const SESSIONS_FILE = resolve(__dirname, '../../data/sessions.jsonl');

/**
 * Read transcript and extract key content
 * @param {string} transcriptPath - Path to the transcript JSONL file
 * @returns {Object|null} Extracted content
 */
function extractTranscriptContent(transcriptPath) {
  try {
    if (!existsSync(transcriptPath)) {
      return null;
    }

    const content = readFileSync(transcriptPath, 'utf8');
    const lines = content.trim().split('\n').filter(Boolean);

    let firstUserPrompt = null;
    let lastUserPrompt = null;
    let turnCount = 0;
    let firstTimestamp = null;
    let lastTimestamp = null;

    for (const line of lines) {
      try {
        const entry = JSON.parse(line);

        // Track timestamps
        if (entry.timestamp) {
          if (!firstTimestamp) firstTimestamp = entry.timestamp;
          lastTimestamp = entry.timestamp;
        }

        // Count turns
        if (entry.type === 'system' && entry.subtype === 'turn_duration') {
          turnCount++;
        }

        // Extract user messages (not tool_result)
        if (entry.type === 'user' && entry.message?.role === 'user') {
          const msgContent = entry.message.content;
          if (typeof msgContent === 'string' && msgContent.trim()) {
            if (!firstUserPrompt) {
              firstUserPrompt = msgContent.trim();
            }
            lastUserPrompt = msgContent.trim();
          }
        }
      } catch {}
    }

    return {
      firstUserPrompt: firstUserPrompt?.substring(0, 200) || '',
      lastUserPrompt: lastUserPrompt?.substring(0, 200) || '',
      turnCount,
      firstTimestamp,
      lastTimestamp
    };
  } catch {
    return null;
  }
}

/**
 * Save session summary to local JSONL file
 */
function saveSummary(entry) {
  try {
    appendFileSync(SESSIONS_FILE, JSON.stringify(entry) + '\n', 'utf8');
    return true;
  } catch {
    return false;
  }
}

/**
 * Check if session already has a summary
 */
function alreadySummarized(sessionId) {
  try {
    if (!existsSync(SESSIONS_FILE)) {
      return false;
    }
    const content = readFileSync(SESSIONS_FILE, 'utf8');
    return content.includes(`"sessionId":"${sessionId}"`);
  } catch {
    return false;
  }
}

async function main() {
  // Read hook input
  let hookInput = {};
  try {
    let input = '';
    for await (const chunk of process.stdin) {
      input += chunk;
    }
    if (input) {
      hookInput = JSON.parse(input);
    }
  } catch {
    process.exit(0);
  }

  const { session_id, transcript_path, cwd, reason } = hookInput;

  // Skip if no transcript or already summarized
  if (!transcript_path || !session_id) {
    process.exit(0);
  }

  const wasAlreadySummarized = alreadySummarized(session_id);

  // Set cwd for config
  if (cwd) {
    process.env.EVERMEM_CWD = cwd;
  }

  const config = getConfig();
  if (!config.isConfigured) {
    process.exit(0);
  }

  // Extract content from transcript
  const content = extractTranscriptContent(transcript_path);
  if (!content || content.turnCount === 0) {
    process.exit(0);
  }

  // Use first user prompt as summary (truncated)
  const summary = content.firstUserPrompt || 'Session with no text prompts';

  // Calculate session duration
  let durationStr = '';
  if (content.firstTimestamp && content.lastTimestamp) {
    const durationMs = new Date(content.lastTimestamp) - new Date(content.firstTimestamp);
    const minutes = Math.floor(durationMs / 60000);
    if (minutes < 1) {
      durationStr = '<1min';
    } else if (minutes < 60) {
      durationStr = `${minutes}min`;
    } else {
      const hours = Math.floor(minutes / 60);
      const remainMins = minutes % 60;
      durationStr = remainMins > 0 ? `${hours}h${remainMins}m` : `${hours}h`;
    }
  }

  // Truncate summary for display
  const displaySummary = summary.length > 50
    ? summary.substring(0, 50) + '...'
    : summary;

  // Build output: turns, duration, summary
  const parts = [`${content.turnCount} turns`];
  if (durationStr) parts.push(durationStr);

  // Save to local file (only if not already saved)
  if (!wasAlreadySummarized) {
    const entry = {
      sessionId: session_id,
      groupId: getGroupId(),
      summary,
      turnCount: content.turnCount,
      reason: reason || 'unknown',
      startTime: content.firstTimestamp,
      endTime: content.lastTimestamp,
      timestamp: new Date().toISOString()
    };
    saveSummary(entry);
  }

  // Always output session summary (whether saved or not)
  const message = `📝 Session (${parts.join(', ')}): "${displaySummary}"`;

  // Log to unified debug file
  debug('output', message);

  console.error(message);  // Direct terminal output
  console.log(JSON.stringify({ systemMessage: message }));
}

main().catch(() => process.exit(0));
