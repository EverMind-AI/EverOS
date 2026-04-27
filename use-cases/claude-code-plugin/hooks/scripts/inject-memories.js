#!/usr/bin/env node

/**
 * Memory Plugin - UserPromptSubmit Hook
 *
 * This hook automatically injects relevant memories from past sessions ，
 * into Claude's context when the user submits a prompt.
 *
 * Flow:
 * 1. Read prompt from stdin
 * 2. Skip if prompt is too short or API not configured
 * 3. Search EverMem Cloud for relevant memories
 * 4. Optionally filter with Claude SDK
 * 5. Display summary to user (via systemMessage)
 * 6. Inject context for Claude (via additionalContext)
 */

import { isConfigured } from './utils/config.js';
import { searchMemories, transformSearchResults } from './utils/evermem-api.js';
import { formatRelativeTime } from './utils/mock-store.js';
import { debug, setDebugPrefix } from './utils/debug.js';

// Set debug prefix for this script
setDebugPrefix('inject');

const MIN_WORDS = 3;
const MAX_MEMORIES = 5;
const MIN_SCORE = 0.1;  // Only show memories with relevance score above this threshold

/**
 * Count words/tokens in a string (multilingual support)
 * - For CJK (Chinese/Japanese/Korean): counts each character as a token
 * - For other languages: counts space-separated words
 * - For mixed text: counts both
 * @param {string} text
 * @returns {number}
 */
function countWords(text) {
  if (!text) return 0;
  const trimmed = text.trim();
  if (!trimmed) return 0;

  // Regex for CJK characters (Chinese, Japanese Kanji, Korean Hanja)
  // Also includes Japanese Hiragana/Katakana and Korean Hangul
  const cjkRegex = /[\u4E00-\u9FFF\u3400-\u4DBF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF]/g;

  // Count CJK characters
  const cjkMatches = trimmed.match(cjkRegex);
  const cjkCount = cjkMatches ? cjkMatches.length : 0;

  // Remove CJK characters and count remaining space-separated words
  const nonCjkText = trimmed.replace(cjkRegex, ' ').trim();
  const wordCount = nonCjkText ? nonCjkText.split(/\s+/).filter(w => w.length > 0).length : 0;

  return cjkCount + wordCount;
}

/**
 * Main hook handler
 */
async function main() {
  try {
    // Read stdin
    const input = await readStdin();
    const data = JSON.parse(input);
    const prompt = data.prompt || '';

    debug('hookInput:', data);

    // Set cwd from hook input for config.getGroupId()
    if (data.cwd) {
      process.env.EVERMEM_CWD = data.cwd;
    }

    // Skip short prompts silently
    const wordCount = countWords(prompt);
    if (wordCount < MIN_WORDS) {
      debug('skipped: prompt too short', { wordCount, minWords: MIN_WORDS });
      process.exit(0);
    }

    // Skip if not configured (silent - don't nag users)
    if (!isConfigured()) {
      debug('skipped: not configured');
      process.exit(0);
    }

    // Search memories from EverMem Cloud
    let memories = [];
    let apiResponse = null;
    try {
      debug('searching memories for prompt:', prompt.slice(0, 100) + (prompt.length > 100 ? '...' : ''));
      apiResponse = await searchMemories(prompt, {
        topK: 15,
        retrieveMethod: 'hybrid'
      });
      memories = transformSearchResults(apiResponse);
      debug("memories:", memories);
      debug('search results:', { total: memories.length, memories: memories.map(m => ({ score: m.score, subject: m.subject })) });
    } catch (error) {
      // Silent on API errors - don't block user workflow
      debug('search error:', error.message);
      process.exit(0);
    }

    // Filter by minimum score threshold
    const relevantMemories = memories.filter(m => m.score >= MIN_SCORE);
    debug('filtered memories:', { total: relevantMemories.length, minScore: MIN_SCORE });

    // No relevant memories above threshold - silently exit (this is normal)
    if (relevantMemories.length === 0) {
      debug('skipped: no relevant memories above threshold');
      process.exit(0);
    }

    // Take top memories
    const selectedMemories = relevantMemories.slice(0, MAX_MEMORIES);
    debug('selected memories:', selectedMemories.map(m => ({ score: m.score, subject: m.subject, timestamp: m.timestamp })));

    // Build context for Claude
    const context = buildContext(selectedMemories);

    // Build display message for user
    const displayMessage = buildDisplayMessage(selectedMemories);

    // Output JSON with systemMessage (user display) and additionalContext (for Claude)
    const output = {
      systemMessage: displayMessage,
      hookSpecificOutput: {
        hookEventName: 'UserPromptSubmit',
        additionalContext: context
      }
    };

    debug('output:', { systemMessage: displayMessage, contextLength: context.length });
    process.stdout.write(JSON.stringify(output));
    process.exit(0);

  } catch (error) {
    // Silent on errors - don't block user workflow
    debug('error:', error.message);
    process.exit(0);
  }
}

/**
 * Read all stdin input
 * @returns {Promise<string>}
 */
function readStdin() {
  return new Promise((resolve, reject) => {
    let data = '';

    process.stdin.setEncoding('utf8');

    process.stdin.on('data', chunk => {
      data += chunk;
    });

    process.stdin.on('end', () => {
      resolve(data);
    });

    process.stdin.on('error', reject);
  });
}

/**
 * Build display message for user (shown via systemMessage)
 * @param {Object[]} memories - Selected memories
 * @returns {string}
 */
function buildDisplayMessage(memories) {
  const header = `📝 Memory Retrieved (${memories.length}):`;

  const lines = [header];

  for (const memory of memories) {
    const relTime = formatRelativeTime(memory.timestamp);
    const score = memory.score ? memory.score.toFixed(2) : '0.00';
    // Use subject as title if available, otherwise truncate text
    const title = memory.subject
      ? memory.subject
      : (memory.text.length > 60 ? memory.text.slice(0, 60) + '...' : memory.text);
    lines.push(`  • [${score}] (${relTime}) ${title}`);
  }

  return lines.join('\n');
}

/**
 * Build context string for Claude
 * Memories are sorted by timestamp (most recent first) to prioritize recent context
 * @param {Object[]} memories - Selected memories
 * @returns {string}
 */
function buildContext(memories) {
  const lines = [];

  // Sort by timestamp descending (most recent first)
  const sortedMemories = [...memories].sort((a, b) => {
    const timeA = a.timestamp ? new Date(a.timestamp).getTime() : 0;
    const timeB = b.timestamp ? new Date(b.timestamp).getTime() : 0;
    return timeB - timeA;
  });

  lines.push('<relevant-memories>');
  lines.push('The following memories from past sessions are relevant to the user\'s current task:');
  lines.push('');
  lines.push('IMPORTANT: Memories are ordered by recency (most recent first). When there are conflicts or updates between memories, prefer the MORE RECENT information as it likely reflects the latest decisions, code changes, or user preferences.');
  lines.push('');

  for (const memory of sortedMemories) {
    // Format timestamp for context
    const timeStr = memory.timestamp
      ? new Date(memory.timestamp).toLocaleString('zh-CN', {
          year: 'numeric',
          month: '2-digit',
          day: '2-digit',
          weekday: 'short',
          hour: '2-digit',
          minute: '2-digit',
          timeZone: 'UTC'
        }) + ' UTC'
      : 'Unknown time';

    lines.push(`[${timeStr}]`);
    lines.push(memory.text);
    lines.push('');
  }

  lines.push('Use this context to inform your response. The user has already seen these memories displayed.');
  lines.push('</relevant-memories>');

  return lines.join('\n');
}

// Run
main();
