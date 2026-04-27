#!/usr/bin/env node

process.on('uncaughtException', () => process.exit(0));
process.on('unhandledRejection', () => process.exit(0));

import { readFileSync, existsSync } from 'fs';
import { isConfigured } from './utils/config.js';  // This loads .env
import { addMemory } from './utils/evermem-api.js';
import { debug, setDebugPrefix } from './utils/debug.js';

// Set debug prefix for this script
setDebugPrefix('store');

try {
  let input = '';
  for await (const chunk of process.stdin) {
    input += chunk;
  }

  const hookInput = JSON.parse(input);
  debug('hookInput:', hookInput);
  const transcriptPath = hookInput.transcript_path;

  // Set cwd from hook input for config.getGroupId()
  if (hookInput.cwd) {
    process.env.EVERMEM_CWD = hookInput.cwd;
  }

  if (!transcriptPath || !existsSync(transcriptPath) || !isConfigured()) {
    process.exit(0);
  }

  /**
   * Read transcript file with retry logic
   * Waits for turn_duration marker which indicates the turn is complete
   */
  async function readTranscriptWithRetry(path, maxRetries = 5, delayMs = 100) {
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      const content = readFileSync(path, 'utf8');
      const lines = content.trim().split('\n');

      // Check if the last line is turn_duration (indicates turn is complete)
      let isComplete = false;
      try {
        const lastLine = JSON.parse(lines[lines.length - 1]);
        isComplete = lastLine.type === 'system' && lastLine.subtype === 'turn_duration';
      } catch {}

      debug(`read attempt ${attempt}:`, {
        totalLines: lines.length,
        isComplete,
        lastLineType: (() => {
          try {
            const e = JSON.parse(lines[lines.length - 1]);
            return e.subtype ? `${e.type}/${e.subtype}` : e.type;
          } catch { return 'unknown'; }
        })()
      });

      if (isComplete) {
        return lines;
      }

      if (attempt < maxRetries) {
        debug(`turn not complete, waiting ${delayMs}ms before retry...`);
        await new Promise(resolve => setTimeout(resolve, delayMs));
      }
    }

    // Return whatever we have after max retries
    debug('max retries reached, proceeding with current content');
    const content = readFileSync(path, 'utf8');
    return content.trim().split('\n');
  }

  const lines = await readTranscriptWithRetry(transcriptPath);

  // Debug: show last 3 lines of the file (just the type)
  debug('last 3 lines types:', lines.slice(-3).map((line, idx) => {
    try {
      const e = JSON.parse(line);
      return { index: lines.length - 3 + idx, type: e.type, subtype: e.subtype, hasContent: !!e.message?.content };
    } catch { return { index: lines.length - 3 + idx, error: 'parse failed' }; }
  }));

  /**
   * Check if content is meaningful (not just whitespace/newlines)
   * @param {string} text
   * @returns {boolean}
   */
  function hasContent(text) {
    return text && text.trim().length > 0;
  }

  /**
   * Extract the last turn's user input and assistant response
   *
   * A Turn = User sends message → Claude responds (may include multiple tool calls)
   * Turn boundary is marked by: {"type":"system","subtype":"turn_duration"}
   *
   * User messages may be:
   *   - Original input: {"type":"user","message":{"content":"string"}}
   *   - Tool result: {"type":"user","message":{"content":[{"type":"tool_result",...}]}}
   *
   * Assistant messages may contain multiple content blocks:
   *   - thinking: Claude's internal reasoning
   *   - tool_use: Tool invocations
   *   - text: Final response to user (this is what we want)
   */
  function extractLastTurn(lines) {
    // IMPORTANT: When Stop hook runs, turn_duration for current turn hasn't been written yet.
    // The turn_duration marker is written AFTER the Stop hook completes.
    // So current turn END is always at the end of the file.
    const turnEndIndex = lines.length;

    // Current turn START is right after the last turn_duration marker.
    // Only turn_duration marks turn boundaries (file-history-snapshot is NOT a turn boundary).
    // If no marker found, start from beginning of file.
    let turnStartIndex = 0;
    for (let i = lines.length - 1; i >= 0; i--) {
      try {
        const e = JSON.parse(lines[i]);
        if (e.type === 'system' && e.subtype === 'turn_duration') {
          turnStartIndex = i + 1;
          break;
        }
      } catch {}
    }

    debug('turn range:', { turnStartIndex, turnEndIndex, totalLines: lines.length });

    // Collect user and assistant content from the turn
    const userTexts = [];
    const assistantTexts = [];

    // Debug: log each line's type in the turn
    const lineTypes = [];

    for (let i = turnStartIndex; i < turnEndIndex; i++) {
      try {
        const e = JSON.parse(lines[i]);
        const content = e.message?.content;

        // Debug: record line type
        const lineInfo = { index: i, type: e.type };
        if (e.type === 'assistant' && Array.isArray(content)) {
          lineInfo.contentTypes = content.map(b => b.type);
        }
        lineTypes.push(lineInfo);

        if (e.type === 'user') {
          // User message - distinguish between original input and tool_result
          if (typeof content === 'string') {
            // Original user input (plain string)
            userTexts.push(content);
          } else if (Array.isArray(content)) {
            // Check if it's a tool_result (skip) or text blocks (include)
            for (const block of content) {
              if (block.type === 'text' && block.text) {
                userTexts.push(block.text);
              }
              // Skip tool_result - it's part of Claude's workflow, not user input
            }
          }
        }

        if (e.type === 'assistant') {
          // Assistant message - extract text blocks only
          if (Array.isArray(content)) {
            for (const block of content) {
              if (block.type === 'text' && block.text) {
                assistantTexts.push(block.text);
              }
              // Skip: thinking (internal), tool_use (workflow)
            }
          } else if (typeof content === 'string') {
            assistantTexts.push(content);
          }
        }
      } catch {}
    }

    // Debug: output line types
    debug('line types in turn:', lineTypes);
    debug('assistantTexts count:', assistantTexts.length);

    return {
      user: userTexts.join('\n\n'),
      assistant: assistantTexts.join('\n\n')
    };
  }

  // Extract the last turn's content
  const lastTurn = extractLastTurn(lines);
  const lastUser = lastTurn.user;
  const lastAssistant = lastTurn.assistant;

  debug('extracted:', {
    userLength: lastUser?.length || 0,
    assistantLength: lastAssistant?.length || 0,
    userPreview: lastUser?.slice(0, 100),
    assistantPreview: lastAssistant?.slice(0, 100)
  });

  // Run both in parallel with Promise.all
  const promises = [];
  const results = [];
  const skipped = [];

  // Check if user content is meaningful
  if (lastUser) {
    if (hasContent(lastUser)) {
      const len = lastUser.length;
      promises.push(
        addMemory({ content: lastUser, role: 'user', messageId: `u_${Date.now()}` })
          .then(r => results.push({ type: 'USER', len, ...r }))
          .catch(e => results.push({ type: 'USER', len, ok: false, error: e.message }))
      );
    } else {
      skipped.push({ type: 'USER', reason: 'whitespace-only content' });
    }
  }

  // Check if assistant content is meaningful
  if (lastAssistant) {
    if (hasContent(lastAssistant)) {
      const len = lastAssistant.length;
      promises.push(
        addMemory({ content: lastAssistant, role: 'assistant', messageId: `a_${Date.now()}` })
          .then(r => results.push({ type: 'ASSISTANT', len, ...r }))
          .catch(e => results.push({ type: 'ASSISTANT', len, ok: false, error: e.message }))
      );
    } else {
      skipped.push({ type: 'ASSISTANT', reason: 'whitespace-only content' });
    }
  }

  await Promise.all(promises);

  // Check if all calls succeeded
  const allSuccess = results.length > 0 && results.every(r => r.ok && !r.error);

  // Debug output
  debug('results:', results);
  debug('skipped:', skipped);

  // Build output message
  let output = '';

  if (allSuccess) {
    const details = results.map(r => `${r.type.toLowerCase()}: ${r.len}`).join(', ');
    output = `💾 Memory saved (${results.length}) [${details}]`;
    // Add skipped info if any
    if (skipped.length > 0) {
      output += `\n⏭️ Skipped: ${skipped.map(s => `${s.type} (${s.reason})`).join(', ')}`;
    }
    process.stdout.write(JSON.stringify({ systemMessage: output }));
    process.exit(0);
  } else if (results.length === 0 && skipped.length > 0) {
    // All content was skipped
    output = `⏭️ EverMem: No content to save\n`;
    for (const s of skipped) {
      output += `  • ${s.type}: ${s.reason}\n`;
    }
    process.stdout.write(JSON.stringify({ systemMessage: output }));
    process.exit(0);
  } else {
    // Failure: show detailed errors via systemMessage
    function truncateBody(body) {
      if (!body) return body;
      const copy = { ...body };
      if (copy.content && typeof copy.content === 'string' && copy.content.length > 100) {
        copy.content = copy.content.substring(0, 100) + '... [truncated]';
      }
      return copy;
    }

    output = '💾 EverMem: Save failed\n';
    for (const r of results) {
      if (r.error) {
        output += `${r.type}: ERROR - ${r.error}\n`;
      } else if (!r.ok) {
        output += `${r.type}: FAILED (${r.status})\n`;
        output += `Request: ${JSON.stringify(truncateBody(r.body), null, 2)}\n`;
        output += `Response: ${JSON.stringify(r.response, null, 2)}\n`;
      }
    }
    // Also show skipped if any
    if (skipped.length > 0) {
      output += `⏭️ Skipped: ${skipped.map(s => `${s.type} (${s.reason})`).join(', ')}\n`;
    }
    process.stdout.write(JSON.stringify({ systemMessage: output }));
  }

} catch (e) {
  // Silent on errors
  process.exit(0);
}
