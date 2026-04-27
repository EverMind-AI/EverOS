/**
 * SDK-based filtering and summarization of memories
 * Uses Claude Agent SDK to intelligently filter and summarize relevant memories
 * Inherits authentication from Claude Code (no API key needed)
 */

import { query } from '@anthropic-ai/claude-agent-sdk';
import { execSync } from 'child_process';

const MAX_MEMORIES = 5;
const TIMEOUT_MS = 10000;

/**
 * @typedef {Object} Memory
 * @property {string} text - The memory content
 * @property {string} timestamp - ISO timestamp when memory was created
 */

/**
 * @typedef {Object} FilteredMemory
 * @property {string} text - Original memory text
 * @property {string} timestamp - Original timestamp
 * @property {string} type - Inferred type (decision, bug_fix, etc.)
 */

/**
 * Find the Claude Code executable path
 * @returns {string|null} Path to claude executable or null if not found
 */
function findClaudeExecutable() {
  try {
    // Try 'which' on Unix-like systems
    const result = execSync('which claude', { encoding: 'utf-8', stdio: ['pipe', 'pipe', 'pipe'] });
    return result.trim();
  } catch {
    try {
      // Try 'where' on Windows
      const result = execSync('where claude', { encoding: 'utf-8', stdio: ['pipe', 'pipe', 'pipe'] });
      return result.trim().split('\n')[0];
    } catch {
      return null;
    }
  }
}

/**
 * Filter memories using Claude Agent SDK
 * @param {string} prompt - User's current prompt
 * @param {Memory[]} candidates - Array of candidate memory objects
 * @returns {Promise<Object>} Filtered result with original memories and types
 */
export async function filterAndSummarize(prompt, candidates) {
  const claudePath = findClaudeExecutable();

  if (!claudePath) {
    throw new Error('Claude Code executable not found');
  }

  const systemPrompt = `You are a JSON-only memory filter. You MUST respond with ONLY a JSON object, nothing else. No explanations, no markdown, no text before or after the JSON. Just the raw JSON object starting with { and ending with }.`;

  const filterPrompt = `Filter these memories for relevance to the user's prompt.

USER PROMPT: "${prompt}"

CANDIDATE MEMORIES:
${candidates.map((c, i) => `[${i + 1}] ${c.text}`).join('\n\n')}

OUTPUT FORMAT (respond with ONLY this JSON, nothing else):
{"selected": [{"index": N, "type": "TYPE"}], "synthesis": "NARRATIVE"}

RULES:
- index is the memory number (1-based)
- type must be one of: decision, bug_fix, implementation, learning, preference
- Maximum ${MAX_MEMORIES} memories
- If no memories are relevant: {"selected": [], "synthesis": null}
- ONLY output the JSON object, no other text`;

  // Create abort controller for timeout
  const abortController = new AbortController();
  const timeoutId = setTimeout(() => abortController.abort(), TIMEOUT_MS);

  try {
    let responseText = '';

    // Use Agent SDK query with Claude Code executable
    const queryResult = query({
      prompt: filterPrompt,
      options: {
        pathToClaudeCodeExecutable: claudePath,
        model: 'claude-sonnet-4-20250514',
        systemPrompt,
        allowedTools: [],  // No tools needed for filtering
        abortController,
        maxTurns: 1  // Single turn only
      }
    });

    // Collect response text from the async generator
    for await (const message of queryResult) {
      if (message.type === 'assistant' && message.message?.content) {
        for (const block of message.message.content) {
          if (block.type === 'text') {
            responseText += block.text;
          }
        }
      }
    }

    clearTimeout(timeoutId);

    // Extract JSON from response (handle potential text wrapping)
    let jsonText = responseText.trim();

    // Remove markdown code block if present
    if (jsonText.includes('```')) {
      const match = jsonText.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
      if (match) {
        jsonText = match[1];
      }
    }

    // Try to find JSON object in the response
    const jsonMatch = jsonText.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      jsonText = jsonMatch[0];
    }

    // Parse JSON response
    const parsed = JSON.parse(jsonText);

    // Validate structure
    if (!parsed.selected || !Array.isArray(parsed.selected)) {
      throw new Error('Invalid response structure');
    }

    // Map back to original memories with type info
    const selected = parsed.selected
      .slice(0, MAX_MEMORIES)
      .map(item => {
        const originalMemory = candidates[item.index - 1];
        if (!originalMemory) return null;
        return {
          text: originalMemory.text,
          timestamp: originalMemory.timestamp,
          type: item.type || 'implementation'
        };
      })
      .filter(Boolean);

    return {
      selected,
      synthesis: parsed.synthesis
    };
  } catch (error) {
    clearTimeout(timeoutId);

    if (error.name === 'AbortError') {
      throw new Error('SDK timeout');
    }

    throw new Error(`SDK filter failed: ${error.message}`);
  }
}

/**
 * Create fallback result from raw candidates
 * @param {Memory[]} candidates - Raw memory candidates
 * @param {number} limit - Max memories to return
 * @returns {Object} Fallback result structure
 */
export function createFallbackResult(candidates, limit = 3) {
  const selected = candidates.slice(0, limit).map(memory => ({
    text: memory.text,
    timestamp: memory.timestamp,
    type: 'implementation'  // Default type
  }));

  return {
    selected,
    synthesis: null,
    isFallback: true
  };
}
