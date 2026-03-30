/**
 * Context Assembler Module
 * Handles query-aware context assembly from EverMemOS memories
 */

import { searchMemories } from "./api.js";
import { buildMemoryPrompt, parseSearchResponse } from "./prompt.js";

/**
 * @typedef {import("./types.js").EverMemOSConfig} EverMemOSConfig
 * @typedef {import("./types.js").Logger} Logger
 * @typedef {import("./types.js").ParsedMemoryResponse} ParsedMemoryResponse
 */

/**
 * Handles query-aware context assembly
 * Retrieves relevant memories based on current query and conversation state
 */
export class ContextAssembler {
  /**
   * @param {EverMemOSConfig} cfg
   * @param {Logger} logger
   */
  constructor(cfg, logger) {
    this.cfg = cfg;
    this.log = logger;
  }

  /**
   * Assemble context from memories based on current query and conversation state
   * @param {string} query - Current user query
   * @param {Array} messages - Full conversation history
   * @param {number} turnCount - Current turn number
   * @returns {Promise<{context: string, memoryCount: number}>}
   */
  async assemble(query, messages, turnCount) {
    // Early turns: retrieve more context for grounding
    const earlyTurnMultiplier = turnCount <= 2 ? 2 : 1;
    const topK = Math.min(this.cfg.topK * earlyTurnMultiplier, 20);

    /** @type {Object} */
    const params = {
      query,
      user_id: this.cfg.userId,
      group_id: this.cfg.groupId || undefined,
      memory_types: this.cfg.memoryTypes,
      retrieve_method: this.cfg.retrieveMethod,
      top_k: topK,
    };

    /** @type {any} */
    const result = await searchMemories(this.cfg, params, this.log);
    /** @type {ParsedMemoryResponse} */
    const parsed = parseSearchResponse(result) || { episodic: [], traits: [], case: null, skill: null };

    // Count total memories (including pending messages)
    const memoryCount =
      (parsed.episodic?.length || 0) +
      (parsed.pending?.length || 0) +
      (parsed.traits?.length || 0) +
      (parsed.case ? 1 : 0) +
      (parsed.skill ? 1 : 0);

    const context = buildMemoryPrompt(parsed, { wrapInCodeBlock: true });

    return { context, memoryCount };
  }

  /**
   * Build minimal context for subagents (smaller context window)
   * @param {string} query - Subagent query
   * @returns {Promise<string>}
   */
  async assembleForSubagent(query) {
    if (!query || query.length < 3) return "";

    const topK = Math.min(this.cfg.topK, 3);

    /** @type {Object} */
    const params = {
      query,
      user_id: this.cfg.userId,
      group_id: this.cfg.groupId || undefined,
      memory_types: this.cfg.memoryTypes,
      retrieve_method: this.cfg.retrieveMethod,
      top_k: topK,
    };

    /** @type {any} */
    const result = await searchMemories(this.cfg, params, this.log);
    /** @type {ParsedMemoryResponse} */
    const parsed = parseSearchResponse(result) || { episodic: [], traits: [], case: null, skill: null };

    // Use no code block for subagents (cleaner format)
    return buildMemoryPrompt(parsed, { wrapInCodeBlock: false });
  }
}
