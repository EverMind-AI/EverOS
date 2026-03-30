/**
 * Subagent Tracker Module
 * Tracks subagent lifecycles for hierarchical memory management
 */

/**
 * @typedef {import("./types.js").EverMemOSConfig} EverMemOSConfig
 * @typedef {import("./types.js").Logger} Logger
 * @typedef {import("./types.js").SubagentInfo} SubagentInfo
 */

/**
 * Handles subagent lifecycle tracking
 * Enables hierarchical memory management across parent and child agent conversations
 */
export class SubagentTracker {
  /**
   * @param {EverMemOSConfig} cfg
   * @param {Logger} logger
   */
  constructor(cfg, logger) {
    this.cfg = cfg;
    this.log = logger;

    /** @type {Map<string, SubagentInfo>} */
    this.activeSubagents = new Map();
  }

  /**
   * Register a new subagent
   * @param {string} subagentId
   * @param {Object} metadata
   * @param {string} [metadata.subagentType]
   * @param {number} [metadata.parentTurnCount]
   */
  register(subagentId, metadata = {}) {
    this.activeSubagents.set(subagentId, {
      startTime: Date.now(),
      ...metadata,
    });
    this.log.info(`[evermind-ai-everos] subagent tracker: registered ${subagentId}`);
  }

  /**
   * Unregister a subagent
   * @param {string} subagentId
   * @returns {boolean} - True if subagent was found and removed
   */
  unregister(subagentId) {
    const removed = this.activeSubagents.delete(subagentId);
    if (removed) {
      this.log.info(`[evermind-ai-everos] subagent tracker: unregistered ${subagentId}`);
    }
    return removed;
  }

  /**
   * Get active subagent count
   * @returns {number}
   */
  getActiveCount() {
    return this.activeSubagents.size;
  }

  /**
   * Check if a subagent is active
   * @param {string} subagentId
   * @returns {boolean}
   */
  isActive(subagentId) {
    return this.activeSubagents.has(subagentId);
  }

  /**
   * Get info for a specific subagent
   * @param {string} subagentId
   * @returns {SubagentInfo|undefined}
   */
  getInfo(subagentId) {
    return this.activeSubagents.get(subagentId);
  }

  /**
   * Get all active subagent IDs
   * @returns {string[]}
   */
  getActiveIds() {
    return Array.from(this.activeSubagents.keys());
  }

  /**
   * Clean up stale subagents (older than specified duration)
   * @param {number} maxAgeMs - Maximum age in milliseconds (default: 1 hour)
   * @returns {string[]} - List of cleaned up subagent IDs
   */
  cleanup(maxAgeMs = 60 * 60 * 1000) {
    const now = Date.now();
    const stale = [];

    for (const [id, info] of this.activeSubagents.entries()) {
      if (now - info.startTime > maxAgeMs) {
        stale.push(id);
      }
    }

    for (const id of stale) {
      this.unregister(id);
    }

    if (stale.length > 0) {
      this.log.info(`[evermind-ai-everos] subagent tracker: cleaned up ${stale.length} stale subagents`);
    }

    return stale;
  }
}
