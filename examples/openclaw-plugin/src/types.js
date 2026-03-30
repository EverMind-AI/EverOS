/**
 * JSDoc type definitions for EverMemOS ContextEngine
 * This file contains type definitions used across all ContextEngine modules
 */

/**
 * @typedef {Object} EverMemOSConfig
 * @property {string} serverUrl - EverMemOS server URL (e.g., "http://localhost:1995")
 * @property {string} userId - User ID for memory storage
 * @property {string} groupId - Group ID for shared memory
 * @property {number} topK - Number of memories to retrieve
 * @property {string[]} memoryTypes - Memory types to retrieve (episodic_memory, profile, agent_skill, agent_case)
 * @property {string} retrieveMethod - Retrieval strategy (keyword, vector, hybrid, agentic)
 */

/**
 * @typedef {Object} Logger
 * @property {(...args: any[]) => void} log - Info level logging
 * @property {(...args: any[]) => void} warn - Warning level logging
 * @property {(...args: any[]) => void} error - Error level logging
 */

/**
 * @typedef {Object} BootstrapContext
 * @property {Object} api - OpenClaw API object
 * @property {EverMemOSConfig} pluginConfig - Plugin configuration
 */

/**
 * @typedef {Object} AssembleContext
 * @property {string|Array} prompt - Current user prompt (can be string or content blocks)
 * @property {Array} messages - Full conversation history
 * @property {string} [sessionId] - Optional session identifier
 */

/**
 * @typedef {Object} AssembleResult
 * @property {string} context - Formatted context string to prepend
 * @property {Object} metadata - Metadata about the assembly
 * @property {number} [metadata.memoryCount] - Number of memories retrieved
 * @property {string} [metadata.retrieveMethod] - Method used for retrieval
 * @property {number} [metadata.turnCount] - Current turn number
 * @property {string} [metadata.skipped] - Reason if assembly was skipped
 * @property {string} [metadata.error] - Error message if failed
 */

/**
 * @typedef {Object} AfterTurnContext
 * @property {Array} messages - Messages from the completed turn
 * @property {boolean} success - Whether the turn completed successfully
 * @property {string} [errorMessage] - Error message if turn failed
 */

/**
 * @typedef {Object} CompactContext
 * @property {Array} messages - Current session messages
 * @property {number} tokenCount - Estimated token count of context
 * @property {string} [sessionId] - Optional session identifier
 */

/**
 * @typedef {Object} CompactResult
 * @property {boolean} shouldCompact - Whether compaction is recommended
 * @property {string} reason - Explanation of the decision
 * @property {Object} [metadata] - Additional metadata
 * @property {string} [metadata.memoryStrategy] - Suggested memory consolidation strategy
 * @property {number} [metadata.turnCount] - Turn count at evaluation time
 */

/**
 * @typedef {Object} PrepareSubagentContext
 * @property {string} subagentId - Unique identifier for the subagent
 * @property {string|Array} prompt - Prompt for the subagent
 * @property {Array} parentMessages - Parent agent's conversation history
 * @property {string} [subagentType] - Type of subagent being spawned
 */

/**
 * @typedef {Object} PrepareSubagentResult
 * @property {string} prependContext - Context to prepend to subagent prompt
 * @property {Object} metadata - Metadata about the preparation
 * @property {string} [metadata.subagentId] - Subagent identifier
 * @property {number} [metadata.parentTurnCount] - Parent's turn count
 */

/**
 * @typedef {Object} SubagentEndedContext
 * @property {string} subagentId - Unique identifier for the subagent
 * @property {Array} messages - Messages from the subagent conversation
 * @property {boolean} success - Whether the subagent completed successfully
 * @property {string} [errorMessage] - Error message if subagent failed
 */

/**
 * @typedef {Object} ParsedMemoryResponse
 * @property {Array<{text: string, timestamp: number|string|null}>} episodic - Episodic memories
 * @property {Array<{text: string, kind: string}>} traits - Profile traits
 * @property {Object|null} case - Top agent case memory
 * @property {Object|null} skill - Top agent skill memory
 */

/**
 * @typedef {Object} SubagentInfo
 * @property {number} startTime - Subagent spawn timestamp
 * @property {string} [subagentType] - Type of subagent
 * @property {number} [parentTurnCount] - Parent's turn count at spawn
 */
