/**
 * JSDoc type definitions for EverOS ContextEngine
 * This file contains type definitions used across all ContextEngine modules
 */

/**
 * @typedef {Object} EverOSConfig
 * @property {string} serverUrl - EverOS server URL (e.g., "http://localhost:1995")
 * @property {string} userId - User ID for memory storage
 * @property {string} groupId - Group ID for shared memory
 * @property {number} topK - Number of memories to retrieve
 * @property {string[]} memoryTypes - Memory types to retrieve (episodic_memory)
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
 * @property {EverOSConfig} pluginConfig - Plugin configuration
 */

/**
 * @typedef {Object} AssembleContext
 * @property {string} [prompt] - Current user prompt (passed by OpenClaw runtime since 2026.3.23)
 * @property {Array} messages - Full conversation history
 * @property {string} [sessionId] - Optional session identifier
 * @property {string} [sessionKey] - Session key for state management
 * @property {number} [tokenBudget] - Token budget for assembled context
 * @property {string} [model] - Current model identifier
 */

/**
 * @typedef {Object} AssembleResult
 * @property {Array} messages - Ordered messages to use as model context
 * @property {number} estimatedTokens - Estimated total tokens in assembled context
 * @property {string} [systemPromptAddition] - Context-engine-provided instructions prepended to system prompt
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
 * @typedef {Object} ParsedMemoryResponse
 * @property {Array<{text: string, timestamp: number|string|null}>} episodic - Episodic memories
 * @property {Array<{text: string, timestamp: number|string|null}>} pending - Recent unconsolidated messages
 */

