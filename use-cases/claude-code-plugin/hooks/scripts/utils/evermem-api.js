/**
 * EverMem Cloud API client
 * Handles memory search and storage operations
 */

import { getConfig } from './config.js';
import { debug, setDebugPrefix } from './debug.js';

// Set debug prefix for this script
setDebugPrefix('EverMemAPI');
const TIMEOUT_MS = 30000; // 30 seconds

/**
 * Search memories from EverMem Cloud (v1)
 * @param {string} query - Search query text
 * @param {Object} options - Additional options
 * @param {number} options.topK - Max results (default: 10)
 * @param {string} options.retrieveMethod - Search method: keyword|vector|hybrid|agentic (default: 'hybrid')
 * @param {string[]} options.memoryTypes - Memory types (default: ['episodic_memory'])
 * @returns {Promise<Object>} Raw API response with _debug envelope
 */
export async function searchMemories(query, options = {}) {
  const config = getConfig();

  if (!config.isConfigured) {
    throw new Error('EverMem API key not configured');
  }

  const {
    topK = 10,
    retrieveMethod = 'hybrid',
    memoryTypes = ['episodic_memory']
  } = options;

  const url = `${config.apiBaseUrl}/api/v1/memories/search`;
  const filters = config.groupId
    ? { group_id: config.groupId }
    : { user_id: config.userId };

  const requestBody = {
    query,
    method: retrieveMethod,
    top_k: topK,
    memory_types: memoryTypes,
    filters
  };

  debug('searchMemories request body', requestBody);

  const debugEnvelope = {
    url,
    requestBody,
    apiKeyMasked: 'API_KEY_HIDDEN'
  };

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${config.apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestBody),
      signal: controller.signal
    });

    clearTimeout(timeoutId);

    const text = await response.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch {
      return { _debug: { ...debugEnvelope, status: response.status, rawBody: text, error: 'non-JSON response' } };
    }

    if (!response.ok) {
      return { _debug: { ...debugEnvelope, status: response.status, error: data } };
    }

    data._debug = debugEnvelope;
    return data;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      throw new Error(`API timeout after ${TIMEOUT_MS}ms`);
    }
    return { _debug: { ...debugEnvelope, error: error.message } };
  }
}

/**
 * Transform v1 search API response to plugin memory format.
 * v1 returns: { data: { episodes: [{ id, user_id, session_id, timestamp, summary, subject, score, participants, group_id? }], ... } }
 * @param {Object} apiResponse - Raw v1 API response
 * @returns {Object[]} Formatted memories sorted by score desc
 */
export function transformSearchResults(apiResponse) {
  const episodes = apiResponse?.data?.episodes;
  if (!Array.isArray(episodes)) {
    return [];
  }

  const memories = [];
  for (const ep of episodes) {
    const content = ep.summary || '';
    if (!content) continue;

    memories.push({
      text: content,
      subject: ep.subject || '',
      timestamp: ep.timestamp || new Date().toISOString(),
      memoryType: ep.memory_type || 'episodic_memory',
      score: ep.score || 0,
      metadata: {
        groupId: ep.group_id,
        type: ep.memory_type,
        participants: ep.participants
      }
    });
  }

  memories.sort((a, b) => b.score - a.score);
  return memories;
}


/**
 * Add a memory to EverMem Cloud (v1).
 * Uses /api/v1/memories/group when config.groupId is set, else /api/v1/memories (personal).
 * @param {Object} message - Message to store
 * @param {string} message.content - Message content
 * @param {string} message.role - 'user' or 'assistant'
 * @param {string} [message.messageId] - (unused in v1; accepted for backward compatibility)
 * @returns {Promise<Object>} Debug envelope { url, body, status, ok, response }
 */
export async function addMemory(message) {
  const config = getConfig();

  if (!config.isConfigured) {
    throw new Error('EverMem API key not configured');
  }

  const role = message.role === 'assistant' ? 'assistant' : 'user';
  const sender_id = role === 'assistant' ? 'claude-assistant' : config.userId;

  const baseMessage = {
    sender_id,
    role,
    timestamp: Date.now(),
    content: message.content
  };

  let url;
  let requestBody;

  if (config.groupId) {
    url = `${config.apiBaseUrl}/api/v1/memories/group`;
    requestBody = {
      group_id: config.groupId,
      messages: [baseMessage],
      async_mode: true
    };
  } else {
    url = `${config.apiBaseUrl}/api/v1/memories`;
    requestBody = {
      user_id: config.userId,
      messages: [baseMessage],
      async_mode: true
    };
  }

  let response, responseText, responseData, status, ok;

  try {
    response = await fetch(url, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${config.apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestBody)
    });
    status = response.status;
    ok = response.ok;
    responseText = await response.text();
    try {
      responseData = JSON.parse(responseText);
    } catch {}
  } catch (fetchError) {
    status = 0;
    ok = false;
    responseText = fetchError.message;
  }

  return {
    url,
    body: requestBody,
    status,
    ok,
    response: responseData || responseText
  };
}

/**
 * Get memories from EverMem Cloud (v1, ordered newest first by default).
 * @param {Object} options - Options
 * @param {number} options.page - Page number (default: 1)
 * @param {number} options.pageSize - Results per page (default: 100, max: 100)
 * @param {string} options.memoryType - Memory type filter (default: 'episodic_memory')
 * @returns {Promise<Object>} Raw v1 response { data: { episodes, total_count, count, ... } }
 */
export async function getMemories(options = {}) {
  const config = getConfig();

  if (!config.isConfigured) {
    throw new Error('EverMem API key not configured');
  }

  const {
    page = 1,
    pageSize = 100,
    memoryType = 'episodic_memory'
  } = options;

  const filters = config.groupId
    ? { group_id: config.groupId }
    : { user_id: config.userId };

  const url = `${config.apiBaseUrl}/api/v1/memories/get`;
  const requestBody = {
    memory_type: memoryType,
    filters,
    page,
    page_size: pageSize,
    rank_by: 'timestamp',
    rank_order: 'desc'
  };

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${config.apiKey}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(requestBody)
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API error ${response.status}: ${errorText}`);
  }

  return await response.json();
}

/**
 * Transform v1 getMemories response to simple format.
 * @param {Object} apiResponse - Raw v1 API response
 * @returns {Object[]} Formatted memories newest-first
 */
export function transformGetMemoriesResults(apiResponse) {
  const episodes = apiResponse?.data?.episodes;
  if (!Array.isArray(episodes)) {
    return [];
  }

  const memories = episodes.map(ep => ({
    text: ep.episode || ep.summary || '',
    subject: ep.subject || '',
    timestamp: ep.timestamp || new Date().toISOString(),
    groupId: ep.group_id
  })).filter(m => m.text);

  memories.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
  return memories;
}
