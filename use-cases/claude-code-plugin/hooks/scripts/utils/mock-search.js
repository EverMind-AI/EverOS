/**
 * Simple substring search for mock memories
 * To be replaced with semantic retrieval in production
 */

const MAX_CANDIDATES = 15;

/**
 * @typedef {Object} Memory
 * @property {string} text - The memory content
 * @property {string} timestamp - ISO timestamp when memory was created
 */

/**
 * Search memories for matches to query terms
 * @param {string} query - User's prompt
 * @param {Memory[]} memories - Array of memory objects
 * @returns {Memory[]} Matching memories (max 15)
 */
export function searchMemories(query, memories) {
  if (!query || !memories || memories.length === 0) {
    return [];
  }

  // Split query into terms, filter out very short terms
  const queryTerms = query
    .toLowerCase()
    .split(/\s+/)
    .filter(term => term.length > 2);

  if (queryTerms.length === 0) {
    return [];
  }

  // Find memories that match any query term
  const matches = memories.filter(memory => {
    const memoryLower = memory.text.toLowerCase();
    return queryTerms.some(term => memoryLower.includes(term));
  });

  // Return up to MAX_CANDIDATES
  return matches.slice(0, MAX_CANDIDATES);
}

/**
 * Count words/tokens in a string (multilingual support)
 * - For CJK (Chinese/Japanese/Korean): counts each character as a token
 * - For other languages: counts space-separated words
 * - For mixed text: counts both
 * @param {string} text - Input text
 * @returns {number} Word/token count
 */
export function countWords(text) {
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
