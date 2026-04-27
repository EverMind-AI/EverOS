/**
 * Terminal output formatting utilities
 */

import { formatRelativeTime } from './mock-store.js';

// Memory type emoji mapping
const TYPE_ICONS = {
  decision: { emoji: '\u{1F3AF}', ascii: '[DECISION]' },      // Target
  bug_fix: { emoji: '\u{1F41B}', ascii: '[BUG]' },            // Bug
  implementation: { emoji: '\u{1F527}', ascii: '[IMPL]' },    // Wrench
  learning: { emoji: '\u{1F4A1}', ascii: '[LEARN]' },         // Lightbulb
  preference: { emoji: '\u{2699}\u{FE0F}', ascii: '[PREF]' }  // Gear
};

/**
 * Detect if terminal likely supports Unicode
 * @returns {boolean}
 */
export function supportsUnicode() {
  const term = process.env.TERM || '';
  const lang = process.env.LANG || '';
  const lcAll = process.env.LC_ALL || '';

  // Check for UTF-8 in locale settings
  if (lang.includes('UTF-8') || lcAll.includes('UTF-8')) {
    return true;
  }

  // Check for modern terminal types
  if (term.includes('xterm') || term.includes('256color') || term.includes('kitty') || term.includes('alacritty')) {
    return true;
  }

  // Default to Unicode on macOS
  if (process.platform === 'darwin') {
    return true;
  }

  return false;
}

/**
 * Get icon for memory type
 * @param {string} type - Memory type
 * @param {boolean} useUnicode - Whether to use Unicode
 * @returns {string}
 */
export function getTypeIcon(type, useUnicode = true) {
  const icons = TYPE_ICONS[type] || TYPE_ICONS.implementation;
  return useUnicode ? icons.emoji : icons.ascii;
}

/**
 * Format the "Searching memories..." spinner
 * @returns {string}
 */
export function formatSpinner() {
  const useUnicode = supportsUnicode();
  const icon = useUnicode ? '\u23F3' : '[...]';  // Hourglass
  return `${icon} Searching memories...\n`;
}

/**
 * @typedef {Object} FilteredMemory
 * @property {string} text - Original memory text
 * @property {string} timestamp - ISO timestamp
 * @property {string} type - Memory type
 */

/**
 * Format the memory summary box with original memories and timestamps
 * @param {Object} result - SDK filter result
 * @param {FilteredMemory[]} result.selected - Selected memories
 * @param {string} result.synthesis - SDK synthesis
 * @param {number} rawCount - Number of raw candidates
 * @param {number} filteredCount - Number after filtering
 * @returns {string}
 */
export function formatSummaryBox(result, rawCount, filteredCount) {
  const useUnicode = supportsUnicode();
  const divider = useUnicode ? '\u2500'.repeat(50) : '-'.repeat(50);

  let output = '\n';
  output += useUnicode ? '\u{1F4AD} Memory Retrieved\n' : '=== Memory Retrieved ===\n';
  output += divider + '\n';

  // Individual memories with original text and timestamp
  for (let i = 0; i < result.selected.length; i++) {
    const memory = result.selected[i];
    const icon = getTypeIcon(memory.type, useUnicode);
    const relativeTime = formatRelativeTime(memory.timestamp);

    output += `${icon} (${relativeTime}) ${memory.text.slice(0, 80)}...\n`;
  }

  output += divider + '\n';
  output += `${filteredCount} memories recalled\n`;

  return output;
}

/**
 * Format "No relevant memories" message
 * @returns {string}
 */
export function formatNoMemories() {
  const useUnicode = supportsUnicode();
  const icon = useUnicode ? '\u{1F4AD}' : '===';

  return `\n${icon} Memory Retrieved: No relevant memories found\n`;
}

/**
 * Format error message
 * @param {string} message - Error message
 * @returns {string}
 */
export function formatError(message) {
  const useUnicode = supportsUnicode();
  const icon = useUnicode ? '\u26A0\u{FE0F}' : '[!]';

  return `\n${icon} Memory Retrieved: ${message}\n   Continuing without memory context\n`;
}

/**
 * Format fallback summary (when SDK fails)
 * @param {FilteredMemory[]} memories - Memory objects with text, timestamp, type
 * @param {number} rawCount - Total raw candidates
 * @returns {string}
 */
export function formatFallbackSummary(memories, rawCount) {
  const useUnicode = supportsUnicode();
  const divider = useUnicode ? '\u2500'.repeat(50) : '-'.repeat(50);

  let output = '\n';
  output += useUnicode ? '\u{1F4AD} Memory Retrieved (Fallback)\n' : '=== Memory Retrieved (Fallback) ===\n';
  output += divider + '\n';

  for (let i = 0; i < memories.length; i++) {
    const memory = memories[i];
    const icon = getTypeIcon(memory.type, useUnicode);
    const relativeTime = formatRelativeTime(memory.timestamp);

    output += `${icon} (${relativeTime}) ${memory.text.slice(0, 80)}...\n`;
  }

  output += divider + '\n';
  output += `Showing top ${memories.length} matches (SDK unavailable)\n`;

  return output;
}
