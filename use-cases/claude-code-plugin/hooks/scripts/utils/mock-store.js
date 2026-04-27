import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const DATA_PATH = join(__dirname, '..', '..', '..', 'data', 'mock-memories.json');

let memoriesCache = null;

/**
 * @typedef {Object} Memory
 * @property {string} text - The memory content
 * @property {string} timestamp - ISO timestamp when memory was created
 */

/**
 * Load mock memories from JSON file
 * @returns {Memory[]} Array of memory objects with text and timestamp
 */
export function loadMemories() {
  if (memoriesCache !== null) {
    return memoriesCache;
  }

  try {
    const data = readFileSync(DATA_PATH, 'utf-8');
    const parsed = JSON.parse(data);
    memoriesCache = parsed.memories || [];
    return memoriesCache;
  } catch (error) {
    console.error(`[Memory Plugin] Failed to load memories: ${error.message}`);
    return [];
  }
}

/**
 * Format a timestamp as relative time (e.g., "2h ago", "3 days ago")
 * @param {string} isoTimestamp - ISO timestamp string
 * @returns {string} Relative time string
 */
export function formatRelativeTime(isoTimestamp) {
  const now = new Date();
  const then = new Date(isoTimestamp);
  const diffMs = now - then;

  const seconds = Math.floor(diffMs / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);
  const weeks = Math.floor(days / 7);
  const months = Math.floor(days / 30);

  if (months > 0) {
    return months === 1 ? '1 month ago' : `${months} months ago`;
  }
  if (weeks > 0) {
    return weeks === 1 ? '1 week ago' : `${weeks} weeks ago`;
  }
  if (days > 0) {
    return days === 1 ? '1 day ago' : `${days} days ago`;
  }
  if (hours > 0) {
    return hours === 1 ? '1 hour ago' : `${hours}h ago`;
  }
  if (minutes > 0) {
    return minutes === 1 ? '1 min ago' : `${minutes}m ago`;
  }
  return 'just now';
}
