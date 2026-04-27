/**
 * Groups Store - Local persistence for memory groups (JSONL format)
 *
 * Each groupId+keyId combination is stored only once (no duplicates).
 * Format: {"keyId":"...","groupId":"...","name":"...","path":"...","timestamp":"..."}
 *
 * keyId: SHA-256 hash (first 12 chars) of the API key - identifies which account owns this group
 */

import { readFileSync, appendFileSync, existsSync } from 'fs';
import { resolve, dirname, basename } from 'path';
import { fileURLToPath } from 'url';
import { getKeyId } from './config.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const GROUPS_FILE = resolve(__dirname, '../../../data/groups.jsonl');

/**
 * Check if the groupId+keyId combination already exists in the file
 * @param {string} groupId - The group ID to check
 * @param {string} keyId - The key ID (hashed API key) to check
 * @returns {boolean} True if already exists (should skip)
 */
function alreadyExists(groupId, keyId) {
  try {
    if (!existsSync(GROUPS_FILE)) {
      return false;
    }

    const content = readFileSync(GROUPS_FILE, 'utf8');
    const lines = content.trim().split('\n').filter(Boolean);

    for (const line of lines) {
      try {
        const entry = JSON.parse(line);
        // Match both groupId AND keyId (same project + same API key)
        if (entry.groupId === groupId && entry.keyId === keyId) {
          return true;
        }
      } catch {}
    }
    return false;
  } catch {
    return false;
  }
}

/**
 * Append a group entry to the JSONL file
 * Only records if the groupId+keyId combination doesn't already exist
 * @param {string} groupId - The group ID
 * @param {string} cwd - The working directory path
 * @returns {Object|null} The entry if saved, null if skipped or error
 */
export function saveGroup(groupId, cwd) {
  try {
    const keyId = getKeyId();

    // Skip if this groupId+keyId already exists
    if (alreadyExists(groupId, keyId)) {
      return null;
    }

    const entry = {
      keyId,  // Hashed API key identifier (null if not configured)
      groupId,
      name: basename(cwd),
      path: cwd,
      timestamp: new Date().toISOString()
    };
    appendFileSync(GROUPS_FILE, JSON.stringify(entry) + '\n', 'utf8');
    return entry;
  } catch (e) {
    // Silent on errors
    return null;
  }
}

/**
 * Load and aggregate groups from the JSONL file
 * @param {string} [filterKeyId] - Optional keyId to filter by (only show groups for this API key)
 * @returns {Array} Aggregated list of groups
 */
export function getGroups(filterKeyId = null) {
  try {
    if (!existsSync(GROUPS_FILE)) {
      return [];
    }

    const content = readFileSync(GROUPS_FILE, 'utf8');
    const lines = content.trim().split('\n').filter(Boolean);

    // Aggregate by groupId+keyId (composite key)
    const groupMap = new Map();

    for (const line of lines) {
      try {
        const entry = JSON.parse(line);

        // Skip if filtering by keyId and this entry doesn't match
        if (filterKeyId && entry.keyId !== filterKeyId) {
          continue;
        }

        // Use composite key: keyId:groupId (to separate same project under different accounts)
        const compositeKey = `${entry.keyId || 'none'}:${entry.groupId}`;
        const existing = groupMap.get(compositeKey);

        if (existing) {
          existing.sessionCount += 1;
          // Update lastSeen if this timestamp is newer
          if (entry.timestamp > existing.lastSeen) {
            existing.lastSeen = entry.timestamp;
          }
          // Update firstSeen if this timestamp is older
          if (entry.timestamp < existing.firstSeen) {
            existing.firstSeen = entry.timestamp;
          }
        } else {
          groupMap.set(compositeKey, {
            id: entry.groupId,
            keyId: entry.keyId || null,
            name: entry.name,
            path: entry.path,
            firstSeen: entry.timestamp,
            lastSeen: entry.timestamp,
            sessionCount: 1
          });
        }
      } catch {}
    }

    // Convert to array and sort by lastSeen (most recent first)
    return Array.from(groupMap.values()).sort((a, b) =>
      new Date(b.lastSeen).getTime() - new Date(a.lastSeen).getTime()
    );
  } catch (e) {
    return [];
  }
}

/**
 * Get groups for the current API key only
 * @returns {Array} Aggregated list of groups for current keyId
 */
export function getMyGroups() {
  const keyId = getKeyId();
  return getGroups(keyId);
}

/**
 * Get a specific group by ID (optionally filtered by current keyId)
 * @param {string} groupId - The group ID
 * @param {boolean} [filterByKey=true] - Whether to filter by current API key
 * @returns {Object|null} The group or null if not found
 */
export function getGroup(groupId, filterByKey = true) {
  const keyId = filterByKey ? getKeyId() : null;
  const groups = getGroups(keyId);
  return groups.find(g => g.id === groupId) || null;
}

/**
 * Load raw groups data (for backward compatibility)
 * @returns {Object} Groups data in old format
 */
export function loadGroups() {
  return { groups: getGroups() };
}

/**
 * Format relative time (e.g., "2h ago", "1d ago")
 * @param {string} isoTime - ISO timestamp
 * @returns {string} Relative time string
 */
export function formatRelativeTime(isoTime) {
  const now = Date.now();
  const then = new Date(isoTime).getTime();
  const diffMs = now - then;

  const minutes = Math.floor(diffMs / 60000);
  const hours = Math.floor(diffMs / 3600000);
  const days = Math.floor(diffMs / 86400000);

  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days < 30) return `${days}d ago`;
  return `${Math.floor(days / 30)}mo ago`;
}
