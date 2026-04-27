/**
 * Shared debug utility for EverMem hooks
 *
 * Usage:
 *   import { debug, setDebugPrefix } from './utils/debug.js';
 *   setDebugPrefix('inject');  // Optional: add prefix to log lines
 *   debug('hookInput:', data);
 *
 * Enable by setting EVERMEM_DEBUG=1 in .env file or environment
 * Logs are written to /tmp/evermem-debug.log
 */

import { appendFileSync } from 'fs';
import { isConfigured } from './config.js';  // This loads .env

const DEBUG_LOG_PATH = '/tmp/evermem-debug.log';

// Check debug flag (after config.js loads .env)
const DEBUG = process.env.EVERMEM_DEBUG === '1';

// Optional prefix for log lines (e.g., 'inject' or 'store')
let debugPrefix = '';

/**
 * Set a prefix for debug log lines
 * @param {string} prefix - Prefix to add (e.g., 'inject', 'store')
 */
export function setDebugPrefix(prefix) {
  debugPrefix = prefix ? `[${prefix}] ` : '';
}

/**
 * Write debug message to log file
 * Only writes when EVERMEM_DEBUG=1
 *
 * @param {...any} args - Arguments to log (objects are JSON stringified)
 */
export function debug(...args) {
  if (!DEBUG) return;

  const msg = args.map(a =>
    typeof a === 'object' ? JSON.stringify(a, null, 2) : a
  ).join(' ');

  const timestamp = new Date().toISOString();
  const line = `[${timestamp}] ${debugPrefix}${msg}\n`;

  try {
    appendFileSync(DEBUG_LOG_PATH, line);
  } catch (e) {
    // Silent on write errors
  }
}

/**
 * Check if debug mode is enabled
 * @returns {boolean}
 */
export function isDebugEnabled() {
  return DEBUG;
}