/**
 * Configuration loader for EverMem plugin
 * Reads settings from .env file and environment variables
 */

import { readFileSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { createHash } from 'crypto';

// Load .env file from plugin root
const __dirname = dirname(fileURLToPath(import.meta.url));
const envPath = resolve(__dirname, '../../../.env');

if (existsSync(envPath)) {
  const envContent = readFileSync(envPath, 'utf8');
  for (const line of envContent.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const [key, ...valueParts] = trimmed.split('=');
    if (key && valueParts.length > 0) {
      const value = valueParts.join('=').replace(/^["']|["']$/g, '');
      if (!process.env[key]) {  // Don't override existing env vars
        process.env[key] = value;
      }
    }
  }
}

const API_BASE_URL = 'https://api.evermind.ai';

/**
 * Get the EverMem API key from environment
 * @returns {string|null} API key or null if not set
 */
export function getApiKey() {
  return process.env.EVERMEM_API_KEY || null;
}

/**
 * Get the user ID for memory operations
 * Defaults to 'claude-code-user' if not set
 * @returns {string} User ID
 */
export function getUserId() {
  return process.env.EVERMEM_USER_ID || 'claude-code-user';
}

/**
 * Get the group ID for memory operations
 * Uses project working directory as default group
 * Format: {project_name_prefix_4}{path_hash_5} = 9 chars max
 * @returns {string} Group ID
 */
export function getGroupId() {
  if (process.env.EVERMEM_GROUP_ID) {
    return process.env.EVERMEM_GROUP_ID;
  }
  // Use EVERMEM_CWD (set from hook input) or fall back to process.cwd()
  const cwd = process.env.EVERMEM_CWD || process.cwd();

  // Extract project name (last part of path)
  const projectName = cwd.split('/').filter(Boolean).pop() || 'proj';
  // Take first 4 chars of project name (lowercase, alphanumeric only)
  const namePrefix = projectName.toLowerCase().replace(/[^a-z0-9]/g, '').substring(0, 4) || 'proj';

  // Hash the full path and take first 5 chars
  const pathHash = createHash('sha256').update(cwd).digest('hex').substring(0, 5);

  // Combine: 4 chars name + 5 chars hash = 9 chars
  return `${namePrefix}${pathHash}`;
}

/**
 * Get the API base URL
 * @returns {string} Base URL
 */
export function getApiBaseUrl() {
  return process.env.EVERMEM_API_URL || API_BASE_URL;
}

/**
 * Check if the plugin is properly configured
 * @returns {boolean} True if API key is set
 */
export function isConfigured() {
  return !!getApiKey();
}

/**
 * Get a hashed identifier for the API key (for local storage association)
 * Uses SHA-256 hash, truncated to 12 characters for compactness
 * @returns {string|null} Key ID (first 12 chars of SHA-256 hash) or null if no API key
 */
export function getKeyId() {
  const apiKey = getApiKey();
  if (!apiKey) {
    return null;
  }
  const hash = createHash('sha256').update(apiKey).digest('hex');
  return hash.substring(0, 12);
}

/**
 * Get full configuration object
 * @returns {Object} Configuration
 */
export function getConfig() {
  return {
    apiKey: getApiKey(),
    userId: getUserId(),
    groupId: getGroupId(),
    apiBaseUrl: getApiBaseUrl(),
    isConfigured: isConfigured()
  };
}
