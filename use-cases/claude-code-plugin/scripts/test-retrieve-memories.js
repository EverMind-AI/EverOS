#!/usr/bin/env node

/**
 * Test script for retrieving memories from EverMem Cloud
 * Simulates what happens when user submits a prompt
 *
 * Usage:
 *   export EVERMEM_API_KEY="your-key"
 *   node scripts/test-retrieve-memories.js
 */

import { getConfig, isConfigured } from '../hooks/scripts/utils/config.js';
import { searchMemories, transformSearchResults } from '../hooks/scripts/utils/evermem-api.js';
import { formatRelativeTime } from '../hooks/scripts/utils/mock-store.js';

// Test queries that should match saved memories
const TEST_QUERIES = [
  "How do we handle authentication?",
  "What's our database setup?",
  "Tell me about rate limiting",
  "How are errors handled in the API?",
  "What was that N+1 query issue?",
  "JWT token configuration",
  "PostgreSQL connection pooling"
];

async function main() {
  console.log('🔍 EverMem Retrieve Memory Test\n');
  console.log('=' .repeat(60));

  // Check configuration
  if (!isConfigured()) {
    console.error('❌ EVERMEM_API_KEY not set');
    console.error('   Run: export EVERMEM_API_KEY="your-key"');
    process.exit(1);
  }

  const config = getConfig();
  console.log(`✓ API Key: ${config.apiKey.slice(0, 10)}...`);
  console.log(`✓ User ID: ${config.userId}`);
  console.log(`✓ Group ID: ${config.groupId}`);
  console.log(`✓ API URL: ${config.apiBaseUrl}`);
  console.log('=' .repeat(60));

  for (const query of TEST_QUERIES) {
    console.log(`\n📝 Query: "${query}"`);
    console.log('-'.repeat(60));

    try {
      // Call API exactly like inject-memories.js does
      const apiResponse = await searchMemories(query, {
        topK: 5,
        retrieveMethod: 'hybrid'
      });

      // Transform response exactly like inject-memories.js does
      const memories = transformSearchResults(apiResponse);

      if (memories.length === 0) {
        console.log('   No memories found');
        continue;
      }

      console.log(`   Found ${memories.length} memories:\n`);

      // Display like the plugin does
      for (let i = 0; i < Math.min(memories.length, 3); i++) {
        const memory = memories[i];
        const relTime = formatRelativeTime(memory.timestamp);
        const shortText = memory.text.length > 70
          ? memory.text.slice(0, 70) + '...'
          : memory.text;

        console.log(`   ${i + 1}. (${relTime}) [${memory.type}]`);
        console.log(`      "${shortText}"`);
        if (memory.score) {
          console.log(`      Score: ${memory.score.toFixed(3)}`);
        }
        console.log('');
      }

    } catch (error) {
      console.log(`   ❌ Error: ${error.message}`);
    }
  }

  console.log('\n' + '=' .repeat(60));
  console.log('\n✅ Retrieval test complete!');
  console.log('\nTo test in Claude Code:');
  console.log('   claude --plugin-dir /Users/hzh/code/memory-plugin');
}

main().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});
