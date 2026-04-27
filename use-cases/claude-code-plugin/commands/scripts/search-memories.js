#!/usr/bin/env node

/**
 * Search memories from EverMem Cloud
 * Usage: node search-memories.js "query string"
 */

import { getConfig, isConfigured } from '../../hooks/scripts/utils/config.js';
import { searchMemories, transformSearchResults } from '../../hooks/scripts/utils/evermem-api.js';

const query = process.argv[2] || '';

if (!query) {
  console.log('Usage: /evermem:search <query>');
  console.log('Example: /evermem:search "how do we handle authentication"');
  process.exit(0);
}

if (!isConfigured()) {
  console.log('Error: EVERMEM_API_KEY not configured');
  console.log('Set it with: export EVERMEM_API_KEY="your-key"');
  process.exit(1);
}

async function main() {
  try {
    const config = getConfig();
    console.log('Searching EverMem Cloud...\n');
    console.log(`Query: "${query}"`);
    console.log(`User: ${config.userId}`);
    console.log(`Group: ${config.groupId}`);
    console.log('');

    const apiResponse = await searchMemories(query, {
      topK: 10,
      retrieveMethod: 'hybrid'
    });

    // Debug: show raw API response
    console.log('--- RAW API RESPONSE ---');
    console.log(JSON.stringify(apiResponse, null, 2));
    console.log('--- END RAW RESPONSE ---\n');

    const memories = transformSearchResults(apiResponse);

    if (memories.length === 0) {
      console.log('No memories found matching your query.');
      process.exit(0);
    }

    console.log(`Found ${memories.length} memories:`);
    console.log('='.repeat(70));

    for (let i = 0; i < memories.length; i++) {
      const m = memories[i];
      const score = m.score ? `${(m.score * 100).toFixed(1)}%` : 'N/A';
      const date = new Date(m.timestamp).toLocaleDateString();
      const time = new Date(m.timestamp).toLocaleTimeString();

      console.log('');
      console.log(`${i + 1}. [Score: ${score}] ${date} ${time}`);
      console.log('-'.repeat(70));

      // Word wrap the content
      const words = m.text.split(' ');
      let line = '';
      for (const word of words) {
        if ((line + ' ' + word).length > 70) {
          console.log(line.trim());
          line = word;
        } else {
          line += ' ' + word;
        }
      }
      if (line.trim()) {
        console.log(line.trim());
      }
    }

    console.log('');
    console.log('='.repeat(70));

  } catch (error) {
    console.log(`Error: ${error.message}`);
    process.exit(1);
  }
}

main();
