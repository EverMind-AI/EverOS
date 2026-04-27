#!/usr/bin/env node

/**
 * EverMem MCP Server
 * Exposes memory search tool for Claude to find relevant context from past sessions
 */

import { createInterface } from 'readline';
import { searchMemories, transformSearchResults } from '../hooks/scripts/utils/evermem-api.js';
import { getConfig } from '../hooks/scripts/utils/config.js';

// Tool definitions - following claude-mem's concise pattern
const TOOLS = [
  {
    name: 'evermem_search',
    description: 'Search past conversation memories. Returns summaries with dates and relevance scores. Use when user asks about previous work, decisions, or context from past sessions. Params: query (required), limit (default: 10, max: 20)',
    inputSchema: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: 'Search query - use keywords, topics, or questions'
        },
        limit: {
          type: 'number',
          description: 'Max results to return (default: 10, max: 20)'
        }
      },
      required: ['query']
    }
  }
];

/**
 * Format date as relative time (e.g., "2 days ago", "today")
 */
function formatRelativeDate(timestamp) {
  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now - date;
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffDays === 0) return 'today';
  if (diffDays === 1) return 'yesterday';
  if (diffDays < 7) return `${diffDays} days ago`;
  if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
  return date.toLocaleDateString();
}

/**
 * Handle evermem_search tool call
 */
async function handleSearch(args) {
  const config = getConfig();

  if (!config.isConfigured) {
    return {
      isError: true,
      content: [{ type: 'text', text: 'EverMem API key not configured. Set EVERMEM_API_KEY environment variable.' }]
    };
  }

  const query = args.query;
  if (!query) {
    return {
      isError: true,
      content: [{ type: 'text', text: 'Missing required parameter: query' }]
    };
  }

  const limit = Math.min(args.limit || 10, 20);

  try {
    const response = await searchMemories(query, { topK: limit });
    const memories = transformSearchResults(response);

    if (memories.length === 0) {
      return {
        content: [{ type: 'text', text: `No memories found for: "${query}"` }]
      };
    }

    // Format as compact table (token-efficient like claude-mem)
    const header = '| # | Score | Date | Summary |';
    const separator = '|---|-------|------|---------|';

    const rows = memories.map((mem, i) => {
      const score = Math.round(mem.score * 100);
      const date = formatRelativeDate(mem.timestamp);
      // Use full subject field
      const summary = (mem.subject || mem.text.substring(0, 150)).replace(/\|/g, '/').replace(/\n/g, ' ');
      return `| ${i + 1} | ${score}% | ${date} | ${summary} |`;
    });

    const table = [header, separator, ...rows].join('\n');

    // Add context about what was found
    const resultText = `Found ${memories.length} memories for "${query}":\n\n${table}\n\nTo get full content of a specific memory, ask me to elaborate on that topic.`;

    return {
      content: [{ type: 'text', text: resultText }]
    };
  } catch (error) {
    return {
      isError: true,
      content: [{ type: 'text', text: `Search error: ${error.message}` }]
    };
  }
}

/**
 * Handle incoming JSON-RPC request
 */
async function handleRequest(request) {
  const { id, method, params } = request;

  switch (method) {
    case 'initialize':
      return {
        jsonrpc: '2.0',
        id,
        result: {
          protocolVersion: '2024-11-05',
          capabilities: {
            tools: {}
          },
          serverInfo: {
            name: 'evermem',
            version: '0.1.0'
          }
        }
      };

    case 'notifications/initialized':
      return null;

    case 'tools/list':
      return {
        jsonrpc: '2.0',
        id,
        result: {
          tools: TOOLS
        }
      };

    case 'tools/call':
      const { name, arguments: args } = params;
      let result;

      switch (name) {
        case 'evermem_search':
          result = await handleSearch(args || {});
          break;
        default:
          return {
            jsonrpc: '2.0',
            id,
            error: {
              code: -32601,
              message: `Unknown tool: ${name}`
            }
          };
      }

      return {
        jsonrpc: '2.0',
        id,
        result
      };

    default:
      return {
        jsonrpc: '2.0',
        id,
        error: {
          code: -32601,
          message: `Method not found: ${method}`
        }
      };
  }
}

/**
 * Main MCP server loop
 */
async function main() {
  const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false
  });

  rl.on('line', async (line) => {
    if (!line.trim()) return;

    try {
      const request = JSON.parse(line);
      const response = await handleRequest(request);

      if (response) {
        console.log(JSON.stringify(response));
      }
    } catch (error) {
      const errorResponse = {
        jsonrpc: '2.0',
        id: null,
        error: {
          code: -32700,
          message: `Parse error: ${error.message}`
        }
      };
      console.log(JSON.stringify(errorResponse));
    }
  });

  rl.on('close', () => {
    process.exit(0);
  });
}

main().catch(error => {
  console.error('MCP server error:', error);
  process.exit(1);
});
