#!/usr/bin/env bun

/**
 * Get Memories Script for EverMind Cloud API
 *
 * Lists memories stored in EverMind Cloud with pagination support.
 *
 * Usage:
 *   bun run get-memories-cloud --api-key <key>
 *   bun run get-memories-cloud --api-key <key> --group-id asoiaf --page 1 --page-size 10
 */

import { parseArgs } from 'util';

// ============================================================================
// Types
// ============================================================================

interface CliArgs {
  apiKey: string;
  apiUrl: string;
  groupId: string;
  page: number;
  pageSize: number;
  memoryType: string | null;
  startTime: string | null;
  endTime: string | null;
  allPages: boolean;
  json: boolean;
}

interface MemoryItem {
  memory_type: string;
  summary?: string | null;
  subject?: string | null;
  episode?: string | null;
  user_id?: string;
  timestamp?: string;
  group_id?: string | null;
  group_name?: string | null;
  keywords?: string[] | null;
  linked_entities?: string[] | null;
  [key: string]: unknown;
}

interface GetMemoriesResponse {
  status: string;
  message?: string;
  result: {
    memories: MemoryItem[];
    total_count: number;
    count: number;
    metadata?: unknown;
  };
}

// ============================================================================
// CLI Argument Parsing
// ============================================================================

function parseCliArgs(): CliArgs | null {
  try {
    const { values } = parseArgs({
      options: {
        'api-key': { type: 'string' },
        'api-url': { type: 'string', default: 'https://api.evermind.ai' },
        'group-id': { type: 'string', default: 'asoiaf' },
        'page': { type: 'string', default: '1' },
        'page-size': { type: 'string', default: '20' },
        'memory-type': { type: 'string' },
        'start-time': { type: 'string' },
        'end-time': { type: 'string' },
        'all': { type: 'boolean', default: false },
        'json': { type: 'boolean', default: false },
        help: { type: 'boolean', default: false },
      },
      strict: true,
      allowPositionals: false,
    });

    if (values.help) {
      printHelp();
      return null;
    }

    const apiKey = values['api-key'] as string || process.env.EVERMIND_API_KEY || '';
    if (!apiKey) {
      console.error('Error: API key required. Use --api-key or set EVERMIND_API_KEY environment variable\n');
      printHelp();
      process.exit(1);
    }

    return {
      apiKey,
      apiUrl: values['api-url'] as string,
      groupId: values['group-id'] as string,
      page: parseInt(values['page'] as string, 10),
      pageSize: parseInt(values['page-size'] as string, 10),
      memoryType: (values['memory-type'] as string) || null,
      startTime: (values['start-time'] as string) || null,
      endTime: (values['end-time'] as string) || null,
      allPages: values['all'] as boolean,
      json: values['json'] as boolean,
    };
  } catch (error) {
    console.error('Error parsing arguments:', error instanceof Error ? error.message : String(error));
    console.error('');
    printHelp();
    process.exit(1);
  }
}

function printHelp(): void {
  console.log(`
Get Memories Script for EverMind Cloud API

Lists memories stored in EverMind Cloud with pagination support.

Usage:
  bun run get-memories-cloud --api-key <key> [options]

Required:
  --api-key <key>         EverMind API key (or set EVERMIND_API_KEY env var)

Options:
  --api-url <url>         EverMind API URL (default: https://api.evermind.ai)
  --group-id <id>         Group ID to query (default: asoiaf)
  --page <num>            Page number (default: 1)
  --page-size <num>       Results per page, 1-100 (default: 20)
  --memory-type <type>    Filter by type: profile, episodic_memory, foresight, event_log
  --start-time <iso>      Filter start time (ISO 8601 with timezone)
  --end-time <iso>        Filter end time (ISO 8601 with timezone)
  --all                   Fetch all pages (overrides --page)
  --json                  Output raw JSON response
  --help                  Show this help message

Examples:
  bun run get-memories-cloud --api-key YOUR_KEY
  bun run get-memories-cloud --api-key YOUR_KEY --page-size 5 --all
  bun run get-memories-cloud --api-key YOUR_KEY --memory-type profile
  bun run get-memories-cloud --api-key YOUR_KEY --json
`);
}

// ============================================================================
// EverMind Cloud API
// ============================================================================

async function getMemories(
  apiUrl: string,
  apiKey: string,
  params: {
    groupId: string;
    page: number;
    pageSize: number;
    memoryType: string | null;
    startTime: string | null;
    endTime: string | null;
  }
): Promise<GetMemoriesResponse> {
  const queryParams = new URLSearchParams({
    group_ids: params.groupId,
    page: params.page.toString(),
    page_size: params.pageSize.toString(),
  });

  if (params.memoryType) {
    queryParams.set('memory_type', params.memoryType);
  }
  if (params.startTime) {
    queryParams.set('start_time', params.startTime);
  }
  if (params.endTime) {
    queryParams.set('end_time', params.endTime);
  }

  const response = await fetch(`${apiUrl}/api/v0/memories?${queryParams}`, {
    method: 'GET',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
    },
    signal: AbortSignal.timeout(15000),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`HTTP ${response.status}: ${errorText}`);
  }

  return await response.json() as GetMemoriesResponse;
}

// ============================================================================
// Display
// ============================================================================

function displayMemory(memory: MemoryItem, index: number): void {
  const type = memory.memory_type || 'unknown';
  const subject = memory.subject || memory.summary || '(no subject)';
  const timestamp = memory.timestamp ? new Date(memory.timestamp).toLocaleString() : 'N/A';

  console.log(`  ${index}. [${type}] ${subject}`);

  if (memory.summary && memory.summary !== memory.subject) {
    const summary = memory.summary.length > 120
      ? memory.summary.slice(0, 120) + '...'
      : memory.summary;
    console.log(`     Summary: ${summary}`);
  }

  if (memory.keywords && memory.keywords.length > 0) {
    console.log(`     Keywords: ${memory.keywords.join(', ')}`);
  }

  console.log(`     Time: ${timestamp} | Group: ${memory.group_id || 'N/A'}`);
  console.log('');
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  const args = parseCliArgs();

  if (!args) {
    return;
  }

  if (!args.json) {
    console.log('');
    console.log('='.repeat(60));
    console.log('EverMind Cloud - Get Memories');
    console.log('='.repeat(60));
    console.log(`API: ${args.apiUrl}`);
    console.log(`Key: ${args.apiKey.slice(0, 8)}...${args.apiKey.slice(-4)}`);
    console.log(`Group: ${args.groupId}`);
    if (args.memoryType) {
      console.log(`Type filter: ${args.memoryType}`);
    }
    console.log('');
  }

  let totalFetched = 0;
  let currentPage = args.page;
  let totalCount = 0;

  do {
    try {
      const data = await getMemories(args.apiUrl, args.apiKey, {
        groupId: args.groupId,
        page: currentPage,
        pageSize: args.pageSize,
        memoryType: args.memoryType,
        startTime: args.startTime,
        endTime: args.endTime,
      });

      if (args.json) {
        console.log(JSON.stringify(data, null, 2));
        if (!args.allPages) break;
      }

      if (data.status !== 'ok') {
        console.error(`API error: ${data.message || 'Unknown error'}`);
        process.exit(1);
      }

      totalCount = data.result.total_count;
      const memories = data.result.memories;

      if (!args.json) {
        if (currentPage === args.page) {
          console.log(`Total memories: ${totalCount}`);
          console.log('');
        }

        if (memories.length === 0) {
          if (currentPage === args.page) {
            console.log('No memories found.');
          }
          break;
        }

        console.log(`--- Page ${currentPage} (${memories.length} results) ---\n`);

        for (let i = 0; i < memories.length; i++) {
          const globalIndex = (currentPage - 1) * args.pageSize + i + 1;
          displayMemory(memories[i], globalIndex);
        }
      }

      totalFetched += memories.length;

      if (!args.allPages || totalFetched >= totalCount) {
        break;
      }

      currentPage++;
    } catch (error) {
      console.error(`\nError fetching memories:`, error instanceof Error ? error.message : String(error));
      process.exit(1);
    }
  } while (args.allPages);

  if (!args.json) {
    console.log('='.repeat(60));
    console.log(`Fetched ${totalFetched} of ${totalCount} memories`);

    if (!args.allPages && totalFetched < totalCount) {
      const totalPages = Math.ceil(totalCount / args.pageSize);
      console.log(`Page ${args.page} of ${totalPages}. Use --all to fetch all pages.`);
    }

    console.log('='.repeat(60));
    console.log('');
  }
}

main().catch((error) => {
  console.error('\nUnexpected error:', error);
  process.exit(1);
});
