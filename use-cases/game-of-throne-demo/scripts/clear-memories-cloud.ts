#!/usr/bin/env bun

/**
 * Clear Memories Script for EverMind Cloud API
 *
 * Deletes all memories from EverMind Cloud and cleans up progress files.
 *
 * Usage:
 *   bun run clear-memories-cloud --api-key <key>
 *   bun run clear-memories-cloud --api-key <key> --dry-run
 */

import { parseArgs } from 'util';
import { existsSync, readdirSync, unlinkSync } from 'fs';
import { resolve, basename } from 'path';

// ============================================================================
// Types
// ============================================================================

interface CliArgs {
  apiKey: string;
  apiUrl: string;
  groupId: string;
  deleteAll: boolean;
  dryRun: boolean;
  keepProgress: boolean;
}

interface DeleteResponse {
  status: string;
  message: string;
  result?: {
    filters: string[];
    count: number;
  };
}

interface DeleteResult {
  success: boolean;
  message: string;
  count: number;
  notFound: boolean;
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
        'delete-all': { type: 'boolean', default: false },
        'dry-run': { type: 'boolean', default: false },
        'keep-progress': { type: 'boolean', default: false },
        help: { type: 'boolean', default: false },
      },
      strict: true,
      allowPositionals: false,
    });

    if (values.help) {
      printHelp();
      return null;
    }

    // API key from argument or environment variable
    const apiKey = values['api-key'] as string || process.env.EVERMIND_API_KEY || '';
    if (!apiKey) {
      console.error('❌ Error: API key required. Use --api-key or set EVERMIND_API_KEY environment variable\n');
      printHelp();
      process.exit(1);
    }

    const deleteAll = values['delete-all'] as boolean;

    return {
      apiKey,
      apiUrl: values['api-url'] as string,
      groupId: deleteAll ? '__all__' : values['group-id'] as string,
      deleteAll,
      dryRun: values['dry-run'] as boolean,
      keepProgress: values['keep-progress'] as boolean,
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
Clear Memories Script for EverMind Cloud API

Deletes all memories from EverMind Cloud and cleans up progress files.

Usage:
  bun run clear-memories-cloud --api-key <key> [options]

Required:
  --api-key <key>       EverMind API key (or set EVERMIND_API_KEY env var)

Options:
  --api-url <url>       EverMind API URL (default: https://api.evermind.ai)
  --group-id <id>       Group ID to delete memories for (default: asoiaf)
  --delete-all          Delete ALL memories (sets group_id to "__all__")
  --dry-run             Show what would be deleted without actually deleting
  --keep-progress       Keep progress files, only delete memories from cloud
  --help                Show this help message

Examples:
  bun run clear-memories-cloud --api-key YOUR_KEY
  bun run clear-memories-cloud --api-key YOUR_KEY --dry-run
  bun run clear-memories-cloud --api-key YOUR_KEY --delete-all
  EVERMIND_API_KEY=your_key bun run clear-memories-cloud
`);
}

// ============================================================================
// Progress File Cleanup
// ============================================================================

function findProgressFiles(): string[] {
  const cwd = process.cwd();
  const files: string[] = [];

  try {
    const entries = readdirSync(cwd);
    for (const entry of entries) {
      // Match both local and cloud progress files
      if ((entry.startsWith('.novel-progress-') || entry.startsWith('.novel-progress-cloud-')) && entry.endsWith('.json')) {
        files.push(resolve(cwd, entry));
      }
    }
  } catch (error) {
    console.error('Error reading directory:', error);
  }

  return files;
}

function deleteProgressFiles(files: string[], dryRun: boolean): number {
  let deleted = 0;

  for (const file of files) {
    if (dryRun) {
      console.log(`  Would delete: ${basename(file)}`);
      deleted++;
    } else {
      try {
        unlinkSync(file);
        console.log(`  Deleted: ${basename(file)}`);
        deleted++;
      } catch (error) {
        console.error(`  Failed to delete ${basename(file)}:`, error);
      }
    }
  }

  return deleted;
}

// ============================================================================
// EverMind Cloud API
// ============================================================================

async function deleteMemories(apiUrl: string, apiKey: string, groupId: string): Promise<DeleteResult> {
  // API expects all three fields: event_id, user_id, group_id
  // Use "__all__" magic value to match all records for that field
  const requestBody = {
    event_id: '__all__',
    user_id: '__all__',
    group_id: groupId,  // Specific group to delete, or "__all__" for everything
  };

  const response = await fetch(`${apiUrl}/api/v0/memories`, {
    method: 'DELETE',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
    },
    body: JSON.stringify(requestBody),
    signal: AbortSignal.timeout(30000),
  });

  const data = await response.json() as DeleteResponse;

  // Handle 404 as success (no memories to delete)
  if (response.status === 404) {
    return {
      success: true,
      message: 'No memories found (already clean)',
      count: 0,
      notFound: true,
    };
  }

  if (response.ok && data.status === 'ok') {
    return {
      success: true,
      message: data.message || 'Memories deleted',
      count: data.result?.count || 0,
      notFound: false,
    };
  }

  return {
    success: false,
    message: data.message || `HTTP ${response.status}`,
    count: 0,
    notFound: false,
  };
}

async function checkHealth(apiUrl: string, apiKey: string): Promise<boolean> {
  try {
    const response = await fetch(`${apiUrl}/health`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
      },
      signal: AbortSignal.timeout(5000),
    });

    return response.ok;
  } catch (error) {
    return false;
  }
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  const args = parseCliArgs();

  if (!args) {
    return;
  }

  console.log('');
  console.log('═'.repeat(60));
  console.log('🧹 Clear Memories - EverMind Cloud');
  console.log('═'.repeat(60));
  console.log(`API: ${args.apiUrl}`);
  console.log(`Key: ${args.apiKey.slice(0, 8)}...${args.apiKey.slice(-4)}`);
  console.log(`Target: ${args.deleteAll ? 'ALL MEMORIES' : `group "${args.groupId}"`}`);

  if (args.dryRun) {
    console.log('\n⚠️  DRY RUN MODE - No changes will be made\n');
  }

  // Step 1: Check EverMind Cloud health
  console.log('\n📡 Checking EverMind Cloud connection...');
  const isHealthy = await checkHealth(args.apiUrl, args.apiKey);

  if (!isHealthy) {
    console.log('  ⚠️  EverMind Cloud is not available at', args.apiUrl);
    console.log('  Skipping memory deletion from cloud.\n');
  } else {
    console.log('  ✓ EverMind Cloud is healthy\n');

    // Step 2: Delete memories from cloud
    console.log(`📦 Deleting memories for group_id: "${args.groupId}"...`);

    if (args.dryRun) {
      console.log(`  Would send DELETE request to ${args.apiUrl}/api/v0/memories`);
      console.log(`  With body: {"event_id": "__all__", "user_id": "__all__", "group_id": "${args.groupId}"}`);
    } else {
      try {
        const result = await deleteMemories(args.apiUrl, args.apiKey, args.groupId);

        if (result.success) {
          if (result.notFound) {
            console.log(`  ✓ ${result.message}`);
          } else {
            console.log(`  ✓ ${result.message}`);
            console.log(`    Memories deleted: ${result.count}`);
          }
        } else {
          console.log(`  ✗ ${result.message}`);
        }
      } catch (error) {
        console.error('  ✗ Failed to delete memories:', error instanceof Error ? error.message : String(error));
      }
    }
  }

  // Step 3: Clean up progress files
  if (!args.keepProgress) {
    console.log('\n📁 Cleaning up progress files...');
    const progressFiles = findProgressFiles();

    if (progressFiles.length === 0) {
      console.log('  No progress files found.');
    } else {
      console.log(`  Found ${progressFiles.length} progress file(s):`);
      const deleted = deleteProgressFiles(progressFiles, args.dryRun);
      console.log(`  ${args.dryRun ? 'Would delete' : 'Deleted'}: ${deleted} file(s)`);
    }
  } else {
    console.log('\n📁 Keeping progress files (--keep-progress flag set)');
  }

  // Summary
  console.log('\n' + '═'.repeat(60));
  if (args.dryRun) {
    console.log('✅ Dry run complete. Run without --dry-run to apply changes.');
  } else {
    console.log('✅ Cleanup complete!');
  }
  console.log('═'.repeat(60) + '\n');
}

main().catch((error) => {
  console.error('\nUnexpected error:', error);
  process.exit(1);
});
