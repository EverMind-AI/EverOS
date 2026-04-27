#!/usr/bin/env bun

/**
 * Novel Loading Script for EverMind Cloud API
 *
 * Processes plain text novel files, detects chapters, splits into paragraphs,
 * and stores them in EverMind Cloud with progress tracking and resumption support.
 *
 * Usage:
 *   bun run load-novel-cloud --file <path> --book-title <title> --book-abbrev <abbrev> --api-key <key>
 */

import { parseArgs } from 'util';
import { existsSync } from 'fs';
import { resolve, basename } from 'path';

// ============================================================================
// Types
// ============================================================================

interface CliArgs {
  file: string;
  bookTitle: string;
  bookAbbrev: string;
  apiKey: string;
  apiUrl: string;
  paragraphLimit: number;
  minParagraphSize: number;
  checkHealth: boolean;
  dryRun: boolean;
  freshStart: boolean;
  progressFile?: string;
}

interface Chapter {
  number: number;
  name: string;
  text: string;
  startPos: number;
}

interface Paragraph {
  messageId: string;
  chapterNumber: number;
  chapterName: string;
  paragraphNumber: number;
  text: string;
}

interface ProgressFile {
  book_title: string;
  book_abbrev: string;
  started_at: string;
  last_updated: string;
  total_chapters: number;
  total_paragraphs: number;
  paragraphs: Record<string, 'success' | 'failed'>;
}

// EverMind Cloud API request format
interface CloudMemorizeRequest {
  message_id: string;
  group_id: string;
  group_name: string;
  create_time: string;
  role: string;
  sender: string;
  sender_name: string;
  content: string;
  refer_list: string[];
}

interface SaveResult {
  success: boolean;
  error?: string;
}

interface LoadingSummary {
  chaptersProcessed: number;
  totalParagraphs: number;
  alreadyLoaded: number;
  newlyLoaded: number;
  failed: number;
  failedParagraphs: Array<{ messageId: string; error: string }>;
}

// ============================================================================
// CLI Argument Parsing
// ============================================================================

function parseCliArgs(): CliArgs | null {
  try {
    const { values } = parseArgs({
      options: {
        file: { type: 'string' },
        'book-title': { type: 'string' },
        'book-abbrev': { type: 'string' },
        'api-key': { type: 'string' },
        'api-url': { type: 'string', default: 'https://api.evermind.ai' },
        'paragraph-limit': { type: 'string', default: '10' },
        'min-paragraph-size': { type: 'string', default: '200' },
        'check-health': { type: 'boolean', default: false },
        'dry-run': { type: 'boolean', default: false },
        'fresh-start': { type: 'boolean', default: false },
        'progress-file': { type: 'string' },
        help: { type: 'boolean', default: false },
      },
      strict: true,
      allowPositionals: false,
    });

    if (values.help) {
      printHelp();
      return null;
    }

    // Validate required arguments
    if (!values.file || !values['book-title'] || !values['book-abbrev']) {
      console.error('❌ Error: Missing required arguments\n');
      printHelp();
      process.exit(1);
    }

    // API key from argument or environment variable
    const apiKey = values['api-key'] as string || process.env.EVERMIND_API_KEY || '';
    if (!apiKey) {
      console.error('❌ Error: API key required. Use --api-key or set EVERMIND_API_KEY environment variable\n');
      printHelp();
      process.exit(1);
    }

    return {
      file: values.file as string,
      bookTitle: values['book-title'] as string,
      bookAbbrev: values['book-abbrev'] as string,
      apiKey,
      apiUrl: values['api-url'] as string,
      paragraphLimit: parseInt(values['paragraph-limit'] as string, 10),
      minParagraphSize: parseInt(values['min-paragraph-size'] as string, 10),
      checkHealth: values['check-health'] as boolean,
      dryRun: values['dry-run'] as boolean,
      freshStart: values['fresh-start'] as boolean,
      progressFile: values['progress-file'] as string | undefined,
    };
  } catch (error) {
    console.error('❌ Error parsing arguments:', error instanceof Error ? error.message : String(error));
    console.error('');
    printHelp();
    process.exit(1);
  }
}

function printHelp(): void {
  console.log(`
Novel Loading Script for EverMind Cloud API

Usage:
  bun run load-novel-cloud --file <path> --book-title <title> --book-abbrev <abbrev> --api-key <key> [options]

Required Arguments:
  --file <path>           Path to novel text file
  --book-title <title>    Full book title (e.g., "A Game of Thrones")
  --book-abbrev <abbrev>  Book abbreviation for message IDs (e.g., "got")
  --api-key <key>         EverMind API key (or set EVERMIND_API_KEY env var)

Optional Arguments:
  --api-url <url>           EverMind API URL (default: https://api.evermind.ai)
  --paragraph-limit <num>   Maximum number of paragraphs to load (default: 10, use 0 for unlimited)
  --min-paragraph-size <n>  Minimum characters per paragraph, groups short ones (default: 200, use 0 to disable)
  --check-health            Check API health before loading
  --dry-run                 Parse and show what would be loaded without actually loading
  --fresh-start             Ignore existing progress file and start from beginning
  --progress-file <path>    Custom progress file path (default: .novel-progress-cloud-{abbrev}.json)
  --help                    Show this help message

Examples:
  bun run load-novel-cloud --file got.txt --book-title "A Game of Thrones" --book-abbrev "got" --api-key YOUR_KEY
  bun run load-novel-cloud --file got.txt --book-title "A Game of Thrones" --book-abbrev "got" --paragraph-limit 50
  EVERMIND_API_KEY=your_key bun run load-novel-cloud --file got.txt --book-title "A Game of Thrones" --book-abbrev "got"
`);
}

// ============================================================================
// Chapter Detection
// ============================================================================

const CHAPTER_PATTERNS = [
  /^PROLOGUE\s*$/m,
  /^EPILOGUE\s*$/m,
  /^([A-Z][A-Z\s]{2,})\s*$/m, // POV character names (EDDARD, JON, ARYA, etc.)
  /^CHAPTER\s+(\d+)/im,
];

interface ChapterBoundary {
  position: number;
  name: string;
  isPrologue: boolean;
  isEpilogue: boolean;
}

function detectChapters(text: string): Chapter[] {
  const boundaries: ChapterBoundary[] = [];

  // Find all chapter boundaries
  for (const pattern of CHAPTER_PATTERNS) {
    const matches = text.matchAll(new RegExp(pattern, 'gm'));

    for (const match of matches) {
      const position = match.index!;
      const matchedText = match[0].trim();

      // Determine chapter name
      let name = matchedText;
      let isPrologue = false;
      let isEpilogue = false;

      if (matchedText === 'PROLOGUE') {
        isPrologue = true;
        name = 'Prologue';
      } else if (matchedText === 'EPILOGUE') {
        isEpilogue = true;
        name = 'Epilogue';
      } else if (match[1]) {
        // Captured group from POV pattern or chapter number
        name = toTitleCase(match[1].trim());
      }

      boundaries.push({ position, name, isPrologue, isEpilogue });
    }
  }

  // Sort by position and remove duplicates
  boundaries.sort((a, b) => a.position - b.position);
  const uniqueBoundaries = boundaries.filter(
    (boundary, index, arr) =>
      index === 0 || boundary.position !== arr[index - 1].position
  );

  // Extract chapters
  const chapters: Chapter[] = [];
  let chapterNumber = 0;

  for (let i = 0; i < uniqueBoundaries.length; i++) {
    const boundary = uniqueBoundaries[i];
    const nextBoundary = uniqueBoundaries[i + 1];

    // Assign chapter number - always increment to ensure unique IDs
    // (The file may contain multiple books, each with their own PROLOGUE/EPILOGUE)
    chapterNumber++;

    // Extract chapter text
    const startPos = boundary.position;
    const endPos = nextBoundary ? nextBoundary.position : text.length;
    const chapterText = text.slice(startPos, endPos);

    // Skip the chapter heading line itself
    const firstNewline = chapterText.indexOf('\n');
    const contentText = firstNewline !== -1 ? chapterText.slice(firstNewline + 1) : chapterText;

    chapters.push({
      number: chapterNumber,
      name: boundary.name,
      text: contentText.trim(),
      startPos,
    });
  }

  return chapters;
}

function toTitleCase(str: string): string {
  return str
    .toLowerCase()
    .split(/\s+/)
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

// ============================================================================
// Paragraph Splitting
// ============================================================================

function splitIntoParagraphs(
  chapter: Chapter,
  bookTitle: string,
  bookAbbrev: string,
  minParagraphSize: number = 0
): Paragraph[] {
  // Split by double newlines (paragraph breaks)
  const rawParagraphs = chapter.text.split(/\n\s*\n/);

  const paragraphs: Paragraph[] = [];
  let paragraphNumber = 1;

  for (const rawText of rawParagraphs) {
    const cleanText = rawText.trim();

    // Skip empty paragraphs
    if (!cleanText) {
      continue;
    }

    const messageId = generateMessageId(bookAbbrev, chapter.number, paragraphNumber);

    paragraphs.push({
      messageId,
      chapterNumber: chapter.number,
      chapterName: chapter.name,
      paragraphNumber,
      text: cleanText,
    });

    paragraphNumber++;
  }

  // Group short paragraphs if minParagraphSize is set
  if (minParagraphSize > 0) {
    return groupShortParagraphs(paragraphs, minParagraphSize, bookAbbrev, chapter.number);
  }

  return paragraphs;
}

/**
 * Group consecutive short paragraphs together until they reach minimum size
 * This helps create more coherent memory chunks with better context
 */
function groupShortParagraphs(
  paragraphs: Paragraph[],
  minSize: number,
  bookAbbrev: string,
  chapterNum: number
): Paragraph[] {
  if (paragraphs.length === 0) {
    return paragraphs;
  }

  const grouped: Paragraph[] = [];
  let currentGroup: Paragraph[] = [];
  let currentSize = 0;

  for (const paragraph of paragraphs) {
    currentGroup.push(paragraph);
    currentSize += paragraph.text.length;

    // Check if we've reached the minimum size or this is the last paragraph
    const isLastParagraph = paragraph === paragraphs[paragraphs.length - 1];
    const reachedMinSize = currentSize >= minSize;

    if (reachedMinSize || isLastParagraph) {
      // Merge the current group into a single paragraph
      if (currentGroup.length === 1) {
        // No grouping needed
        grouped.push(currentGroup[0]);
      } else {
        // Merge multiple paragraphs
        const mergedText = currentGroup.map(p => p.text).join('\n\n');
        const firstParagraphNum = currentGroup[0].paragraphNumber;

        grouped.push({
          messageId: generateMessageId(bookAbbrev, chapterNum, firstParagraphNum),
          chapterNumber: currentGroup[0].chapterNumber,
          chapterName: currentGroup[0].chapterName,
          paragraphNumber: firstParagraphNum,
          text: mergedText,
        });
      }

      // Reset for next group
      currentGroup = [];
      currentSize = 0;
    }
  }

  return grouped;
}

function generateMessageId(bookAbbrev: string, chapterNum: number, paragraphNum: number): string {
  const chStr = chapterNum.toString().padStart(2, '0');
  const pStr = paragraphNum.toString().padStart(3, '0');
  return `asoiaf-${bookAbbrev}-ch${chStr}-p${pStr}`;
}

// ============================================================================
// Progress File Management
// ============================================================================

function getProgressFilePath(args: CliArgs): string {
  if (args.progressFile) {
    return resolve(args.progressFile);
  }
  return resolve(`.novel-progress-cloud-${args.bookAbbrev}.json`);
}

async function readProgressFile(filePath: string): Promise<ProgressFile | null> {
  if (!existsSync(filePath)) {
    return null;
  }

  try {
    const content = await Bun.file(filePath).text();
    return JSON.parse(content) as ProgressFile;
  } catch (error) {
    console.error(`⚠ Warning: Failed to read progress file: ${error}`);
    return null;
  }
}

async function writeProgressFile(filePath: string, progress: ProgressFile): Promise<void> {
  try {
    await Bun.write(filePath, JSON.stringify(progress, null, 2));
  } catch (error) {
    console.error(`⚠ Warning: Failed to write progress file: ${error}`);
  }
}

async function updateProgressFile(
  filePath: string,
  messageId: string,
  status: 'success' | 'failed',
  progress: ProgressFile
): Promise<void> {
  progress.paragraphs[messageId] = status;
  progress.last_updated = new Date().toISOString();
  await writeProgressFile(filePath, progress);
}

function initializeProgressFile(args: CliArgs, totalChapters: number, totalParagraphs: number): ProgressFile {
  return {
    book_title: args.bookTitle,
    book_abbrev: args.bookAbbrev,
    started_at: new Date().toISOString(),
    last_updated: new Date().toISOString(),
    total_chapters: totalChapters,
    total_paragraphs: totalParagraphs,
    paragraphs: {},
  };
}

// ============================================================================
// EverMind Cloud API Interaction
// ============================================================================

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
    console.error(`❌ Health check failed: ${error instanceof Error ? error.message : String(error)}`);
    return false;
  }
}

async function saveParagraphWithRetry(
  paragraph: Paragraph,
  bookTitle: string,
  apiUrl: string,
  apiKey: string,
  maxRetries: number = 3
): Promise<SaveResult> {
  // Create chapter metadata prefix
  const chapterMetadata = `[${bookTitle} - Ch${paragraph.chapterNumber}: ${paragraph.chapterName}]`;
  const content = `${chapterMetadata}\n\n${paragraph.text}`;

  // EverMind Cloud API request format
  const request: CloudMemorizeRequest = {
    message_id: paragraph.messageId,
    group_id: 'asoiaf',
    group_name: 'A Song of Ice and Fire',
    create_time: new Date().toISOString(),
    role: 'assistant',  // Using 'assistant' for narrator content
    sender: 'asoiaf_narrator',
    sender_name: 'Narrator',
    content,
    refer_list: [],
  };

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch(`${apiUrl}/api/v0/memories`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`,
        },
        body: JSON.stringify(request),
        signal: AbortSignal.timeout(30000), // 30 second timeout for cloud API
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      await response.json(); // Parse response to ensure it's valid
      return { success: true };
    } catch (error) {
      const isLastAttempt = attempt === maxRetries;
      const errorMsg = error instanceof Error ? error.message : String(error);

      // Determine error type for better logging
      const isTimeout = errorMsg.includes('timeout') || errorMsg.includes('abort');
      const errorType = isTimeout ? 'timeout' : 'error';

      if (isLastAttempt) {
        return { success: false, error: errorMsg };
      }

      // Exponential backoff: 1s, 2s, 4s
      const delayMs = Math.pow(2, attempt - 1) * 1000;
      console.log(`  ⚠ Retry ${attempt}/${maxRetries} (${errorType}) after ${delayMs}ms...`);
      console.log(`    Error: ${errorMsg}`);
      await new Promise(resolve => setTimeout(resolve, delayMs));
    }
  }

  return { success: false, error: 'Max retries exceeded' };
}

// ============================================================================
// Main Loading Logic
// ============================================================================

async function loadNovel(args: CliArgs): Promise<void> {
  const filePath = resolve(args.file);

  // Check if file exists
  if (!existsSync(filePath)) {
    console.error(`❌ Error: File not found: ${filePath}`);
    process.exit(1);
  }

  console.log('');
  console.log('═'.repeat(60));
  console.log('📚 EverMind Cloud - Novel Loading Script');
  console.log('═'.repeat(60));
  console.log(`API: ${args.apiUrl}`);
  console.log(`Key: ${args.apiKey.slice(0, 8)}...${args.apiKey.slice(-4)}`);
  console.log('');

  // Health check if requested
  if (args.checkHealth) {
    console.log('🔍 Checking EverMind Cloud API...');
    const isHealthy = await checkHealth(args.apiUrl, args.apiKey);

    if (isHealthy) {
      console.log('✓ EverMind Cloud API: OK\n');
    } else {
      console.error('❌ EverMind Cloud API is not available or API key is invalid.');
      process.exit(1);
    }
  }

  // Read novel file
  console.log(`📖 Reading novel file: ${basename(filePath)}`);
  const text = await Bun.file(filePath).text();

  // Detect chapters
  console.log('🔍 Detecting chapters...');
  const chapters = detectChapters(text);

  if (chapters.length === 0) {
    console.error('❌ Error: No chapters detected in the file.');
    console.error('Make sure the file contains chapter markers like:');
    console.error('  - PROLOGUE');
    console.error('  - Character names in ALL CAPS (e.g., EDDARD, JON)');
    console.error('  - CHAPTER X');
    process.exit(1);
  }

  console.log(`✓ Found ${chapters.length} chapters\n`);

  // Split into paragraphs
  const allParagraphs: Paragraph[] = [];
  for (const chapter of chapters) {
    const paragraphs = splitIntoParagraphs(chapter, args.bookTitle, args.bookAbbrev, args.minParagraphSize);
    allParagraphs.push(...paragraphs);
  }

  console.log(`✓ Total paragraphs in novel: ${allParagraphs.length}`);
  if (args.minParagraphSize > 0) {
    console.log(`✓ Grouped short paragraphs (min size: ${args.minParagraphSize} chars)`);
  }

  // Apply paragraph limit
  const paragraphsToLoad = args.paragraphLimit > 0
    ? allParagraphs.slice(0, args.paragraphLimit)
    : allParagraphs;

  if (args.paragraphLimit > 0 && allParagraphs.length > args.paragraphLimit) {
    console.log(`⚠ Paragraph limit applied: loading first ${args.paragraphLimit} paragraphs\n`);
  } else {
    console.log('');
  }

  // Dry run mode
  if (args.dryRun) {
    console.log('🔎 DRY RUN MODE - Showing exact memories that would be saved:\n');
    console.log('═'.repeat(80));
    console.log(`Total paragraphs to load: ${paragraphsToLoad.length}\n`);

    for (let i = 0; i < paragraphsToLoad.length; i++) {
      const paragraph = paragraphsToLoad[i];

      // Create the exact memory object that would be saved
      const chapterMetadata = `[${args.bookTitle} - Ch${paragraph.chapterNumber}: ${paragraph.chapterName}]`;
      const content = `${chapterMetadata}\n\n${paragraph.text}`;

      const memoryObject: CloudMemorizeRequest = {
        message_id: paragraph.messageId,
        group_id: 'asoiaf',
        group_name: 'A Song of Ice and Fire',
        create_time: new Date().toISOString(),
        role: 'assistant',
        sender: 'asoiaf_narrator',
        sender_name: 'Narrator',
        content,
        refer_list: [],
      };

      console.log(`\n[${i + 1}/${paragraphsToLoad.length}] Memory Object:`);
      console.log('─'.repeat(80));
      console.log(JSON.stringify(memoryObject, null, 2));
      console.log('─'.repeat(80));

      // Show a preview of the content for readability
      const contentPreview = paragraph.text.slice(0, 150);
      console.log(`Content preview: ${contentPreview}${paragraph.text.length > 150 ? '...' : ''}`);
      console.log(`Content length: ${content.length} characters`);
    }

    console.log('\n' + '═'.repeat(80));
    console.log(`\nSummary:`);
    console.log(`  Total chapters detected: ${chapters.length}`);
    console.log(`  Total paragraphs in novel: ${allParagraphs.length}`);
    console.log(`  Paragraphs to load: ${paragraphsToLoad.length}`);
    console.log('\nRun without --dry-run to actually save these memories to EverMind Cloud.');
    return;
  }

  // Initialize or load progress file
  const progressFilePath = getProgressFilePath(args);
  let progress: ProgressFile;

  if (args.freshStart || !existsSync(progressFilePath)) {
    if (args.freshStart && existsSync(progressFilePath)) {
      console.log(`🗑️  Fresh start: Ignoring existing progress file\n`);
    }
    progress = initializeProgressFile(args, chapters.length, paragraphsToLoad.length);
    await writeProgressFile(progressFilePath, progress);
    console.log(`✓ Created progress file: ${basename(progressFilePath)}\n`);
  } else {
    const existingProgress = await readProgressFile(progressFilePath);
    if (existingProgress) {
      progress = existingProgress;
      console.log(`✓ Resuming from existing progress file: ${basename(progressFilePath)}`);
      const successCount = Object.values(progress.paragraphs).filter(s => s === 'success').length;
      console.log(`  Already loaded: ${successCount} paragraphs\n`);
    } else {
      progress = initializeProgressFile(args, chapters.length, paragraphsToLoad.length);
      await writeProgressFile(progressFilePath, progress);
    }
  }

  // Load paragraphs
  const summary: LoadingSummary = {
    chaptersProcessed: 0,
    totalParagraphs: paragraphsToLoad.length,
    alreadyLoaded: 0,
    newlyLoaded: 0,
    failed: 0,
    failedParagraphs: [],
  };

  console.log('📚 Loading novel into EverMind Cloud...\n');

  // Create a Set of message IDs to load for quick lookup
  const messageIdsToLoad = new Set(paragraphsToLoad.map(p => p.messageId));

  for (const chapter of chapters) {
    const paragraphs = splitIntoParagraphs(chapter, args.bookTitle, args.bookAbbrev, args.minParagraphSize);

    // Filter paragraphs to only those in our load list
    const paragraphsInChapterToLoad = paragraphs.filter(p => messageIdsToLoad.has(p.messageId));

    // Skip chapter if no paragraphs to load
    if (paragraphsInChapterToLoad.length === 0) {
      continue;
    }

    console.log(`Loading Chapter ${chapter.number}: ${chapter.name}`);

    for (const paragraph of paragraphsInChapterToLoad) {
      const existingStatus = progress.paragraphs[paragraph.messageId];

      // Skip already loaded paragraphs
      if (existingStatus === 'success') {
        console.log(`  ⊘ Skipping ${paragraph.messageId} (already loaded)`);
        summary.alreadyLoaded++;
        continue;
      }

      // Try to save
      const result = await saveParagraphWithRetry(paragraph, args.bookTitle, args.apiUrl, args.apiKey);

      // Update progress file immediately
      await updateProgressFile(
        progressFilePath,
        paragraph.messageId,
        result.success ? 'success' : 'failed',
        progress
      );

      if (result.success) {
        console.log(`  ✓ Saved ${paragraph.messageId}`);
        summary.newlyLoaded++;
      } else {
        console.log(`  ✗ Failed ${paragraph.messageId}: ${result.error}`);
        summary.failed++;
        summary.failedParagraphs.push({
          messageId: paragraph.messageId,
          error: result.error || 'Unknown error',
        });
      }
    }

    summary.chaptersProcessed++;
    console.log(''); // Empty line between chapters
  }

  // Print summary
  console.log('═'.repeat(60));
  console.log('📊 Loading Summary');
  console.log('═'.repeat(60));
  console.log(`Chapters processed:     ${summary.chaptersProcessed}`);
  console.log(`Total paragraphs:       ${summary.totalParagraphs}`);
  console.log(`Already loaded:         ${summary.alreadyLoaded}`);
  console.log(`Newly loaded:           ${summary.newlyLoaded}`);
  console.log(`Failed:                 ${summary.failed}`);
  console.log('');

  if (summary.failedParagraphs.length > 0) {
    console.log('❌ Failed paragraphs (can retry by running script again):');
    for (const failed of summary.failedParagraphs) {
      console.log(`  - ${failed.messageId}: ${failed.error}`);
    }
    console.log('');
  }

  console.log(`Progress saved to: ${basename(progressFilePath)}`);

  if (summary.failed > 0) {
    console.log('\n⚠ Some paragraphs failed to load. Run the script again to retry.');
    process.exit(1);
  } else {
    console.log('\n✅ Novel loaded successfully to EverMind Cloud!');
  }
}

// ============================================================================
// Entry Point
// ============================================================================

async function main() {
  const args = parseCliArgs();

  if (!args) {
    return; // Help was shown or args were invalid
  }

  try {
    await loadNovel(args);
  } catch (error) {
    console.error('\n❌ Unexpected error:', error instanceof Error ? error.message : String(error));
    process.exit(1);
  }
}

main();
