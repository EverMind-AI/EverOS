import type { IMemoryService, Memory } from './IMemoryService';

interface EverCoreMemoryItem {
  memory_type: string;
  summary: string | null;
  subject?: string;      // Concise title/headline
  episode?: string;      // Detailed narrative with timestamps
  user_id?: string;
  timestamp?: string;
  group_id?: string | null;
  group_name?: string | null;
  keywords?: string[] | null;
  linked_entities?: string[] | null;
  score?: number | null;
  original_data?: OriginalDataItem[];  // Nested inside each memory item
  [key: string]: unknown;
}

interface OriginalDataMessage {
  content: string;
  extend?: {
    message_id?: string;
    speaker_name?: string;
    [key: string]: unknown;
  };
}

interface OriginalDataItem {
  data_type: string;
  messages: OriginalDataMessage[];
}

interface ProfileSearchItem {
  item_type: 'explicit_info' | 'implicit_trait';
  category?: string;
  trait_name?: string;
  description: string;
  score: number;
}

interface EverCoreSearchResponse {
  status: string;
  message?: string;
  result: {
    profiles: ProfileSearchItem[];
    memories: EverCoreMemoryItem[];
    total_count: number;
    scores: number[];
    has_more: boolean;
    pending_messages?: unknown[];
    query_metadata?: unknown;
    metadata?: unknown;
  };
}

interface EverCoreHealthResponse {
  status: string;
  [key: string]: unknown;
}

/**
 * Book abbreviation to full title mapping
 */
const BOOK_TITLES: Record<string, string> = {
  'got': 'A Game of Thrones',
  'cok': 'A Clash of Kings',
  'sos': 'A Storm of Swords',
  'ffc': 'A Feast for Crows',
  'dwd': 'A Dance with Dragons',
};

/**
 * Configuration for EverCore/EverMind Cloud service
 */
interface EverCoreConfig {
  baseUrl: string;
  apiKey?: string;      // Required for cloud API
  groupId?: string;     // Group ID for search (default: 'asoiaf')
}

/**
 * EverCore service implementation for memory retrieval
 * Supports both local EverCore and EverMind Cloud API
 */
export class EverCoreService implements IMemoryService {
  private baseUrl: string;
  private apiKey?: string;
  private groupId: string;
  private isCloudMode: boolean;

  constructor(config: string | EverCoreConfig) {
    if (typeof config === 'string') {
      // Legacy: just a URL string (local mode)
      this.baseUrl = config.replace(/\/$/, '');
      this.apiKey = undefined;
      this.groupId = 'asoiaf';
      this.isCloudMode = false;
    } else {
      this.baseUrl = config.baseUrl.replace(/\/$/, '');
      this.apiKey = config.apiKey;
      this.groupId = config.groupId || 'asoiaf';
      this.isCloudMode = !!config.apiKey;
    }
  }

  /**
   * Retrieve relevant memories for a query using EverCore search
   */
  async retrieveMemories(query: string, limit: number = 5): Promise<Memory[]> {
    try {
      const searchUrl = `${this.baseUrl}/api/v0/memories/search`;

      const params = new URLSearchParams({
        query,
        retrieve_method: 'hybrid',
        top_k: limit.toString(),
        include_metadata: 'true',
      });

      // Add group_ids for cloud mode
      if (this.isCloudMode) {
        params.set('group_ids', this.groupId);
      }

      const headers: Record<string, string> = {};

      // Add auth header for cloud mode
      if (this.apiKey) {
        headers['Authorization'] = `Bearer ${this.apiKey}`;
      }

      const response = await fetch(`${searchUrl}?${params}`, {
        method: 'GET',
        headers,
        signal: AbortSignal.timeout(this.isCloudMode ? 15000 : 10000),
      });

      if (!response.ok) {
        console.error(`EverCore search failed: HTTP ${response.status}`);
        return [];
      }

      const data = await response.json() as EverCoreSearchResponse;
      return this.mapSearchResultsToMemories(data);
    } catch (error) {
      console.error('Error retrieving memories from EverCore:', error);
      return []; // Graceful degradation
    }
  }

  /**
   * Check if EverCore service is available
   */
  async isAvailable(): Promise<boolean> {
    try {
      const headers: Record<string, string> = {};
      if (this.apiKey) {
        headers['Authorization'] = `Bearer ${this.apiKey}`;
      }

      const response = await fetch(`${this.baseUrl}/health`, {
        method: 'GET',
        headers,
        signal: AbortSignal.timeout(5000),
      });

      if (!response.ok) {
        return false;
      }

      const data = await response.json() as EverCoreHealthResponse;
      // Cloud API returns "ok" status, local returns "healthy"
      return data.status === 'healthy' || data.status === 'ok';
    } catch (error) {
      console.warn('EverCore health check failed:', error);
      return false;
    }
  }

  /**
   * Map EverCore search results to our Memory interface
   */
  private mapSearchResultsToMemories(data: EverCoreSearchResponse): Memory[] {
    const memories: Memory[] = [];

    const result = data.result;
    if (!result || !result.memories || result.memories.length === 0) {
      return memories;
    }

    const memoryItems = result.memories;
    const scores = result.scores || [];

    for (let i = 0; i < memoryItems.length; i++) {
      const item = memoryItems[i];
      const score = item.score ?? scores[i] ?? 0;
      const originalContents = item.original_data || [];

      const memory = this.mapMemoryItem(item, score, originalContents);
      if (memory) {
        memories.push(memory);
      }
    }

    return memories;
  }

  /**
   * Map a single EverCore memory item to our Memory interface
   */
  private mapMemoryItem(
    item: EverCoreMemoryItem,
    score: number,
    originalContents: OriginalDataItem[] = []
  ): Memory | null {
    try {
      // Extract original book content and metadata from original_data
      const originalTexts: string[] = [];
      const cleanedTexts: string[] = [];
      let firstMessageId: string | undefined;
      let metadata: Memory['metadata'] = {
        bookTitle: 'Unknown Book',
        chapterNumber: undefined,
        chapterName: undefined,
      };

      for (const orig of originalContents) {
        for (const msg of orig.messages || []) {
          if (msg.content) {
            // Store raw content for original display
            originalTexts.push(msg.content);

            // Parse metadata from the first message's content prefix
            if (cleanedTexts.length === 0) {
              const parsed = this.parseContent(msg.content, msg.extend?.message_id || '');
              metadata = parsed.metadata;
              cleanedTexts.push(parsed.content);
              firstMessageId = msg.extend?.message_id;
            } else {
              const parsed = this.parseContent(msg.content, '');
              cleanedTexts.push(parsed.content);
            }
          }
        }
      }

      // Join original texts (with prefix) for "Show original" feature
      const originalContent = cleanedTexts.join('\n\n');

      // Generate a unique ID
      const memoryId = firstMessageId || `memory-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

      // Clean summary by removing date artifacts
      const cleanedSummary = this.cleanDateArtifacts(item.summary || '');

      // Use summary as the main content for display
      return {
        id: memoryId,
        content: cleanedSummary,
        metadata,
        relevanceScore: score,
        // New rich fields
        subject: this.cleanDateArtifacts(item.subject || ''),
        summary: cleanedSummary,
        episode: item.episode,
        originalContent: originalContent || undefined,
      };
    } catch (error) {
      console.error('Error mapping memory item:', error, item);
      return null;
    }
  }

  /**
   * Parse content to extract metadata and clean paragraph text
   * Expected format: "[Book Title - ChX: POV Name]\n\nParagraph text..."
   */
  private parseContent(content: string, messageId: string): {
    content: string;
    metadata: Memory['metadata'];
  } {
    // Try to extract metadata from content prefix
    const prefixMatch = content.match(/^\[(.+?)\s+-\s+Ch(\d+):\s+(.+?)\]\n\n/);

    if (prefixMatch) {
      const [fullMatch, bookTitle, chapterNum, chapterName] = prefixMatch;
      const cleanContent = content.slice(fullMatch.length); // Remove prefix

      return {
        content: cleanContent,
        metadata: {
          bookTitle: bookTitle.trim(),
          chapterNumber: parseInt(chapterNum, 10),
          chapterName: chapterName.trim(),
        },
      };
    }

    // Fallback: try to parse from message_id and use full content
    const fallbackMetadata = this.parseMessageId(messageId);

    return {
      content,
      metadata: {
        bookTitle: fallbackMetadata.bookTitle,
        chapterNumber: fallbackMetadata.chapterNumber,
        chapterName: fallbackMetadata.chapterName,
      },
    };
  }

  /**
   * Parse message ID to extract book and chapter info as fallback
   * Format: "asoiaf-{book}-ch{num}-p{paragraph}"
   * Example: "asoiaf-got-ch01-p001"
   */
  private parseMessageId(messageId: string): {
    bookTitle: string;
    chapterNumber?: number;
    chapterName?: string;
  } {
    const match = messageId.match(/asoiaf-(\w+)-ch(\d+)-p(\d+)/);

    if (match) {
      const [, bookAbbrev, chapterNum] = match;
      const bookTitle = BOOK_TITLES[bookAbbrev] || `Unknown Book (${bookAbbrev})`;

      return {
        bookTitle,
        chapterNumber: parseInt(chapterNum, 10),
        chapterName: undefined,
      };
    }

    // Complete fallback
    return {
      bookTitle: 'Unknown Book',
      chapterNumber: undefined,
      chapterName: undefined,
    };
  }

  /**
   * Remove date artifacts from text generated by EverMind Cloud
   * Examples:
   *   "On January 18, 2026, a Night's Watch..." -> "A Night's Watch..."
   *   "Bran Witnesses His Father Execute a Deserted Night's Watchman - January 18, 2026" -> "Bran Witnesses..."
   */
  private cleanDateArtifacts(text: string): string {
    if (!text) return text;

    // Remove date prefixes like "On January 18, 2026, " or "On Sunday, January 18, 2026, "
    let cleaned = text.replace(
      /^On\s+(?:Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday)?,?\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4},?\s*/i,
      ''
    );

    // Remove date suffixes like " - January 18, 2026", ", January 18, 2026", or " on January 18, 2026"
    cleaned = cleaned.replace(
      /\s*[-–—,]\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}\.?$/i,
      ''
    );

    // Remove inline date references like "(January 18, 2026)" or "on January 18, 2026"
    cleaned = cleaned.replace(
      /\s+on\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}/gi,
      ''
    );

    // Capitalize first letter if it was lowercased after removing prefix
    if (cleaned.length > 0 && cleaned[0] !== cleaned[0].toUpperCase()) {
      cleaned = cleaned[0].toUpperCase() + cleaned.slice(1);
    }

    return cleaned.trim();
  }
}
