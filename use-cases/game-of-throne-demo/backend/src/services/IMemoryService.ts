export interface Memory {
  id: string;
  content: string;
  metadata: {
    bookTitle: string;
    chapterNumber?: number;
    chapterName?: string;
  };
  relevanceScore?: number;
  // Rich fields from EverMind Cloud API
  subject?: string;           // Concise title/headline
  summary?: string;           // Short summary paragraph
  episode?: string;           // Detailed narrative with timestamps
  originalContent?: string;   // The actual source text from the book
}

export interface IMemoryService {
  /**
   * Retrieve relevant memories for a query
   */
  retrieveMemories(query: string, limit?: number): Promise<Memory[]>;

  /**
   * Health check
   */
  isAvailable(): Promise<boolean>;

  /**
   * Clear all memories (Stage 2)
   */
  clearMemories?(): Promise<void>;
}
