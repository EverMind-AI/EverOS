import { IMemoryService, Memory } from './IMemoryService.js';
import { mockMemories } from '../data/mockMemories.js';

export class MockMemoryService implements IMemoryService {
  constructor() {
    // No API needed for keyword-based retrieval
  }

  async retrieveMemories(query: string, limit: number = 5): Promise<Memory[]> {
    // Fast keyword-based retrieval for PoC
    // Calculate relevance score for each memory based on keyword matching
    const queryLower = query.toLowerCase();
    const queryWords = queryLower.split(/\s+/).filter(word => word.length > 3);

    const scoredMemories = mockMemories.map((memory, index) => {
      const contentLower = memory.content.toLowerCase();
      const chapterLower = (memory.metadata.chapterName || '').toLowerCase();

      let score = 0;

      // Score based on query word matches
      queryWords.forEach(word => {
        const contentMatches = (contentLower.match(new RegExp(word, 'g')) || []).length;
        const chapterMatches = (chapterLower.match(new RegExp(word, 'g')) || []).length;
        score += contentMatches * 10 + chapterMatches * 5;
      });

      // Bonus for exact phrase match
      if (contentLower.includes(queryLower)) {
        score += 100;
      }

      return { memory, score, index };
    });

    // Sort by score (highest first) and return top N
    const topMemories = scoredMemories
      .sort((a, b) => b.score - a.score)
      .slice(0, limit)
      .map(item => item.memory);

    // If no matches found (all scores are 0), return random selection
    if (topMemories.every((_, i) => scoredMemories[i].score === 0)) {
      return mockMemories.slice(0, limit);
    }

    return topMemories;
  }

  async isAvailable(): Promise<boolean> {
    return true;
  }
}
