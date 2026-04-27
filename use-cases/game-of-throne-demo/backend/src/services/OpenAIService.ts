import OpenAI from 'openai';
import { Memory } from './IMemoryService.js';

export interface Message {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export class OpenAIService {
  private openai: OpenAI;
  private model: string;
  private systemPromptWithMemory: string;
  private systemPromptWithoutMemory: string;

  constructor(apiKey: string, model: string = 'anthropic/claude-3-haiku') {
    this.openai = new OpenAI({
      apiKey,
      baseURL: 'https://openrouter.ai/api/v1',
      defaultHeaders: {
        'HTTP-Referer': 'https://github.com/your-repo', // Optional, for OpenRouter rankings
        'X-Title': 'EverMem Story Memory Demo', // Optional, for OpenRouter rankings
      }
    });
    this.model = model;

    // System prompt when memories are provided
    this.systemPromptWithMemory = `You are an expert on "A Game of Thrones" (Book 1) by George R.R. Martin.
You have access to numbered excerpts from the book to answer user questions accurately.

Guidelines:
- ONLY use the provided memory excerpts to answer questions. Do NOT add information from general knowledge.
- When your answer is based on a specific memory, cite it using [1], [2], etc. at the end of the relevant sentence or paragraph.
- You can cite multiple sources for the same statement, e.g., [1][2].
- If the provided memories don't contain enough information to fully answer the question, just answer with what's available in the memories.
- Be concise and accurate. Stick strictly to what's in the excerpts.

Example format:
"Ned Stark executed the deserter before the family discovered the direwolves [1]. The pups were found near their dead mother [2]."
`;

    // System prompt when no memories are provided (general knowledge only)
    this.systemPromptWithoutMemory = `You are a helpful assistant answering questions about "A Game of Thrones" (Book 1) by George R.R. Martin.

IMPORTANT CONSTRAINTS:
- You must ONLY use knowledge from your training data. Do NOT search the internet, use tools, or access any external sources.
- Answer based solely on what you remember from your training about the book.
- If you don't remember specific details (exact quotes, chapter numbers, minor character names, specific scenes), be honest and say you're not certain rather than guessing.
- Do NOT make up specific details like page numbers, exact quotes, or precise plot points if you're unsure.

Guidelines:
- Provide a helpful answer using your general knowledge of the story, characters, and plot.
- Be concise and conversational.
- It's okay to give a general answer if you don't recall specifics.
`;
  }

  /**
   * Stream chat completion with memories as context
   */
  async *streamChatCompletion(
    query: string,
    memories: Memory[],
    conversationHistory: Message[] = []
  ): AsyncGenerator<string, void, unknown> {
    // Use appropriate system prompt based on whether memories are provided
    // Same model is used for both to show that the difference comes from memory, not model
    const systemPrompt = memories.length > 0
      ? this.systemPromptWithMemory
      : this.systemPromptWithoutMemory;

    // Build the messages array with sliding window
    const messages: Message[] = [
      { role: 'system', content: systemPrompt },
    ];

    // Add memories as context with citation numbers
    if (memories.length > 0) {
      const memoriesText = memories
        .map((m, index) => {
          const chapterInfo = m.metadata.chapterNumber
            ? `Chapter ${m.metadata.chapterNumber}${m.metadata.chapterName ? `: ${m.metadata.chapterName}` : ''}`
            : m.metadata.chapterName || '';
          const source = chapterInfo
            ? `${m.metadata.bookTitle} - ${chapterInfo}`
            : m.metadata.bookTitle;

          // Include both summary and original text for better context
          let content = `Summary: ${m.content}`;
          if (m.originalContent) {
            content += `\n\nOriginal Text:\n${m.originalContent}`;
          }

          return `[${index + 1}] ${source}\n${content}`;
        })
        .join('\n\n');

      messages.push({
        role: 'system',
        content: `Retrieved Memories (cite using [1], [2], etc.):\n\n${memoriesText}`,
      });
    }

    // Add conversation history (sliding window - last messages that fit in ~4000 tokens)
    // Rough estimate: 1 token ≈ 4 characters
    const maxHistoryTokens = 4000;
    const maxHistoryChars = maxHistoryTokens * 4;

    let historyChars = 0;
    const recentHistory: Message[] = [];

    for (let i = conversationHistory.length - 1; i >= 0; i--) {
      const msg = conversationHistory[i];
      const msgChars = msg.content.length;

      if (historyChars + msgChars > maxHistoryChars) {
        break;
      }

      recentHistory.unshift(msg);
      historyChars += msgChars;
    }

    messages.push(...recentHistory);

    // Add current query
    messages.push({ role: 'user', content: query });

    // Stream the response
    const stream = await this.openai.chat.completions.create({
      model: this.model,
      messages,
      stream: true,
    });

    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content;
      if (content) {
        yield content;
      }
    }
  }

  /**
   * Generate follow-up questions based on the conversation
   */
  async generateFollowUps(
    query: string,
    assistantResponse: string,
    memories: Memory[]
  ): Promise<string[]> {
    try {
      const memoryContext = memories
        .slice(0, 3)
        .map(m => m.subject || m.content.slice(0, 100))
        .join('; ');

      const response = await this.openai.chat.completions.create({
        model: this.model,
        messages: [
          {
            role: 'system',
            content: `You generate follow-up questions for a Q&A about "A Game of Thrones" (Book 1).
Given the user's question and the assistant's response, suggest 2-3 natural follow-up questions the user might want to ask.

Rules:
- Questions should be specific and interesting
- Questions should relate to the current topic or naturally expand on it
- Keep questions concise (under 15 words each)
- Return ONLY the questions, one per line, no numbering or bullets`
          },
          {
            role: 'user',
            content: `User asked: "${query}"

Assistant responded about: ${assistantResponse.slice(0, 500)}

Related context: ${memoryContext}

Generate 2-3 follow-up questions:`
          }
        ],
        max_tokens: 150,
      });

      const content = response.choices[0]?.message?.content || '';
      const questions = content
        .split('\n')
        .map(q => q.trim())
        .filter(q => q.length > 0 && q.endsWith('?'))
        .slice(0, 3);

      return questions;
    } catch (error) {
      console.error('Error generating follow-ups:', error);
      return [];
    }
  }

  /**
   * Check if OpenAI API is available
   */
  async isAvailable(): Promise<boolean> {
    try {
      await this.openai.models.list();
      return true;
    } catch {
      return false;
    }
  }
}
