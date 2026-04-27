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

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  followUps?: string[];  // AI-generated follow-up questions
}

export interface ChatState {
  messages: Message[];
  currentMemories: Memory[];
  isLoading: boolean;
  error: string | null;
}

export interface SSEEvent {
  type: 'memories' | 'token' | 'done' | 'followups' | 'error' | 'complete';
  stream?: 'withMemory' | 'withoutMemory';
  memories?: Memory[];
  token?: string;
  message?: string;
  followUps?: string[];
}

export interface ComparisonStreamState {
  content: string;
  isStreaming: boolean;
  isDone: boolean;
}

export interface ComparisonState {
  withMemory: ComparisonStreamState;
  withoutMemory: ComparisonStreamState;
}
