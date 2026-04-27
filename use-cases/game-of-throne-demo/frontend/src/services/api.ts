import { Memory, Message, SSEEvent } from '../types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3001';

export interface ChatStreamCallbacks {
  onMemories: (memories: Memory[]) => void;
  onToken: (token: string) => void;
  onDone: () => void;
  onFollowUps: (followUps: string[]) => void;
  onError: (error: string) => void;
}

export async function sendChatMessage(
  message: string,
  conversationHistory: Message[],
  callbacks: ChatStreamCallbacks
): Promise<void> {
  try {
    const response = await fetch(`${API_URL}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        conversationHistory,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Response body is not readable');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });

      // Process complete SSE messages
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep incomplete line in buffer

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          try {
            const event = JSON.parse(data) as SSEEvent;

            switch (event.type) {
              case 'memories':
                if (event.memories) {
                  callbacks.onMemories(event.memories);
                }
                break;
              case 'token':
                if (event.token) {
                  callbacks.onToken(event.token);
                }
                break;
              case 'done':
                callbacks.onDone();
                break;
              case 'followups':
                if (event.followUps) {
                  callbacks.onFollowUps(event.followUps);
                }
                break;
              case 'error':
                callbacks.onError(event.message || 'An error occurred');
                break;
            }
          } catch (e) {
            console.error('Error parsing SSE event:', e);
          }
        }
      }
    }
  } catch (error) {
    console.error('Error in chat stream:', error);
    callbacks.onError(
      error instanceof Error ? error.message : 'Connection lost. Please check your internet.'
    );
  }
}

export interface CompareStreamCallbacks {
  onMemories: (memories: Memory[]) => void;
  onToken: (stream: 'withMemory' | 'withoutMemory', token: string) => void;
  onStreamDone: (stream: 'withMemory' | 'withoutMemory') => void;
  onFollowUps: (followUps: string[]) => void;
  onComplete: () => void;
  onError: (error: string) => void;
}

export async function sendCompareMessage(
  message: string,
  conversationHistory: Message[],
  callbacks: CompareStreamCallbacks
): Promise<void> {
  try {
    const response = await fetch(`${API_URL}/api/chat/compare`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        conversationHistory,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Response body is not readable');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });

      // Process complete SSE messages
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep incomplete line in buffer

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          try {
            const event = JSON.parse(data) as SSEEvent;

            switch (event.type) {
              case 'memories':
                if (event.memories) {
                  callbacks.onMemories(event.memories);
                }
                break;
              case 'token':
                if (event.token && event.stream) {
                  callbacks.onToken(event.stream, event.token);
                }
                break;
              case 'done':
                if (event.stream) {
                  callbacks.onStreamDone(event.stream);
                }
                break;
              case 'followups':
                if (event.followUps) {
                  callbacks.onFollowUps(event.followUps);
                }
                break;
              case 'complete':
                callbacks.onComplete();
                break;
              case 'error':
                callbacks.onError(event.message || 'An error occurred');
                break;
            }
          } catch (e) {
            console.error('Error parsing SSE event:', e);
          }
        }
      }
    }
  } catch (error) {
    console.error('Error in compare stream:', error);
    callbacks.onError(
      error instanceof Error ? error.message : 'Connection lost. Please check your internet.'
    );
  }
}

export async function checkHealth(): Promise<{
  status: string;
  backend: string;
  openai: string;
  memory: string;
}> {
  const response = await fetch(`${API_URL}/api/health`);
  return response.json();
}
