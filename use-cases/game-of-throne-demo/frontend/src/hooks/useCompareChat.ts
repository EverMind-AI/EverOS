import { useState, useEffect, useCallback } from 'react';
import { Message, Memory, ComparisonState } from '../types';
import { sendCompareMessage } from '../services/api';

const STORAGE_KEY = 'compare_chat_history';
const MEMORIES_STORAGE_KEY = 'compare_chat_memories';

const initialComparisonState: ComparisonState = {
  withMemory: { content: '', isStreaming: false, isDone: false },
  withoutMemory: { content: '', isStreaming: false, isDone: false },
};

export function useCompareChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentMemories, setCurrentMemories] = useState<Memory[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isRetrievingMemories, setIsRetrievingMemories] = useState(false);
  const [isLoadingFollowUps, setIsLoadingFollowUps] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [comparison, setComparison] = useState<ComparisonState>(initialComparisonState);

  // Load chat history and memories from localStorage on mount
  useEffect(() => {
    const storedMessages = localStorage.getItem(STORAGE_KEY);
    if (storedMessages) {
      try {
        setMessages(JSON.parse(storedMessages));
      } catch (e) {
        console.error('Error loading chat history:', e);
      }
    }

    const storedMemories = localStorage.getItem(MEMORIES_STORAGE_KEY);
    if (storedMemories) {
      try {
        setCurrentMemories(JSON.parse(storedMemories));
      } catch (e) {
        console.error('Error loading memories:', e);
      }
    }
  }, []);

  // Save chat history to localStorage
  const saveChatHistory = useCallback((msgs: Message[]) => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(msgs));
  }, []);

  // Save memories to localStorage
  const saveMemories = useCallback((memories: Memory[]) => {
    localStorage.setItem(MEMORIES_STORAGE_KEY, JSON.stringify(memories));
  }, []);

  const sendMessage = useCallback(
    async (content: string) => {
      if (!content.trim() || isLoading) return;

      setError(null);
      setIsLoading(true);
      setIsRetrievingMemories(true);
      setCurrentMemories([]); // Clear memories to show loading state

      // Reset comparison state for new message
      setComparison({
        withMemory: { content: '', isStreaming: true, isDone: false },
        withoutMemory: { content: '', isStreaming: true, isDone: false },
      });

      // Add user message
      const userMessage: Message = { role: 'user', content };
      const updatedMessages = [...messages, userMessage];
      setMessages(updatedMessages);
      saveChatHistory(updatedMessages);

      try {
        let withMemoryContent = '';
        let withoutMemoryContent = '';
        let pendingFollowUps: string[] | undefined;

        await sendCompareMessage(content, messages, {
          onMemories: (memories) => {
            setCurrentMemories(memories);
            saveMemories(memories);
            setIsRetrievingMemories(false);
          },
          onToken: (stream, token) => {
            if (stream === 'withMemory') {
              withMemoryContent += token;
              setComparison(prev => ({
                ...prev,
                withMemory: { ...prev.withMemory, content: withMemoryContent },
              }));
            } else {
              withoutMemoryContent += token;
              setComparison(prev => ({
                ...prev,
                withoutMemory: { ...prev.withoutMemory, content: withoutMemoryContent },
              }));
            }
          },
          onStreamDone: (stream) => {
            setComparison(prev => {
              const updated = {
                ...prev,
                [stream]: { ...prev[stream], isStreaming: false, isDone: true },
              };
              // When withMemory stream is done, start loading follow-ups indicator
              if (stream === 'withMemory') {
                setIsLoadingFollowUps(true);
              }
              return updated;
            });
          },
          onFollowUps: (followUps) => {
            // Store follow-ups to be added when assistant message is created
            pendingFollowUps = followUps;
            setIsLoadingFollowUps(false);
          },
          onComplete: () => {
            // Add complete assistant message with follow-ups
            const assistantMessage: Message = {
              role: 'assistant',
              content: withMemoryContent,
              followUps: pendingFollowUps,
            };
            const finalMessages = [...updatedMessages, assistantMessage];
            setMessages(finalMessages);
            saveChatHistory(finalMessages);
            setIsLoading(false);
            setIsLoadingFollowUps(false);
          },
          onError: (errorMessage) => {
            setError(errorMessage);
            setIsLoading(false);
            setIsRetrievingMemories(false);
            setIsLoadingFollowUps(false);
            setComparison(initialComparisonState);
          },
        });
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
        setIsLoading(false);
        setIsRetrievingMemories(false);
        setIsLoadingFollowUps(false);
        setComparison(initialComparisonState);
      }
    },
    [messages, isLoading, saveChatHistory, saveMemories]
  );

  const clearChat = useCallback(() => {
    setMessages([]);
    setCurrentMemories([]);
    setError(null);
    setComparison(initialComparisonState);
    localStorage.removeItem(STORAGE_KEY);
    localStorage.removeItem(MEMORIES_STORAGE_KEY);
  }, []);

  return {
    messages,
    currentMemories,
    isLoading,
    isRetrievingMemories,
    isLoadingFollowUps,
    error,
    comparison,
    sendMessage,
    clearChat,
  };
}
