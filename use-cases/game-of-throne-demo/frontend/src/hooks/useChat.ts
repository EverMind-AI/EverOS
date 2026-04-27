import { useState, useEffect, useCallback } from 'react';
import { Message, Memory } from '../types';
import { sendChatMessage } from '../services/api';

const STORAGE_KEY = 'chat_history';
const MEMORIES_STORAGE_KEY = 'chat_memories';

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentMemories, setCurrentMemories] = useState<Memory[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isRetrievingMemories, setIsRetrievingMemories] = useState(false);
  const [isLoadingFollowUps, setIsLoadingFollowUps] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [streamingContent, setStreamingContent] = useState('');

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
      setStreamingContent('');
      setCurrentMemories([]); // Clear memories to show loading state

      // Add user message
      const userMessage: Message = { role: 'user', content };
      const updatedMessages = [...messages, userMessage];
      setMessages(updatedMessages);
      saveChatHistory(updatedMessages);

      try {
        let assistantContent = '';

        await sendChatMessage(content, messages, {
          onMemories: (memories) => {
            setCurrentMemories(memories);
            saveMemories(memories);
            setIsRetrievingMemories(false);
          },
          onToken: (token) => {
            assistantContent += token;
            setStreamingContent(assistantContent);
          },
          onDone: () => {
            // Add complete assistant message (follow-ups will be added when received)
            const assistantMessage: Message = {
              role: 'assistant',
              content: assistantContent,
            };
            const finalMessages = [...updatedMessages, assistantMessage];
            setMessages(finalMessages);
            saveChatHistory(finalMessages);
            setStreamingContent('');
            setIsLoading(false);
            setIsLoadingFollowUps(true); // Start loading follow-ups
          },
          onFollowUps: (followUps) => {
            setIsLoadingFollowUps(false);
            // Update the last assistant message with follow-ups
            setMessages(prev => {
              if (prev.length === 0) return prev;
              const updated = [...prev];
              const lastIndex = updated.length - 1;
              if (updated[lastIndex].role === 'assistant') {
                updated[lastIndex] = { ...updated[lastIndex], followUps };
                saveChatHistory(updated);
              }
              return updated;
            });
          },
          onError: (errorMessage) => {
            setError(errorMessage);
            setIsLoading(false);
            setIsRetrievingMemories(false);
            setIsLoadingFollowUps(false);
            setStreamingContent('');
          },
        });
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
        setIsLoading(false);
        setIsRetrievingMemories(false);
        setIsLoadingFollowUps(false);
        setStreamingContent('');
      }
    },
    [messages, isLoading, saveChatHistory, saveMemories]
  );

  const clearChat = useCallback(() => {
    setMessages([]);
    setCurrentMemories([]);
    setError(null);
    setStreamingContent('');
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
    streamingContent,
    sendMessage,
    clearChat,
  };
}
