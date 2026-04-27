import { MessageList } from './MessageList';
import { ChatInput } from './ChatInput';
import { ExampleQueries } from './ExampleQueries';
import { Memory, Message } from '../types';

interface ChatInterfaceProps {
  messages: Message[];
  streamingContent: string;
  isLoading: boolean;
  isRetrievingMemories: boolean;
  isLoadingFollowUps: boolean;
  error: string | null;
  memories: Memory[];
  onSendMessage: (message: string) => void;
  onClearChat: () => void;
}

export function ChatInterface({
  messages,
  streamingContent,
  isLoading,
  isRetrievingMemories,
  isLoadingFollowUps,
  error,
  memories,
  onSendMessage,
  onClearChat,
}: ChatInterfaceProps) {
  const showWelcome = messages.length === 0 && !streamingContent;

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <div className="chat-header-title">
          <div className="chat-header-main">
            <span className="evermem-logo-text">EverMind</span>
            <h1><span className="brand-evermem">EverMem</span> Story Memory Demo</h1>
          </div>
          <span className="chat-header-subtitle">A Game of Thrones</span>
        </div>
        <button onClick={onClearChat} className="clear-button" disabled={isLoading}>
          Clear
        </button>
      </div>

      <div className="chat-messages">
        {showWelcome && (
          <div className="welcome-message">
            <h2>Welcome</h2>
            <p>
              See how <strong>EverMem</strong> memorizes and retrieves story details.
              Ask any question about <strong>A Game of Thrones</strong> (Book 1) and watch relevant memories surface in real-time.
            </p>
            <ExampleQueries onSelectQuery={onSendMessage} disabled={isLoading} />
          </div>
        )}

        {!showWelcome && (
          <MessageList
            messages={messages}
            streamingContent={streamingContent}
            isLoading={isLoading}
            isRetrievingMemories={isRetrievingMemories}
            isLoadingFollowUps={isLoadingFollowUps}
            memories={memories}
            onFollowUpClick={onSendMessage}
          />
        )}

        {error && (
          <div className="error-message">
            <strong>Error:</strong> {error}
          </div>
        )}
      </div>

      <ChatInput onSend={onSendMessage} disabled={isLoading} />
    </div>
  );
}
