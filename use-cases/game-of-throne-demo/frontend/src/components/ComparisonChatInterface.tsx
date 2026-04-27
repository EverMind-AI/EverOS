import { ChatInput } from './ChatInput';
import { ExampleQueries } from './ExampleQueries';
import { ComparisonView } from './ComparisonView';
import { Memory, Message, ComparisonState } from '../types';

interface ComparisonChatInterfaceProps {
  messages: Message[];
  comparison: ComparisonState;
  isLoading: boolean;
  isRetrievingMemories: boolean;
  isLoadingFollowUps: boolean;
  error: string | null;
  memories: Memory[];
  onSendMessage: (message: string) => void;
  onClearChat: () => void;
}

export function ComparisonChatInterface({
  messages,
  comparison,
  isLoading,
  isRetrievingMemories,
  isLoadingFollowUps,
  error,
  memories,
  onSendMessage,
  onClearChat,
}: ComparisonChatInterfaceProps) {
  const showWelcome = messages.length === 0 && !comparison.withMemory.content && !comparison.withoutMemory.content;
  const showComparison = comparison.withMemory.content || comparison.withoutMemory.content || comparison.withMemory.isStreaming || comparison.withoutMemory.isStreaming;

  // Get the last user message to display
  const lastUserMessage = [...messages].reverse().find(m => m.role === 'user');

  // Get follow-ups from the last assistant message
  const lastMessage = messages[messages.length - 1];
  const followUps = lastMessage?.role === 'assistant' ? lastMessage.followUps : undefined;

  return (
    <div className="comparison-chat-interface">
      <div className="chat-header">
        <div className="chat-header-title">
          <div className="chat-header-main">
            <span className="evermem-logo-text">EverMind</span>
            <h1><span className="brand-evermem">EverMem</span> Story Memory Demo</h1>
          </div>
          <span className="chat-header-subtitle">A Game of Thrones - Side-by-Side Comparison · Powered by Claude Haiku</span>
        </div>
        <button onClick={onClearChat} className="clear-button" disabled={isLoading}>
          Clear
        </button>
      </div>

      <div className="comparison-main-content">
        {showWelcome && (
          <div className="comparison-welcome">
            <h2>See the Difference Memory Makes</h2>
            <p>
              Ask any question about <strong>A Game of Thrones</strong> and watch two responses stream side-by-side:
            </p>
            <ul className="comparison-feature-list">
              <li><span className="comparison-badge with-memory">With Memory</span> Uses EverMem to retrieve relevant story details</li>
              <li><span className="comparison-badge without-memory">Without Memory</span> Standard LLM response with no context</li>
            </ul>
            <ExampleQueries onSelectQuery={onSendMessage} disabled={isLoading} />
          </div>
        )}

        {/* Show user's question */}
        {lastUserMessage && (
          <div className="comparison-user-question">
            <div className="comparison-user-label">Your Question</div>
            <div className="comparison-user-content">{lastUserMessage.content}</div>
          </div>
        )}

        {showComparison && (
          <ComparisonView
            comparison={comparison}
            memories={memories}
            isRetrievingMemories={isRetrievingMemories}
            followUps={followUps}
            isLoadingFollowUps={isLoadingFollowUps}
            onFollowUpClick={onSendMessage}
            isLoading={isLoading}
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
