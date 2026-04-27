import { useCompareChat } from './hooks/useCompareChat';
import { ComparisonChatInterface } from './components/ComparisonChatInterface';
import './App.css';

function App() {
  const {
    messages,
    currentMemories,
    isLoading,
    isRetrievingMemories,
    isLoadingFollowUps,
    error,
    comparison,
    sendMessage,
    clearChat,
  } = useCompareChat();

  return (
    <div className="app comparison-mode">
      <ComparisonChatInterface
        messages={messages}
        comparison={comparison}
        isLoading={isLoading}
        isRetrievingMemories={isRetrievingMemories}
        isLoadingFollowUps={isLoadingFollowUps}
        error={error}
        memories={currentMemories}
        onSendMessage={sendMessage}
        onClearChat={clearChat}
      />
    </div>
  );
}

export default App;
