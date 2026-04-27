import { useEffect, useRef, useState, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import type { Components } from 'react-markdown';
import { Message, Memory } from '../types';

interface MessageListProps {
  messages: Message[];
  streamingContent: string;
  isLoading: boolean;
  isRetrievingMemories: boolean;
  isLoadingFollowUps: boolean;
  memories: Memory[];
  onFollowUpClick: (question: string) => void;
}

interface CitationProps {
  citationNumber: number;
  memory: Memory | undefined;
}

function Citation({ citationNumber, memory }: CitationProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [showOriginal, setShowOriginal] = useState(false);

  if (!memory) {
    // Fallback if memory not found - just show the citation number
    return <span className="citation-badge citation-missing">memory [{citationNumber}]</span>;
  }

  const handleClick = () => {
    setIsExpanded(!isExpanded);
    // Also highlight in side panel
    const sidePanel = document.querySelector(`[data-memory-id="${memory.id}"]`);
    if (sidePanel) {
      sidePanel.classList.add('memory-highlighted');
      sidePanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      setTimeout(() => sidePanel.classList.remove('memory-highlighted'), 2000);
    }
  };

  return (
    <span className="citation-wrapper">
      <span
        className={`citation-badge ${isExpanded ? 'citation-expanded' : ''}`}
        onClick={handleClick}
        title={memory.subject || `Memory ${citationNumber}`}
      >
        memory [{citationNumber}]
      </span>
      {isExpanded && (
        <span className="citation-expanded-block">
          <span className="citation-expanded-header">
            <span className="citation-expanded-badge">[{citationNumber}]</span>
            <span className="citation-expanded-title">{memory.subject || 'Memory'}</span>
            <button
              className="citation-close-btn"
              onClick={(e) => { e.stopPropagation(); setIsExpanded(false); }}
            >
              ×
            </button>
          </span>
          <span className="citation-expanded-meta">
            {memory.metadata.bookTitle}
            {memory.metadata.chapterNumber && ` - Chapter ${memory.metadata.chapterNumber}`}
            {memory.metadata.chapterName && `: ${memory.metadata.chapterName}`}
          </span>
          <span className="citation-expanded-content">
            {memory.summary || memory.content}
          </span>
          {memory.originalContent && (
            <span className="citation-original-section">
              <button
                className="citation-toggle-original"
                onClick={(e) => { e.stopPropagation(); setShowOriginal(!showOriginal); }}
              >
                {showOriginal ? 'Hide original text' : 'Show original text'}
              </button>
              {showOriginal && (
                <span className="citation-original-text">{memory.originalContent}</span>
              )}
            </span>
          )}
        </span>
      )}
    </span>
  );
}

interface CitationContentProps {
  content: string;
  memories: Memory[];
}

function CitationContent({ content, memories }: CitationContentProps) {
  // Parse text and replace [1], [2], etc. with Citation components
  const parts: (string | JSX.Element)[] = [];
  const regex = /\[(\d+)\]/g;
  let lastIndex = 0;
  let match;

  while ((match = regex.exec(content)) !== null) {
    // Add text before the citation
    if (match.index > lastIndex) {
      parts.push(content.slice(lastIndex, match.index));
    }

    // Add the citation component
    const citationNumber = parseInt(match[1], 10);
    const memory = memories[citationNumber - 1]; // Citations are 1-indexed
    parts.push(
      <Citation
        key={`citation-${match.index}`}
        citationNumber={citationNumber}
        memory={memory}
      />
    );

    lastIndex = regex.lastIndex;
  }

  // Add remaining text after last citation
  if (lastIndex < content.length) {
    parts.push(content.slice(lastIndex));
  }

  return <>{parts}</>;
}

export function MessageList({ messages, streamingContent, isLoading, isRetrievingMemories, isLoadingFollowUps, memories, onFollowUpClick }: MessageListProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingContent]);

  // Create custom markdown components to handle citations in text
  const createMarkdownComponents = useCallback((mems: Memory[]): Components => ({
    p: ({ children }) => {
      // Process children to handle citations
      const processedChildren = processChildren(children, mems);
      return <p>{processedChildren}</p>;
    },
    li: ({ children }) => {
      const processedChildren = processChildren(children, mems);
      return <li>{processedChildren}</li>;
    },
    strong: ({ children }) => {
      const processedChildren = processChildren(children, mems);
      return <strong>{processedChildren}</strong>;
    },
    em: ({ children }) => {
      const processedChildren = processChildren(children, mems);
      return <em>{processedChildren}</em>;
    },
  }), []);

  // Helper to process children and replace citation patterns
  const processChildren = (children: React.ReactNode, mems: Memory[]): React.ReactNode => {
    if (!children) return children;

    if (typeof children === 'string') {
      // Check if string contains citations
      if (/\[\d+\]/.test(children)) {
        return <CitationContent content={children} memories={mems} />;
      }
      return children;
    }

    if (Array.isArray(children)) {
      return children.map((child, index) => {
        if (typeof child === 'string' && /\[\d+\]/.test(child)) {
          return <CitationContent key={index} content={child} memories={mems} />;
        }
        return child;
      });
    }

    return children;
  };

  const markdownComponents = createMarkdownComponents(memories);

  return (
    <div className="message-list">
      {messages.map((message, index) => (
        <div key={index} className={`message message-${message.role}`}>
          <div className="message-role">
            {message.role === 'user' ? 'You' : 'The Maester'}
          </div>
          <div className="message-content">
            {message.role === 'assistant' ? (
              <ReactMarkdown components={markdownComponents}>
                {message.content}
              </ReactMarkdown>
            ) : (
              <ReactMarkdown>{message.content}</ReactMarkdown>
            )}
          </div>
          {message.role === 'assistant' && (
            <>
              {/* Show follow-up questions if available */}
              {message.followUps && message.followUps.length > 0 && (
                <div className="follow-ups">
                  <div className="follow-ups-label">Follow-up questions:</div>
                  <div className="follow-ups-list">
                    {message.followUps.map((question, qIndex) => (
                      <button
                        key={qIndex}
                        className="follow-up-btn"
                        onClick={() => onFollowUpClick(question)}
                        disabled={isLoading}
                      >
                        {question}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              {/* Show loading indicator for follow-ups on the last message */}
              {isLoadingFollowUps && index === messages.length - 1 && !message.followUps && (
                <div className="follow-ups follow-ups-loading">
                  <div className="follow-ups-label">
                    <span className="follow-ups-spinner"></span>
                    Generating follow-up questions...
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      ))}

      {streamingContent && (
        <div className="message message-assistant">
          <div className="message-role">The Maester</div>
          <div className="message-content">
            <ReactMarkdown components={markdownComponents}>
              {streamingContent}
            </ReactMarkdown>
            <span className="typing-indicator"></span>
          </div>
        </div>
      )}

      {isLoading && !streamingContent && (
        <div className="message message-assistant">
          <div className="message-role">The Maester</div>
          <div className="message-content">
            <span className="typing-indicator">
              {isRetrievingMemories ? 'Retrieving memories...' : 'Thinking...'}
            </span>
          </div>
        </div>
      )}

      <div ref={messagesEndRef} />
    </div>
  );
}
