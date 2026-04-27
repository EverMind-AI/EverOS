import { useState, useCallback, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import type { Components } from 'react-markdown';
import { Memory, ComparisonState } from '../types';

interface ComparisonViewProps {
  comparison: ComparisonState;
  memories: Memory[];
  isRetrievingMemories: boolean;
  followUps?: string[];
  isLoadingFollowUps: boolean;
  onFollowUpClick: (question: string) => void;
  isLoading: boolean;
}

interface CitationProps {
  citationNumber: number;
  memory: Memory | undefined;
}

function Citation({ citationNumber, memory }: CitationProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [showOriginal, setShowOriginal] = useState(false);

  if (!memory) {
    return <span className="citation-badge citation-missing">memory [{citationNumber}]</span>;
  }

  const handleClick = () => {
    setIsExpanded(!isExpanded);
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

interface MemoryChipProps {
  memory: Memory;
  citationNumber: number;
}

function MemoryChip({ memory, citationNumber }: MemoryChipProps) {
  const title = memory.subject || memory.metadata.chapterName || 'Memory';
  // Truncate to first ~30 chars
  const truncatedTitle = title.length > 30 ? title.slice(0, 30) + '...' : title;

  return (
    <div className="comparison-memory-chip" data-memory-id={memory.id}>
      <span className="comparison-memory-badge">[{citationNumber}]</span>
      <span className="comparison-memory-title">{truncatedTitle}</span>

      {/* Hover popover */}
      <div className="comparison-memory-popover">
        <div className="comparison-memory-popover-header">
          <span className="comparison-memory-popover-badge">[{citationNumber}]</span>
          <span className="comparison-memory-popover-title">{title}</span>
        </div>
        <div className="comparison-memory-meta">
          {memory.metadata.bookTitle}
          {memory.metadata.chapterNumber && ` - Chapter ${memory.metadata.chapterNumber}`}
          {memory.metadata.chapterName && `: ${memory.metadata.chapterName}`}
        </div>
        <div className="comparison-memory-summary">
          {memory.summary || memory.content}
        </div>
      </div>
    </div>
  );
}

interface CitationContentProps {
  content: string;
  memories: Memory[];
}

function CitationContent({ content, memories }: CitationContentProps) {
  const parts: (string | JSX.Element)[] = [];
  const regex = /\[(\d+)\]/g;
  let lastIndex = 0;
  let match;

  while ((match = regex.exec(content)) !== null) {
    if (match.index > lastIndex) {
      parts.push(content.slice(lastIndex, match.index));
    }

    const citationNumber = parseInt(match[1], 10);
    const memory = memories[citationNumber - 1];
    parts.push(
      <Citation
        key={`citation-${match.index}`}
        citationNumber={citationNumber}
        memory={memory}
      />
    );

    lastIndex = regex.lastIndex;
  }

  if (lastIndex < content.length) {
    parts.push(content.slice(lastIndex));
  }

  return <>{parts}</>;
}

interface ComparisonPanelProps {
  title: string;
  badgeClass: string;
  content: string;
  isStreaming: boolean;
  isDone: boolean;
  memories: Memory[];
  showMemories: boolean;
  isRetrievingMemories: boolean;
  followUps?: string[];
  isLoadingFollowUps: boolean;
  onFollowUpClick: (question: string) => void;
  isLoading: boolean;
}

function ComparisonPanel({
  title,
  badgeClass,
  content,
  isStreaming,
  memories,
  showMemories,
  isRetrievingMemories,
  followUps,
  isLoadingFollowUps,
  onFollowUpClick,
  isLoading,
}: ComparisonPanelProps) {
  const contentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (contentRef.current) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight;
    }
  }, [content]);

  const createMarkdownComponents = useCallback((mems: Memory[]): Components => ({
    p: ({ children }) => {
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

  const processChildren = (children: React.ReactNode, mems: Memory[]): React.ReactNode => {
    if (!children) return children;

    if (typeof children === 'string') {
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

  const markdownComponents = createMarkdownComponents(showMemories ? memories : []);

  return (
    <div className="comparison-panel">
      <div className="comparison-panel-header">
        <span className={`comparison-badge ${badgeClass}`}>{title}</span>
      </div>

      {/* Compact memory panel for "With Memory" side */}
      {showMemories && (
        <div className="comparison-memories">
          {isRetrievingMemories && memories.length === 0 ? (
            <div className="comparison-memories-loading">
              <span className="follow-ups-spinner"></span>
              <span>Retrieving memories...</span>
            </div>
          ) : memories.length > 0 ? (
            <div className="comparison-memories-list">
              {memories.map((memory, index) => (
                <MemoryChip key={memory.id} memory={memory} citationNumber={index + 1} />
              ))}
            </div>
          ) : (
            <div className="comparison-memories-empty">No memories</div>
          )}
        </div>
      )}

      {/* No memory placeholder for "Without Memory" side */}
      {!showMemories && (
        <div className="comparison-memories comparison-memories-none">
          <span className="comparison-no-memory-text">No memory context provided</span>
        </div>
      )}

      <div className="comparison-content" ref={contentRef}>
        {content ? (
          <>
            <div className="message-role">The Maester</div>
            <div className="message-content">
              <ReactMarkdown components={markdownComponents}>
                {content}
              </ReactMarkdown>
              {isStreaming && <span className="typing-indicator"></span>}
            </div>

            {/* Follow-ups only for "With Memory" side */}
            {showMemories && !isStreaming && followUps && followUps.length > 0 && (
              <div className="follow-ups">
                <div className="follow-ups-label">Follow-up questions:</div>
                <div className="follow-ups-list">
                  {followUps.map((question, qIndex) => (
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

            {showMemories && !isStreaming && isLoadingFollowUps && !followUps && (
              <div className="follow-ups follow-ups-loading">
                <div className="follow-ups-label">
                  <span className="follow-ups-spinner"></span>
                  Generating follow-up questions...
                </div>
              </div>
            )}
          </>
        ) : isStreaming ? (
          <>
            <div className="message-role">The Maester</div>
            <div className="message-content">
              <span className="typing-indicator">Thinking...</span>
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
}

export function ComparisonView({
  comparison,
  memories,
  isRetrievingMemories,
  followUps,
  isLoadingFollowUps,
  onFollowUpClick,
  isLoading,
}: ComparisonViewProps) {
  return (
    <div className="comparison-panels">
      <ComparisonPanel
        title="With Memory"
        badgeClass="with-memory"
        content={comparison.withMemory.content}
        isStreaming={comparison.withMemory.isStreaming && !comparison.withMemory.isDone}
        isDone={comparison.withMemory.isDone}
        memories={memories}
        showMemories={true}
        isRetrievingMemories={isRetrievingMemories}
        followUps={followUps}
        isLoadingFollowUps={isLoadingFollowUps}
        onFollowUpClick={onFollowUpClick}
        isLoading={isLoading}
      />

      <div className="comparison-divider"></div>

      <ComparisonPanel
        title="Without Memory"
        badgeClass="without-memory"
        content={comparison.withoutMemory.content}
        isStreaming={comparison.withoutMemory.isStreaming && !comparison.withoutMemory.isDone}
        isDone={comparison.withoutMemory.isDone}
        memories={[]}
        showMemories={false}
        isRetrievingMemories={false}
        followUps={undefined}
        isLoadingFollowUps={false}
        onFollowUpClick={onFollowUpClick}
        isLoading={isLoading}
      />
    </div>
  );
}
