import { useState } from 'react';
import { Memory } from '../types';

interface MemoryPanelProps {
  memories: Memory[];
  isLoading: boolean;
}

interface MemoryCardProps {
  memory: Memory;
  citationNumber: number;
}

function MemoryCard({ memory, citationNumber }: MemoryCardProps) {
  const [showOriginal, setShowOriginal] = useState(false);

  return (
    <div className="memory-card" data-memory-id={memory.id}>
      {/* Citation badge */}
      <div className="memory-citation-badge">[{citationNumber}]</div>

      {/* Header with book/chapter info */}
      <div className="memory-metadata">
        <span className="memory-book">{memory.metadata.bookTitle}</span>
        {(memory.metadata.chapterNumber || memory.metadata.chapterName) && (
          <span className="memory-chapter">
            {memory.metadata.chapterNumber ? `Chapter ${memory.metadata.chapterNumber}` : ''}
            {memory.metadata.chapterNumber && memory.metadata.chapterName ? ': ' : ''}
            {memory.metadata.chapterName || ''}
          </span>
        )}
      </div>

      {/* Subject/Title */}
      {memory.subject && (
        <div className="memory-subject">{memory.subject}</div>
      )}

      {/* Main content - summary or content */}
      <div className="memory-summary">
        {memory.summary || memory.content || '(no content)'}
      </div>

      {/* Original content toggle */}
      {memory.originalContent && (
        <div className="memory-original-section">
          <button
            className="memory-toggle-btn"
            onClick={() => setShowOriginal(!showOriginal)}
          >
            {showOriginal ? 'Hide original text' : 'Show original text'}
          </button>
          {showOriginal && (
            <div className="memory-original">
              {memory.originalContent}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function MemoryPanel({ memories, isLoading }: MemoryPanelProps) {
  return (
    <div className="memory-panel">
      <div className="memory-panel-header">
        <span className="memory-panel-icon">✦</span>
        <span className="memory-panel-title">Retrieved Memories</span>
        <span className="memory-panel-count">{memories.length > 0 && memories.length}</span>
      </div>

      {isLoading && memories.length === 0 ? (
        <div className="memory-loading">
          <div className="loading-spinner"></div>
          <p>Retrieving memories...</p>
        </div>
      ) : memories.length === 0 ? (
        <div className="memory-empty">
          <p>No memories retrieved yet. Ask a question to see relevant excerpts from the books.</p>
        </div>
      ) : (
        <div className="memory-list">
          {memories.map((memory, index) => (
            <MemoryCard key={memory.id} memory={memory} citationNumber={index + 1} />
          ))}
        </div>
      )}
    </div>
  );
}
