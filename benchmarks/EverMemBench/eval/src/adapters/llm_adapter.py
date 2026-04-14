"""
LLM Adapter for multi-person group chat evaluation.

Unlike memory system adapters (memos, mem0, etc.), the LLM adapter uses
the FULL DIALOGUE as context instead of searching for relevant memories.
This is for direct LLM evaluation without memory systems.

Key differences from memory adapters:
- add(): No-op, just stores the dialogue data for later use
- search(): Returns full dialogue formatted as context (no actual search)

Usage:
    python -m eval.cli --system llm --dataset ... --qa ... --stages answer evaluate
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

from eval.src.adapters.base import BaseAdapter
from eval.src.core.data_models import Dataset, AddResult, SearchResult
from eval.src.utils.logger import get_console, print_success


class LLMAdapter(BaseAdapter):
    """
    LLM adapter for direct dialogue-based evaluation.
    
    This adapter does not use any memory system. Instead, it:
    1. Stores the full dialogue dataset during add() (no-op)
    2. Returns the full dialogue as context during search()
    
    The full dialogue context allows LLMs to be evaluated on their ability
    to comprehend and extract information from long conversations.
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: Optional[Path] = None):
        """
        Initialize LLM adapter.
        
        Args:
            config: Configuration dictionary (minimal, no system-specific yaml)
            output_dir: Output directory for results
        """
        super().__init__(config, output_dir)
        
        self.console = get_console()
        
        # Store dialogue data for later use in search()
        self._dialogue_data: Optional[Dataset] = None
        self._dialogue_context: Optional[str] = None
        
        self.console.print("✅ LLMAdapter initialized", style="bold green")
        self.console.print("   Mode: Full dialogue as context (no memory system)")
    
    async def add(
        self,
        dataset: Dataset,
        user_id: str,
        days_to_process: Optional[List[str]] = None,
        **kwargs
    ) -> AddResult:
        """
        Store dataset for later use (no actual add operation).
        
        For LLM system, add() is a no-op that just stores the dialogue
        data for use in search() later.
        
        Args:
            dataset: Dataset with group chat data
            user_id: User ID (not used for LLM)
            days_to_process: Optional list of dates to process
            **kwargs: Additional parameters (ignored)
            
        Returns:
            AddResult indicating success
        """
        self.console.print(f"\n{'='*60}", style="bold cyan")
        self.console.print("Stage: Add (LLM - No-op)", style="bold cyan")
        self.console.print(f"{'='*60}", style="bold cyan")
        
        # Store the dataset for use in search()
        self._dialogue_data = dataset
        
        # Pre-format the dialogue as context string
        self._dialogue_context = self._format_dialogue_as_context(dataset, days_to_process)
        
        # Count messages for reporting
        if days_to_process:
            days = [d for d in dataset.days if d.date in days_to_process]
        else:
            days = dataset.days
        
        total_messages = sum(day.total_messages for day in days)
        
        self.console.print(f"   Dataset: {dataset.name}")
        self.console.print(f"   Days: {len(days)}")
        self.console.print(f"   Total messages: {total_messages}")
        self.console.print(f"   Context length: {len(self._dialogue_context):,} characters")
        
        print_success("LLM adapter: dialogue stored for context (no memory system)")
        
        return AddResult(
            success=True,
            days_processed=len(days),
            messages_sent=0,  # No messages sent to any memory system
            metadata={
                "mode": "llm_full_dialogue",
                "context_length": len(self._dialogue_context),
                "total_messages": total_messages,
            }
        )
    
    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
        **kwargs
    ) -> SearchResult:
        """
        Return full dialogue as context (no actual search).
        
        For LLM system, search() returns the pre-formatted full dialogue
        as context, allowing the LLM to find relevant information itself.
        
        Args:
            query: Search query (the question text)
            user_id: User ID (not used for LLM)
            top_k: Number of memories (not used for LLM)
            **kwargs: Additional parameters
                - question_id: ID of the question (for result tracking)
                
        Returns:
            SearchResult with full dialogue as context
        """
        question_id = kwargs.get("question_id", "unknown")
        
        if self._dialogue_context is None:
            raise ValueError(
                "Dialogue context not initialized. "
                "Call add() first to load the dataset."
            )
        
        return SearchResult(
            question_id=question_id,
            query=query,
            retrieved_memories=[],  # No memory retrieval
            context=self._dialogue_context,
            search_duration_ms=0,  # No search time
            metadata={
                "mode": "llm_full_dialogue",
                "context_length": len(self._dialogue_context),
            }
        )
    
    def _format_dialogue_as_context(
        self,
        dataset: Dataset,
        days_to_process: Optional[List[str]] = None
    ) -> str:
        """
        Format the entire dialogue dataset as a context string.
        
        Format:
        === Date: 2025-01-09 ===
        
        [Group 1]
        [09:32:15] Weihua Zhang: Good morning everyone...
        [09:35:02] Mingzhi Li: Received, Boss Zhang!...
        
        [Group 2]
        [10:00:00] Alice: Hello...
        
        === Date: 2025-01-10 ===
        ...
        
        Args:
            dataset: Dataset with group chat data
            days_to_process: Optional list of dates to include
            
        Returns:
            Formatted dialogue string
        """
        lines: List[str] = []
        
        # Determine which days to include
        if days_to_process:
            days = [d for d in dataset.days if d.date in days_to_process]
        else:
            days = dataset.days
        
        for day in days:
            # Add date header
            lines.append(f"=== Date: {day.date} ===")
            lines.append("")
            
            # Process each group
            for group_name in sorted(day.groups.keys()):
                messages = day.groups[group_name]
                
                # Add group header
                lines.append(f"[{group_name}]")
                
                # Format each message
                for msg in messages:
                    # Extract time from timestamp (HH:MM:SS)
                    time_str = msg.timestamp.strftime("%H:%M:%S")
                    # Format: [HH:MM:SS] Speaker: Content
                    lines.append(f"[{time_str}] {msg.speaker}: {msg.content}")
                
                lines.append("")  # Blank line between groups
            
            lines.append("")  # Blank line between days
        
        return "\n".join(lines)
    
    async def close(self):
        """
        Clean up resources.
        
        LLM adapter has no external resources to clean up.
        """
        self._dialogue_data = None
        self._dialogue_context = None

