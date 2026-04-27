"""
Search stage - retrieve relevant memories.
"""
import asyncio
from typing import List, Any, Optional
from logging import Logger
from tqdm import tqdm

from evaluation.src.core.data_models import QAPair, SearchResult
from evaluation.src.adapters.base import BaseAdapter
from evaluation.src.utils.checkpoint import CheckpointManager


async def run_search_stage(
    adapter: BaseAdapter,
    qa_pairs: List[QAPair],
    index: Any,
    conversations: List,
    checkpoint_manager: Optional[CheckpointManager],
    logger: Logger,
) -> List[SearchResult]:
    """
    Execute concurrent search with fine-grained checkpointing.
    
    Process by conversation groups, save checkpoint after each conversation.
    
    Args:
        adapter: System adapter
        qa_pairs: List of QA pairs
        index: Index
        conversations: Conversation list (for online API cache rebuild)
        checkpoint_manager: Checkpoint manager for resume
        logger: Logger
        
    Returns:
        List of search results
    """
    print(f"\n{'='*60}")
    print(f"Stage 2/4: Search")
    print(f"{'='*60}")
    
    # Load fine-grained checkpoint
    all_search_results_dict = {}
    if checkpoint_manager:
        all_search_results_dict = checkpoint_manager.load_search_progress()
    
    # Group QA pairs by conversation
    conv_to_qa = {}
    for qa in qa_pairs:
        conv_id = qa.metadata.get("conversation_id", "unknown")
        if conv_id not in conv_to_qa:
            conv_to_qa[conv_id] = []
        conv_to_qa[conv_id].append(qa)
    
    total_convs = len(conv_to_qa)
    processed_convs = set(all_search_results_dict.keys())
    remaining_convs = set(conv_to_qa.keys()) - processed_convs
    
    print(f"Total conversations: {total_convs}")
    print(f"Total questions: {len(qa_pairs)}")
    if processed_convs:
        print(f"Already processed: {len(processed_convs)} conversations (from checkpoint)")
        print(f"Remaining: {len(remaining_convs)} conversations")
    
    # Build conversation_id to conversation mapping (for online API cache rebuild)
    conv_id_to_conv = {conv.conversation_id: conv for conv in conversations}
    
    # Search-stage concurrency can be configured separately via system config:
    #   search.num_workers (fallback to adapter.num_workers, then 20)
    search_cfg = adapter.config.get("search", {})
    num_workers = int(search_cfg.get("num_workers", getattr(adapter, "num_workers", 20)))
    semaphore = asyncio.Semaphore(num_workers)
    print(f"Search concurrency: {num_workers} workers")
    
    # Create fine-grained progress bar (track by questions)
    total_questions = len(qa_pairs)
    processed_questions = sum(len(all_search_results_dict.get(conv_id, [])) for conv_id in processed_convs)
    
    pbar = tqdm(
        total=total_questions,
        initial=processed_questions,
        desc="🔍 Search Progress",
        unit="qa"
    )
    
    async def search_single_with_tracking(qa):
        async with semaphore:
            conv_id = qa.metadata.get("conversation_id", "0")
            conversation = conv_id_to_conv.get(conv_id)
            
            # Search with timeout and retry (similar to answer_stage.py)
            max_retries = 3
            timeout_seconds = 300.0  # Increased from 120s for complex agentic retrieval
            result = None
            
            for attempt in range(max_retries):
                try:
                    result = await asyncio.wait_for(
                        adapter.search(qa.question, conv_id, index, conversation=conversation),
                        timeout=timeout_seconds
                    )
                    break  # Success, exit retry loop
                    
                except asyncio.TimeoutError:
                    if attempt < max_retries - 1:
                        tqdm.write(f"  ⏱️  Search timeout ({timeout_seconds}s) for question in {conv_id}, retry {attempt + 1}/{max_retries}...")
                        await asyncio.sleep(2)  # Short delay before retry
                    else:
                        tqdm.write(f"  ❌ Search timeout after {max_retries} attempts for question in {conv_id}: {qa.question[:60]}...")
                        # Return empty search result on timeout
                        from evaluation.src.core.data_models import SearchResult
                        result = SearchResult(
                            query=qa.question,
                            conversation_id=conv_id,
                            results=[],
                            retrieval_metadata={"error": "Search timeout after retries"}
                        )
                
                except Exception as e:
                    if attempt < max_retries - 1:
                        tqdm.write(f"  ⚠️  Search failed for question in {conv_id}: {str(e)}, retry {attempt + 1}/{max_retries}...")
                        await asyncio.sleep(2)
                    else:
                        tqdm.write(f"  ❌ Search failed after {max_retries} attempts for question in {conv_id}: {str(e)}")
                        # Return empty search result on error
                        from evaluation.src.core.data_models import SearchResult
                        result = SearchResult(
                            query=qa.question,
                            conversation_id=conv_id,
                            results=[],
                            retrieval_metadata={"error": f"Search error: {str(e)}"}
                        )
            
            pbar.update(1)  # Update progress bar after each question
            return result
    
    # Process by conversation (use numeric sort for conversation IDs like "longmemeval_10")
    def sort_key(item):
        """Sort by numeric part of conversation_id if possible, else alphabetically."""
        conv_id = item[0]
        # Try to extract numeric suffix (e.g., "longmemeval_10" -> 10)
        parts = conv_id.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return (parts[0], int(parts[1]))
        return (conv_id, 0)
    
    for idx, (conv_id, qa_list) in enumerate(sorted(conv_to_qa.items(), key=sort_key)):
        # Skip already processed conversations
        if conv_id in processed_convs:
            tqdm.write(f"⏭️  Skipping Conversation ID: {conv_id} (already processed)")
            continue
        
        tqdm.write(f"Processing Conversation ID: {conv_id} ({idx+1}/{total_convs}) - {len(qa_list)} questions")
        
        # Process all questions for this conversation concurrently
        tasks = [search_single_with_tracking(qa) for qa in qa_list]
        results_for_conv = await asyncio.gather(*tasks)
        
        # Save results in dict format
        results_for_conv_dict = [
            {
                "question_id": qa.question_id,
                "query": qa.question,
                "conversation_id": conv_id,
                "results": result.results,
                "retrieval_metadata": result.retrieval_metadata
            }
            for qa, result in zip(qa_list, results_for_conv)
        ]
        
        all_search_results_dict[conv_id] = results_for_conv_dict
        
        # Save checkpoint after each conversation
        if checkpoint_manager:
            checkpoint_manager.save_search_progress(all_search_results_dict)
    
    # Close progress bar
    pbar.close()
    
    # Delete fine-grained checkpoint after completion
    if checkpoint_manager:
        checkpoint_manager.delete_search_checkpoint()
    
    # Convert dict format to SearchResult object list (maintain original return format)
    # Use same numeric sort as above to ensure consistent ordering
    def sort_key_conv_id(conv_id):
        """Sort by numeric part of conversation_id if possible, else alphabetically."""
        parts = conv_id.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return (parts[0], int(parts[1]))
        return (conv_id, 0)
    
    all_results = []
    for conv_id in sorted(conv_to_qa.keys(), key=sort_key_conv_id):
        if conv_id in all_search_results_dict:
            for result_dict in all_search_results_dict[conv_id]:
                all_results.append(SearchResult(
                    query=result_dict["query"],
                    conversation_id=result_dict["conversation_id"],
                    results=result_dict["results"],
                    retrieval_metadata=result_dict.get("retrieval_metadata", {})
                ))
    
    print(f"\n{'='*60}")
    print(f"🎉 All conversations processed!")
    print(f"{'='*60}")
    print(f"✅ Search completed: {len(all_results)} results\n")
    return all_results

