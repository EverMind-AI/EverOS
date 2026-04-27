"""
Answer stage - generate answers.
"""
import asyncio
import time
from typing import List, Optional
from logging import Logger
from tqdm import tqdm

from evaluation.src.core.data_models import QAPair, SearchResult, AnswerResult
from evaluation.src.adapters.base import BaseAdapter
from evaluation.src.utils.checkpoint import CheckpointManager


def build_context(search_result: SearchResult) -> str:
    """
    Build context from search results.
    
    Prefer pre-formatted context (dual-speaker scenarios), else use simple numbering (single-speaker scenarios).
    
    Args:
        search_result: Search result
        
    Returns:
        Context string
    """
    # Prefer pre-formatted context (provided by adapter)
    formatted_context = search_result.retrieval_metadata.get("formatted_context", "")
    if formatted_context:
        return formatted_context
    
    # Single speaker scenario: simple formatting
    context_parts = []
    
    # Get top_k from retrieval_metadata, default to len(results) if not specified
    top_k = search_result.retrieval_metadata.get("top_k", len(search_result.results))
    
    # Add memory content (use top_k instead of hardcoded 10)
    for idx, result in enumerate(search_result.results[:top_k], 1):
        content = result.get("content", "")
        context_parts.append(f"{idx}. {content}")
    
    context = "\n\n".join(context_parts)
    
    # For systems supporting preferences (e.g., Memos), add formatted pref_string
    preferences = search_result.retrieval_metadata.get("preferences", {})
    pref_string = preferences.get("pref_string", "")
    
    if pref_string:
        context += "\n\n" + pref_string
    
    return context


async def run_answer_stage(
    adapter: BaseAdapter,
    qa_pairs: List[QAPair],
    search_results: List[SearchResult],
    checkpoint_manager: Optional[CheckpointManager],
    logger: Logger,
) -> List[AnswerResult]:
    """
    Generate answers with fine-grained checkpointing.
    
    Save checkpoint every SAVE_INTERVAL questions.
    
    Args:
        adapter: System adapter
        qa_pairs: List of QA pairs
        search_results: List of search results
        checkpoint_manager: Checkpoint manager for resume
        logger: Logger
        
    Returns:
        List of answer results
    """
    print(f"\n{'='*60}")
    print(f"Stage 3/4: Answer")
    print(f"{'='*60}")
    
    SAVE_INTERVAL = 400  # Save every 400 tasks
    MAX_CONCURRENT = 50  # Max concurrency
    
    # Load fine-grained checkpoint
    all_answer_results = {}
    if checkpoint_manager:
        loaded_results = checkpoint_manager.load_answer_progress()
        # Convert to {question_id: AnswerResult} format
        for result in loaded_results.values():
            all_answer_results[result["question_id"]] = result
    
    total_qa_count = len(qa_pairs)
    processed_count = len(all_answer_results)
    
    print(f"Total questions: {total_qa_count}")
    if processed_count > 0:
        print(f"Already processed: {processed_count} questions (from checkpoint)")
        print(f"Remaining: {total_qa_count - processed_count} questions")
    
    # Prepare pending tasks
    # qa_pairs and search_results should have matching order (both use numeric sort by conversation_id)
    pending_tasks = []
    for qa, sr in zip(qa_pairs, search_results):
        if qa.question_id not in all_answer_results:
            pending_tasks.append((qa, sr))
    
    if not pending_tasks:
        print(f"✅ All questions already processed!")
        # Convert to AnswerResult object list (original order)
        results = []
        for qa in qa_pairs:
            if qa.question_id in all_answer_results:
                result_dict = all_answer_results[qa.question_id]
                results.append(AnswerResult(
                    question_id=result_dict["question_id"],
                    question=result_dict["question"],
                    answer=result_dict["answer"],
                    golden_answer=result_dict["golden_answer"],
                    category=result_dict.get("category"),
                    conversation_id=result_dict.get("conversation_id", ""),
                    formatted_context=result_dict.get("formatted_context", ""),  # Load formatted_context
                    # search_results not loaded to save space
                ))
        return results
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    completed = processed_count
    failed = 0
    start_time = time.time()
    
    # Use tqdm progress bar
    pbar = tqdm(
        total=total_qa_count,
        initial=processed_count,
        desc="💬 Answer Progress",
        unit="qa"
    )
    
    async def answer_single_with_tracking(qa, search_result):
        nonlocal completed, failed
        
        async with semaphore:
            try:
                # Build context
                context = build_context(search_result)
                
                # Detect multiple-choice and enhance question if needed
                query = qa.question
                if "all_options" in qa.metadata:
                    options = qa.metadata["all_options"]
                    options_text = "\n".join([f"{key} {value}" for key, value in options.items()])
                    
                    # Integrate options and requirements into question
                    query = f"""{qa.question}

OPTIONS:
{options_text}

IMPORTANT: This is a multiple-choice question. You MUST analyze the context and select the BEST option. In your FINAL ANSWER, return ONLY the option letter like (a), (b), (c), or (d), nothing else."""
                
                # Call adapter's answer method with timeout and retry
                max_retries = 3
                timeout_seconds = 120.0  # 3 minutes timeout per attempt
                answer = None
                
                for attempt in range(max_retries):
                    try:
                        answer = await asyncio.wait_for(
                            adapter.answer(
                                query=query,
                                context=context,
                                conversation_id=search_result.conversation_id,
                            ),
                            timeout=timeout_seconds
                        )
                        answer = answer.strip()
                        break  # Success, exit retry loop
                        
                    except asyncio.TimeoutError:
                        if attempt < max_retries - 1:
                            tqdm.write(f"  ⏱️  Timeout (180s) for {qa.question_id}, retry {attempt + 1}/{max_retries}...")
                            await asyncio.sleep(2)  # Short delay before retry
                        else:
                            tqdm.write(f"  ❌ Timeout after {max_retries} attempts for {qa.question_id}: {qa.question[:50]}...")
                            answer = "Error: Answer generation timeout after retries"
                            failed += 1
            
            except Exception as e:
                tqdm.write(f"  ⚠️ Answer generation failed for {qa.question_id}: {e}")
                answer = "Error: Failed to generate answer"
                failed += 1
            
            result = AnswerResult(
                question_id=qa.question_id,
                question=qa.question,
                answer=answer,
                golden_answer=qa.answer,
                category=qa.category,
                conversation_id=search_result.conversation_id,
                formatted_context=context,  # Save actual context used
                metadata=qa.metadata,  # Pass metadata (contains all_options for multiple-choice)
            )
            
            # Save result
            all_answer_results[qa.question_id] = {
                "question_id": result.question_id,
                "question": result.question,
                "answer": result.answer,
                "golden_answer": result.golden_answer,
                "category": result.category,
                "conversation_id": result.conversation_id,
                "formatted_context": result.formatted_context,  # Save formatted_context
                "metadata": result.metadata,  # Save metadata (contains all_options)
            }
            
            completed += 1
            pbar.update(1)  # Update progress bar
            
            # Save checkpoint periodically
            if checkpoint_manager and (completed % SAVE_INTERVAL == 0 or completed == total_qa_count):
                elapsed = time.time() - start_time
                speed = completed / elapsed if elapsed > 0 else 0
                eta = (total_qa_count - completed) / speed if speed > 0 else 0
                
                tqdm.write(f"Progress: {completed}/{total_qa_count} ({completed/total_qa_count*100:.1f}%) | "
                          f"Speed: {speed:.1f} qa/s | Failed: {failed} | ETA: {eta/60:.1f} min")
                
                checkpoint_manager.save_answer_progress(all_answer_results, completed, total_qa_count)
            
            return result
    
    # Create all pending tasks
    tasks = [
        answer_single_with_tracking(qa, sr)
        for qa, sr in pending_tasks
    ]
    
    # Execute concurrently
    await asyncio.gather(*tasks)
    
    # Close progress bar
    pbar.close()
    
    # Statistics
    elapsed_time = time.time() - start_time
    success_rate = (completed - failed) / completed * 100 if completed > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"✅ All responses generated!")
    print(f"   - Total questions: {total_qa_count}")
    print(f"   - Successful: {completed - failed}")
    print(f"   - Failed: {failed}")
    print(f"   - Success rate: {success_rate:.1f}%")
    print(f"   - Time elapsed: {elapsed_time/60:.1f} minutes ({elapsed_time:.0f}s)")
    print(f"   - Average speed: {total_qa_count/elapsed_time:.1f} qa/s")
    print(f"{'='*60}\n")
    
    # Delete fine-grained checkpoints after completion
    if checkpoint_manager:
        checkpoint_manager.delete_answer_checkpoints()
    
    # Convert to AnswerResult object list (original order)
    results = []
    for qa in qa_pairs:
        if qa.question_id in all_answer_results:
            result_dict = all_answer_results[qa.question_id]
            results.append(AnswerResult(
                question_id=result_dict["question_id"],
                question=result_dict["question"],
                answer=result_dict["answer"],
                golden_answer=result_dict["golden_answer"],
                category=result_dict.get("category"),
                conversation_id=result_dict.get("conversation_id", ""),
                formatted_context=result_dict.get("formatted_context", ""),
                search_results=result_dict.get("search_results", []),
                metadata=result_dict.get("metadata", {}),  # Restore metadata
            ))
    
    return results

