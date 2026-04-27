import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from time import time
from typing import List, Dict, Optional

import pandas as pd
from tqdm import tqdm


from evaluation.src.adapters.evermemos.config import ExperimentConfig
from evaluation.src.adapters.evermemos.prompts.answer_prompts import ANSWER_PROMPT

# Use Memory Layer's LLMProvider
from memory_layer.llm.llm_provider import LLMProvider


# Context building template (migrated from stage3)
TEMPLATE = """Episodes memories for conversation between {speaker_1} and {speaker_2}:

    {speaker_memories}
"""


def load_memcells_by_conversation(conv_idx: int, memcells_dir: Path) -> Dict[str, dict]:
    """
    Load all memcells for specified conversation, return event_id -> memcell mapping.

    Args:
        conv_idx: Conversation index
        memcells_dir: Memcells directory path

    Returns:
        Mapping of {event_id: memcell_dict}
    """
    memcell_file = memcells_dir / f"memcell_list_conv_{conv_idx}.json"

    if not memcell_file.exists():
        print(f"Warning: Memcell file not found: {memcell_file}")
        return {}

    try:
        with open(memcell_file, "r", encoding="utf-8") as f:
            memcells = json.load(f)

        # Build event_id -> memcell mapping
        memcell_map = {}
        for memcell in memcells:
            event_id = memcell.get("event_id")
            if event_id:
                memcell_map[event_id] = memcell

        return memcell_map

    except Exception as e:
        print(f"Error loading memcells from {memcell_file}: {e}")
        return {}


def build_context_from_event_ids(
    event_ids: List[str],
    memcell_map: Dict[str, dict],
    speaker_a: str,
    speaker_b: str,
    top_k: int = 10,
) -> str:
    """
    Extract corresponding episode memory from memcell_map based on event_ids and build context.

    Args:
        event_ids: Retrieved event_ids list (sorted by relevance)
        memcell_map: Mapping of event_id -> memcell
        speaker_a: Speaker A
        speaker_b: Speaker B
        top_k: Select top k event_ids (default 10)

    Returns:
        Formatted context string
    """
    # Select top-k event_ids
    selected_event_ids = event_ids[:top_k]

    # Extract corresponding episode memory from memcell_map
    retrieved_docs_text = []
    for event_id in selected_event_ids:
        memcell = memcell_map.get(event_id)
        if not memcell:
            # Cannot find corresponding memcell, skip
            continue

        subject = memcell.get('subject', 'N/A')
        episode = memcell.get('episode', 'N/A')
        doc_text = f"{subject}: {episode}\n---"
        retrieved_docs_text.append(doc_text)

    # Concatenate all documents
    speaker_memories = "\n\n".join(retrieved_docs_text)

    # Format final context using template
    context = TEMPLATE.format(
        speaker_1=speaker_a, speaker_2=speaker_b, speaker_memories=speaker_memories
    )

    return context


async def locomo_response(
    llm_provider: LLMProvider,  # Use LLMProvider
    context: str,
    question: str,
    experiment_config: ExperimentConfig,
) -> str:
    """Generate answer (using LLMProvider).

    Args:
        llm_provider: LLM Provider
        context: Retrieved context
        question: User question
        experiment_config: Experiment configuration

    Returns:
        Generated answer
    """
    prompt = ANSWER_PROMPT.format(context=context, question=question)

    for i in range(experiment_config.max_retries):
        try:
            result = await llm_provider.generate(prompt=prompt, temperature=0)

            # Safe parse FINAL ANSWER (avoid index out of range)
            if "FINAL ANSWER:" in result:
                parts = result.split("FINAL ANSWER:")
                if len(parts) > 1:
                    result = parts[1].strip()
                else:
                    # Split failed, use original result
                    result = result.strip()
            else:
                # No FINAL ANSWER marker, use original result
                result = result.strip()

            if result == "":
                continue
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

    return result


async def process_qa(
    qa,
    search_result,
    llm_provider,
    experiment_config,
    memcell_map: Dict[str, dict],
    speaker_a: str,
    speaker_b: str,
):
    """
    Process single QA pair (new version: build context from event_ids).

    Args:
        qa: Question and answer pair
        search_result: Retrieval result (contains event_ids)
        llm_provider: LLM Provider
        experiment_config: Experiment configuration
        memcell_map: Mapping of event_id -> memcell
        speaker_a: Speaker A
        speaker_b: Speaker B

    Returns:
        Dictionary containing question, answer, category, etc.
    """
    start = time()
    query = qa.get("question")
    gold_answer = qa.get("answer")
    qa_category = qa.get("category")

    # Build context from event_ids (using top_k)
    event_ids = search_result.get("event_ids", [])
    top_k = experiment_config.response_top_k

    context = build_context_from_event_ids(
        event_ids=event_ids,
        memcell_map=memcell_map,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        top_k=top_k,
    )

    answer = await locomo_response(llm_provider, context, query, experiment_config)

    response_duration_ms = (time() - start) * 1000

    # Only output in verbose mode (reduce logs)
    # print(f"Processed question: {query}")
    # print(f"Answer: {answer}")

    return {
        "question": query,
        "answer": answer,
        "category": qa_category,
        "golden_answer": gold_answer,
        "search_context": context,  # Save built context
        "event_ids_used": event_ids[:top_k],  # Record actually used event_ids
        "response_duration_ms": response_duration_ms,
        "search_duration_ms": search_result.get("retrieval_metadata", {}).get(
            "total_latency_ms", 0
        ),
    }


async def main(search_path, save_path):
    """
    Optimized main function.

    Performance optimizations:
    1. Global concurrent processing: all QA pairs processed concurrently instead of serial by conversation
    2. Concurrency control: use Semaphore to control maximum concurrency
    3. Progress monitoring: real-time progress display
    4. Incremental saving: save immediately after each conversation completes (avoid data loss on crash)

    Optimization results:
    - Before: 77 minutes (serial)
    - After: ~8 minutes (concurrent 50)
    - Speedup: ~10x
    """
    llm_config = ExperimentConfig.llm_config["openai"]
    experiment_config = ExperimentConfig()

    # Create LLM Provider (replaces AsyncOpenAI)
    llm_provider = LLMProvider(
        provider_type="openai",
        model=llm_config["model"],
        api_key=llm_config["api_key"],
        base_url=llm_config["base_url"],
        temperature=llm_config.get("temperature", 0.0),
        max_tokens=llm_config.get("max_tokens", 32768),
    )

    locomo_df = pd.read_json(experiment_config.datase_path)
    with open(search_path) as file:
        locomo_search_results = json.load(file)

    num_users = len(locomo_df)

    # Load memcells directory
    memcells_dir = Path(search_path).parent / "memcells"
    if not memcells_dir.exists():
        print(f"Error: Memcells directory not found: {memcells_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Stage4: LLM Response Generation")
    print(f"{'='*60}")
    print(f"Total conversations: {num_users}")
    print(f"Response top-k: {experiment_config.response_top_k}")
    print(f"Memcells directory: {memcells_dir}")

    # Global concurrency control (key optimization)
    # Control number of QA pairs processed simultaneously to avoid API rate limiting
    MAX_CONCURRENT = 50  # Adjust based on API limits (10-100)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Collect all QA pairs (across conversations)
    all_tasks = []
    task_to_group = {}  # Track which group each task belongs to

    # Define processing function with concurrency control
    async def process_qa_with_semaphore(
        qa, search_result, group_id, memcell_map, speaker_a, speaker_b
    ):
        """QA processing with concurrency control."""
        async with semaphore:
            result = await process_qa(
                qa,
                search_result,
                llm_provider,
                experiment_config,
                memcell_map,
                speaker_a,
                speaker_b,
            )
            return (group_id, result)

    total_qa_count = 0
    for group_idx in range(num_users):
        qa_set = locomo_df["qa"].iloc[group_idx]
        qa_set_filtered = [qa for qa in qa_set if qa.get("category") != 5]

        group_id = f"locomo_exp_user_{group_idx}"
        search_results = locomo_search_results.get(group_id)

        # Load memcells for current conversation
        memcell_map = load_memcells_by_conversation(group_idx, memcells_dir)
        print(f"Loaded {len(memcell_map)} memcells for conversation {group_idx}")

        # Get speaker information
        conversation_data = locomo_df["conversation"].iloc[group_idx]
        speaker_a = conversation_data.get("speaker_a", "Speaker A")
        speaker_b = conversation_data.get("speaker_b", "Speaker B")

        matched_pairs = []
        for qa in qa_set_filtered:
            question = qa.get("question")
            matching_result = next(
                (
                    result
                    for result in search_results
                    if result.get("query") == question
                ),
                None,
            )
            if matching_result:
                matched_pairs.append((qa, matching_result))
            else:
                print(
                    f"Warning: No matching search result found for question: {question}"
                )

        total_qa_count += len(matched_pairs)

        # Create tasks (global concurrency)
        for qa, search_result in matched_pairs:
            task = process_qa_with_semaphore(
                qa, search_result, group_id, memcell_map, speaker_a, speaker_b
            )
            all_tasks.append(task)

    print(f"Total questions to process: {total_qa_count}")
    print(f"Max concurrent requests: {MAX_CONCURRENT}")
    print(f"Estimated time: {total_qa_count * 3 / MAX_CONCURRENT / 60:.1f} minutes")
    print(f"\n{'='*60}")
    print(f"Starting parallel processing...")
    print(f"{'='*60}\n")

    # Execute all tasks globally concurrently (with progress monitoring)
    all_responses = {f"locomo_exp_user_{i}": [] for i in range(num_users)}

    import time as time_module

    start_time = time_module.time()
    completed = 0
    failed = 0

    # Batch processing + incremental saving (avoid data loss on crash)
    CHUNK_SIZE = 200  # Process 200 tasks at a time
    SAVE_INTERVAL = 400  # Save every 400 tasks

    for chunk_start in range(0, len(all_tasks), CHUNK_SIZE):
        chunk_tasks = all_tasks[chunk_start : chunk_start + CHUNK_SIZE]
        chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)

        # Group results into each conversation
        for result in chunk_results:
            if isinstance(result, Exception):
                print(f"  ❌ Task failed: {result}")
                failed += 1
                continue

            group_id, qa_result = result
            all_responses[group_id].append(qa_result)

        completed += len(chunk_tasks)
        elapsed = time_module.time() - start_time
        speed = completed / elapsed if elapsed > 0 else 0
        eta = (total_qa_count - completed) / speed if speed > 0 else 0

        print(
            f"Progress: {completed}/{total_qa_count} ({completed/total_qa_count*100:.1f}%) | "
            f"Speed: {speed:.1f} qa/s | Failed: {failed} | ETA: {eta/60:.1f} min"
        )

        # Incremental saving (save every SAVE_INTERVAL tasks)
        if completed % SAVE_INTERVAL == 0 or completed == total_qa_count:
            temp_save_path = (
                Path(save_path).parent / f"responses_checkpoint_{completed}.json"
            )
            with open(temp_save_path, "w", encoding="utf-8") as f:
                json.dump(all_responses, f, indent=2, ensure_ascii=False)
            print(f"  💾 Checkpoint saved: {temp_save_path.name}")

    elapsed_time = time_module.time() - start_time
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

    # Save final results
    os.makedirs(Path(save_path).parent, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_responses, f, indent=2, ensure_ascii=False)
        print(f"✅ Final results saved to: {save_path}")

    # Clean up checkpoint files
    checkpoint_files = list(Path(save_path).parent.glob("responses_checkpoint_*.json"))
    for checkpoint_file in checkpoint_files:
        checkpoint_file.unlink()
        print(f"  🗑️  Removed checkpoint: {checkpoint_file.name}")


if __name__ == "__main__":
    config = ExperimentConfig()
    search_result_path = str(
        Path(__file__).parent / config.experiment_name / "search_results.json"
    )
    save_path = Path(__file__).parent / config.experiment_name / "responses.json"
    asyncio.run(main(search_result_path, save_path))
