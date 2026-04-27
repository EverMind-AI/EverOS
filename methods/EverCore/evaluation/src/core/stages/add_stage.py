"""
Add stage - ingest conversation data and build index.
"""
from pathlib import Path
from typing import List, Any, Optional
from logging import Logger

from evaluation.src.core.data_models import Conversation, Dataset
from evaluation.src.adapters.base import BaseAdapter
from evaluation.src.utils.checkpoint import CheckpointManager


async def run_add_stage(
    adapter: BaseAdapter,
    dataset: Dataset,
    output_dir: Path,
    checkpoint_manager: Optional[CheckpointManager],
    logger: Logger,
    console: Any,
    completed_stages: set,
) -> dict:
    """
    Execute Add stage.
    
    Args:
        adapter: System adapter
        dataset: Standard format dataset
        output_dir: Output directory
        checkpoint_manager: Checkpoint manager for resume
        logger: Logger
        console: Console object
        completed_stages: Set of completed stages
        
    Returns:
        Dict containing index
    """
    # Pass checkpoint_manager for fine-grained resume support
    index = await adapter.add(
        conversations=dataset.conversations,
        output_dir=output_dir,
        checkpoint_manager=checkpoint_manager
    )
    
    # Index metadata (lazy load, no need to persist)
    logger.info("✅ Stage 1 completed")
    
    # Save checkpoint
    completed_stages.add("add")
    if checkpoint_manager:
        checkpoint_manager.save_checkpoint(completed_stages)
    
    return {"index": index}

