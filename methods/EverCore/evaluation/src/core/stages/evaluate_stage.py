"""
Evaluate stage - evaluate answer quality.
"""
from typing import List, Optional
from logging import Logger

from evaluation.src.core.data_models import AnswerResult, EvaluationResult
from evaluation.src.evaluators.base import BaseEvaluator
from evaluation.src.utils.checkpoint import CheckpointManager


async def run_evaluate_stage(
    evaluator: BaseEvaluator,
    answer_results: List[AnswerResult],
    checkpoint_manager: Optional[CheckpointManager],
    logger: Logger,
) -> EvaluationResult:
    """
    Execute Evaluate stage.
    
    Args:
        evaluator: Evaluator
        answer_results: List of answer results
        checkpoint_manager: Checkpoint manager for resume
        logger: Logger
        
    Returns:
        Evaluation result
    """
    logger.info("Starting Stage 4: Evaluate")
    
    eval_result = await evaluator.evaluate(answer_results)
    
    logger.info("✅ Stage 4 completed")
    
    return eval_result


