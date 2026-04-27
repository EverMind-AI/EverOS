"""
Hybrid evaluator - automatically choose evaluation method based on question type.

- Multiple-choice questions (with all_options) → Exact Match
- Open-ended questions (without all_options) → LLM Judge
"""
from typing import List
from collections import defaultdict

from evaluation.src.evaluators.base import BaseEvaluator
from evaluation.src.evaluators.registry import register_evaluator
from evaluation.src.core.data_models import AnswerResult, EvaluationResult
from evaluation.src.evaluators.exact_match import ExactMatch
from evaluation.src.evaluators.llm_judge import LLMJudge


@register_evaluator("hybrid")
class HybridEvaluator(BaseEvaluator):
    """
    Hybrid evaluator that combines Exact Match and LLM Judge.
    
    Automatically detects question type and uses appropriate evaluation method:
    - Questions with all_options in metadata → Exact Match (for multiple-choice)
    - Questions without all_options → LLM Judge (for open-ended)
    """
    
    def __init__(self, config: dict):
        """
        Initialize hybrid evaluator.
        
        Args:
            config: Evaluation config containing settings for both evaluators
        """
        super().__init__(config)
        
        # Initialize both evaluators
        self.exact_match_evaluator = ExactMatch(config)
        self.llm_judge_evaluator = LLMJudge(config)
    
    async def evaluate(
        self, 
        answer_results: List[AnswerResult]
    ) -> EvaluationResult:
        """
        Evaluate answers using hybrid approach.
        
        Args:
            answer_results: List of answer results
            
        Returns:
            Combined evaluation result
        """
        print(f"\n{'='*60}")
        print(f"Evaluation: Hybrid (Exact Match + LLM Judge)")
        print(f"{'='*60}")
        
        # Separate questions by type, preserving original indices
        choice_questions = []
        open_questions = []
        question_type_map = {}  # Maps index to question type
        
        for idx, answer_result in enumerate(answer_results):
            # Check if question has all_options (multiple-choice)
            has_options = "all_options" in answer_result.metadata
            
            if has_options:
                choice_questions.append(answer_result)
                question_type_map[idx] = 'choice'
            else:
                open_questions.append(answer_result)
                question_type_map[idx] = 'open'
        
        print(f"\n📊 Question Distribution:")
        print(f"   - Multiple-choice (Exact Match): {len(choice_questions)}")
        print(f"   - Open-ended (LLM Judge): {len(open_questions)}")
        
        # Evaluate multiple-choice questions with Exact Match
        choice_detailed_results = []
        choice_correct = 0
        
        if choice_questions:
            print(f"\n{'='*60}")
            print(f"Evaluating Multiple-Choice Questions ({len(choice_questions)})")
            print(f"{'='*60}")
            
            choice_eval_result = await self.exact_match_evaluator.evaluate(choice_questions)
            choice_detailed_results = choice_eval_result.detailed_results
            choice_correct = choice_eval_result.correct
        
        # Evaluate open-ended questions with LLM Judge
        open_detailed_results = []
        open_correct = 0
        open_metadata = {}
        
        if open_questions:
            print(f"\n{'='*60}")
            print(f"Evaluating Open-Ended Questions ({len(open_questions)})")
            print(f"{'='*60}")
            
            open_eval_result = await self.llm_judge_evaluator.evaluate(open_questions)
            # LLM Judge returns grouped results (dict), need to flatten to list
            if isinstance(open_eval_result.detailed_results, dict):
                # Flatten grouped results to list
                for conv_results in open_eval_result.detailed_results.values():
                    open_detailed_results.extend(conv_results)
            else:
                open_detailed_results = open_eval_result.detailed_results
            open_correct = open_eval_result.correct
            open_metadata = open_eval_result.metadata
        
        # Combine results in original order
        # Create maps for quick lookup
        choice_results_map = {result['question_id']: result for result in choice_detailed_results}
        open_results_map = {result['question_id']: result for result in open_detailed_results}
        
        # Reconstruct results in original order
        all_detailed_results = []
        choice_idx = 0
        open_idx = 0
        
        for idx in range(len(answer_results)):
            q_type = question_type_map[idx]
            if q_type == 'choice':
                # Use question_id from original answer_result to look up evaluated result
                original_result = choice_questions[choice_idx]
                question_id = original_result.question_id
                all_detailed_results.append(choice_results_map[question_id])
                choice_idx += 1
            else:  # q_type == 'open'
                original_result = open_questions[open_idx]
                question_id = original_result.question_id
                all_detailed_results.append(open_results_map[question_id])
                open_idx += 1
        
        total_correct = choice_correct + open_correct
        total_questions = len(answer_results)
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
        
        # Calculate per-category statistics
        category_stats = self._calculate_category_stats(all_detailed_results)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Combined Results")
        print(f"{'='*60}")
        print(f"   - Total questions: {total_questions}")
        print(f"   - Correct: {total_correct}")
        print(f"   - Overall accuracy: {overall_accuracy:.2%}")
        print(f"\n   Breakdown:")
        print(f"   - Multiple-choice: {choice_correct}/{len(choice_questions)} ({choice_correct/len(choice_questions)*100:.1f}%)" if choice_questions else "   - Multiple-choice: 0/0")
        print(f"   - Open-ended: {open_correct}/{len(open_questions)} ({open_correct/len(open_questions)*100:.1f}%)" if open_questions else "   - Open-ended: 0/0")
        
        if category_stats:
            print(f"\n📊 Category Statistics:")
            for cat, stats in sorted(category_stats.items()):
                print(f"   Category {cat}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.2%})")
        
        # Construct metadata
        combined_metadata = {
            "evaluator": "hybrid",
            "total_questions": total_questions,
            "choice_questions": len(choice_questions),
            "open_questions": len(open_questions),
            "choice_correct": choice_correct,
            "open_correct": open_correct,
            "choice_accuracy": choice_correct / len(choice_questions) if choice_questions else 0.0,
            "open_accuracy": open_correct / len(open_questions) if open_questions else 0.0,
            "category_stats": category_stats,
        }
        
        # Add LLM Judge specific metadata if available
        if open_metadata:
            combined_metadata["llm_judge_metadata"] = {
                "mean_accuracy": open_metadata.get("mean_accuracy"),
                "std_accuracy": open_metadata.get("std_accuracy"),
                "num_runs": open_metadata.get("num_runs"),
            }
        
        return EvaluationResult(
            total_questions=total_questions,
            correct=total_correct,
            accuracy=overall_accuracy,
            detailed_results=all_detailed_results,
            metadata=combined_metadata
        )
    
    def _calculate_category_stats(self, detailed_results: List[dict]) -> dict:
        """Calculate per-category statistics."""
        category_data = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for result in detailed_results:
            category = result.get("category", "unknown")
            category_data[category]["total"] += 1
            if result.get("is_correct", False):
                category_data[category]["correct"] += 1
        
        # Add accuracy
        category_stats = {}
        for cat, data in category_data.items():
            category_stats[cat] = {
                "correct": data["correct"],
                "total": data["total"],
                "accuracy": data["correct"] / data["total"] if data["total"] > 0 else 0.0
            }
        
        return category_stats

