"""
Exact Match evaluator - direct answer comparison, suitable for multiple-choice scenarios.
"""
import re
from typing import List

from evaluation.src.evaluators.base import BaseEvaluator
from evaluation.src.evaluators.registry import register_evaluator
from evaluation.src.core.data_models import AnswerResult, EvaluationResult


@register_evaluator("exact_match")
class ExactMatch(BaseEvaluator):
    """Exact match evaluator."""
    
    def __init__(self, config: dict):
        """
        Initialize evaluator.
        
        Args:
            config: Evaluation config (optional: case_sensitive, normalize_whitespace)
        """
        super().__init__(config)
        
        # Config options
        self.case_sensitive = config.get("case_sensitive", False)  # Default: case insensitive
        self.normalize_whitespace = config.get("normalize_whitespace", True)  # Default: normalize whitespace
        self.extract_choice = config.get("extract_choice", True)  # Default: extract choices like (a), (b), (c)
    
    async def evaluate(
        self, 
        answer_results: List[AnswerResult]
    ) -> EvaluationResult:
        """
        Evaluate answers using exact match.
        
        Args:
            answer_results: List of answer results
            
        Returns:
            Evaluation result
        """
        print(f"\n{'='*60}")
        print(f"Evaluation: Exact Match")
        print(f"  - Case sensitive: {self.case_sensitive}")
        print(f"  - Normalize whitespace: {self.normalize_whitespace}")
        print(f"  - Extract choice: {self.extract_choice}")
        print(f"{'='*60}")
        
        detailed_results = []
        total_correct = 0
        
        # Evaluate each answer
        for answer_result in answer_results:
            is_correct = self._check_match(
                answer_result.golden_answer,
                answer_result.answer
            )
            
            if is_correct:
                total_correct += 1
            
            detailed_results.append({
                "question_id": answer_result.question_id,
                "question": answer_result.question,
                "golden_answer": answer_result.golden_answer,
                "generated_answer": answer_result.answer,
                "is_correct": is_correct,
                "category": answer_result.category,
            })
        
        accuracy = total_correct / len(answer_results) if answer_results else 0.0
        
        print(f"\n✅ Evaluation complete:")
        print(f"   - Total questions: {len(answer_results)}")
        print(f"   - Correct: {total_correct}")
        print(f"   - Accuracy: {accuracy:.2%}")
        
        return EvaluationResult(
            total_questions=len(answer_results),
            correct=total_correct,
            accuracy=accuracy,
            detailed_results=detailed_results,
            metadata={
                "evaluator": "exact_match",
                "case_sensitive": self.case_sensitive,
                "normalize_whitespace": self.normalize_whitespace,
                "extract_choice": self.extract_choice,
            }
        )
    
    def _check_match(self, golden: str, generated: str) -> bool:
        """
        Check if two answers match.
        
        Args:
            golden: Golden answer
            generated: Generated answer
            
        Returns:
            Whether answers match
        """
        # Preprocess
        golden_processed = self._preprocess(golden)
        generated_processed = self._preprocess(generated)
        
        # If choice extraction enabled, try to extract choice from generated answer
        if self.extract_choice:
            extracted_choice = self._extract_choice(generated_processed)
            if extracted_choice:
                generated_processed = extracted_choice
        
        # Compare
        if self.case_sensitive:
            return golden_processed == generated_processed
        else:
            return golden_processed.lower() == generated_processed.lower()
    
    def _preprocess(self, text: str) -> str:
        """
        Preprocess text.
        
        Args:
            text: Raw text
            
        Returns:
            Processed text
        """
        if not text:
            return ""
        
        # Normalize whitespace
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_choice(self, text: str) -> str:
        """
        Extract choice from text (supports formats: (a), a), a., A, etc.).
        
        Returns:
            Normalized choice format "(a)", empty string if not found
        """
        # Try matching (a), (b), (c), (d) format
        match = re.search(r'\(([a-zA-Z])\)', text)
        if match:
            return f"({match.group(1).lower()})"
        
        # Try matching a), b), c), d) format
        match = re.search(r'\b([a-zA-Z])\)', text)
        if match:
            return f"({match.group(1).lower()})"
        
        # Try matching a., b., c., d. format
        match = re.search(r'\b([a-zA-Z])\.', text)
        if match:
            return f"({match.group(1).lower()})"
        
        # Try matching standalone letter (at start or surrounded by whitespace)
        match = re.search(r'(?:^|\s)([a-zA-Z])(?:\s|$)', text)
        if match:
            letter = match.group(1).lower()
            # Only accept first few letters (options typically don't exceed f)
            if letter in 'abcdefgh':
                return f"({letter})"
        
        # Return empty string if no match
        return ""


