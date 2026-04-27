"""
Evaluator base class - define unified evaluator interface.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from evaluation.src.core.data_models import AnswerResult, EvaluationResult


class BaseEvaluator(ABC):
    """Evaluator base class."""
    
    def __init__(self, config: dict):
        """
        Initialize evaluator.
        
        Args:
            config: Evaluation config
        """
        self.config = config
    
    @abstractmethod
    async def evaluate(
        self, 
        answer_results: List[AnswerResult]
    ) -> EvaluationResult:
        """
        Evaluate answer results.
        
        Args:
            answer_results: List of answer results
            
        Returns:
            Evaluation result
        """
        pass
    
    def get_name(self) -> str:
        """Return evaluator name."""
        return self.__class__.__name__

