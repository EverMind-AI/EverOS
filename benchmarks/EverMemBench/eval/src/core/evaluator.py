"""
Evaluator for QA evaluation.

Evaluates generated answers against golden answers.
- Multiple choice: Direct comparison (predicted letter == correct letter)
- Open-ended: LLM judge comparison

Features:
- Concurrent API calls with configurable concurrency
- Progress display with rich progress bar
- Exponential backoff retry mechanism
- Support for both AnswerResult and LightAnswerResult (memory-efficient)
"""
import asyncio
import json
import os
import random
from typing import Dict, Any, List, Optional, Union

from openai import AsyncOpenAI
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from eval.src.core.data_models import AnswerResult, LightAnswerResult, EvaluationResult
from eval.src.utils.config import load_yaml, get_config_path
from eval.src.utils.logger import get_console

# Type alias for answer results (supports both full and lightweight)
AnswerResultType = Union[AnswerResult, LightAnswerResult]


class Evaluator:
    """
    Evaluator for QA results.
    
    Supports two evaluation modes:
    - Multiple choice: Direct string comparison
    - Open-ended: LLM judge with configurable number of runs
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, num_runs: int = 1, system_name: str = "default"):
        """
        Initialize evaluator.

        Args:
            config: Optional config dict. If not provided, loads from pipeline.yaml
            num_runs: Number of LLM judge runs for open-ended evaluation
            system_name: System name for prompt selection
        """
        self.console = get_console()
        self.num_runs = num_runs
        self.system_name = system_name

        # Load prompts config
        prompts_path = get_config_path("prompts.yaml")
        self.prompts_config = load_yaml(str(prompts_path))

        # Get LLM judge prompts
        judge_config = self.prompts_config.get("llm_judge", {})
        self.judge_system_prompt = judge_config.get("system_prompt", "")
        self.judge_user_prompt = judge_config.get("user_prompt", "")

        # Load pipeline config (evaluate section)
        try:
            pipeline_config_path = get_config_path("pipeline.yaml")
            pipeline_config = load_yaml(str(pipeline_config_path))
        except Exception:
            pipeline_config = {}

        evaluate_config = pipeline_config.get("evaluate", {})
        retry_config = pipeline_config.get("retry", {})

        # Allow explicit config param to override pipeline.yaml
        if config:
            evaluate_config = {**evaluate_config, **config}

        # OpenRouter configuration via environment variables
        api_key = evaluate_config.get("api_key") or os.environ.get("LLM_API_KEY")
        base_url = evaluate_config.get("base_url") or os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")

        if not api_key:
            raise ValueError("LLM API key required. Set LLM_API_KEY env var or provide in config.")

        api_timeout = evaluate_config.get("timeout", 300)
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=api_timeout
        )

        # Model configuration
        self.model = evaluate_config.get("model", "openai/gpt-4.1-nano")

        # Provider configuration for OpenRouter (CRITICAL for cache hits)
        self.provider_config = evaluate_config.get("provider", None)

        # Concurrency settings
        self.concurrency = evaluate_config.get("concurrency", 20)

        # Retry settings
        self.max_retries = retry_config.get("max_retries", 3)
        self.retry_delay = retry_config.get("retry_delay", 1.0)
        self.max_delay = retry_config.get("max_delay", 30.0)

        self.console.print("✅ Evaluator initialized", style="bold green")
        self.console.print(f"   Base URL: {base_url}")
        self.console.print(f"   Model: {self.model}")
        if self.provider_config:
            provider_order = self.provider_config.get("order", [])
            allow_fallbacks = self.provider_config.get("allow_fallbacks", True)
            self.console.print(f"   Provider: {provider_order} (fallbacks: {allow_fallbacks})")
        self.console.print(f"   Concurrency: {self.concurrency}")
        self.console.print(f"   Max Retries: {self.max_retries}")
        self.console.print(f"   Num Runs (OE): {self.num_runs}")
    
    async def evaluate(
        self,
        answer_results: List[AnswerResultType],
        concurrency: Optional[int] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate all answer results with progress display.
        
        Supports both AnswerResult (memory systems) and LightAnswerResult (LLM system).
        
        Args:
            answer_results: List of answer results to evaluate (AnswerResult or LightAnswerResult)
            concurrency: Max concurrent LLM calls (overrides config if provided)
            **kwargs: Additional parameters
            
        Returns:
            EvaluationResult with accuracy statistics
        """
        self.console.print(f"\n{'='*60}", style="bold magenta")
        self.console.print("Stage: Evaluate", style="bold magenta")
        self.console.print(f"{'='*60}", style="bold magenta")
        self.console.print(f"Total questions: {len(answer_results)}")
        
        detailed_results = []
        
        # Separate by question type
        mc_results = [r for r in answer_results if r.question_type == "multiple_choice"]
        oe_results = [r for r in answer_results if r.question_type == "open_ended"]
        
        self.console.print(f"   Multiple choice: {len(mc_results)}")
        self.console.print(f"   Open-ended: {len(oe_results)}")
        
        # Evaluate multiple choice (direct comparison, no LLM needed)
        mc_correct = 0
        for result in mc_results:
            is_correct = self._evaluate_mc(result)
            mc_correct += int(is_correct)
            detailed_results.append({
                "question_id": result.question_id,
                "question": result.question,
                "question_type": result.question_type,
                "golden_answer": result.golden_answer,
                "generated_answer": result.generated_answer,
                "is_correct": is_correct,
            })
        
        # Evaluate open-ended (LLM judge) with progress
        oe_correct = 0
        if oe_results:
            self.console.print("\n   Evaluating open-ended questions with LLM judge...")
            actual_concurrency = concurrency or self.concurrency
            oe_judgments = await self._evaluate_oe_batch_with_progress(oe_results, actual_concurrency)
            for result, is_correct in zip(oe_results, oe_judgments):
                oe_correct += int(is_correct)
                detailed_results.append({
                    "question_id": result.question_id,
                    "question": result.question,
                    "question_type": result.question_type,
                    "golden_answer": result.golden_answer,
                    "generated_answer": result.generated_answer,
                    "is_correct": is_correct,
                })
        
        # Calculate accuracy
        total = len(answer_results)
        correct = mc_correct + oe_correct
        accuracy = correct / total if total > 0 else 0.0
        
        # Accuracy by type
        accuracy_by_type = {}
        if mc_results:
            mc_accuracy = mc_correct / len(mc_results)
            accuracy_by_type["multiple_choice"] = {
                "total": len(mc_results),
                "correct": mc_correct,
                "accuracy": mc_accuracy,
            }
        if oe_results:
            oe_accuracy = oe_correct / len(oe_results)
            accuracy_by_type["open_ended"] = {
                "total": len(oe_results),
                "correct": oe_correct,
                "accuracy": oe_accuracy,
            }
        
        # Print summary
        self.console.print(f"\n{'='*60}", style="bold magenta")
        self.console.print(f"✅ Evaluation complete:", style="bold green")
        self.console.print(f"   Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2%}")
        if "multiple_choice" in accuracy_by_type:
            mc = accuracy_by_type["multiple_choice"]
            self.console.print(f"   MC: {mc['correct']}/{mc['total']} = {mc['accuracy']:.2%}")
        if "open_ended" in accuracy_by_type:
            oe = accuracy_by_type["open_ended"]
            self.console.print(f"   OE: {oe['correct']}/{oe['total']} = {oe['accuracy']:.2%}")
        
        return EvaluationResult(
            total_questions=total,
            correct=correct,
            accuracy=accuracy,
            accuracy_by_type=accuracy_by_type,
            detailed_results=detailed_results,
            metadata={
                "model": self.model,
                "num_runs": self.num_runs,
            }
        )
    
    def _evaluate_mc(self, result: AnswerResultType) -> bool:
        """
        Evaluate multiple choice answer.

        Direct comparison of generated answer letter with correct option.
        Uses robust multi-pattern parsing (same logic as answerer._parse_mc_answer)
        to correctly extract the answer letter from LLM responses.

        Args:
            result: Answer result to evaluate (AnswerResult or LightAnswerResult)

        Returns:
            True if correct, False otherwise
        """
        generated = result.generated_answer.strip()

        # Handle failure markers — always incorrect
        if generated.startswith("[") and generated.endswith("]"):
            return False

        # Get correct option from golden_answer
        golden = result.golden_answer.strip().upper()

        # Extract first letter if golden is like "A. Option text"
        if len(golden) > 1 and golden[0] in "ABCD" and golden[1] in ".):":
            golden = golden[0]

        # Parse generated answer using robust multi-pattern extraction
        generated = self._parse_mc_answer(generated)

        return generated == golden

    @staticmethod
    def _parse_mc_answer(response: str) -> str:
        """
        Parse multiple choice answer from LLM response.

        Uses multi-pattern regex extraction to robustly find the answer letter,
        avoiding false matches from common words like "ACCORDING", "CANNOT", etc.

        Args:
            response: Raw LLM response

        Returns:
            Single uppercase letter (A/B/C/D) or the original response if unparseable
        """
        import re

        response = response.strip().upper()

        if not response:
            return ""

        # Direct single letter
        if len(response) == 1 and response in "ABCD":
            return response

        # Pattern 1: Letter followed by delimiter (most reliable)
        match = re.search(r'\b([ABCD])[.):,\s]', response)
        if match:
            return match.group(1)

        # Pattern 2: "answer is X" or "choose X"
        match = re.search(r'(?:answer|choice|option|select)[:\s]+([ABCD])\b', response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Pattern 3: Standalone letter at start or end (not part of a word)
        if response[0] in "ABCD" and (len(response) == 1 or not response[1].isalpha()):
            return response[0]
        if response[-1] in "ABCD" and (len(response) == 1 or not response[-2].isalpha()):
            return response[-1]

        return response
    
    async def _evaluate_oe_batch_with_progress(
        self,
        results: List[AnswerResultType],
        concurrency: int = 10
    ) -> List[bool]:
        """
        Evaluate open-ended answers using LLM judge with progress display.
        
        Args:
            results: List of answer results to evaluate (AnswerResult or LightAnswerResult)
            concurrency: Max concurrent LLM calls
            
        Returns:
            List of boolean judgments
        """
        semaphore = asyncio.Semaphore(concurrency)
        total = len(results)
        completed = 0
        judgments: List[Optional[bool]] = [None] * total
        
        async def evaluate_with_progress(idx: int, result: AnswerResultType, progress, task_id) -> None:
            nonlocal completed
            async with semaphore:
                is_correct = await self._evaluate_oe_single(result)
                judgments[idx] = is_correct
                completed += 1
                progress.update(task_id, completed=completed)
        
        # Create progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold magenta]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[cyan]{task.completed}/{task.total}"),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task_id = progress.add_task("LLM Judge evaluating...", total=total)
            
            tasks = [
                evaluate_with_progress(idx, result, progress, task_id)
                for idx, result in enumerate(results)
            ]
            
            await asyncio.gather(*tasks)
        
        return judgments
    
    async def _evaluate_oe_single(self, result: AnswerResultType) -> bool:
        """
        Evaluate single open-ended answer using LLM judge with retry.
        
        Args:
            result: Answer result to evaluate (AnswerResult or LightAnswerResult)
            
        Returns:
            True if correct, False otherwise
        """
        # Run multiple times if num_runs > 1
        judgments = []
        for _ in range(self.num_runs):
            is_correct = await self._call_llm_judge_with_retry(
                question=result.question,
                golden_answer=result.golden_answer,
                generated_answer=result.generated_answer
            )
            judgments.append(is_correct)
        
        # Majority vote
        return sum(judgments) > len(judgments) / 2
    
    async def _call_llm_judge_with_retry(
        self,
        question: str,
        golden_answer: str,
        generated_answer: str
    ) -> bool:
        """
        Call LLM judge to evaluate answer with exponential backoff retry.
        
        Args:
            question: Original question
            golden_answer: Expected correct answer
            generated_answer: Model-generated answer
            
        Returns:
            True if LLM judges as CORRECT, False otherwise
        """
        # Build prompt
        user_prompt = self.judge_user_prompt.format(
            question=question,
            golden_answer=golden_answer,
            generated_answer=generated_answer
        )
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Build request kwargs
                request_kwargs = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.judge_system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0,
                }
                
                if self.provider_config:
                    request_kwargs["extra_body"] = {
                        "provider": {
                            "order": self.provider_config.get("order", []),
                            "allow_fallbacks": self.provider_config.get("allow_fallbacks", True)
                        }
                    }
                
                response = await self.client.chat.completions.create(**request_kwargs)
                
                content = response.choices[0].message.content or ""
                
                # Parse JSON response
                return self._parse_judge_response(content)
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    # Exponential backoff with jitter
                    delay = min(self.retry_delay * (2 ** attempt) + random.uniform(0, 1), self.max_delay)
                    self.console.print(
                        f"   ⚠️  LLM judge failed (attempt {attempt + 1}/{self.max_retries}), "
                        f"retrying in {delay:.1f}s: {str(e)[:50]}...",
                        style="yellow"
                    )
                    await asyncio.sleep(delay)
        
        # All retries exhausted
        self.console.print(f"   ❌ LLM judge failed after {self.max_retries} attempts: {last_error}", style="red")
        return False
    
    def _parse_judge_response(self, content: str) -> bool:
        """
        Parse LLM judge response to extract CORRECT/WRONG label.
        
        Args:
            content: Raw LLM response content
            
        Returns:
            True if CORRECT, False otherwise
        """
        try:
            # Handle case where JSON might be wrapped in markdown code block
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                content_json = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                content_json = content[json_start:json_end].strip()
            elif "{" in content and "}" in content:
                # Extract JSON object from content
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                content_json = content[json_start:json_end]
            else:
                content_json = content
            
            result = json.loads(content_json)
            label = result.get("label", "WRONG")
            # Handle case where label might be a dict
            if isinstance(label, dict):
                label = label.get("label", "WRONG")
            if isinstance(label, str):
                label = label.strip().upper()
            else:
                label = "WRONG"
                
            return label == "CORRECT"
            
        except (json.JSONDecodeError, KeyError):
            # Fallback: look for CORRECT/WRONG in response
            label = "CORRECT" if "CORRECT" in content.upper() and "WRONG" not in content.upper() else "WRONG"
            return label == "CORRECT"
