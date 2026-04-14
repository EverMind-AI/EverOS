"""
Answer generator for QA evaluation.

Generates answers using LLM based on retrieved memories.
Supports two question types:
- multiple_choice: Returns a single letter (A/B/C/D)
- open_ended: Returns free-text answer

Features:
- Concurrent API calls with configurable concurrency
- Progress display with rich progress bar
- Exponential backoff retry mechanism
- Cache hit tracking for LLM API calls
- Support for LLM system (full dialogue) vs memory system prompts
"""
import asyncio
import os
import re
import time
import random
from typing import Dict, Any, List, Optional, Tuple

from openai import AsyncOpenAI
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from eval.src.core.data_models import QAItem, SearchResult, AnswerResult
from eval.src.utils.config import load_yaml, get_config_path
from eval.src.utils.logger import get_console


class Answerer:
    """
    Answer generator using LLM.
    
    Loads prompts from config/prompts.yaml and generates answers
    based on retrieved memories and question type.
    
    Supports two modes:
    - Memory system mode: Uses standard prompts with retrieved memories
    - LLM system mode: Uses llm_answer prompts with full dialogue as context
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        system_name: str = "default",
    ):
        """
        Initialize answerer with LLM configuration.

        Args:
            config: Optional config dict. If not provided, loads from pipeline.yaml
            system_name: System name ("llm" uses special prompts, others use standard)
        """
        self.console = get_console()
        self.system_name = system_name

        # Load prompts config
        prompts_path = get_config_path("prompts.yaml")
        self.prompts_config = load_yaml(str(prompts_path))

        # Get prompts based on system type
        # For "llm" system, use llm_answer prompts (full dialogue)
        # For memory systems, use standard answer prompts
        if system_name == "llm":
            self.mc_prompt = self.prompts_config.get("llm_answer", {}).get("multiple_choice", "")
            self.oe_prompt = self.prompts_config.get("llm_answer", {}).get("open_ended", "")
        else:
            self.mc_prompt = self.prompts_config.get("answer", {}).get("multiple_choice", "")
            self.oe_prompt = self.prompts_config.get("answer", {}).get("open_ended", "")

        # Load pipeline config (answer section)
        try:
            pipeline_config_path = get_config_path("pipeline.yaml")
            pipeline_config = load_yaml(str(pipeline_config_path))
        except Exception:
            pipeline_config = {}

        answer_config = pipeline_config.get("answer", {})
        retry_config = pipeline_config.get("retry", {})
        debug_config = pipeline_config.get("debug", {})

        # Allow explicit config param to override pipeline.yaml
        if config:
            answer_config = {**answer_config, **config}

        # OpenRouter configuration via environment variables
        api_key = answer_config.get("api_key") or os.environ.get("LLM_API_KEY")
        base_url = answer_config.get("base_url") or os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")

        if not api_key:
            raise ValueError("LLM API key required. Set LLM_API_KEY env var or provide in config.")

        # Set timeout for long context processing (3MB+ dialogue requires longer time)
        api_timeout = answer_config.get("timeout", 300)
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=api_timeout
        )

        # Model configuration
        self.model = answer_config.get("model", "openai/gpt-4.1-mini")

        self.temperature = answer_config.get("temperature", 0)
        self.max_tokens = answer_config.get("max_tokens", 1000)

        # Provider configuration for OpenRouter (CRITICAL for cache hits)
        self.provider_config = answer_config.get("provider", None)

        # Concurrency settings
        self.concurrency = answer_config.get("concurrency", 10)

        # Retry settings
        self.max_retries = retry_config.get("max_retries", 3)
        self.retry_delay = retry_config.get("retry_delay", 1.0)
        self.max_delay = retry_config.get("max_delay", 30.0)

        # Debug settings
        self.show_usage = debug_config.get("show_usage", False)

        # Cache hit tracking
        self.cache_hits = 0
        self.total_requests = 0

        self.console.print("✅ Answerer initialized", style="bold green")
        self.console.print(f"   Base URL: {base_url}")
        self.console.print(f"   Model: {self.model}")
        if self.provider_config:
            provider_order = self.provider_config.get("order", [])
            allow_fallbacks = self.provider_config.get("allow_fallbacks", True)
            self.console.print(f"   Provider: {provider_order} (fallbacks: {allow_fallbacks})")
        self.console.print(f"   System: {system_name}")
        self.console.print(f"   Concurrency: {self.concurrency}")
        self.console.print(f"   Max Retries: {self.max_retries}")
        if self.show_usage:
            self.console.print(f"   Debug: show_usage enabled")
    
    async def generate_answer(
        self,
        qa_item: QAItem,
        search_result: SearchResult,
        **kwargs
    ) -> AnswerResult:
        """
        Generate answer for a single QA item with retry.
        
        Args:
            qa_item: QA item with question and options
            search_result: Search result with retrieved memories
            **kwargs: Additional parameters
            
        Returns:
            AnswerResult with generated answer
        """
        start_time = time.time()
        
        # Select prompt based on question type
        if qa_item.question_type == "multiple_choice":
            answer = await self._answer_multiple_choice(qa_item, search_result)
        else:
            answer = await self._answer_open_ended(qa_item, search_result)
        
        duration_ms = (time.time() - start_time) * 1000
        
        return AnswerResult(
            question_id=qa_item.question_id,
            question=qa_item.question,
            question_type=qa_item.question_type,
            golden_answer=qa_item.answer,
            generated_answer=answer,
            search_result=search_result,
            answer_duration_ms=duration_ms,
        )
    
    async def generate_answers_batch(
        self,
        qa_items: List[QAItem],
        search_results: List[SearchResult],
        concurrency: Optional[int] = None,
        **kwargs
    ) -> List[AnswerResult]:
        """
        Generate answers for multiple QA items concurrently with progress display.
        
        Shows progress bar with cache hit statistics.
        
        Args:
            qa_items: List of QA items
            search_results: List of corresponding search results
            concurrency: Max concurrent LLM calls (overrides config if provided)
            **kwargs: Additional parameters
            
        Returns:
            List of AnswerResult objects
        """
        if len(qa_items) != len(search_results):
            raise ValueError("qa_items and search_results must have same length")
        
        actual_concurrency = concurrency or self.concurrency
        semaphore = asyncio.Semaphore(actual_concurrency)
        
        # Reset cache tracking for this batch
        self.cache_hits = 0
        self.total_requests = 0
        
        # Track completed count for progress
        completed = 0
        total = len(qa_items)
        results: List[Optional[AnswerResult]] = [None] * total
        
        async def generate_with_progress(idx: int, qa_item: QAItem, search_result: SearchResult, progress, task_id) -> None:
            nonlocal completed
            async with semaphore:
                result = await self.generate_answer(qa_item, search_result, **kwargs)
                results[idx] = result
                completed += 1
                # Update progress with cache hit info
                progress.update(
                    task_id, 
                    completed=completed,
                    description=f"Generating answers... [green]Cache: {self.cache_hits}/{self.total_requests}[/green]"
                )
        
        # Create progress bar with cache hit display
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[cyan]{task.completed}/{task.total}"),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task_id = progress.add_task("Generating answers...", total=total)
            
            tasks = [
                generate_with_progress(idx, qa, sr, progress, task_id)
                for idx, (qa, sr) in enumerate(zip(qa_items, search_results))
            ]
            
            await asyncio.gather(*tasks)
        
        # Print final cache stats
        cache_rate = (self.cache_hits / self.total_requests * 100) if self.total_requests > 0 else 0
        self.console.print(f"   📊 Cache Statistics: {self.cache_hits}/{self.total_requests} hits ({cache_rate:.1f}%)")
        
        return results
    
    async def _answer_multiple_choice(
        self,
        qa_item: QAItem,
        search_result: SearchResult
    ) -> str:
        """
        Generate answer for multiple choice question.
        
        Args:
            qa_item: QA item with question and options
            search_result: Search result with context
            
        Returns:
            Single letter answer (A/B/C/D)
        """
        # Format options
        options_text = "\n".join(qa_item.options) if qa_item.options else ""
        
        # Build prompt
        prompt = self.mc_prompt.format(
            context=search_result.context,
            question=qa_item.question,
            options=options_text
        )
        
        # Call LLM with retry
        response, _ = await self._call_llm_with_retry(prompt)
        
        # Parse response - extract single letter
        answer = self._parse_mc_answer(response)
        
        return answer
    
    async def _answer_open_ended(
        self,
        qa_item: QAItem,
        search_result: SearchResult
    ) -> str:
        """
        Generate answer for open-ended question.
        
        Args:
            qa_item: QA item with question
            search_result: Search result with context
            
        Returns:
            Free-text answer
        """
        # Build prompt
        prompt = self.oe_prompt.format(
            context=search_result.context,
            question=qa_item.question
        )
        
        # Call LLM with retry
        response, _ = await self._call_llm_with_retry(prompt)
        
        # Handle API failure marker
        if response.startswith("[LLM_CALL_FAILED]"):
            return "[FAILED]"
        
        return response.strip() if response else "[EMPTY]"
    
    async def _call_llm_with_retry(self, prompt: str) -> Tuple[str, bool]:
        """
        Call LLM with exponential backoff retry.
        
        Args:
            prompt: Full prompt text
            
        Returns:
            Tuple of (LLM response text, cache_hit boolean)
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Build request kwargs
                request_kwargs = {
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                }
                
                # Add provider configuration for OpenRouter (CRITICAL for cache hits)
                # This ensures all requests go to the same provider backend
                if self.provider_config:
                    request_kwargs["extra_body"] = {
                        "provider": {
                            "order": self.provider_config.get("order", []),
                            "allow_fallbacks": self.provider_config.get("allow_fallbacks", True)
                        }
                    }
                
                response = await self.client.chat.completions.create(**request_kwargs)
                
                # Track total requests
                self.total_requests += 1
                
                # Check for cache hit: OpenRouter returns cached_tokens in prompt_tokens_details
                cache_hit = False
                cached_tokens = 0
                
                if hasattr(response, 'usage') and response.usage:
                    usage = response.usage
                    details = getattr(usage, 'prompt_tokens_details', None)
                    if details:
                        cached_tokens = getattr(details, 'cached_tokens', 0) or 0
                    
                    if self.show_usage:
                        prompt_tokens = getattr(usage, 'prompt_tokens', 0) or 0
                        completion_tokens = getattr(usage, 'completion_tokens', 0) or 0
                        self.console.print(
                            f"   📊 Usage: prompt={prompt_tokens}, completion={completion_tokens}, cached={cached_tokens}",
                            style="dim"
                        )
                
                if cached_tokens > 0:
                    cache_hit = True
                    self.cache_hits += 1
                
                return response.choices[0].message.content or "", cache_hit
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    # Exponential backoff with jitter
                    delay = min(self.retry_delay * (2 ** attempt) + random.uniform(0, 1), self.max_delay)
                    self.console.print(
                        f"   ⚠️  LLM call failed (attempt {attempt + 1}/{self.max_retries}), "
                        f"retrying in {delay:.1f}s: {str(e)[:50]}...",
                        style="yellow"
                    )
                    await asyncio.sleep(delay)
        
        # All retries exhausted
        self.total_requests += 1
        error_msg = str(last_error)[:100] if last_error else "Unknown error"
        self.console.print(f"   ❌ LLM call failed after {self.max_retries} attempts: {error_msg}", style="red")
        # Return a marker to indicate failure (not just empty string)
        return "[LLM_CALL_FAILED]", False
    
    def _parse_mc_answer(self, response: str) -> str:
        """
        Parse multiple choice answer from LLM response.
        
        Extracts single letter (A/B/C/D) from response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Single uppercase letter, "[FAILED]" for API failures, or "[INVALID]" for unparseable responses
        """
        # Handle API failure marker
        if response.startswith("[LLM_CALL_FAILED]"):
            return "[FAILED]"
        
        response = response.strip().upper()
        
        # Empty response
        if not response:
            return "[EMPTY]"
        
        # Direct single letter
        if len(response) == 1 and response in "ABCD":
            return response
        
        # Find first occurrence of A/B/C/D (but not in common words like "CANNOT", "ANSWER")
        # Look for patterns like "A.", "A)", "A:", "Answer: A", or standalone A/B/C/D
        
        # Pattern 1: Letter followed by delimiter (most reliable)
        match = re.search(r'\b([ABCD])[.):,\s]', response)
        if match:
            return match.group(1)
        
        # Pattern 2: "answer is X" or "choose X"
        match = re.search(r'(?:answer|choice|option|select)[:\s]+([ABCD])\b', response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Pattern 3: Standalone letter at start or end
        if response[0] in "ABCD" and (len(response) == 1 or not response[1].isalpha()):
            return response[0]
        if response[-1] in "ABCD" and (len(response) == 1 or not response[-2].isalpha()):
            return response[-1]
        
        # Cannot parse - return marker
        return "[INVALID]"
