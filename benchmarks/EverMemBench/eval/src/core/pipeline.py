"""
Pipeline orchestrator for multi-person group chat evaluation.

Manages the evaluation workflow stages:
- Add: Ingest data into memory system
- Search: Retrieve relevant memories
- Answer: Generate answers
- Evaluate: Assess answer quality
"""
import asyncio
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List, Optional

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from eval.src.core.data_models import (
    Dataset, AddResult, QAItem, SearchResult, AnswerResult, LightAnswerResult, EvaluationResult
)
from typing import Union
from eval.src.adapters.base import BaseAdapter
from eval.src.utils.logger import get_console, print_header
from eval.src.utils.config import load_yaml, get_config_path


class Pipeline:
    """
    Evaluation pipeline orchestrator.
    
    Coordinates the execution of evaluation stages and manages
    adapter lifecycle.
    """
    
    def __init__(
        self,
        adapter: BaseAdapter,
        output_dir: Optional[Path] = None,
        system_name: str = "unknown",
    ):
        """
        Initialize pipeline.
        
        Args:
            adapter: Memory system adapter
            output_dir: Output directory for results
            system_name: Name of the memory system (for result file naming)
        """
        self.adapter = adapter
        self.system_name = system_name

        # Use output directory as-is (CLI already appends system subdirectory)
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.console = get_console()

        # Load pipeline config once
        try:
            pipeline_config_path = get_config_path("pipeline.yaml")
            self.pipeline_config = load_yaml(str(pipeline_config_path))
        except Exception:
            self.pipeline_config = {}

        # Lazy-loaded components
        self._answerer = None
        self._evaluator = None
    
    async def run(
        self,
        dataset: Optional[Dataset],
        user_id: str,
        stages: Optional[List[str]] = None,
        smoke_days: Optional[int] = None,
        smoke_date: Optional[str] = None,
        qa_path: Optional[str] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run pipeline stages.

        Args:
            dataset: Dataset to process
            user_id: User ID for memory system
            stages: List of stages to run (default: ["add"])
            smoke_days: Limit to first N days for smoke test
            smoke_date: Run smoke test for a specific date (YYYY-MM-DD)
            qa_path: Path to QA JSON file (required for search/answer/evaluate)
            top_k: Number of memories to retrieve for search (None = use adapter config default)
            **kwargs: Additional parameters

        Returns:
            Dict with stage results
        """
        start_time = time.time()
        
        # Default stages
        if stages is None:
            stages = ["add"]
        
        print_header("Multi-Person Group Chat Evaluation Pipeline")
        if dataset is not None:
            self.console.print(f"Dataset: {dataset.name}")
            self.console.print(f"Total Days: {dataset.total_days}")
            self.console.print(f"Total Messages: {dataset.total_messages}")
        self.console.print(f"User ID: {user_id}")
        self.console.print(f"Stages: {stages}")
        
        # Apply smoke date filter (takes precedence over smoke_days)
        if smoke_date and dataset is not None:
            day = dataset.get_day(smoke_date)
            if day is None:
                raise ValueError(f"Smoke date not found in dataset: {smoke_date}")
            dataset = Dataset(
                name=f"{dataset.name}_date_{smoke_date}",
                days=[day],
                metadata={**dataset.metadata, "is_subset": True, "smoke_date": smoke_date},
            )
            self.console.print(f"\n[yellow]🧪 Smoke Test Mode: date {smoke_date}[/yellow]")
            self.console.print(f"   Subset: {dataset.total_days} days, {dataset.total_messages} messages")

        # Apply smoke test limit
        if smoke_days is not None and smoke_days > 0 and dataset is not None:
            dataset = dataset.get_days_subset(smoke_days)
            self.console.print(f"\n[yellow]🧪 Smoke Test Mode: {smoke_days} day(s)[/yellow]")
            self.console.print(f"   Subset: {dataset.total_days} days, {dataset.total_messages} messages")
        
        results = {}
        qa_items = None
        search_results = None
        answer_results = None
        
        # Stage: Add
        if "add" in stages:
            add_result = await self._run_add_stage(dataset, user_id, **kwargs)
            results["add"] = add_result
        
        # Load QA data if needed for search/answer/evaluate
        if any(s in stages for s in ["search", "answer", "evaluate"]):
            if not qa_path:
                raise ValueError("--qa argument required for search/answer/evaluate stages")
            from eval.src.core.qa_loader import load_qa
            qa_limit = kwargs.get("qa_limit")
            qa_items = load_qa(qa_path, limit=qa_limit)
            self.console.print(f"\nLoaded {len(qa_items)} QA items")
        
        # Stage: Search
        if "search" in stages:
            if qa_items is None:
                raise ValueError("QA items required for search stage")
            search_results = await self._run_search_stage(qa_items, user_id, top_k, **kwargs)
            results["search"] = search_results
            # Skip saving search results for LLM system (no memory retrieval)
            if self.system_name != "llm":
                self._save_search_results(search_results, user_id)
        
        # Stage: Answer
        if "answer" in stages:
            if qa_items is None:
                raise ValueError("QA items required for answer stage")
            
            # Load search results if not just completed
            if search_results is None:
                search_results = self._load_search_results(user_id)
                if search_results is None:
                    raise ValueError("Search results required. Run 'search' stage first or ensure results file exists.")
            
            answer_results = await self._run_answer_stage(qa_items, search_results, user_id=user_id, **kwargs)
            results["answer"] = answer_results
            # Save answer results (including LLM system for resume support)
            if self.system_name == "llm":
                self._save_light_answer_results(answer_results, user_id)
            else:
                self._save_answer_results(answer_results, user_id)
        
        # Stage: Evaluate
        if "evaluate" in stages:
            # Load answer results if not just completed
            if answer_results is None:
                answer_results = self._load_answer_results(user_id)
                if answer_results is None:
                    raise ValueError("Answer results required. Run 'answer' stage first or ensure results file exists.")
            
            eval_result = await self._run_evaluate_stage(answer_results, **kwargs)
            results["evaluate"] = eval_result
            self._save_evaluation_results(eval_result, user_id)
        
        # Cleanup
        await self.adapter.close()
        
        # Summary
        elapsed = time.time() - start_time
        self._print_summary(results, elapsed)
        
        return results
    
    async def _run_add_stage(
        self,
        dataset: Dataset,
        user_id: str,
        **kwargs
    ) -> AddResult:
        """
        Execute Add stage.
        
        Args:
            dataset: Dataset to ingest
            user_id: User ID for memory system
            **kwargs: Additional parameters
            
        Returns:
            AddResult
        """
        return await self.adapter.add(dataset, user_id, **kwargs)
    
    async def _run_search_stage(
        self,
        qa_items: List[QAItem],
        user_id: str,
        top_k: Optional[int],
        **kwargs
    ) -> List[SearchResult]:
        """
        Execute Search stage with concurrency, retry, and progress display.

        Args:
            qa_items: List of QA items to search for
            user_id: User ID for memory system
            top_k: Number of memories to retrieve (None = use adapter config default)
            **kwargs: Additional parameters

        Returns:
            List of SearchResult objects
        """
        self.console.print(f"\n{'='*60}", style="bold blue")
        self.console.print("Stage: Search", style="bold blue")
        self.console.print(f"{'='*60}", style="bold blue")
        self.console.print(f"Questions: {len(qa_items)}")
        self.console.print(f"Top K: {top_k if top_k is not None else '(from config)'}")

        search_config = self.pipeline_config.get("search", {})
        retry_config = self.pipeline_config.get("retry", {})

        # Concurrency settings
        search_concurrency = search_config.get("concurrency", 3)

        # Retry settings
        max_retries = retry_config.get("max_retries", 3)
        retry_delay = retry_config.get("retry_delay", 1.0)
        max_delay = retry_config.get("max_delay", 30.0)
        search_timeout = search_config.get("timeout", 120.0)
        
        self.console.print(f"   Concurrency: {search_concurrency}")
        self.console.print(f"   Max Retries: {max_retries}")
        self.console.print(f"   Timeout: {search_timeout}s")
        
        # Setup concurrent execution
        semaphore = asyncio.Semaphore(search_concurrency)
        total = len(qa_items)
        completed = 0
        results: List[Optional[SearchResult]] = [None] * total
        
        async def search_with_retry(idx: int, qa: QAItem, progress, task_id) -> None:
            """Execute single search with retry and timeout."""
            nonlocal completed
            
            async with semaphore:
                last_error = None
                result = None
                
                for attempt in range(max_retries):
                    try:
                        # Add timeout wrapper
                        result = await asyncio.wait_for(
                            self.adapter.search(
                                query=qa.question,
                                user_id=user_id,
                                top_k=top_k,
                                question_id=qa.question_id,
                                **kwargs
                            ),
                            timeout=search_timeout
                        )
                        break  # Success, exit retry loop
                        
                    except asyncio.TimeoutError:
                        last_error = f"Timeout after {search_timeout}s"
                        if attempt < max_retries - 1:
                            delay = min(retry_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                            self.console.print(
                                f"   ⏱️  Search timeout for {qa.question_id}, "
                                f"retry {attempt + 1}/{max_retries} in {delay:.1f}s...",
                                style="yellow"
                            )
                            await asyncio.sleep(delay)
                            
                    except Exception as e:
                        last_error = str(e)
                        if attempt < max_retries - 1:
                            delay = min(retry_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                            self.console.print(
                                f"   ⚠️  Search failed for {qa.question_id}: {str(e)[:50]}..., "
                                f"retry {attempt + 1}/{max_retries} in {delay:.1f}s",
                                style="yellow"
                            )
                            await asyncio.sleep(delay)
                
                # If all retries exhausted, create empty result
                if result is None:
                    self.console.print(
                        f"   ❌ Search failed after {max_retries} attempts for {qa.question_id}: {last_error}",
                        style="red"
                    )
                    result = SearchResult(
                        question_id=qa.question_id,
                        query=qa.question,
                        retrieved_memories=[],
                        context="(Search failed)",
                        search_duration_ms=0,
                        metadata={"error": last_error}
                    )
                
                results[idx] = result
                completed += 1
                progress.update(task_id, completed=completed)
        
        # Execute with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[cyan]{task.completed}/{task.total}"),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task_id = progress.add_task("Searching memories...", total=total)
            
            tasks = [
                search_with_retry(idx, qa, progress, task_id)
                for idx, qa in enumerate(qa_items)
            ]
            
            await asyncio.gather(*tasks)
        
        self.console.print(f"\n{'='*60}", style="bold blue")
        self.console.print(f"✅ Search completed: {len(results)} results", style="bold green")
        
        return results
    
    async def _run_answer_stage(
        self,
        qa_items: List[QAItem],
        search_results: List[SearchResult],
        user_id: str = "",
        **kwargs
    ) -> Union[List[AnswerResult], List[LightAnswerResult]]:
        """
        Execute Answer stage with concurrent processing and resume support.
        
        For LLM system:
        - Uses memory-efficient LightAnswerResult (no context storage)
        - Includes warmup phase before batch processing
        - Supports question-level resume (skips already answered questions)
        
        Args:
            qa_items: List of QA items
            search_results: List of search results
            user_id: User ID for loading existing results (resume support)
            **kwargs: Additional parameters
            
        Returns:
            List of AnswerResult (memory systems) or LightAnswerResult (LLM system)
        """
        self.console.print(f"\n{'='*60}", style="bold yellow")
        self.console.print("Stage: Answer", style="bold yellow")
        self.console.print(f"{'='*60}", style="bold yellow")
        self.console.print(f"Total questions: {len(qa_items)}")
        
        # Try to load existing answer results for resume
        existing_results = {}
        if user_id:
            if self.system_name == "llm":
                existing_list = self._load_light_answer_results(user_id)
            else:
                existing_list = self._load_answer_results(user_id)
            
            if existing_list:
                existing_results = {r.question_id: r for r in existing_list}
                self.console.print(f"   📂 Loaded {len(existing_results)} existing answers (resume mode)", style="dim cyan")
        
        # Initialize answerer if needed (pass system_name for prompt selection)
        if self._answerer is None:
            from eval.src.core.answerer import Answerer
            self._answerer = Answerer(system_name=self.system_name)
        
        # Map search results by question_id
        sr_map = {sr.question_id: sr for sr in search_results}
        
        # Prepare paired lists for batch processing (filter out already answered)
        paired_qa_items = []
        paired_search_results = []
        skipped_count = 0
        
        for qa in qa_items:
            # Skip if already answered
            if qa.question_id in existing_results:
                skipped_count += 1
                continue
            
            sr = sr_map.get(qa.question_id)
            if sr is None:
                self.console.print(f"   ⚠️  No search result for {qa.question_id}", style="yellow")
                sr = SearchResult(
                    question_id=qa.question_id,
                    query=qa.question,
                    retrieved_memories=[],
                    context="(No search result)",
                    search_duration_ms=0
                )
            paired_qa_items.append(qa)
            paired_search_results.append(sr)
        
        if skipped_count > 0:
            self.console.print(f"   ⏭️  Skipped {skipped_count} already answered questions", style="dim green")
            self.console.print(f"   🔄 Remaining to process: {len(paired_qa_items)}", style="dim yellow")
        
        # If all questions already answered, return existing results in original order
        if len(paired_qa_items) == 0:
            self.console.print(f"   ✅ All questions already answered, nothing to do", style="green")
            # Return results in original qa_items order
            return [existing_results[qa.question_id] for qa in qa_items if qa.question_id in existing_results]
        
        # Warmup phase for LLM system (improves cache hit rate)
        if self.system_name == "llm" and len(paired_qa_items) > 1:
            await self._warmup_llm_cache(paired_qa_items[0], paired_search_results[0])
        
        # For LLM system, use memory-efficient processing with incremental save
        if self.system_name == "llm":
            new_results = await self._run_llm_answer_stage_lightweight(
                paired_qa_items, 
                paired_search_results,
                user_id=user_id,
                existing_results=existing_results,
                all_qa_items=qa_items,
                **kwargs
            )
        else:
            # Standard batch processing for memory systems
            new_results = await self._answerer.generate_answers_batch(
                paired_qa_items, 
                paired_search_results, 
                **kwargs
            )
        
        # Merge existing and new results
        new_results_map = {r.question_id: r for r in new_results}
        merged_results_map = {**existing_results, **new_results_map}
        
        # Return results in original qa_items order
        answer_results = [merged_results_map[qa.question_id] for qa in qa_items if qa.question_id in merged_results_map]
        
        self.console.print(f"\n{'='*60}", style="bold yellow")
        self.console.print(f"✅ Answer completed: {len(answer_results)} total ({len(new_results)} new, {skipped_count} resumed)", style="bold green")
        
        return answer_results
    
    async def _run_llm_answer_stage_lightweight(
        self,
        qa_items: List[QAItem],
        search_results: List[SearchResult],
        user_id: str = "",
        existing_results: Dict[str, LightAnswerResult] = None,
        all_qa_items: List[QAItem] = None,
        **kwargs
    ) -> List[LightAnswerResult]:
        """
        Memory-efficient answer generation for LLM system with incremental save.
        
        Uses shared context reference and converts to LightAnswerResult
        immediately to avoid storing large context in each result.
        Saves progress incrementally to support resume on interruption.
        
        Args:
            qa_items: List of QA items to process (excluding already answered)
            search_results: List of search results (all share same context)
            user_id: User ID for saving results
            existing_results: Dict of already answered results (question_id -> result)
            all_qa_items: Full list of QA items (for saving in correct order)
            **kwargs: Additional parameters
            
        Returns:
            List of LightAnswerResult objects (memory-efficient)
        """
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
        import threading
        
        existing_results = existing_results or {}
        all_qa_items = all_qa_items or qa_items
        
        # Get shared context reference (all search_results have same context for LLM)
        shared_context = search_results[0].context if search_results else ""
        context_size_mb = len(shared_context) / (1024 * 1024)
        self.console.print(f"   📦 Shared context: {context_size_mb:.2f} MB (memory-efficient mode)")
        
        # Create a single SearchResult to reuse (avoid creating N copies)
        shared_search_result = SearchResult(
            question_id="shared",
            query="",
            retrieved_memories=[],
            context=shared_context,
            search_duration_ms=0,
            metadata={"mode": "llm_shared_context"}
        )
        
        # Clear original search_results to free memory
        search_results.clear()
        
        # Get concurrency settings from pipeline.yaml
        concurrency = self.pipeline_config.get("answer", {}).get("concurrency", 1)
        
        semaphore = asyncio.Semaphore(concurrency)
        
        # Reset answerer cache tracking
        self._answerer.cache_hits = 0
        self._answerer.total_requests = 0
        
        # Track progress
        completed = 0
        total = len(qa_items)
        results: List[LightAnswerResult] = [None] * total
        
        # For incremental save - track new results
        new_results_map: Dict[str, LightAnswerResult] = {}
        save_lock = threading.Lock()
        save_interval = 5  # Save every N completed questions
        last_saved_count = 0
        
        def _save_incremental():
            """Save current progress (existing + new results)."""
            nonlocal last_saved_count
            if not user_id:
                return
            
            with save_lock:
                # Merge existing and new results
                merged = {**existing_results, **new_results_map}
                # Build ordered list based on all_qa_items
                ordered_results = [merged[qa.question_id] for qa in all_qa_items if qa.question_id in merged]
                if ordered_results:
                    self._save_light_answer_results(ordered_results, user_id)
                    last_saved_count = len(new_results_map)
        
        async def generate_lightweight(idx: int, qa: QAItem, progress, task_id) -> None:
            """Generate answer and convert to lightweight result."""
            nonlocal completed
            async with semaphore:
                # Generate answer using shared context
                answer_result = await self._answerer.generate_answer(qa, shared_search_result, **kwargs)
                
                # Convert to lightweight result immediately (discard search_result reference)
                light_result = LightAnswerResult(
                    question_id=answer_result.question_id,
                    question=answer_result.question,
                    question_type=answer_result.question_type,
                    golden_answer=answer_result.golden_answer,
                    generated_answer=answer_result.generated_answer,
                    answer_duration_ms=answer_result.answer_duration_ms,
                )
                results[idx] = light_result
                
                # Track for incremental save
                with save_lock:
                    new_results_map[qa.question_id] = light_result
                
                # Clear reference to AnswerResult
                del answer_result
                
                completed += 1
                progress.update(
                    task_id, 
                    completed=completed,
                    description=f"Generating answers... [green]Cache: {self._answerer.cache_hits}/{self._answerer.total_requests}[/green]"
                )
                
                # Incremental save every N questions
                if completed % save_interval == 0 and completed > last_saved_count:
                    _save_incremental()
        
        # Execute with progress bar
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
                generate_lightweight(idx, qa, progress, task_id)
                for idx, qa in enumerate(qa_items)
            ]
            
            await asyncio.gather(*tasks)
        
        # Final save to ensure all results are persisted
        _save_incremental()
        
        # Print cache stats
        cache_rate = (self._answerer.cache_hits / self._answerer.total_requests * 100) if self._answerer.total_requests > 0 else 0
        self.console.print(f"   📊 Cache Statistics: {self._answerer.cache_hits}/{self._answerer.total_requests} hits ({cache_rate:.1f}%)")
        
        return results
    
    async def _warmup_llm_cache(self, first_qa: QAItem, first_sr: SearchResult):
        """
        Warmup LLM cache by sending first request and waiting.
        
        This improves cache hit rate for subsequent requests as the
        provider can cache the prompt prefix.
        
        Args:
            first_qa: First QA item
            first_sr: First search result
        """
        # Load warmup config from pipeline.yaml
        warmup_config = self.pipeline_config.get("warmup", {})
        warmup_enabled = warmup_config.get("enabled", True)
        warmup_delay = warmup_config.get("delay_seconds", 5)
        
        if not warmup_enabled:
            return
        
        self.console.print(f"\n[cyan]🔥 Warmup Phase: Sending first request to warm cache...[/cyan]")
        
        # Generate answer for first question (warmup request)
        await self._answerer.generate_answer(first_qa, first_sr)
        
        self.console.print(f"[cyan]   ⏳ Waiting {warmup_delay}s for cache propagation...[/cyan]")
        await asyncio.sleep(warmup_delay)
        
        self.console.print(f"[cyan]   ✅ Warmup complete, starting batch processing[/cyan]\n")
    
    async def _run_evaluate_stage(
        self,
        answer_results: Union[List[AnswerResult], List[LightAnswerResult]],
        **kwargs
    ) -> EvaluationResult:
        """
        Execute Evaluate stage.
        
        Args:
            answer_results: List of answer results
            **kwargs: Additional parameters (num_runs for LLM judge)
            
        Returns:
            EvaluationResult
        """
        # Initialize evaluator if needed
        if self._evaluator is None:
            from eval.src.core.evaluator import Evaluator
            num_runs = kwargs.get("num_runs", 1)
            self._evaluator = Evaluator(num_runs=num_runs, system_name=self.system_name)
        
        return await self._evaluator.evaluate(answer_results, **kwargs)
    
    def _save_search_results(self, results: List[SearchResult], user_id: str):
        """Save search results to JSON file."""
        # Results are saved in system-specific folder: results/{system}/search_results_{user_id}.json
        output_path = self.output_dir / f"search_results_{user_id}.json"
        data = [self._search_result_to_dict(r) for r in results]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.console.print(f"   📁 Saved: {output_path}")
    
    def _save_answer_results(self, results: List[AnswerResult], user_id: str):
        """Save answer results to JSON file."""
        # Results are saved in system-specific folder: results/{system}/answer_results_{user_id}.json
        output_path = self.output_dir / f"answer_results_{user_id}.json"
        data = [self._answer_result_to_dict(r) for r in results]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.console.print(f"   📁 Saved: {output_path}")
    
    def _save_evaluation_results(self, result: EvaluationResult, user_id: str):
        """Save evaluation results to JSON file."""
        # Results are saved in system-specific folder: results/{system}/evaluation_results_{user_id}.json
        output_path = self.output_dir / f"evaluation_results_{user_id}.json"
        data = asdict(result)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.console.print(f"   📁 Saved: {output_path}")
    
    def _load_search_results(self, user_id: str) -> Optional[List[SearchResult]]:
        """Load search results from JSON file."""
        input_path = self.output_dir / f"search_results_{user_id}.json"
        if not input_path.exists():
            return None
        
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return [SearchResult(
            question_id=r.get("question_id", ""),
            query=r.get("query", ""),
            retrieved_memories=r.get("retrieved_memories", []),
            context=r.get("context", ""),
            search_duration_ms=r.get("search_duration_ms", 0),
            metadata=r.get("metadata", {}),
        ) for r in data]
    
    def _load_answer_results(self, user_id: str) -> Optional[List[AnswerResult]]:
        """Load answer results from JSON file."""
        input_path = self.output_dir / f"answer_results_{user_id}.json"
        if not input_path.exists():
            return None
        
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return [AnswerResult(
            question_id=r.get("question_id", ""),
            question=r.get("question", ""),
            question_type=r.get("question_type", "open_ended"),
            golden_answer=r.get("golden_answer", ""),
            generated_answer=r.get("generated_answer", ""),
            search_result=SearchResult(
                question_id=r.get("question_id", ""),
                query=r.get("question", ""),
                retrieved_memories=r.get("search_result", {}).get("retrieved_memories", []),
                context=r.get("search_result", {}).get("context", ""),
                search_duration_ms=r.get("search_result", {}).get("search_duration_ms", 0),
            ),
            answer_duration_ms=r.get("answer_duration_ms", 0),
            metadata=r.get("metadata", {}),
        ) for r in data]
    
    def _save_light_answer_results(self, results: List[LightAnswerResult], user_id: str):
        """Save LightAnswerResult list to JSON file (for LLM system resume support)."""
        output_path = self.output_dir / f"answer_results_{user_id}.json"
        data = [self._light_answer_result_to_dict(r) for r in results]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.console.print(f"   📁 Saved: {output_path}")
    
    def _load_light_answer_results(self, user_id: str) -> Optional[List[LightAnswerResult]]:
        """Load LightAnswerResult list from JSON file."""
        input_path = self.output_dir / f"answer_results_{user_id}.json"
        if not input_path.exists():
            return None
        
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return [LightAnswerResult(
            question_id=r.get("question_id", ""),
            question=r.get("question", ""),
            question_type=r.get("question_type", "open_ended"),
            golden_answer=r.get("golden_answer", ""),
            generated_answer=r.get("generated_answer", ""),
            answer_duration_ms=r.get("answer_duration_ms", 0),
        ) for r in data]
    
    def _light_answer_result_to_dict(self, r: LightAnswerResult) -> Dict[str, Any]:
        """Convert LightAnswerResult to dict."""
        return {
            "question_id": r.question_id,
            "question": r.question,
            "question_type": r.question_type,
            "golden_answer": r.golden_answer,
            "generated_answer": r.generated_answer,
            "answer_duration_ms": r.answer_duration_ms,
        }
    
    def _search_result_to_dict(self, r: SearchResult) -> Dict[str, Any]:
        """Convert SearchResult to dict."""
        return {
            "question_id": r.question_id,
            "query": r.query,
            "retrieved_memories": r.retrieved_memories,
            "context": r.context,
            "search_duration_ms": r.search_duration_ms,
            "metadata": r.metadata,
        }
    
    def _answer_result_to_dict(self, r: AnswerResult) -> Dict[str, Any]:
        """Convert AnswerResult to dict."""
        return {
            "question_id": r.question_id,
            "question": r.question,
            "question_type": r.question_type,
            "golden_answer": r.golden_answer,
            "generated_answer": r.generated_answer,
            "answer_duration_ms": r.answer_duration_ms,
            "search_result": self._search_result_to_dict(r.search_result),
            "metadata": r.metadata,
        }
    
    def _print_summary(self, results: Dict[str, Any], elapsed: float):
        """Print pipeline summary."""
        self.console.print(f"\n{'='*60}", style="bold green")
        self.console.print("📊 Pipeline Summary", style="bold green")
        self.console.print(f"{'='*60}", style="bold green")
        
        if "add" in results:
            add_result: AddResult = results["add"]
            status = "✅ Success" if add_result.success else "⚠️  With Errors"
            self.console.print(f"Add Stage: {status}")
            self.console.print(f"  Days Processed: {add_result.days_processed}")
            self.console.print(f"  Messages Sent: {add_result.messages_sent}")
            if add_result.errors:
                self.console.print(f"  Errors: {len(add_result.errors)}")
        
        if "search" in results:
            search_results: List[SearchResult] = results["search"]
            self.console.print(f"Search Stage: ✅ {len(search_results)} queries")
        
        if "answer" in results:
            answer_results: List[AnswerResult] = results["answer"]
            self.console.print(f"Answer Stage: ✅ {len(answer_results)} answers")
        
        if "evaluate" in results:
            eval_result: EvaluationResult = results["evaluate"]
            self.console.print(f"Evaluate Stage: ✅ Accuracy: {eval_result.accuracy:.2%}")
        
        self.console.print(f"\nTotal Time: {elapsed:.2f}s")
        self.console.print(f"{'='*60}\n", style="bold green")
