"""
Pipeline core module.

Orchestrates the evaluation workflow across four stages: Add → Search → Answer → Evaluate.
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from evaluation.src.core.data_models import (
    Dataset,
    SearchResult,
    AnswerResult,
    EvaluationResult,
)
from evaluation.src.adapters.base import BaseAdapter
from evaluation.src.evaluators.base import BaseEvaluator
from evaluation.src.utils.logger import setup_logger, get_console
from evaluation.src.utils.saver import ResultSaver
from evaluation.src.utils.checkpoint import CheckpointManager

# Import components for answer generation
from memory_layer.llm.llm_provider import LLMProvider

# Import stage execution functions
from evaluation.src.core.stages.add_stage import run_add_stage
from evaluation.src.core.stages.search_stage import run_search_stage
from evaluation.src.core.stages.answer_stage import run_answer_stage
from evaluation.src.core.stages.evaluate_stage import run_evaluate_stage


class Pipeline:
    """
    Evaluation Pipeline.

    Four-stage workflow:
    1. Add: Ingest conversation data and build indices
    2. Search: Retrieve relevant memories
    3. Answer: Generate answers
    4. Evaluate: Evaluate answer quality
    """

    def __init__(
        self,
        adapter: BaseAdapter,
        evaluator: BaseEvaluator,
        llm_provider: LLMProvider,
        output_dir: Path,
        run_name: str = "default",
        use_checkpoint: bool = True,
        filter_categories: Optional[List[int]] = None,
    ):
        """
        Initialize Pipeline.

        Args:
            adapter: System adapter
            evaluator: Evaluator
            llm_provider: LLM Provider for answer generation
            output_dir: Output directory
            run_name: Run name to distinguish different runs
            use_checkpoint: Enable checkpoint/resume functionality
            filter_categories: List of question categories to filter out (e.g., [5] filters Category 5)
        """
        self.adapter = adapter
        self.evaluator = evaluator
        self.llm_provider = llm_provider
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger(self.output_dir / "pipeline.log")
        self.saver = ResultSaver(self.output_dir)
        self.console = get_console()

        # Checkpoint/resume support
        self.use_checkpoint = use_checkpoint
        self.checkpoint = (
            CheckpointManager(output_dir=output_dir, run_name=run_name)
            if use_checkpoint
            else None
        )
        self.completed_stages: set = set()

        # Question category filter configuration (read from dataset config)
        self.filter_categories = filter_categories or []

    async def run(
        self,
        dataset: Dataset,
        stages: Optional[List[str]] = None,
        smoke_test: bool = False,
        smoke_messages: int = 10,
        smoke_questions: int = 3,
        from_conv: int = 0,
        to_conv: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run complete Pipeline.

        Args:
            dataset: Standard format dataset
            stages: List of stages to execute, None means all
                   Options: ["add", "search", "answer", "evaluate"]
            smoke_test: Enable smoke test mode
            smoke_messages: Number of messages in smoke test (default 10)
            smoke_questions: Number of questions in smoke test (default 3)
            from_conv: Starting conversation index to process (inclusive, 0-based)
            to_conv: Ending conversation index to process (exclusive), None means all

        Returns:
            Evaluation results dictionary
        """
        start_time = time.time()

        self.console.print(f"\n{'='*60}", style="bold cyan")
        self.console.print("🚀 Evaluation Pipeline", style="bold cyan")
        self.console.print(f"{'='*60}", style="bold cyan")
        self.console.print(f"Dataset: {dataset.dataset_name}")
        self.console.print(f"System: {self.adapter.get_system_info()['name']}")
        self.console.print(f"Stages: {stages or 'all'}")
        if smoke_test:
            self.console.print(
                f"[yellow]🧪 Smoke Test Mode: {smoke_messages} messages, {smoke_questions} questions[/yellow]"
            )
        self.console.print(f"{'='*60}\n", style="bold cyan")

        # Apply conversation range filter (before smoke test)
        # This allows processing a subset of conversations for incremental/distributed testing
        if from_conv > 0 or to_conv is not None:
            dataset = self._apply_conversation_range(dataset, from_conv, to_conv)
            self.console.print(f"[cyan]📌 Conversation Range Filter Applied:[/cyan]")
            self.console.print(
                f"[cyan]   Range: [{from_conv}:{to_conv or 'end'}][/cyan]"
            )
            self.console.print(
                f"[cyan]   Conversations: {len(dataset.conversations)}[/cyan]"
            )
            self.console.print(f"[cyan]   Questions: {len(dataset.qa_pairs)}[/cyan]\n")

        # Smoke test: trim messages and questions for quick validation
        if smoke_test:
            dataset = self._apply_smoke_test(dataset, smoke_messages, smoke_questions)
            self.console.print(f"[yellow]✂️  Smoke test applied:[/yellow]")
            self.console.print(
                f"[yellow]   - Conversations: {len(dataset.conversations)}[/yellow]"
            )
            if len(dataset.conversations) == 0:
                self.console.print(
                    f"[red]   ⚠️  No conversations selected! Check your filters.[/red]"
                )
            elif len(dataset.conversations) == 1:
                self.console.print(
                    f"[yellow]   - Conversation ID: {dataset.conversations[0].conversation_id}[/yellow]"
                )
            else:
                first_id = dataset.conversations[0].conversation_id
                last_id = dataset.conversations[-1].conversation_id
                self.console.print(
                    f"[yellow]   - Range: {first_id} to {last_id}[/yellow]"
                )
            total_messages = sum(len(conv.messages) for conv in dataset.conversations)
            msg_limit = (
                f"max {smoke_messages} per conv" if smoke_messages > 0 else "all"
            )
            qa_limit = (
                f"max {smoke_questions} per conv" if smoke_questions > 0 else "all"
            )
            self.console.print(
                f"[yellow]   - Messages: {total_messages} ({msg_limit})[/yellow]"
            )
            self.console.print(
                f"[yellow]   - Questions: {len(dataset.qa_pairs)} ({qa_limit})[/yellow]\n"
            )

        # Check if we have any conversations to process
        if len(dataset.conversations) == 0:
            self.console.print(
                f"[red]❌ No conversations to process! Check your --from-conv and --to-conv parameters.[/red]"
            )
            self.console.print(
                f"[yellow]💡 Tip: --to-conv should be greater than --from-conv (uses Python slice [from:to))[/yellow]"
            )
            return {
                "error": "No conversations selected",
                "stages_completed": [],
                "total_conversations": 0,
                "total_questions": 0,
            }

        # Filter question categories based on config (e.g., filter out Category 5 adversarial questions)
        original_qa_count = len(dataset.qa_pairs)

        if self.filter_categories:
            # Normalize categories to strings (support both int and str configs)
            filter_set = {str(cat) for cat in self.filter_categories}

            # Filter out specified categories
            dataset.qa_pairs = [
                qa for qa in dataset.qa_pairs if qa.category not in filter_set
            ]

            filtered_count = original_qa_count - len(dataset.qa_pairs)

            if filtered_count > 0:
                filtered_categories_str = ", ".join(sorted(filter_set))
                self.console.print(
                    f"[dim]🔍 Filtered out {filtered_count} questions from categories: {filtered_categories_str}[/dim]"
                )
                self.console.print(
                    f"[dim]   Remaining questions: {len(dataset.qa_pairs)}[/dim]\n"
                )

        # Try loading checkpoint
        search_results_data = None
        answer_results_data = None

        if self.use_checkpoint and self.checkpoint:
            checkpoint_data = self.checkpoint.load_checkpoint()
            if checkpoint_data:
                self.completed_stages = set(checkpoint_data.get('completed_stages', []))
                # Load saved intermediate results
                if 'search_results' in checkpoint_data:
                    search_results_data = checkpoint_data['search_results']
                if 'answer_results' in checkpoint_data:
                    answer_results_data = checkpoint_data['answer_results']

        # Default: execute all stages
        if stages is None:
            stages = ["add", "search", "answer", "evaluate"]

        results = {}

        # Stage 1: Add
        add_just_completed = False  # Track if add just completed

        if "add" in stages and "add" not in self.completed_stages:
            self.logger.info("Starting Stage 1: Add")

            stage_results = await run_add_stage(
                adapter=self.adapter,
                dataset=dataset,
                output_dir=self.output_dir,
                checkpoint_manager=self.checkpoint,
                logger=self.logger,
                console=self.console,
                completed_stages=self.completed_stages,
            )
            results.update(stage_results)
            add_just_completed = True  # Add just completed

        elif "add" in self.completed_stages:
            self.console.print(
                "\n[yellow]⏭️  Skip Add stage (already completed)[/yellow]"
            )
            # Rebuild index metadata (handled by adapter, only needed for local systems)
            # For online APIs, returns None but still need to set results["index"]
            index = self.adapter.build_lazy_index(
                dataset.conversations, self.output_dir
            )
            results["index"] = index  # Set even if None
        else:
            # Rebuild index metadata (handled by adapter, only needed for local systems)
            # For online APIs, returns None but still need to set results["index"]
            index = self.adapter.build_lazy_index(
                dataset.conversations, self.output_dir
            )
            results["index"] = index  # Set even if None
            if index is not None:
                self.logger.info("⏭️  Skipped Stage 1, using lazy loading")

        # Post-Add Wait: for online API systems, wait for backend indexing to complete
        # Only wait if add just completed
        if add_just_completed:
            wait_seconds = self.adapter.config.get("post_add_wait_seconds", 0)
            if wait_seconds > 0 and "search" in stages:
                self.console.print(
                    f"\n[yellow]⏰ Waiting {wait_seconds}s for backend indexing to complete...[/yellow]"
                )
                self.logger.info(f"⏰ Waiting {wait_seconds}s for backend indexing")

                # Show countdown progress bar
                from rich.progress import (
                    Progress,
                    SpinnerColumn,
                    TextColumn,
                    BarColumn,
                    TimeRemainingColumn,
                )

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("{task.percentage:>3.0f}%"),
                    TimeRemainingColumn(),
                    console=self.console,
                ) as progress:
                    task = progress.add_task(
                        f"⏰ Backend indexing in progress...", total=wait_seconds
                    )
                    for i in range(wait_seconds):
                        time.sleep(1)
                        progress.update(task, advance=1)

                self.console.print(
                    f"[green]✅ Wait completed, ready for search[/green]\n"
                )
                self.logger.info("✅ Post-add wait completed")

        # Stage 2: Search
        if "search" in stages and "search" not in self.completed_stages:
            self.logger.info("Starting Stage 2: Search")

            search_results = await run_search_stage(
                adapter=self.adapter,
                qa_pairs=dataset.qa_pairs,
                index=results["index"],
                conversations=dataset.conversations,  # Pass conversations for cache rebuilding
                checkpoint_manager=self.checkpoint,
                logger=self.logger,
            )

            self.saver.save_json(
                [self._search_result_to_dict(sr) for sr in search_results],
                "search_results.json",
            )
            results["search_results"] = search_results
            self.logger.info("✅ Stage 2 completed")

            # Save checkpoint
            self.completed_stages.add("search")
            if self.checkpoint:
                search_results_data = [
                    self._search_result_to_dict(sr) for sr in search_results
                ]
                self.checkpoint.save_checkpoint(
                    self.completed_stages, search_results=search_results_data
                )
        elif "search" in self.completed_stages:
            self.console.print(
                f"\n[yellow]⏭️  Skip Search stage (already completed)[/yellow]"
            )
            if search_results_data:
                # Load from checkpoint
                search_results = [
                    self._dict_to_search_result(d) for d in search_results_data
                ]
                results["search_results"] = search_results
            elif self.saver.file_exists("search_results.json"):
                # Load from file
                search_data = self.saver.load_json("search_results.json")
                search_results = [self._dict_to_search_result(d) for d in search_data]
                results["search_results"] = search_results
        elif "answer" in stages or "evaluate" in stages:
            # Only try loading when subsequent stages need search_results
            if self.saver.file_exists("search_results.json"):
                search_data = self.saver.load_json("search_results.json")
                search_results = [self._dict_to_search_result(d) for d in search_data]
                results["search_results"] = search_results
                self.logger.info("⏭️  Skipped Stage 2, loaded existing results")
            else:
                raise FileNotFoundError(
                    "Search results not found. Please run 'search' stage first."
                )
        else:
            # Don't need search_results (e.g., only running add stage)
            search_results = None

        # Stage 3: Answer
        if "answer" in stages and "answer" not in self.completed_stages:
            self.logger.info("Starting Stage 3: Answer")

            answer_results = await run_answer_stage(
                adapter=self.adapter,
                qa_pairs=dataset.qa_pairs,
                search_results=search_results,
                checkpoint_manager=self.checkpoint,
                logger=self.logger,
            )

            self.saver.save_json(
                [self._answer_result_to_dict(ar) for ar in answer_results],
                "answer_results.json",
            )
            results["answer_results"] = answer_results
            self.logger.info("✅ Stage 3 completed")

            # Save checkpoint
            self.completed_stages.add("answer")
            if self.checkpoint:
                answer_results_dict = [
                    self._answer_result_to_dict(ar) for ar in answer_results
                ]
                self.checkpoint.save_checkpoint(
                    self.completed_stages,
                    search_results=(
                        search_results_data
                        if search_results_data
                        else [self._search_result_to_dict(sr) for sr in search_results]
                    ),
                    answer_results=answer_results_dict,
                )
                # Sync answer_results_data to ensure subsequent stages use correct data
                answer_results_data = answer_results_dict
        elif "answer" in self.completed_stages:
            self.console.print(
                f"\n[yellow]⏭️  Skip Answer stage (already completed)[/yellow]"
            )
            if answer_results_data:
                # Load from checkpoint
                answer_results = [
                    self._dict_to_answer_result(d) for d in answer_results_data
                ]
                results["answer_results"] = answer_results
            elif self.saver.file_exists("answer_results.json"):
                # Load from file
                answer_data = self.saver.load_json("answer_results.json")
                answer_results = [self._dict_to_answer_result(d) for d in answer_data]
                results["answer_results"] = answer_results
        elif "evaluate" in stages:
            # Only try loading when evaluate stage needs answer_results
            if self.saver.file_exists("answer_results.json"):
                answer_data = self.saver.load_json("answer_results.json")
                answer_results = [self._dict_to_answer_result(d) for d in answer_data]
                results["answer_results"] = answer_results
                self.logger.info("⏭️  Skipped Stage 3, loaded existing results")
            else:
                raise FileNotFoundError(
                    "Answer results not found. Please run 'answer' stage first."
                )
        else:
            # Don't need answer_results (e.g., only running add or search)
            answer_results = None

        # Stage 4: Evaluate
        if "evaluate" in stages and "evaluate" not in self.completed_stages:
            eval_result = await run_evaluate_stage(
                evaluator=self.evaluator,
                answer_results=answer_results,
                checkpoint_manager=self.checkpoint,
                logger=self.logger,
            )

            self.saver.save_json(
                self._eval_result_to_dict(eval_result), "eval_results.json"
            )
            results["eval_result"] = eval_result

            # Save checkpoint
            self.completed_stages.add("evaluate")
            if self.checkpoint:
                # Handle None cases for search_results and answer_results
                if search_results_data:
                    sr_data = search_results_data
                elif search_results:
                    sr_data = [self._search_result_to_dict(sr) for sr in search_results]
                else:
                    sr_data = []

                if answer_results_data:
                    ar_data = answer_results_data
                elif answer_results:
                    ar_data = [self._answer_result_to_dict(ar) for ar in answer_results]
                else:
                    ar_data = []

                self.checkpoint.save_checkpoint(
                    self.completed_stages,
                    search_results=sr_data,
                    answer_results=ar_data,
                    eval_results=self._eval_result_to_dict(eval_result),
                )
        elif "evaluate" in self.completed_stages:
            self.console.print(
                "\n[yellow]⏭️  Skip Evaluate stage (already completed)[/yellow]"
            )

        # Generate report
        elapsed_time = time.time() - start_time
        self._generate_report(results, elapsed_time)

        return results

    def _apply_smoke_test(
        self, dataset: Dataset, num_messages: int, num_questions: int
    ) -> Dataset:
        """
        Apply smoke test: trim messages and questions for quick validation.

        This allows quick validation of the complete workflow (Add → Search → Answer → Evaluate)
        using only a small subset of data to save time.

        Strategy:
        - If dataset has multiple conversations (e.g., from conversation range filter):
          Apply smoke limits to ALL conversations in the range
        - If dataset has only one conversation:
          Apply smoke limits to that conversation (legacy behavior)

        Args:
            dataset: Original dataset (may be pre-filtered by conversation range)
            num_messages: Number of messages to keep per conversation (for Add stage), 0 means all
            num_questions: Number of questions to keep per conversation (for Search/Answer/Evaluate stages), 0 means all

        Returns:
            Trimmed dataset
        """
        if not dataset.conversations:
            return dataset

        # Process all conversations (respecting conversation range filter if applied)
        trimmed_conversations = []
        trimmed_qa_pairs = []

        total_messages_before = 0
        total_messages_after = 0
        total_questions_before = 0
        total_questions_after = 0

        for conv in dataset.conversations:
            conv_id = conv.conversation_id

            # Trim messages for this conversation
            if num_messages > 0:
                total_messages_before += len(conv.messages)
                conv.messages = conv.messages[:num_messages]
                total_messages_after += len(conv.messages)
            else:
                total_messages_after += len(conv.messages)
                total_messages_before += len(conv.messages)

            trimmed_conversations.append(conv)

            # Trim questions for this conversation
            conv_qa_pairs = [
                qa
                for qa in dataset.qa_pairs
                if qa.metadata.get("conversation_id") == conv_id
            ]

            if num_questions > 0:
                total_questions_before += len(conv_qa_pairs)
                selected_qa_pairs = conv_qa_pairs[:num_questions]
                total_questions_after += len(selected_qa_pairs)
            else:
                selected_qa_pairs = conv_qa_pairs
                total_questions_after += len(selected_qa_pairs)
                total_questions_before += len(selected_qa_pairs)

            trimmed_qa_pairs.extend(selected_qa_pairs)

        # Log summary
        if len(trimmed_conversations) == 1:
            conv_desc = f"Conv {trimmed_conversations[0].conversation_id}"
        else:
            conv_desc = f"{len(trimmed_conversations)} conversations"

        msg_desc = (
            f"{total_messages_after}/{total_messages_before}"
            if num_messages > 0
            else f"{total_messages_after} (all)"
        )
        qa_desc = (
            f"{total_questions_after}/{total_questions_before}"
            if num_questions > 0
            else f"{total_questions_after} (all)"
        )

        self.logger.info(
            f"Smoke test: {conv_desc} - "
            f"{msg_desc} messages, "
            f"{qa_desc} questions"
        )

        return Dataset(
            dataset_name=dataset.dataset_name + "_smoke",
            conversations=trimmed_conversations,
            qa_pairs=trimmed_qa_pairs,
            metadata={
                **dataset.metadata,
                "smoke_test": True,
                "smoke_messages": num_messages,
                "smoke_questions": num_questions,
                "total_conversations": len(trimmed_conversations),
            },
        )

    def _apply_conversation_range(
        self, dataset: Dataset, from_conv: int, to_conv: Optional[int]
    ) -> Dataset:
        """
        Filter conversations by index range.

        This allows processing a subset of conversations for incremental testing
        or distributed processing. The conversation_id attribute of each Conversation
        object remains unchanged, ensuring consistent user_id generation for online APIs.

        Args:
            dataset: Original dataset
            from_conv: Starting conversation index (inclusive, 0-based)
            to_conv: Ending conversation index (exclusive), None means all

        Returns:
            Filtered dataset with selected conversations and their QA pairs

        Example:
            - Original: 100 conversations (locomo_0 to locomo_99)
            - from_conv=10, to_conv=20: select conversations[10:20]
            - Result: 10 conversations (locomo_10 to locomo_19)
            - conversation_id attributes remain: "locomo_10", "locomo_11", ..., "locomo_19"
        """
        if not dataset.conversations:
            return dataset

        # Apply range slicing
        total_convs = len(dataset.conversations)
        end_idx = to_conv if to_conv is not None else total_convs

        # Validation
        if from_conv < 0:
            self.logger.warning(f"from_conv < 0, resetting to 0")
            from_conv = 0
        if from_conv >= total_convs:
            self.logger.warning(
                f"from_conv ({from_conv}) >= total conversations ({total_convs}), no data to process"
            )
            return Dataset(
                dataset_name=dataset.dataset_name,
                conversations=[],
                qa_pairs=[],
                metadata={
                    **dataset.metadata,
                    "conversation_range": [from_conv, end_idx],
                    "original_conversation_count": total_convs,
                    "original_qa_count": len(dataset.qa_pairs),
                },
            )

        # Slice conversations (conversation_id attributes remain unchanged)
        selected_convs = dataset.conversations[from_conv:end_idx]
        selected_conv_ids = {conv.conversation_id for conv in selected_convs}

        # Filter QA pairs for selected conversations
        selected_qa_pairs = [
            qa
            for qa in dataset.qa_pairs
            if qa.metadata.get("conversation_id") in selected_conv_ids
        ]

        self.logger.info(
            f"Conversation range [{from_conv}:{end_idx}] - "
            f"selected {len(selected_convs)}/{total_convs} conversations, "
            f"{len(selected_qa_pairs)}/{len(dataset.qa_pairs)} questions"
        )

        return Dataset(
            dataset_name=dataset.dataset_name,
            conversations=selected_convs,
            qa_pairs=selected_qa_pairs,
            metadata={
                **dataset.metadata,
                "conversation_range": [from_conv, end_idx],
                "original_conversation_count": total_convs,
                "original_qa_count": len(dataset.qa_pairs),
            },
        )

    def _generate_report(self, results: Dict[str, Any], elapsed_time: float):
        """Generate evaluation report."""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("📊 Evaluation Report")
        report_lines.append("=" * 60)
        report_lines.append("")

        # System information
        system_info = self.adapter.get_system_info()
        report_lines.append(f"System: {system_info['name']}")
        report_lines.append(f"Time Elapsed: {elapsed_time:.2f}s")
        report_lines.append("")

        # Evaluation results
        if "eval_result" in results:
            eval_result = results["eval_result"]
            report_lines.append(f"Total Questions: {eval_result.total_questions}")
            report_lines.append(f"Correct: {eval_result.correct}")
            report_lines.append(f"Accuracy: {eval_result.accuracy:.2%}")
            report_lines.append("")

        report_lines.append("=" * 60)

        report_text = "\n".join(report_lines)

        # Save report
        report_path = self.output_dir / "report.txt"
        with open(report_path, "w") as f:
            f.write(report_text)

        # Print to console
        self.console.print("\n" + report_text, style="bold green")
        self.logger.info(f"Report saved to: {report_path}")

    # Serialization helper methods
    def _search_result_to_dict(self, sr: SearchResult) -> dict:
        """Convert SearchResult object to dictionary."""
        return {
            "query": sr.query,
            "conversation_id": sr.conversation_id,
            "results": sr.results,
            "retrieval_metadata": sr.retrieval_metadata,
        }

    def _dict_to_search_result(self, d: dict) -> SearchResult:
        """Convert dictionary to SearchResult object."""
        return SearchResult(**d)

    def _answer_result_to_dict(self, ar: AnswerResult) -> dict:
        """Convert AnswerResult object to dictionary."""
        # Handle empty search_results
        return {
            "question_id": ar.question_id,
            "question": ar.question,
            "answer": ar.answer,
            "golden_answer": ar.golden_answer,
            "category": ar.category,
            "conversation_id": ar.conversation_id,
            "formatted_context": ar.formatted_context,
            "metadata": ar.metadata,
        }

    def _dict_to_answer_result(self, d: dict) -> AnswerResult:
        """Convert dictionary to AnswerResult object."""
        return AnswerResult(**d)

    def _eval_result_to_dict(self, er: EvaluationResult) -> dict:
        """Convert EvaluationResult object to dictionary."""
        return {
            "total_questions": er.total_questions,
            "correct": er.correct,
            "accuracy": er.accuracy,
            "detailed_results": er.detailed_results,
            "metadata": er.metadata,
        }
