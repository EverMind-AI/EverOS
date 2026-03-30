"""
EverMemOS Adapter - connects evaluation framework with EverMemOS implementation.
"""

import asyncio
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.console import Console

from evaluation.src.adapters.base import BaseAdapter
from evaluation.src.adapters.registry import register_adapter
from evaluation.src.core.data_models import Conversation, SearchResult
from common_utils.datetime_utils import to_iso_format

# Import EverMemOS implementation
from evaluation.src.adapters.evermemos import (
    stage1_memcells_extraction,
    stage2_index_building,
    stage3_memory_retrivel,
    stage4_response,
)

# Import Memory Layer components
from memory_layer.llm.llm_provider import LLMProvider
from memory_layer.memory_extractor.event_log_extractor import EventLogExtractor


@register_adapter("evermemos")
class EverMemOSAdapter(BaseAdapter):
    """
    EverMemOS adapter.

    Responsibilities:
    1. Receive calls from evaluation framework
    2. Convert data formats (evaluation framework ↔ EverMemOS)
    3. Call stage*.py implementations
    4. Return results in evaluation framework format

    Implementation details:
    - MemCell extraction (stage1)
    - Index building (stage2)
    - Retrieval logic (stage3)
    - Answer generation (stage4)
    """

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config)
        self.output_dir = Path(output_dir) if output_dir else Path(".")

        # Initialize LLM Provider (shared across all stages)
        # Read from YAML llm configuration
        llm_config = config.get("llm", {})

        self.llm_provider = LLMProvider(
            provider_type=llm_config.get("provider", "openai"),
            model=llm_config.get("model", "gpt-4o-mini"),
            api_key=llm_config.get("api_key", ""),
            base_url=llm_config.get("base_url", "https://api.openai.com/v1"),
            temperature=llm_config.get("temperature", 0.3),
            max_tokens=llm_config.get("max_tokens", 32768),
        )

        # Initialize Event Log Extractor
        self.event_log_extractor = EventLogExtractor(llm_provider=self.llm_provider)

        # Ensure NLTK data is available
        stage2_index_building.ensure_nltk_data()

        print(f"✅ EverMemOS Adapter initialized")
        print(f"   LLM Model: {llm_config.get('model')}")
        print(f"   Output Dir: {self.output_dir}")

    @staticmethod
    def _extract_conv_index(conversation_id: str) -> str:
        """
        Extract numeric index part from conversation_id.

        Examples:
        - "locomo_0" -> "0"
        - "personamem_42" -> "42"
        - "123" -> "123"
        - "test_abc_5" -> "5"

        Strategy: Take the part after the last underscore, or return original if no underscore
        """
        if "_" in conversation_id:
            return conversation_id.split("_")[-1]
        return conversation_id

    def _check_missing_indexes(
        self, index_dir: Path, num_conv: int, index_type: str = "bm25"
    ) -> List[int]:
        """
        Check for missing index files.

        Args:
            index_dir: Index directory
            num_conv: Total number of conversations
            index_type: Index type ("bm25" or "embedding")

        Returns:
            List of conversation indices with missing indexes
        """
        missing_indexes = []

        for i in range(num_conv):
            if index_type == "bm25":
                index_file = index_dir / f"bm25_index_conv_{i}.pkl"
            else:  # embedding
                index_file = index_dir / f"embedding_index_conv_{i}.pkl"

            if not index_file.exists():
                missing_indexes.append(i)

        return missing_indexes

    async def add(
        self,
        conversations: List[Conversation],
        output_dir: Path = None,
        checkpoint_manager=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Add stage: Extract MemCells and build indexes.

        Call flow:
        1. Stage 1: Extract MemCells (stage1_memcells_extraction.py) - concurrent processing
        2. Stage 2: Build BM25 and Embedding indexes (stage2_index_building.py)

        Returns: Index metadata (Plan A: lazy loading)
        """
        output_dir = Path(output_dir) if output_dir else self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        memcells_dir = output_dir / "memcells"
        memcells_dir.mkdir(parents=True, exist_ok=True)
        bm25_index_dir = output_dir / "bm25_index"
        emb_index_dir = output_dir / "vectors"
        bm25_index_dir.mkdir(parents=True, exist_ok=True)
        emb_index_dir.mkdir(parents=True, exist_ok=True)

        console = Console()

        # ========== Stage 1: MemCell Extraction (concurrent processing) ==========
        console.print(f"\n{'='*60}", style="bold cyan")
        console.print(f"Stage 1: MemCell Extraction", style="bold cyan")
        console.print(f"{'='*60}", style="bold cyan")

        # Convert data format: evaluation framework → EverMemOS
        raw_data_dict = {}
        for conv in conversations:
            conv_id = conv.conversation_id
            raw_data = []

            for idx, msg in enumerate(conv.messages):
                # Handle timestamp: if None, use index-based pseudo timestamp
                if msg.timestamp is not None:
                    timestamp_str = to_iso_format(msg.timestamp)
                else:
                    # Generate pseudo timestamp using message index (maintain relative order)
                    # Base time: 2023-01-01 00:00:00, 30 seconds interval per message
                    from datetime import datetime, timedelta

                    base_time = datetime(2023, 1, 1, 0, 0, 0)
                    pseudo_time = base_time + timedelta(seconds=idx * 30)
                    timestamp_str = to_iso_format(pseudo_time)

                message_dict = {
                    "speaker_id": msg.speaker_id,
                    "user_name": msg.speaker_name or msg.speaker_id,
                    "speaker_name": msg.speaker_name or msg.speaker_id,
                    "content": msg.content,
                    "timestamp": timestamp_str,
                }

                # Add optional fields
                for optional_field in ["img_url", "blip_caption", "query"]:
                    if (
                        optional_field in msg.metadata
                        and msg.metadata[optional_field] is not None
                    ):
                        message_dict[optional_field] = msg.metadata[optional_field]

                raw_data.append(message_dict)

            raw_data_dict[conv_id] = raw_data

        # Check completed conversations (checkpoint resume)
        # Use extracted index to check files (stage1 saves using extracted index)
        completed_convs = set()
        if checkpoint_manager:
            all_conv_indices = [
                self._extract_conv_index(conv.conversation_id) for conv in conversations
            ]
            completed_indices = checkpoint_manager.load_add_progress(
                memcells_dir, all_conv_indices
            )
            # Map completed indices back to original conversation_id
            for conv in conversations:
                if self._extract_conv_index(conv.conversation_id) in completed_indices:
                    completed_convs.add(conv.conversation_id)

        # Filter conversations to process
        pending_conversations = [
            conv
            for conv in conversations
            if conv.conversation_id not in completed_convs
        ]

        console.print(
            f"\n📊 Total conversations: {len(conversations)}", style="bold cyan"
        )
        console.print(f"✅ Completed: {len(completed_convs)}", style="bold green")
        console.print(f"⏳ Pending: {len(pending_conversations)}", style="bold yellow")

        if len(pending_conversations) == 0:
            console.print(
                f"\n🎉 All conversations completed, skipping MemCell extraction!",
                style="bold green",
            )
        else:
            total_messages = sum(
                len(raw_data_dict[c.conversation_id]) for c in pending_conversations
            )
            console.print(f"📝 Pending messages: {total_messages}", style="bold blue")
            console.print(f"🚀 Starting concurrent processing...\n", style="bold green")

            # Use Rich progress bar for concurrent processing
            start_time = time.time()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TaskProgressColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                TextColumn("•"),
                TextColumn("[bold blue]{task.fields[status]}"),
                console=console,
                transient=False,
                refresh_per_second=1,
            ) as progress:
                # Create main progress task
                main_task = progress.add_task(
                    "[bold cyan]🎯 Overall Progress",
                    total=len(conversations),
                    completed=len(completed_convs),
                    status="Processing",
                )

                # Create progress bars for completed conversations (show as complete)
                conversation_tasks = {}
                for conv_id in completed_convs:
                    conv_index = self._extract_conv_index(conv_id)
                    conv_task_id = progress.add_task(
                        f"[green]Conv-{conv_index}",
                        total=len(raw_data_dict.get(conv_id, [])),
                        completed=len(raw_data_dict.get(conv_id, [])),
                        status="✅ (Skipped)",
                    )
                    conversation_tasks[conv_id] = conv_task_id

                # Create progress bars and tasks for pending conversations
                processing_tasks = []
                for conv in pending_conversations:
                    conv_id = conv.conversation_id
                    conv_index = self._extract_conv_index(
                        conv_id
                    )  # Extract numeric index
                    conv_task_id = progress.add_task(
                        f"[yellow]Conv-{conv_index}",
                        total=len(raw_data_dict[conv_id]),
                        completed=0,
                        status="Waiting",
                    )
                    conversation_tasks[conv_id] = conv_task_id

                    # Create processing task, pass extracted index
                    task = stage1_memcells_extraction.process_single_conversation(
                        conv_id=conv_index,  # Use extracted index
                        conversation=raw_data_dict[conv_id],  # Data uses original ID
                        save_dir=str(memcells_dir),
                        llm_provider=self.llm_provider,
                        event_log_extractor=self.event_log_extractor,
                        progress_counter=None,
                        progress=progress,
                        task_id=conv_task_id,
                        config=self._convert_config_to_experiment_config(),
                    )
                    processing_tasks.append((conv_id, task))

                # Define completion update function
                async def run_with_completion(conv_id, task):
                    result = await task
                    progress.update(
                        conversation_tasks[conv_id],
                        status="✅",
                        completed=progress.tasks[conversation_tasks[conv_id]].total,
                    )
                    progress.update(main_task, advance=1)
                    return result

                # Execute all pending tasks concurrently
                if processing_tasks:
                    results = await asyncio.gather(
                        *[
                            run_with_completion(conv_id, task)
                            for conv_id, task in processing_tasks
                        ]
                    )
                else:
                    results = []

                progress.update(main_task, status="✅ Complete")

            end_time = time.time()
            elapsed = end_time - start_time

            # Statistics
            successful_convs = sum(1 for _, memcell_list in results if memcell_list)
            total_memcells = sum(len(memcell_list) for _, memcell_list in results)

            console.print("\n" + "=" * 60, style="dim")
            console.print("📊 MemCell Extraction Statistics:", style="bold")
            console.print(
                f"   ✅ Successfully processed: {successful_convs}/{len(pending_conversations)}",
                style="green",
            )
            console.print(f"   📝 Total memcells: {total_memcells}", style="blue")
            console.print(f"   ⏱️  Total time: {elapsed:.2f}s", style="yellow")
            if len(pending_conversations) > 0:
                console.print(
                    f"   🚀 Average per conversation: {elapsed/len(pending_conversations):.2f}s",
                    style="cyan",
                )
            console.print("=" * 60, style="dim")

        # ========== Stage 2: Index Building ==========
        console.print(f"\n{'='*60}", style="bold cyan")
        console.print(f"Stage 2: Index Building", style="bold cyan")
        console.print(f"{'='*60}", style="bold cyan")

        # Call stage2 implementation to build indexes
        exp_config = self._convert_config_to_experiment_config()
        exp_config.num_conv = len(conversations)  # Set conversation count

        # Smart skip logic: check existing index files
        bm25_need_build = self._check_missing_indexes(
            index_dir=bm25_index_dir, num_conv=len(conversations), index_type="bm25"
        )

        emb_need_build = []
        use_hybrid = self.config.get("search", {}).get("use_hybrid_search", True)
        if use_hybrid:
            emb_need_build = self._check_missing_indexes(
                index_dir=emb_index_dir,
                num_conv=len(conversations),
                index_type="embedding",
            )

        # Statistics
        total_convs = len(conversations)
        bm25_to_build = len(bm25_need_build)
        emb_to_build = len(emb_need_build) if use_hybrid else 0

        console.print(f"\n📊 Index Building Statistics:")
        console.print(f"   Total conversations: {total_convs}")
        console.print(
            f"   BM25 index: need to build {bm25_to_build}, existing {total_convs - bm25_to_build}"
        )
        if use_hybrid:
            console.print(
                f"   Embedding index: need to build {emb_to_build}, existing {total_convs - emb_to_build}"
            )

        # Build BM25 index
        if bm25_to_build > 0:
            console.print(
                f"\n🔨 Building BM25 index ({bm25_to_build} conversations)...",
                style="yellow",
            )
            stage2_index_building.build_bm25_index(
                config=exp_config, data_dir=memcells_dir, bm25_save_dir=bm25_index_dir
            )
            console.print("✅ BM25 index building completed", style="green")
        else:
            console.print("✅ All BM25 indexes exist, skipping build", style="green")

        # Build Embedding index (if enabled)
        if use_hybrid:
            if emb_to_build > 0:
                console.print(
                    f"\n🔨 Building Embedding index ({emb_to_build} conversations)...",
                    style="yellow",
                )
                await stage2_index_building.build_emb_index(
                    config=exp_config, data_dir=memcells_dir, emb_save_dir=emb_index_dir
                )
                console.print("✅ Embedding index building completed", style="green")
            else:
                console.print(
                    "✅ All Embedding indexes exist, skipping build", style="green"
                )

        # ========== Plan A: Return index metadata (lazy loading) ==========
        # Don't load indexes into memory, only return paths and metadata
        index_metadata = {
            "type": "lazy_load",  # Mark as lazy loading
            "memcells_dir": str(memcells_dir),
            "bm25_index_dir": str(bm25_index_dir),
            "emb_index_dir": str(emb_index_dir),
            "conversation_ids": [conv.conversation_id for conv in conversations],
            "use_hybrid_search": use_hybrid,
            "total_conversations": len(conversations),
        }

        console.print(f"\n{'='*60}", style="dim")
        console.print(f"✅ Add stage completed", style="bold green")
        console.print(f"   📁 MemCells: {memcells_dir}", style="dim")
        console.print(f"   📁 BM25 index: {bm25_index_dir}", style="dim")
        if use_hybrid:
            console.print(f"   📁 Embedding index: {emb_index_dir}", style="dim")
        console.print(
            f"   💡 Using lazy loading strategy (memory-friendly)", style="cyan"
        )
        console.print(f"{'='*60}\n", style="dim")

        return index_metadata

    async def search(
        self, query: str, conversation_id: str, index: Any, **kwargs
    ) -> SearchResult:
        """
        Search stage: Retrieve relevant MemCells.

        Lazy loading: Load indexes from files on demand (memory-friendly).
        """
        # Lazy loading - read indexes from files
        bm25_index_dir = Path(index["bm25_index_dir"])
        emb_index_dir = Path(index["emb_index_dir"])

        # Extract numeric index from conversation_id to find index files
        # Example: conversation_id = "locomo_0" -> conv_index = "0"
        conv_index = self._extract_conv_index(conversation_id)

        # Load BM25 index on demand (using numeric index)
        bm25_file = bm25_index_dir / f"bm25_index_conv_{conv_index}.pkl"
        if not bm25_file.exists():
            return SearchResult(
                query=query,
                conversation_id=conversation_id,
                results=[],
                retrieval_metadata={"error": f"BM25 index not found: {bm25_file.name}"},
            )

        with open(bm25_file, "rb") as f:
            bm25_index_data = pickle.load(f)

        bm25 = bm25_index_data.get("bm25")
        docs = bm25_index_data.get("docs")

        # Load Embedding index on demand (using numeric index)
        emb_index = None
        if index.get("use_hybrid_search"):
            emb_file = emb_index_dir / f"embedding_index_conv_{conv_index}.pkl"
            if emb_file.exists():
                with open(emb_file, "rb") as f:
                    emb_index = pickle.load(f)

        # Call stage3 retrieval implementation
        search_config = self.config.get("search", {})
        retrieval_mode = search_config.get("mode", "agentic")

        exp_config = self._convert_config_to_experiment_config()
        # Get correct format llm_config from exp_config
        llm_config = exp_config.llm_config.get(exp_config.llm_service, {})

        if retrieval_mode == "agentic":
            # Agentic retrieval
            top_results, metadata = await stage3_memory_retrivel.agentic_retrieval(
                query=query,
                config=exp_config,
                llm_provider=self.llm_provider,
                llm_config=llm_config,
                emb_index=emb_index,
                bm25=bm25,
                docs=docs,
            )
        elif retrieval_mode == "lightweight":
            # Lightweight retrieval
            top_results, metadata = await stage3_memory_retrivel.lightweight_retrieval(
                query=query,
                emb_index=emb_index,
                bm25=bm25,
                docs=docs,
                config=exp_config,
            )
        else:
            # Default to hybrid retrieval
            top_results = await stage3_memory_retrivel.hybrid_search_with_rrf(
                query=query,
                emb_index=emb_index,
                bm25=bm25,
                docs=docs,
                top_n=20,
                emb_candidates=search_config.get("hybrid_emb_candidates", 100),
                bm25_candidates=search_config.get("hybrid_bm25_candidates", 100),
                rrf_k=search_config.get("hybrid_rrf_k", 60),
            )
            metadata = {}

        # Get response_top_k from config (use early for consistency)
        response_top_k = exp_config.response_top_k

        # Convert to evaluation framework format (use response_top_k to be consistent with formatted_context)
        results = []
        for doc, score in top_results[:response_top_k]:
            results.append(
                {
                    "content": doc.get("episode", ""),
                    "score": float(score),
                    "metadata": {
                        "subject": doc.get("subject", ""),
                        "summary": doc.get("summary", ""),
                    },
                }
            )

        # Build formatted_context
        formatted_context = ""
        conversation = kwargs.get("conversation")
        if conversation and top_results:
            # Get speaker information
            speaker_a = conversation.metadata.get("speaker_a", "Speaker A")
            speaker_b = conversation.metadata.get("speaker_b", "Speaker B")

            # Build context using response_top_k
            retrieved_docs_text = []
            for doc, score in top_results[:response_top_k]:
                subject = doc.get('subject', 'N/A')
                episode = doc.get('episode', 'N/A')
                doc_text = f"{subject}: {episode}\n---"
                retrieved_docs_text.append(doc_text)

            speaker_memories = "\n\n".join(retrieved_docs_text)

            TEMPLATE = """Episodes memories for conversation between {speaker_1} and {speaker_2}:

    {speaker_memories}
"""
            formatted_context = TEMPLATE.format(
                speaker_1=speaker_a,
                speaker_2=speaker_b,
                speaker_memories=speaker_memories,
            )

        # Add formatted_context to metadata
        metadata["formatted_context"] = formatted_context

        return SearchResult(
            query=query,
            conversation_id=conversation_id,
            results=results,
            retrieval_metadata=metadata,
        )

    async def answer(self, query: str, context: str, **kwargs) -> str:
        """
        Answer stage: Generate answer.

        Calls stage4_response.py implementation.
        """
        # Call stage4 answer generation implementation
        exp_config = self._convert_config_to_experiment_config()

        answer = await stage4_response.locomo_response(
            llm_provider=self.llm_provider,
            context=context,
            question=query,
            experiment_config=exp_config,
        )

        return answer

    def get_system_info(self) -> Dict[str, Any]:
        """Return system info."""
        return {
            "name": "EverMemOS",
            "version": "1.0",
            "description": "EverMemOS memory system with agentic retrieval",
            "adapter": "Adapter connecting framework to EverMemOS implementation",
        }

    def _convert_config_to_experiment_config(self):
        """
        Convert evaluation framework config to ExperimentConfig format.
        """
        from evaluation.src.adapters.evermemos.config import ExperimentConfig
        import os

        exp_config = ExperimentConfig()

        # Map LLM configuration: convert YAML llm to ExperimentConfig llm_config format
        llm_cfg = self.config.get("llm", {})
        provider = llm_cfg.get("provider", "openai")

        exp_config.llm_service = provider
        exp_config.llm_config = {
            provider: {
                "llm_provider": provider,
                "model": llm_cfg.get("model", "gpt-4o-mini"),
                "api_key": llm_cfg.get("api_key") or os.getenv("LLM_API_KEY", ""),
                "base_url": llm_cfg.get("base_url")
                or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
                "temperature": llm_cfg.get("temperature", 0.3),
                "max_tokens": llm_cfg.get("max_tokens", 32768),
            }
        }

        # Map Add stage configuration (only override explicitly specified in YAML)
        add_config = self.config.get("add", {})
        if "enable_foresight_extraction" in add_config:
            exp_config.enable_foresight_extraction = add_config[
                "enable_foresight_extraction"
            ]
        if "enable_clustering" in add_config:
            exp_config.enable_clustering = add_config["enable_clustering"]
        if "enable_profile_extraction" in add_config:
            exp_config.enable_profile_extraction = add_config[
                "enable_profile_extraction"
            ]

        # Map Search stage configuration (only override explicitly specified in YAML)
        search_config = self.config.get("search", {})
        if "mode" in search_config:
            exp_config.retrieval_mode = search_config["mode"]
            exp_config.use_agentic_retrieval = exp_config.retrieval_mode == "agentic"

        # Map lightweight_search_mode (controls search method in lightweight mode)
        # Options: "bm25_only" | "hybrid" | "emb_only"
        if "lightweight_search_mode" in search_config:
            exp_config.lightweight_search_mode = search_config[
                "lightweight_search_mode"
            ]

        return exp_config

    def build_lazy_index(
        self, conversations: List[Conversation], output_dir: Any
    ) -> Dict[str, Any]:
        """
        Build EverMemOS lazy-load index metadata.

        EverMemOS specifics:
        - Local indexes (memcells, bm25, embeddings)
        - Lazy loading (only save metadata, don't load actual index files)

        Args:
            conversations: Conversation list
            output_dir: Output directory

        Returns:
            Index metadata dict
        """
        return {
            "type": "lazy_load",
            "memcells_dir": str(output_dir / "memcells"),
            "bm25_index_dir": str(output_dir / "bm25_index"),
            "emb_index_dir": str(output_dir / "vectors"),
            "conversation_ids": [conv.conversation_id for conv in conversations],
            "use_hybrid_search": True,
            "total_conversations": len(conversations),
        }
