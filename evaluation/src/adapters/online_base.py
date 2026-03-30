"""
Online API Adapter base class.

Provides common functionality for all online memory system APIs (Mem0, Memos, Memu, etc.).
All online API adapters can inherit from this class.

Design principles:
- Provide default answer() implementation (using generic prompt)
- Subclasses can override answer() to use their own specific prompts
- Provide helper methods for data format conversion
"""

import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, List, Dict, Optional

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from evaluation.src.adapters.base import BaseAdapter
from evaluation.src.core.data_models import Conversation, SearchResult
from evaluation.src.utils.config import load_yaml

# Import Memory Layer components
from memory_layer.llm.llm_provider import LLMProvider


class OnlineAPIAdapter(BaseAdapter):
    """
    Online API Adapter base class.

    Provides common functionality:
    1. LLM Provider initialization
    2. Answer generation (reuses EverMemOS implementation)
    3. Standard format conversion helper methods

    Subclasses only need to implement:
    - add(): Call online API to ingest data
    - search(): Call online API for retrieval
    """

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config)
        self.output_dir = Path(output_dir) if output_dir else Path(".")

        # Initialize LLM Provider (for answer generation)
        llm_config = config.get("llm", {})

        self.llm_provider = LLMProvider(
            provider_type=llm_config.get("provider", "openai"),
            model=llm_config.get("model", "gpt-4o-mini"),
            api_key=llm_config.get("api_key", ""),
            base_url=llm_config.get("base_url", "https://api.openai.com/v1"),
            temperature=llm_config.get("temperature", 0.3),
            max_tokens=llm_config.get("max_tokens", 32768),
        )

        # Load prompts (from YAML file)
        evaluation_root = Path(__file__).parent.parent.parent
        prompts_path = evaluation_root / "config" / "prompts.yaml"
        self._prompts = load_yaml(str(prompts_path))

        # Set num_workers (conversation-level concurrency)
        # Can be overridden by subclass or config
        self.num_workers = self._get_num_workers(config)

        print(f"✅ {self.__class__.__name__} initialized")
        print(f"   LLM Model: {llm_config.get('model')}")
        print(f"   Output Dir: {self.output_dir}")
        print(f"   Num Workers: {self.num_workers}")

    def _get_num_workers(self, config: dict) -> int:
        """
        Get num_workers from config.

        Args:
            config: Configuration dict (should contain num_workers)

        Returns:
            Number of workers for conversation-level concurrency
        """
        return config.get("num_workers", 10)  # Default to 10 if not specified

    async def add(self, conversations: List[Conversation], **kwargs) -> Dict[str, Any]:
        """
        Ingest conversation data (call online API) with concurrency control.

        Template method that implements the common add flow:
        1. Determine perspective (single or dual)
        2. Organize messages for each user
        3. Call subclass _add_user_messages for each user (with concurrency control)
        4. Post-processing (e.g., wait for tasks)

        Concurrency is controlled by self.num_workers (conversation-level).

        Subclasses can override this method for custom behavior,
        or implement _add_user_messages for standard flow.
        """
        import asyncio

        conversation_ids = []
        add_results = []

        console = Console()
        console.print(f"\n{'='*60}", style="bold cyan")
        console.print("Stage 1: Add", style="bold cyan")
        console.print(f"{'='*60}", style="bold cyan")

        def _conv_label(conv_id: str) -> str:
            parts = conv_id.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                return parts[1]
            return conv_id

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.num_workers)

        async def process_single_conversation(conv, progress, main_task):
            """Process a single conversation with concurrency control."""
            async with semaphore:
                conv_id = conv.conversation_id

                # Extract conversation info (speaker names, user_ids, perspective mode)
                conv_info = self._extract_conversation_info(
                    conversation=conv, conversation_id=conv_id
                )

                # Get format type (subclass can override)
                format_type = self._get_format_type()

                # Organize messages based on perspective
                if conv_info["need_dual_perspective"]:
                    # Dual perspective: prepare messages for both speakers
                    speaker_a_messages = self._conversation_to_messages(
                        conv, format_type=format_type, perspective="speaker_a"
                    )
                    speaker_b_messages = self._conversation_to_messages(
                        conv, format_type=format_type, perspective="speaker_b"
                    )
                    total_messages = len(speaker_a_messages) + len(speaker_b_messages)
                    conv_task_id = progress.add_task(
                        f"[yellow]Conv-{_conv_label(conv_id)}",
                        total=total_messages,
                        completed=0,
                        status="Processing",
                    )

                    # Add messages for both users
                    result_a = await self._add_user_messages(
                        conv,
                        speaker_a_messages,
                        speaker="speaker_a",
                        progress=progress,
                        task_id=conv_task_id,
                        **kwargs,
                    )
                    result_b = await self._add_user_messages(
                        conv,
                        speaker_b_messages,
                        speaker="speaker_b",
                        progress=progress,
                        task_id=conv_task_id,
                        **kwargs,
                    )

                    # Wait for tasks to complete (per-conversation, before releasing semaphore)
                    # This is important for systems like Memu that need to limit concurrent tasks
                    await self._wait_for_conversation_tasks(
                        [result_a, result_b], conversation_id=conv_id, **kwargs
                    )

                    progress.update(
                        conv_task_id, completed=total_messages, status="✅"
                    )
                    progress.update(main_task, advance=1)
                    return conv_id, [result_a, result_b]
                else:
                    # Single perspective: prepare messages for speaker_a only
                    messages = self._conversation_to_messages(
                        conv, format_type=format_type, perspective=None
                    )
                    total_messages = len(messages)
                    conv_task_id = progress.add_task(
                        f"[yellow]Conv-{_conv_label(conv_id)}",
                        total=total_messages,
                        completed=0,
                        status="Processing",
                    )

                    # Add messages for single user
                    result = await self._add_user_messages(
                        conv,
                        messages,
                        speaker="speaker_a",
                        progress=progress,
                        task_id=conv_task_id,
                        **kwargs,
                    )

                    # Wait for tasks to complete (per-conversation, before releasing semaphore)
                    await self._wait_for_conversation_tasks(
                        [result], conversation_id=conv_id, **kwargs
                    )

                    progress.update(
                        conv_task_id, completed=total_messages, status="✅"
                    )
                    progress.update(main_task, advance=1)
                    return conv_id, [result]

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
            main_task = progress.add_task(
                "[bold cyan]🎯 Overall Progress",
                total=len(conversations),
                completed=0,
                status="Processing",
            )

            # Process all conversations concurrently (with semaphore control)
            tasks = [
                process_single_conversation(conv, progress, main_task)
                for conv in conversations
            ]
            results = await asyncio.gather(*tasks)
            progress.update(main_task, status="✅ Complete")

        # Collect results
        for conv_id, conv_results in results:
            conversation_ids.append(conv_id)
            add_results.extend(conv_results)

        # Post-processing (e.g., wait for async tasks)
        await self._post_add_process(add_results, **kwargs)

        # Build and return result
        return self._build_add_result(conversation_ids, add_results, **kwargs)

    @abstractmethod
    async def _add_user_messages(
        self, conv: Conversation, messages: List[Dict[str, Any]], speaker: str, **kwargs
    ) -> Any:
        """
        Add messages for a single user (subclass implementation).

        Args:
            conv: Original conversation object (for extracting extra info)
            messages: Formatted message list (ready to send)
            speaker: "speaker_a" or "speaker_b"
            **kwargs: Extra parameters (may include user_id, timestamp, etc.)

        Returns:
            Subclass-specific result (e.g., task_id for Memu, None for others)
        """
        pass

    async def _wait_for_conversation_tasks(
        self, task_results: List[Any], **kwargs
    ) -> None:
        """
        Wait for tasks from a single conversation to complete (per-conversation hook).

        This is called BEFORE releasing the semaphore, ensuring that systems like Memu
        which create async tasks don't exceed their concurrency limits.

        For systems that complete work immediately (Mem0, Memos), this is a no-op.
        For systems with async tasks (Memu), override this to wait for task completion.

        Args:
            task_results: Results from _add_user_messages for this conversation
            **kwargs: Extra parameters
        """
        # Default: no-op (most systems don't need per-conversation waiting)
        pass

    async def search(
        self, query: str, conversation_id: str, index: Any, **kwargs
    ) -> SearchResult:
        """
        Retrieve relevant memories (call online API).

        Template method that orchestrates the search process:
        1. Extract conversation info (determine perspective)
        2. Call single or dual perspective search
        3. Subclasses implement actual API calls and result building

        Args:
            query: Query text
            conversation_id: Conversation ID
            index: Index metadata (contains conversation_ids)
            **kwargs: Optional parameters (top_k, conversation, etc.)

        Returns:
            SearchResult with standard format
        """
        # Extract conversation information (speakers, user_ids, dual perspective)
        conv_info = self._extract_conversation_info(
            conversation_id=conversation_id, **kwargs
        )

        # Get top_k from kwargs, or fallback to config, or default to 10
        default_top_k = self.config.get("search", {}).get("top_k", 10)
        top_k = kwargs.get("top_k", default_top_k)

        if conv_info["need_dual_perspective"]:
            # Dual perspective: search from both speakers' perspectives
            return await self._search_dual_perspective(
                query=query,
                conversation_id=conversation_id,
                speaker_a=conv_info["speaker_a"],
                speaker_b=conv_info["speaker_b"],
                speaker_a_user_id=conv_info["speaker_a_user_id"],
                speaker_b_user_id=conv_info["speaker_b_user_id"],
                top_k=top_k,
                **kwargs,
            )
        else:
            # Single perspective: search from one user's perspective
            return await self._search_single_perspective(
                query=query,
                conversation_id=conversation_id,
                user_id=conv_info["speaker_a_user_id"],
                top_k=top_k,
                **kwargs,
            )

    async def _search_single_perspective(
        self, query: str, conversation_id: str, user_id: str, top_k: int, **kwargs
    ) -> SearchResult:
        """
        Single perspective search flow (base class implementation).

        Subclasses should NOT override this unless necessary.
        Instead, implement _search_single_user and _build_single_search_result.

        Args:
            query: Query text
            conversation_id: Conversation ID
            user_id: User ID to search for
            top_k: Number of results to retrieve
            **kwargs: Additional parameters

        Returns:
            SearchResult
        """
        # Call subclass to perform search (API call + conversion + special processing)
        results = await self._search_single_user(
            query, conversation_id, user_id, top_k, **kwargs
        )

        # Call subclass to build SearchResult (including formatted_context)
        return self._build_single_search_result(
            query=query,
            conversation_id=conversation_id,
            results=results,
            user_id=user_id,
            top_k=top_k,
            **kwargs,
        )

    async def _search_dual_perspective(
        self,
        query: str,
        conversation_id: str,
        speaker_a: str,
        speaker_b: str,
        speaker_a_user_id: str,
        speaker_b_user_id: str,
        top_k: int,
        **kwargs,
    ) -> SearchResult:
        """
        Dual perspective search flow (base class implementation).

        Subclasses should NOT override this unless necessary.
        Instead, implement _search_single_user and _build_dual_search_result.

        Args:
            query: Query text
            conversation_id: Conversation ID
            speaker_a: Speaker A name
            speaker_b: Speaker B name
            speaker_a_user_id: Speaker A user ID
            speaker_b_user_id: Speaker B user ID
            top_k: Number of results per user
            **kwargs: Additional parameters

        Returns:
            SearchResult
        """
        # Search both users separately
        results_a = await self._search_single_user(
            query, conversation_id, speaker_a_user_id, top_k, **kwargs
        )
        results_b = await self._search_single_user(
            query, conversation_id, speaker_b_user_id, top_k, **kwargs
        )

        # Merge results (for fallback, not re-sorted)
        all_results = results_a + results_b

        # Call subclass to build SearchResult (including formatted_context)
        return self._build_dual_search_result(
            query=query,
            conversation_id=conversation_id,
            all_results=all_results,
            results_a=results_a,
            results_b=results_b,
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            speaker_a_user_id=speaker_a_user_id,
            speaker_b_user_id=speaker_b_user_id,
            top_k=top_k,
            **kwargs,
        )

    @abstractmethod
    async def _search_single_user(
        self, query: str, conversation_id: str, user_id: str, top_k: int, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search memories for a single user (subclass must implement).

        This method should:
        1. Call the system's search API
        2. Convert raw results to standard format
        3. Apply system-specific processing (e.g., timezone, preference, summary)

        Standard result format:
        [
            {
                "content": str,      # Display content (may include timestamp, etc.)
                "score": float,      # Relevance score
                "user_id": str,      # User ID
                "metadata": dict     # System-specific metadata
            },
            ...
        ]

        System-specific processing:
        - Mem0: Apply timezone conversion to timestamps
        - Memos: Extract and include preference information
        - Memu: Fetch and include categories summary

        Args:
            query: Query text
            conversation_id: Conversation ID (some systems may need it for context)
            user_id: User ID to search for
            top_k: Number of results to retrieve
            **kwargs: System-specific parameters (e.g., min_similarity)

        Returns:
            List of search results in standard format
        """
        pass

    @abstractmethod
    def _build_single_search_result(
        self,
        query: str,
        conversation_id: str,
        results: List[Dict[str, Any]],
        user_id: str,
        top_k: int,
        **kwargs,
    ) -> SearchResult:
        """
        Build SearchResult for single perspective (subclass must implement).

        This method should:
        1. Construct retrieval_metadata (system name, parameters, etc.)
        2. Build formatted_context (using template or custom logic)

        Args:
            query: Query text
            conversation_id: Conversation ID
            results: Search results from _search_single_user
            user_id: User ID
            top_k: Number of results requested
            **kwargs: Additional parameters

        Returns:
            SearchResult with formatted_context
        """
        pass

    @abstractmethod
    def _build_dual_search_result(
        self,
        query: str,
        conversation_id: str,
        all_results: List[Dict[str, Any]],
        results_a: List[Dict[str, Any]],
        results_b: List[Dict[str, Any]],
        speaker_a: str,
        speaker_b: str,
        speaker_a_user_id: str,
        speaker_b_user_id: str,
        top_k: int,
        **kwargs,
    ) -> SearchResult:
        """
        Build SearchResult for dual perspective (subclass must implement).

        This method should:
        1. Construct retrieval_metadata (system name, parameters, dual flag, etc.)
        2. Build formatted_context using both speakers' results
           - Use template or custom logic
           - Include system-specific information (preferences, summaries, etc.)

        Args:
            query: Query text
            conversation_id: Conversation ID
            all_results: Merged results (for fallback)
            results_a: Speaker A's search results
            results_b: Speaker B's search results
            speaker_a: Speaker A name
            speaker_b: Speaker B name
            speaker_a_user_id: Speaker A user ID
            speaker_b_user_id: Speaker B user ID
            top_k: Number of results per user
            **kwargs: Additional parameters

        Returns:
            SearchResult with formatted_context
        """
        pass

    async def answer(self, query: str, context: str, **kwargs) -> str:
        """
        Generate answer (using generic MEMOS prompt).

        Subclasses can override this method to use their own specific prompt.
        Defaults to ANSWER_PROMPT_MEMOS (suitable for most systems).
        """
        # Get answer prompt (subclasses can override _get_answer_prompt)
        prompt = self._get_answer_prompt().format(context=context, question=query)

        # Get retry count
        max_retries = self.config.get("answer", {}).get("max_retries", 3)

        # Generate answer
        for i in range(max_retries):
            try:
                result = await self.llm_provider.generate(prompt=prompt, temperature=0)

                # Clean result (remove possible "FINAL ANSWER:" prefix)
                if "FINAL ANSWER:" in result:
                    parts = result.split("FINAL ANSWER:")
                    if len(parts) > 1:
                        result = parts[1].strip()
                    else:
                        result = result.strip()
                else:
                    result = result.strip()

                if result == "":
                    continue

                return result
            except Exception as e:
                print(f"⚠️  Answer generation error (attempt {i+1}/{max_retries}): {e}")
                if i == max_retries - 1:
                    raise
                continue

        return ""

    def _get_answer_prompt(self) -> str:
        """
        Get answer prompt.

        Subclasses can override this method to return their own prompt.
        Defaults to generic default prompt.
        """
        return self._prompts["online_api"]["default"]["answer_prompt_memos"]

    # ===== Helper methods: format conversion =====

    def _conversation_to_messages(
        self,
        conversation: Conversation,
        format_type: str = "basic",
        perspective: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convert standard Conversation to message list.

        Args:
            conversation: Standard conversation object
            format_type: Format type (basic, mem0, memos, memu)
            perspective: Perspective (speaker_a or speaker_b), used for dual-perspective systems like Memos

        Returns:
            Message list
        """
        messages = []
        speaker_a = conversation.metadata.get("speaker_a", "")
        speaker_b = conversation.metadata.get("speaker_b", "")

        for msg in conversation.messages:
            # Intelligently determine role and content
            role, content = self._determine_role_and_content(
                msg.speaker_name, msg.content, speaker_a, speaker_b, perspective
            )

            # Base message
            message = {"role": role, "content": content}

            # Add extra fields based on different system requirements
            if format_type == "memos":
                # Memos format: needs chat_time
                # Note: Memos directly sends messages to API, so this field is used
                if msg.timestamp:
                    from common_utils.datetime_utils import to_iso_format

                    message["chat_time"] = to_iso_format(msg.timestamp)

            elif format_type == "memu":
                # Memu format: needs name and time
                message["name"] = msg.speaker_name
                message["time"] = (
                    msg.timestamp.isoformat() + "Z" if msg.timestamp else None
                )

            # Note: Mem0 extracts timestamps directly from conv.messages in _add_user_messages

            messages.append(message)

        return messages

    def _determine_role_and_content(
        self,
        speaker_name: str,
        content: str,
        speaker_a: str,
        speaker_b: str,
        perspective: Optional[str] = None,
    ) -> tuple:
        """
        Intelligently determine message role and content.

        For systems that only support user/assistant (e.g., Memos), special handling is needed:
        1. If speaker is standard role (user/assistant and variants), use directly
        2. If custom name, convert based on perspective:
           - From speaker_a perspective: speaker_a messages are "user", speaker_b are "assistant"
           - From speaker_b perspective: speaker_b messages are "user", speaker_a are "assistant"
        3. Content for custom speakers needs "speaker: " prefix

        Args:
            speaker_name: Speaker name
            content: Message content
            speaker_a: speaker_a in conversation
            speaker_b: speaker_b in conversation
            perspective: Perspective (for dual-perspective systems)

        Returns:
            (role, content) tuple
        """
        # Case 1: Standard roles (user/assistant and variants)
        speaker_lower = speaker_name.lower()

        # Check if standard role or variant
        if speaker_lower in ["user", "assistant"]:
            # Exact match: "user", "User", "assistant", "Assistant"
            return speaker_lower, content
        elif speaker_lower.startswith("user"):
            # Variants: "user_123", "User_456", etc.
            return "user", content
        elif speaker_lower.startswith("assistant"):
            # Variants: "assistant_123", "Assistant_456", etc.
            return "assistant", content

        # Case 2: Custom speaker, needs conversion
        # Default behavior: speaker_a is user, speaker_b is assistant
        if perspective == "speaker_b":
            # From speaker_b's perspective
            if speaker_name == speaker_b:
                role = "user"
            elif speaker_name == speaker_a:
                role = "assistant"
            else:
                # Unknown speaker, default to assistant
                role = "assistant"
        else:
            # From speaker_a's perspective (default)
            if speaker_name == speaker_a:
                role = "user"
            elif speaker_name == speaker_b:
                role = "assistant"
            else:
                # Unknown speaker, default to user
                role = "user"

        # For custom speakers, content needs prefix
        formatted_content = f"{speaker_name}: {content}"

        return role, formatted_content

    def _extract_user_id(
        self, conversation: Conversation, speaker: str = "speaker_a"
    ) -> str:
        """
        Extract user_id from Conversation (for online API).

        Logic: Combine conversation_id and speaker name to ensure conversation isolation.

        Args:
            conversation: Standard conversation object
            speaker: Speaker identifier (speaker_a or speaker_b)

        Returns:
            user_id string, format: {conv_id}_{speaker_name}

        Examples:
            - LoCoMo: speaker_a="Caroline" → user_id="locomo_0_Caroline"
            - PersonaMem: speaker_a="Kanoa Manu" → user_id="personamem_0_Kanoa_Manu"
            - No speaker: → user_id="locomo_0_speaker_a"

        Design rationale:
            - Include conv_id: Ensure memory isolation between conversations (evaluation accuracy)
            - Include speaker name: More intuitive for backend viewing (e.g., Caroline vs speaker_a)
            - Replace spaces with underscores: Avoid spaces in user_id
        """
        conv_id = conversation.conversation_id
        speaker_name = conversation.metadata.get(speaker)

        if speaker_name:
            # Has speaker name: replace spaces with underscores
            speaker_name_normalized = speaker_name.replace(" ", "_")
            user_id = f"{conv_id}_{speaker_name_normalized}"
        else:
            # No speaker name: locomo_0_speaker_a
            user_id = f"{conv_id}_{speaker}"

        return user_id

    def _get_user_id_from_conversation_id(self, conversation_id: str) -> str:
        """
        Derive user_id from conversation_id (simplified version).

        Args:
            conversation_id: Conversation ID

        Returns:
            user_id string
        """
        # Simplified implementation: directly use conversation_id
        # May need more complex mapping logic in actual use
        return conversation_id

    def _get_format_type(self) -> str:
        """
        Get format type for _conversation_to_messages.

        Subclasses can override this method to specify their format type.
        Default implementation infers from class name.

        Returns:
            Format type string (e.g., "mem0", "memos", "memu", "basic")
        """
        class_name = self.__class__.__name__.lower()

        # Infer format type from class name
        if "mem0" in class_name:
            return "mem0"
        elif "memos" in class_name:
            return "memos"
        elif "memu" in class_name:
            return "memu"
        else:
            return "basic"

    async def _post_add_process(self, add_results: List[Any], **kwargs) -> None:
        """
        Post-processing after adding all conversations.

        Subclasses can override this method for custom post-processing
        (e.g., Memu waiting for async tasks to complete).

        Args:
            add_results: List of results from _add_user_messages calls
            **kwargs: Extra parameters
        """
        # Default: no post-processing
        pass

    def _build_add_result(
        self, conversation_ids: List[str], add_results: List[Any], **kwargs
    ) -> Dict[str, Any]:
        """
        Build the final result dict for add method.

        Subclasses can override this method to customize the return structure.

        Args:
            conversation_ids: List of conversation IDs that were added
            add_results: List of results from _add_user_messages calls
            **kwargs: Extra parameters

        Returns:
            Result dictionary
        """
        system_name = self.__class__.__name__.replace("Adapter", "").lower()

        result = {
            "type": "online_api",
            "system": system_name,
            "conversation_ids": conversation_ids,
        }

        # If add_results contains non-None values, include them
        # (e.g., Memu's task_ids)
        non_none_results = [r for r in add_results if r is not None]
        if non_none_results:
            result["add_results"] = non_none_results

        return result

    def _batch_messages_with_retry(
        self,
        messages: List[Dict[str, Any]],
        batch_size: int,
        add_func: callable,
        max_retries: int = None,
        description: str = "Batch",
    ) -> None:
        """
        Helper method for batching messages with retry logic.

        Subclasses can use this method to simplify batch processing.

        Args:
            messages: Message list to batch
            batch_size: Batch size
            add_func: Function to call for each batch (should accept List[Dict])
            max_retries: Max retry attempts (defaults to self.max_retries)
            description: Description for logging
        """
        if max_retries is None:
            max_retries = getattr(self, 'max_retries', 3)

        for i in range(0, len(messages), batch_size):
            batch_messages = messages[i : i + batch_size]

            # Retry mechanism
            for attempt in range(max_retries):
                try:
                    add_func(batch_messages)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(
                            f"   ⚠️  [{description}] Retry {attempt + 1}/{max_retries}: {e}"
                        )
                        time.sleep(2**attempt)  # Exponential backoff
                    else:
                        print(
                            f"   ❌ [{description}] Failed after {max_retries} retries: {e}"
                        )
                        raise e

    def _need_dual_perspective(self, speaker_a: str, speaker_b: str) -> bool:
        """
        Determine if dual-perspective handling is needed.

        Single perspective (no dual-perspective needed):
        - Standard roles: "user"/"assistant"
        - Case variants: "User"/"Assistant"
        - With suffix: "user_123"/"assistant_456"

        Dual perspective (dual-perspective needed):
        - Custom names: "Elena Rodriguez"/"Alex"

        Args:
            speaker_a: Speaker A name
            speaker_b: Speaker B name

        Returns:
            True if dual perspective is needed, False otherwise
        """

        def is_standard_role(speaker: str) -> bool:
            speaker = speaker.lower()
            # Exact match
            if speaker in ["user", "assistant"]:
                return True
            # Starts with user or assistant
            if speaker.startswith("user") or speaker.startswith("assistant"):
                return True
            return False

        # Only need dual perspective when both speakers are not standard roles
        return not (is_standard_role(speaker_a) or is_standard_role(speaker_b))

    def _extract_conversation_info(
        self,
        conversation: Optional[Conversation] = None,
        conversation_id: str = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Extract conversation information.

        This helper method extracts speaker information and determines if dual
        perspective handling is needed. Used by both add and search methods.

        Args:
            conversation: Conversation object (if directly available)
            conversation_id: Conversation ID (for fallback)
            **kwargs: May contain 'conversation' key if not passed directly

        Returns:
            Dictionary with keys:
            - speaker_a: Speaker A name
            - speaker_b: Speaker B name
            - speaker_a_user_id: User ID for speaker A
            - speaker_b_user_id: User ID for speaker B
            - need_dual_perspective: Whether dual perspective is needed
        """
        # Get conversation from parameter or kwargs
        if conversation is None:
            conversation = kwargs.get("conversation")

        if conversation:
            speaker_a = conversation.metadata.get("speaker_a", "")
            speaker_b = conversation.metadata.get("speaker_b", "")
            speaker_a_user_id = self._extract_user_id(conversation, speaker="speaker_a")
            speaker_b_user_id = self._extract_user_id(conversation, speaker="speaker_b")
            need_dual_perspective = self._need_dual_perspective(speaker_a, speaker_b)
        else:
            # Fallback: use default values (for search when conversation not available)
            if conversation_id is None:
                conversation_id = "unknown"
            speaker_a_user_id = f"{conversation_id}_speaker_a"
            speaker_b_user_id = f"{conversation_id}_speaker_b"
            speaker_a = "speaker_a"
            speaker_b = "speaker_b"
            need_dual_perspective = False

        return {
            "speaker_a": speaker_a,
            "speaker_b": speaker_b,
            "speaker_a_user_id": speaker_a_user_id,
            "speaker_b_user_id": speaker_b_user_id,
            "need_dual_perspective": need_dual_perspective,
        }

    def get_system_info(self) -> Dict[str, Any]:
        """Return system info."""
        return {
            "name": self.__class__.__name__,
            "type": "online_api",
            "description": f"{self.__class__.__name__} adapter for online memory API",
        }
