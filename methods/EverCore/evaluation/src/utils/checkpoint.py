"""
Checkpoint management module - supports resume from interruption.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Set
from datetime import datetime

from common_utils.datetime_utils import get_now_with_timezone


class CheckpointManager:
    """
    Checkpoint manager.

    Two-layer mechanism:
    1. Cross-stage: track completed stages (add/search/answer/evaluate)
    2. Within-stage: track fine-grained progress (search by session, answer by question count)
    """

    def __init__(self, output_dir: Path, run_name: str = "default"):
        """
        Initialize Checkpoint manager.

        Args:
            output_dir: Output directory
            run_name: Run name
        """
        self.output_dir = Path(output_dir)
        self.run_name = run_name

        # Cross-stage checkpoint (record which stages are completed)
        self.checkpoint_file = self.output_dir / f"checkpoint_{run_name}.json"

        # Fine-grained checkpoints (one per stage, track progress within stage)
        self.search_checkpoint = self.output_dir / f"search_results_checkpoint.json"
        self.answer_checkpoint = self.output_dir / f"answer_results_checkpoint.json"

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load existing checkpoint.

        Returns:
            Checkpoint data, or None if not exists
        """
        if not self.checkpoint_file.exists():
            return None

        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)

            print(f"\n🔄 Found checkpoint file: {self.checkpoint_file.name}")
            print(f"   Last updated: {checkpoint.get('last_updated', 'Unknown')}")
            print(
                f"   Completed stages: {', '.join(checkpoint.get('completed_stages', []))}"
            )

            if 'search_results' in checkpoint:
                completed_convs = len(checkpoint['search_results'])
                print(f"   Processed conversations: {completed_convs}")

            return checkpoint

        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")
            print(f"   Starting from scratch")
            return None

    def save_checkpoint(
        self,
        completed_stages: Set[str],
        search_results: Optional[Dict] = None,
        answer_results: Optional[Dict] = None,
        eval_results: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ):
        """
        Save checkpoint.

        Args:
            completed_stages: Set of completed stages
            search_results: Search results (optional)
            answer_results: Answer results (optional)
            eval_results: Evaluation results (optional)
            metadata: Other metadata (optional)
        """
        checkpoint = {
            "run_name": self.run_name,
            "last_updated": get_now_with_timezone().isoformat(),
            "completed_stages": list(completed_stages),
        }

        if search_results is not None:
            checkpoint["search_results"] = search_results

        if answer_results is not None:
            checkpoint["answer_results"] = answer_results

        if eval_results is not None:
            checkpoint["eval_results"] = eval_results

        if metadata is not None:
            checkpoint["metadata"] = metadata

        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)

            print(f"💾 Checkpoint saved: {self.checkpoint_file.name}")

        except Exception as e:
            print(f"⚠️ Failed to save checkpoint: {e}")

    def get_completed_conversations(self) -> Set[str]:
        """
        Get set of completed conversation IDs.

        Returns:
            Set of completed conversation IDs
        """
        checkpoint = self.load_checkpoint()
        if not checkpoint:
            return set()

        completed = set()

        # Get from search_results
        if 'search_results' in checkpoint:
            completed.update(checkpoint['search_results'].keys())

        return completed

    def should_skip_stage(self, stage: str) -> bool:
        """
        Check whether a stage should be skipped.

        Args:
            stage: Stage name (add, search, answer, evaluate)

        Returns:
            True if should skip
        """
        checkpoint = self.load_checkpoint()
        if not checkpoint:
            return False

        completed_stages = set(checkpoint.get('completed_stages', []))
        return stage in completed_stages

    def delete_checkpoint(self):
        """Delete checkpoint file."""
        if self.checkpoint_file.exists():
            try:
                self.checkpoint_file.unlink()
                print(f"🗑️  Checkpoint deleted: {self.checkpoint_file.name}")
            except Exception as e:
                print(f"⚠️ Failed to delete checkpoint: {e}")

    def get_search_results(self) -> Optional[Dict]:
        """Get saved search results."""
        checkpoint = self.load_checkpoint()
        if checkpoint and 'search_results' in checkpoint:
            return checkpoint['search_results']
        return None

    def get_answer_results(self) -> Optional[Dict]:
        """Get saved answer results."""
        checkpoint = self.load_checkpoint()
        if checkpoint and 'answer_results' in checkpoint:
            return checkpoint['answer_results']
        return None

    # ==================== Fine-grained Checkpoint Methods ====================

    def save_add_progress(self, completed_convs: set, memcells_dir: Path):
        """
        Save fine-grained progress for Add stage (record completed session IDs).

        Args:
            completed_convs: Set of completed session IDs
            memcells_dir: MemCells save directory (for checking file existence)
        """
        # Add stage checkpoint strategy:
        # After processing each session, save MemCells to {output_dir}/memcells/{conv_id}.json
        # No additional checkpoint file needed, just check memcells directory
        pass  # Files themselves are checkpoints

    def load_add_progress(self, memcells_dir: Path, all_conv_ids: list) -> set:
        """
        Load fine-grained progress for Add stage (check which sessions are completed).

        Returns:
            Set of completed session IDs
        """
        import json

        completed_convs = set()

        if not memcells_dir.exists():
            print(f"\n🆕 No previous memcells found, starting from scratch")
            return completed_convs

        print(f"\n🔍 Checking for completed conversations in: {memcells_dir}")

        for conv_id in all_conv_ids:
            # Match stage1 actual file name format
            output_file = memcells_dir / f"memcell_list_conv_{conv_id}.json"
            if output_file.exists():
                # Validate file (non-empty and parseable)
                try:
                    with open(output_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if data and len(data) > 0:  # Ensure has data
                            completed_convs.add(conv_id)
                            print(
                                f"✅ Skipped completed session: {conv_id} ({len(data)} memcells)"
                            )
                except Exception as e:
                    print(f"⚠️  Session {conv_id} file corrupted, will reprocess: {e}")

        if completed_convs:
            print(
                f"\n📊 Found {len(completed_convs)}/{len(all_conv_ids)} completed sessions"
            )

        return completed_convs

    def save_search_progress(self, search_results: Dict[str, Any]):
        """
        Save fine-grained progress for Search stage (save after each session).

        Args:
            search_results: Current accumulated search results
                Format: {conv_id: [{"question_id": ..., "results": ...}, ...], ...}
        """
        try:
            with open(self.search_checkpoint, 'w', encoding='utf-8') as f:
                json.dump(search_results, f, indent=2, ensure_ascii=False)

            print(f"💾 Checkpoint saved: {len(search_results)} conversations")

        except Exception as e:
            print(f"⚠️  Failed to save search checkpoint: {e}")

    def load_search_progress(self) -> Dict[str, Any]:
        """
        Load fine-grained progress for Search stage.

        Returns:
            Saved search results, or empty dict if not exists
        """
        if not self.search_checkpoint.exists():
            print(f"\n🆕 No checkpoint found, starting from scratch")
            return {}

        try:
            print(f"\n🔄 Found checkpoint file: {self.search_checkpoint}")
            with open(self.search_checkpoint, 'r', encoding='utf-8') as f:
                search_results = json.load(f)

            print(f"✅ Loaded {len(search_results)} conversations from checkpoint")
            print(f"   Already processed: {sorted(search_results.keys())}")

            return search_results

        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
            print(f"   Starting from scratch...")
            return {}

    def delete_search_checkpoint(self):
        """Delete fine-grained checkpoint for Search stage."""
        if self.search_checkpoint.exists():
            try:
                self.search_checkpoint.unlink()
                print(f"🗑️  Checkpoint file removed (task completed)")
            except Exception as e:
                print(f"⚠️  Failed to remove checkpoint: {e}")

    def save_answer_progress(
        self, answer_results: Dict[str, Any], completed: int, total: int
    ):
        """
        Save fine-grained progress for Answer stage (save every SAVE_INTERVAL questions).

        Args:
            answer_results: Current accumulated answer results
            completed: Number of completed questions
            total: Total number of questions
        """
        try:
            checkpoint_path = self.output_dir / f"responses_checkpoint_{completed}.json"
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(answer_results, f, indent=2, ensure_ascii=False)

            print(f"  💾 Checkpoint saved: {checkpoint_path.name}")

        except Exception as e:
            print(f"⚠️  Failed to save answer checkpoint: {e}")

    def load_answer_progress(self) -> Dict[str, Any]:
        """
        Load fine-grained progress for Answer stage (find latest checkpoint file).

        Returns:
            Saved answer results, or empty dict if not exists
        """
        # Find all responses_checkpoint_*.json files
        checkpoint_files = list(self.output_dir.glob("responses_checkpoint_*.json"))

        if not checkpoint_files:
            print(f"\n🆕 No answer checkpoint found, starting from scratch")
            return {}

        # Find latest checkpoint file (sort by number in filename)
        try:
            latest_checkpoint = max(
                checkpoint_files, key=lambda p: int(p.stem.split('_')[-1])
            )

            print(f"\n🔄 Found checkpoint file: {latest_checkpoint.name}")
            with open(latest_checkpoint, 'r', encoding='utf-8') as f:
                answer_results = json.load(f)

            print(f"✅ Loaded {len(answer_results)} answers from checkpoint")

            return answer_results

        except Exception as e:
            print(f"⚠️  Failed to load answer checkpoint: {e}")
            print(f"   Starting from scratch...")
            return {}

    def delete_answer_checkpoints(self):
        """Delete all fine-grained checkpoints for Answer stage."""
        checkpoint_files = list(self.output_dir.glob("responses_checkpoint_*.json"))

        for checkpoint_file in checkpoint_files:
            try:
                checkpoint_file.unlink()
                print(f"  🗑️  Removed checkpoint: {checkpoint_file.name}")
            except Exception as e:
                print(f"⚠️  Failed to remove checkpoint {checkpoint_file.name}: {e}")
