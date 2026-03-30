"""In-memory cluster storage for evaluation.

This storage implementation is used by the evaluation framework for clustering
during MemCell extraction. It keeps cluster states in memory with optional
file persistence for checkpointing.
"""

from typing import Any, Dict, Optional
from pathlib import Path
import json
import numpy as np

from core.observation.logger import get_logger

logger = get_logger(__name__)


class InMemoryClusterStorage:
    """In-memory cluster storage with optional file persistence."""
    
    def __init__(
        self,
        enable_persistence: bool = False,
        persist_dir: Optional[Path] = None
    ):
        """Initialize in-memory storage.
        
        Args:
            enable_persistence: Whether to persist to disk
            persist_dir: Directory for JSON files (required if enable_persistence=True)
        """
        self._states: Dict[str, Dict[str, Any]] = {}
        self._enable_persistence = enable_persistence
        self._persist_dir = Path(persist_dir) if persist_dir else None
        
        if enable_persistence and not persist_dir:
            raise ValueError("persist_dir is required when enable_persistence=True")
        
        if enable_persistence and self._persist_dir:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()
    
    async def save_cluster_state(
        self,
        group_id: str,
        state: Dict[str, Any]
    ) -> bool:
        """Save cluster state for a group."""
        try:
            serializable_state = self._make_serializable(state)
            self._states[group_id] = serializable_state
            
            if self._enable_persistence:
                self._persist_to_disk(group_id, serializable_state)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save cluster state for group {group_id}: {e}")
            return False
    
    async def load_cluster_state(
        self,
        group_id: str
    ) -> Optional[Dict[str, Any]]:
        """Load cluster state for a group."""
        return self._states.get(group_id)
    
    async def get_cluster_assignments(
        self,
        group_id: str
    ) -> Dict[str, str]:
        """Get event_id -> cluster_id mapping for a group."""
        state = self._states.get(group_id, {})
        return state.get("eventid_to_cluster", {})
    
    async def clear(self, group_id: Optional[str] = None) -> bool:
        """Clear cluster state."""
        try:
            if group_id is None:
                self._states.clear()
            elif group_id in self._states:
                del self._states[group_id]
            return True
        except Exception as e:
            logger.error(f"Failed to clear cluster state: {e}")
            return False
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-serializable objects to JSON-compatible types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    def _persist_to_disk(
        self,
        group_id: str,
        state: Dict[str, Any]
    ) -> None:
        """Persist cluster state to JSON file."""
        if not self._persist_dir:
            return
        
        try:
            state_file = self._persist_dir / f"cluster_state_{group_id}.json"
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2, default=str)
            
            assignments_file = self._persist_dir / f"cluster_map_{group_id}.json"
            assignments = state.get("eventid_to_cluster", {})
            with open(assignments_file, "w", encoding="utf-8") as f:
                json.dump({"assignments": assignments}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist cluster state for group {group_id}: {e}")
    
    def _load_from_disk(self) -> None:
        """Load cluster states from disk on initialization."""
        if not self._persist_dir or not self._persist_dir.exists():
            return
        
        try:
            for state_file in self._persist_dir.glob("cluster_state_*.json"):
                try:
                    group_id = state_file.stem.replace("cluster_state_", "")
                    with open(state_file, "r", encoding="utf-8") as f:
                        state = json.load(f)
                    self._states[group_id] = state
                    logger.info(f"Loaded cluster state for group {group_id} from disk")
                except Exception as e:
                    logger.warning(f"Failed to load cluster state from {state_file}: {e}")
        except Exception as e:
            logger.error(f"Failed to load cluster states from disk: {e}")
