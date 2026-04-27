"""In-memory profile storage for evaluation.

This storage implementation is used by the evaluation framework for profile
management during MemCell extraction. It keeps profiles in memory with optional
file persistence for checkpointing.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import json

from common_utils.datetime_utils import get_now_with_timezone, to_iso_format
from core.observation.logger import get_logger

logger = get_logger(__name__)


class InMemoryProfileStorage:
    """In-memory profile storage with optional file persistence."""
    
    def __init__(
        self,
        enable_persistence: bool = False,
        persist_dir: Optional[Path] = None,
        enable_versioning: bool = True
    ):
        """Initialize in-memory storage.
        
        Args:
            enable_persistence: Whether to persist to disk
            persist_dir: Directory for JSON files (required if enable_persistence=True)
            enable_versioning: Whether to keep version history
        """
        self._profiles: Dict[str, Any] = {}
        self._history: Dict[str, List[Dict[str, Any]]] = {}
        self._enable_persistence = enable_persistence
        self._persist_dir = persist_dir
        self._enable_versioning = enable_versioning
        
        if enable_persistence and not persist_dir:
            raise ValueError("persist_dir is required when enable_persistence=True")
        
        if enable_persistence and persist_dir:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()
    
    async def save_profile(
        self,
        user_id: str,
        profile: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Save or update a user profile."""
        try:
            self._profiles[user_id] = profile
            
            if self._enable_versioning:
                history_entry = {
                    "profile": profile,
                    "metadata": metadata or {},
                    "timestamp": to_iso_format(get_now_with_timezone()),
                }
                if user_id not in self._history:
                    self._history[user_id] = []
                self._history[user_id].append(history_entry)
            
            if self._enable_persistence:
                self._persist_to_disk(user_id, profile, metadata)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save profile for user {user_id}: {e}")
            return False
    
    async def get_profile(self, user_id: str) -> Optional[Any]:
        """Retrieve the latest profile for a user."""
        return self._profiles.get(user_id)
    
    async def get_all_profiles(self) -> Dict[str, Any]:
        """Retrieve all user profiles."""
        return dict(self._profiles)
    
    async def get_profile_history(
        self,
        user_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve profile version history for a user."""
        history = self._history.get(user_id, [])
        history_reversed = list(reversed(history))
        if limit is not None and limit > 0:
            return history_reversed[:limit]
        return history_reversed
    
    async def clear(self) -> bool:
        """Clear all stored profiles."""
        try:
            self._profiles.clear()
            self._history.clear()
            return True
        except Exception as e:
            logger.error(f"Failed to clear profiles: {e}")
            return False
    
    def _serialize_for_json(self, obj: Any) -> Any:
        """Recursively serialize object to JSON-compatible format."""
        import datetime as dt
        
        if isinstance(obj, dt.datetime):
            return to_iso_format(obj)
        elif isinstance(obj, (dt.date, dt.time)):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._serialize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_for_json(item) for item in obj]
        elif hasattr(obj, 'value'):  # Enum
            return obj.value
        elif hasattr(obj, 'to_dict'):
            return self._serialize_for_json(obj.to_dict())
        elif hasattr(obj, '__dict__'):
            return self._serialize_for_json(obj.__dict__)
        else:
            return obj
    
    def _persist_to_disk(
        self,
        user_id: str,
        profile: Any,
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Persist profile to JSON file."""
        if not self._persist_dir:
            return
        
        try:
            if hasattr(profile, "to_dict"):
                try:
                    payload = profile.to_dict()
                except (AttributeError, TypeError) as e:
                    error_msg = str(e).lower()
                    if 'tzinfo' in error_msg or 'isoformat' in error_msg:
                        payload = profile.__dict__.copy()
                        if hasattr(payload.get('memory_type'), 'value'):
                            payload['memory_type'] = payload['memory_type'].value
                        ts = payload.get('timestamp')
                        if ts is not None:
                            if hasattr(ts, 'isoformat'):
                                payload['timestamp'] = ts.isoformat()
                            elif not isinstance(ts, str):
                                payload['timestamp'] = str(ts)
                    else:
                        raise
            elif hasattr(profile, "__dict__"):
                payload = profile.__dict__.copy()
                if hasattr(payload.get('memory_type'), 'value'):
                    payload['memory_type'] = payload['memory_type'].value
                ts = payload.get('timestamp')
                if ts is not None:
                    if hasattr(ts, 'isoformat'):
                        payload['timestamp'] = ts.isoformat()
                    elif not isinstance(ts, str):
                        payload['timestamp'] = str(ts)
            else:
                payload = profile
            
            payload = self._serialize_for_json(payload)
            
            latest_file = self._persist_dir / f"profile_{user_id}.json"
            with open(latest_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
            
            if self._enable_versioning:
                history_dir = self._persist_dir / "history" / user_id
                history_dir.mkdir(parents=True, exist_ok=True)
                
                now = get_now_with_timezone()
                timestamp_str = to_iso_format(now).replace(":", "-").replace("+", "_")
                version_file = history_dir / f"profile_{user_id}_{timestamp_str}.json"
                with open(version_file, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to persist profile for user {user_id}: {e}")
    
    def _load_from_disk(self) -> None:
        """Load profiles from disk on initialization."""
        if not self._persist_dir or not self._persist_dir.exists():
            return
        
        try:
            for profile_file in self._persist_dir.glob("profile_*.json"):
                if profile_file.stem.startswith("profile_") and not profile_file.parent.name == "history":
                    try:
                        with open(profile_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        
                        user_id = data.get("user_id")
                        profile = data.get("profile")
                        
                        if user_id and profile:
                            self._profiles[user_id] = profile
                            logger.info(f"Loaded profile for user {user_id} from disk")
                    except Exception as e:
                        logger.warning(f"Failed to load profile from {profile_file}: {e}")
            
            if self._enable_versioning:
                history_base = self._persist_dir / "history"
                if history_base.exists():
                    for user_dir in history_base.iterdir():
                        if user_dir.is_dir():
                            user_id = user_dir.name
                            user_history = []
                            
                            for version_file in sorted(user_dir.glob("profile_*.json")):
                                try:
                                    with open(version_file, "r", encoding="utf-8") as f:
                                        data = json.load(f)
                                    user_history.append({
                                        "profile": data.get("profile"),
                                        "metadata": data.get("metadata", {}),
                                        "timestamp": data.get("last_updated"),
                                    })
                                except Exception as e:
                                    logger.warning(f"Failed to load history from {version_file}: {e}")
                            
                            if user_history:
                                self._history[user_id] = user_history
        except Exception as e:
            logger.error(f"Failed to load profiles from disk: {e}")
