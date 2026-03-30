# ProfileManager - Automatic Profile Extraction

`ProfileManager` is a core component that automatically extracts and maintains user profiles from clustered conversations. It integrates seamlessly with `ConvMemCellExtractor` to provide hands-free profile management.

## Features

- ✨ **Automatic Profile Extraction**: Profiles are extracted automatically as conversations are processed
- 🎯 **Value Discrimination**: Uses LLM to filter high-quality profile-worthy content
- 🔄 **Incremental Updates**: Profiles are merged incrementally as new information arrives
- 📚 **Version History**: Optional versioning to track profile evolution over time
- 💾 **Flexible Storage**: Pluggable storage backends (in-memory, file-based, or custom)
- 🎭 **Multi-Scenario**: Supports both group_chat and assistant scenarios

## Quick Start

### Basic Usage

```python
from memory_layer.llm.llm_provider import LLMProvider
from memory_layer.memcell_extractor.conv_memcell_extractor import ConvMemCellExtractor
from memory_layer.profile_manager import ProfileManager, ProfileManagerConfig

# Initialize LLM provider
llm_provider = LLMProvider(
    provider_type="openai",
    model="gpt-4",
    api_key="your-api-key"
)

# Configure ProfileManager
config = ProfileManagerConfig(
    scenario="group_chat",  # or "assistant"
    min_confidence=0.6,
    enable_versioning=True,
    auto_extract=True
)

# Create ProfileManager
profile_mgr = ProfileManager(
    llm_provider=llm_provider,
    config=config,
    group_id="my_group_001",
    group_name="My Team Chat"
)

# Attach to MemCellExtractor for automatic updates
memcell_extractor = ConvMemCellExtractor(llm_provider=llm_provider)
profile_mgr.attach_to_extractor(memcell_extractor)

# That's it! Profiles will now be automatically extracted and updated
# as conversations are processed through the extractor
```

### Accessing Profiles

```python
# Get a single user's profile
profile = await profile_mgr.get_profile(user_id="user_123")

# Get all profiles
all_profiles = await profile_mgr.get_all_profiles()

# Get profile version history
history = await profile_mgr.get_profile_history(user_id="user_123", limit=5)

# Get statistics
stats = profile_mgr.get_stats()
print(f"Processed {stats['total_memcells']} memcells")
print(f"Found {stats['high_value_memcells']} high-value memcells")
print(f"Updated profiles {stats['profile_extractions']} times")
```

### Manual Profile Updates

If you want more control, you can manually trigger profile extraction:

```python
# Disable auto-extraction
config = ProfileManagerConfig(auto_extract=False)
profile_mgr = ProfileManager(llm_provider, config)

# Manually trigger profile extraction when a memcell is clustered
result = await profile_mgr.on_memcell_clustered(
    memcell=memcell,
    cluster_id="cluster_001",
    recent_memcells=[previous_mc1, previous_mc2],
    user_id_list=["user_123", "user_456"]
)

print(f"Is high-value: {result['is_high_value']}")
print(f"Confidence: {result['confidence']}")
print(f"Updated {result['profiles_updated']} profiles")
```

### File Persistence

```python
from pathlib import Path
from memory_layer.profile_manager import InMemoryProfileStorage

# Create storage with file persistence
storage = InMemoryProfileStorage(
    enable_persistence=True,
    persist_dir=Path("./profiles"),
    enable_versioning=True
)

profile_mgr = ProfileManager(
    llm_provider=llm_provider,
    config=config,
    storage=storage
)

# Profiles will be automatically saved to ./profiles/
# History will be saved to ./profiles/history/{user_id}/
```

### Export Profiles

```python
# Export all profiles to JSON files
output_dir = Path("./exported_profiles")
count = await profile_mgr.export_profiles(
    output_dir=output_dir,
    include_history=True
)
print(f"Exported {count} profiles to {output_dir}")
```

## Configuration

### ProfileManagerConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scenario` | `ScenarioType` or `str` | `"group_chat"` | Extraction scenario: "group_chat" or "assistant" |
| `min_confidence` | `float` | `0.6` | Minimum confidence threshold for value discrimination (0.0-1.0) |
| `enable_versioning` | `bool` | `True` | Whether to keep profile version history |
| `auto_extract` | `bool` | `True` | Whether to automatically extract profiles on cluster updates |
| `batch_size` | `int` | `50` | Maximum memcells per batch for profile extraction |
| `max_retries` | `int` | `3` | Maximum retry attempts for failed extractions |

### DiscriminatorConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_confidence` | `float` | `0.6` | Minimum confidence for high-value judgment |
| `use_context` | `bool` | `True` | Whether to use previous memcells as context |
| `context_window` | `int` | `2` | Number of previous memcells to include |

## Scenarios

### Group Chat (Work Context)

Optimized for professional/team conversations. Extracts:
- Role & responsibilities
- Hard skills (technologies, tools)
- Soft skills (communication, leadership)
- Project participation
- Working habits & preferences
- Personality traits
- Decision-making patterns

```python
config = ProfileManagerConfig(scenario="group_chat")
```

### Assistant (Companion Context)

Optimized for personal conversations. Focuses on:
- Stable personality traits
- Enduring preferences
- Interests & hobbies
- Decision-making style
- Value system
- Motivations & fears
- Routines & habits

```python
config = ProfileManagerConfig(scenario="assistant")
```

## Custom Storage Backend

Implement the `ProfileStorage` interface for custom storage:

```python
from memory_layer.profile_manager import ProfileStorage

class MyCustomStorage(ProfileStorage):
    async def save_profile(self, user_id: str, profile: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        # Your implementation
        pass
    
    async def get_profile(self, user_id: str) -> Optional[Any]:
        # Your implementation
        pass
    
    async def get_all_profiles(self) -> Dict[str, Any]:
        # Your implementation
        pass
    
    async def get_profile_history(self, user_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        # Your implementation
        pass
    
    async def clear(self) -> bool:
        # Your implementation
        pass

# Use your custom storage
storage = MyCustomStorage()
profile_mgr = ProfileManager(llm_provider, storage=storage)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ProfileManager                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────────┐    ┌──────────────────┐    ┌──────────┐  │
│  │ValueDiscriminator│  │ProfileExtractor  │  │ Storage  │  │
│  └───────────────┘    └──────────────────┘    └──────────┘  │
│         ▲                      ▲                     ▲        │
│         │                      │                     │        │
│         └──────────┬───────────┴─────────────────────┘        │
│                    │                                           │
│              ┌─────▼──────┐                                   │
│              │   Manager   │                                   │
│              │   Core      │                                   │
│              └─────┬──────┘                                   │
└────────────────────┼──────────────────────────────────────────┘
                     │
                     ▼
         ┌──────────────────────┐
         │  ClusteringWorker    │
         │  (via attachment)    │
         └──────────────────────┘
                     │
                     ▼
         ┌──────────────────────┐
         │  MemCellExtractor    │
         └──────────────────────┘
```

## Migration Guide

### From `extract_memory.py`

**Before:**
```python
# Old approach: separate scripts, manual coordination
python extract_memory.py  # Extract memcells
# Then manually run profile extraction
# Then manually manage profile updates
```

**After:**
```python
# New approach: integrated, automatic
profile_mgr = ProfileManager(llm_provider, config)
profile_mgr.attach_to_extractor(memcell_extractor)
# Done! Profiles auto-update as conversations are processed
```

### From `ClusterProfileCoordinator`

**Before:**
```python
# Manual coordinator usage
coordinator, discriminator = build_coordinator(provider, group_id)
for mc in memcells:
    result = await coordinator.on_new_memcell(mc, cluster_id, discriminator, recent)
```

**After:**
```python
# Automatic with ProfileManager
profile_mgr = ProfileManager(llm_provider, config)
profile_mgr.attach_to_extractor(memcell_extractor)
# Profiles update automatically
```

## Best Practices

1. **Choose the right scenario**: Use "group_chat" for work contexts and "assistant" for personal/companion contexts
2. **Tune confidence threshold**: Start with 0.6 and adjust based on your data quality
3. **Enable versioning for important use cases**: Helps track profile evolution and debug issues
4. **Use file persistence for production**: In-memory storage is fast but volatile
5. **Monitor statistics**: Use `get_stats()` to track extraction quality and frequency
6. **Batch appropriately**: Default batch_size=50 works for most cases, increase if you have very long conversations

## Performance

- **Memory**: ~10-50MB per 1000 memcells (depending on versioning and storage backend)
- **Latency**: Profile extraction adds ~2-5s per high-value cluster (depends on LLM)
- **Throughput**: Processes 100-500 memcells/minute (with clustering and profile extraction)

## Troubleshooting

### Profiles not updating

1. Check if `auto_extract=True` in config
2. Verify ProfileManager is attached to extractor via `attach_to_extractor()`
3. Check logs for discrimination failures
4. Verify memcells are being clustered (check cluster_worker)

### Too many/few profile updates

1. Adjust `min_confidence` threshold (higher = fewer updates, lower = more updates)
2. Check discrimination logs to see what's being flagged
3. Verify scenario matches your use case (group_chat vs assistant)

### Storage errors

1. Check directory permissions for file-based storage
2. Verify storage backend is properly initialized
3. Check logs for specific storage errors

## License

Part of the EverMemOS project.

