"""Profile Manager - Pure computation component for profile extraction.

Usage:
    from memory_layer.profile_manager import ProfileManager, ProfileManagerConfig

    config = ProfileManagerConfig(min_confidence=0.6)
    profile_mgr = ProfileManager(llm_provider, config)

    old_profiles = list((await storage.get_all_profiles()).values())

    new_profiles = await profile_mgr.extract_profiles(
        memcells=memcell_list,
        old_profiles=old_profiles,
        user_id_list=["user1", "user2"],
    )

    for profile in new_profiles:
        await storage.save_profile(profile.user_id, profile)
"""

from memory_layer.profile_manager.config import ProfileManagerConfig
from memory_layer.profile_manager.manager import ProfileManager

__all__ = ["ProfileManager", "ProfileManagerConfig"]
