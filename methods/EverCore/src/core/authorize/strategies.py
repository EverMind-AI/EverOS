from typing import Optional, Any, Dict
from .interfaces import AuthorizationStrategy
from .enums import Role

import asyncio


class DefaultAuthorizationStrategy(AuthorizationStrategy):
    """Default authorization strategy"""

    async def check_permission(
        self, user_info: Optional[Dict[str, Any]], required_role: Role, **kwargs
    ) -> bool:
        """
        Default permission check logic

        Args:
            user_info: User information
            required_role: Required role
            **kwargs: Additional parameters

        Returns:
            bool: Whether the user has permission
        """
        # Anonymous users can only access anonymous resources
        if required_role == Role.ANONYMOUS:
            return True

        # Deny access if no user information is provided
        if not user_info:
            return False

        # Check user role
        user_role = user_info.get('role', Role.USER)
        user_role = Role(user_role)

        # Role-based permission check
        if required_role == Role.USER:
            return user_role in [Role.USER, Role.ADMIN]
        elif required_role == Role.ADMIN:
            return user_role == Role.ADMIN
        elif required_role == Role.SIGNATURE:
            return user_role == Role.SIGNATURE

        return False


class RoleBasedAuthorizationStrategy(AuthorizationStrategy):
    """Role-based authorization strategy"""

    def __init__(self):
        # Define role hierarchy
        self.role_hierarchy = {
            Role.ANONYMOUS: 0,
            Role.USER: 1,
            Role.ADMIN: 2,
            Role.SIGNATURE: 1,  # SIGNATURE has the same level as USER, can access resources requiring USER permission
        }

    async def check_permission(
        self, user_info: Optional[Dict[str, Any]], required_role: Role, **kwargs
    ) -> bool:
        """
        Role-based permission check

        Args:
            user_info: User information
            required_role: Required role
            **kwargs: Additional parameters

        Returns:
            bool: Whether the user has permission
        """
        # Anonymous users can only access anonymous resources
        if required_role == Role.ANONYMOUS:
            return True

        # Deny access if no user information is provided
        if not user_info:
            return False

        # Get user role
        user_role_str = user_info.get('role', Role.USER.value)
        try:
            user_role = Role(user_role_str)
        except ValueError:
            # If role is invalid, default to USER
            user_role = Role.USER

        # Check role hierarchy
        required_level = self.role_hierarchy.get(required_role, 0)
        user_level = self.role_hierarchy.get(user_role, 0)

        return user_level >= required_level


class CustomAuthorizationStrategy(AuthorizationStrategy):
    """Custom authorization strategy that allows users to define custom check logic"""

    def __init__(self, custom_check_func):
        """
        Initialize custom strategy

        Args:
            custom_check_func: Custom check function that takes user_info and required_role as parameters
        """
        self.custom_check_func = custom_check_func

    async def check_permission(
        self, user_info: Optional[Dict[str, Any]], required_role: Role, **kwargs
    ) -> bool:
        """
        Perform permission check using custom function

        Args:
            user_info: User information
            required_role: Required role
            **kwargs: Additional parameters

        Returns:
            bool: Whether the user has permission
        """
        try:
            if asyncio.iscoroutinefunction(self.custom_check_func):
                return await self.custom_check_func(user_info, required_role, **kwargs)
            else:
                return self.custom_check_func(user_info, required_role, **kwargs)
        except (ValueError, TypeError, AttributeError):
            # Return False if custom check fails
            return False
