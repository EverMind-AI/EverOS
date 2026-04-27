"""
Authorization module

Provides a role-based authorization system supporting anonymous, user, and admin roles,
as well as custom authorization strategies.
"""

from .enums import Role
from .interfaces import AuthorizationStrategy, AuthorizationContext
from .strategies import (
    DefaultAuthorizationStrategy,
    RoleBasedAuthorizationStrategy,
    CustomAuthorizationStrategy,
)
from .decorators import (
    authorize,
    require_anonymous,
    require_user,
    require_admin,
    custom_authorize,
    check_and_apply_default_auth,
)

__all__ = [
    # Enums
    'Role',
    # Interfaces
    'AuthorizationStrategy',
    'AuthorizationContext',
    # Strategy implementations
    'DefaultAuthorizationStrategy',
    'RoleBasedAuthorizationStrategy',
    'CustomAuthorizationStrategy',
    # Decorators
    'authorize',
    'require_anonymous',
    'require_user',
    'require_admin',
    'custom_authorize',
    'check_and_apply_default_auth',
]
