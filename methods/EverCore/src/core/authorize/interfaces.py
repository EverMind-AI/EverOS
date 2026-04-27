from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
from .enums import Role


class AuthorizationStrategy(ABC):
    """Authorization strategy interface"""

    @abstractmethod
    async def check_permission(
        self, user_info: Optional[Dict[str, Any]], required_role: Role, **kwargs
    ) -> bool:
        """
        Check user permissions

        Args:
            user_info: User information, may be None (anonymous user)
            required_role: Required role
            **kwargs: Additional parameters

        Returns:
            bool: Whether the user has permission
        """
        pass


class AuthorizationContext:
    """Authorization context, containing information required for authorization checks"""

    def __init__(
        self,
        user_info: Optional[Dict[str, Any]] = None,
        required_role: Role = Role.ANONYMOUS,
        strategy: Optional[AuthorizationStrategy] = None,
        **kwargs,
    ):
        self.user_info = user_info
        self.required_role = required_role
        self.strategy = strategy
        self.extra_kwargs = kwargs

    def need_auth(self) -> bool:
        """
        Check if authorization is needed
        """
        return self.required_role != Role.ANONYMOUS
