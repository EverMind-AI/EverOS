from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from fastapi import Request, HTTPException

from core.di.decorators import component


class AuthProvider(ABC):
    """Authentication provider interface, responsible for handling authorization header and user context"""

    @abstractmethod
    async def get_optional_user_data_from_request(
        self, request: Request
    ) -> Optional[Dict[str, Any]]:
        """
        Extract full user data from the request (optional)

        Args:
            request: FastAPI request object

        Returns:
            Optional[Dict[str, Any]]: User data, including user_id, role, etc. Return None if not present or invalid
        """


@component(name="auth_provider")
class TestAuthProviderImpl(AuthProvider):
    """Authentication provider implementation, responsible for handling authorization header and user context"""

    def __init__(self):
        """Initialize the authentication provider"""

    async def get_user_id_from_request(self, request: Request) -> int:
        """
        Extract user ID from the request

        Current implementation: directly obtain user ID from the authorization header (temporary solution)
        Future extension: can support JWT token parsing, etc.

        Args:
            request: FastAPI request object

        Returns:
            int: User ID

        Raises:
            HTTPException: When the authorization header is missing or invalid
        """
        # Get user ID from the authorization header
        auth_header = request.headers.get("authorization")

        if not auth_header:
            raise HTTPException(status_code=401, detail="Missing authorization header")

        # Remove possible "Bearer " prefix
        user_id_str = auth_header.replace("Bearer ", "").strip()

        try:
            user_id = int(user_id_str)
            if user_id <= 0:
                raise ValueError("User ID must be a positive integer")
            return user_id
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid user ID format in authorization header, should be a positive integer",
            )

    async def get_optional_user_data_from_request(
        self, request: Request
    ) -> Optional[Dict[str, Any]]:
        """
        Extract full user data from the request (optional)

        Args:
            request: FastAPI request object

        Returns:
            Optional[Dict[str, Any]]: User data, including user_id, role, etc. Return None if not present or invalid
        """
        try:
            user_id = await self.get_user_id_from_request(request)
            # Import Role enum
            from core.authorize.enums import Role

            return {"user_id": user_id, "role": Role.USER.value}
        except HTTPException:
            return None
