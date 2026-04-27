"""
Health check controller

Provides system health status check interface
"""

from common_utils.datetime_utils import get_now_with_timezone
from typing import Dict, Any
from core.interface.controller.base_controller import BaseController, get
from core.observation.logger import get_logger
from core.di.decorators import component

logger = get_logger(__name__)


@component(name="healthController")
class HealthController(BaseController):
    """
    Health check controller

    Provides system health status check functionality
    """

    def __init__(self):
        super().__init__(
            prefix="/health",
            tags=["Health"],
            default_auth="none",  # Health check does not require authentication
        )

    @get("", summary="Health check", description="Check system health status")
    def health_check(self) -> Dict[str, Any]:
        """
        Health check interface

        Returns:
            Dict[str, Any]: Health status information

        Raises:
            HTTPException: Throws 500 error when system is unhealthy
        """
        try:
            # Log health check request
            logger.debug("Health check request")

            # Return simple health status
            return {
                "status": "healthy",
                "timestamp": get_now_with_timezone().isoformat(),
                "message": "System running normally",
            }
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            # Throw 500 error when exception occurs
            from fastapi import HTTPException

            raise HTTPException(
                status_code=500,
                detail={
                    "status": "unhealthy",
                    "timestamp": get_now_with_timezone().isoformat(),
                    "message": f"System check exception: {str(e)}",
                },
            )
