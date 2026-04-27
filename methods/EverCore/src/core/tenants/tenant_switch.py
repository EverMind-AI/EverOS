# ============================================================================
# Application ready listener (automatically discovered via DI)
# ============================================================================

from core.di.decorators import component
from core.lifespan.lifespan_factory import AppReadyListener
from core.observation.logger import get_logger
from core.tenants.tenant_config import get_tenant_config

logger = get_logger(__name__)


@component(name="tenant_config_app_ready_listener")
class TenantConfigAppReadyListener(AppReadyListener):
    """
    Tenant configuration application ready listener

    Automatically enables strict tenant checking mode when application startup is complete.
    Automatically discovered and invoked by the DI container, no manual registration required.
    """

    def on_app_ready(self) -> None:
        """Enable strict tenant check when application is ready"""
        get_tenant_config().mark_app_ready()
