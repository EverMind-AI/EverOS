import os
from typing import AsyncGenerator
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from core.di.decorators import component
from common_utils.datetime_utils import get_timezone


@component(name="database_session_provider", primary=True)
class DatabaseSessionProvider:
    """Database session provider"""

    def __init__(self):
        """Initialize the database session provider"""
        self.database_url = os.getenv("DATABASE_URL", "")

        timezone = get_timezone()

        # Replace postgresql:// with postgresql+asyncpg:// to support async
        if self.database_url.startswith("postgresql://"):
            self.async_database_url = self.database_url.replace(
                "postgresql://", "postgresql+asyncpg://", 1
            )
        else:
            self.async_database_url = self.database_url

        # Create async engine
        self.async_engine = create_async_engine(
            self.async_database_url,
            echo=False,  # Set to True to see SQL logs
            future=True,
            pool_pre_ping=True,
            pool_recycle=int(
                os.getenv("DB_POOL_RECYCLE", "300")
            ),  # Recycle connections every 5 minutes
            pool_size=int(
                os.getenv("DB_POOL_SIZE", "40")
            ),  # 🔧 Increase connection pool size (default 5 → 10)
            max_overflow=int(
                os.getenv("DB_MAX_OVERFLOW", "25")
            ),  # 🔧 Increase maximum overflow connections (default 10 → 15)
            connect_args={"server_settings": {"timezone": timezone}},
        )

        # Create async session factory
        self.async_session_factory = async_sessionmaker(
            bind=self.async_engine, class_=AsyncSession, expire_on_commit=False
        )

    def create_session(self) -> AsyncSession:
        """Create a new async database session"""
        return self.async_session_factory()

    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session (in context manager form)"""
        async with self.async_session_factory() as session:
            try:
                yield session
            finally:
                await session.close()
