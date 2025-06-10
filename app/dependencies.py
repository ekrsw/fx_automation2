"""
Dependency injection for FastAPI
"""

from functools import lru_cache
from fastapi import Depends
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.database import get_db


@lru_cache()
def get_settings_cached() -> Settings:
    """Get cached settings instance"""
    return get_settings()


def get_database() -> Session:
    """Get database session dependency"""
    return Depends(get_db)


def get_config() -> Settings:
    """Get configuration dependency"""
    return Depends(get_settings_cached)