"""
Logging configuration for FX Auto Trading System
"""

import logging
import logging.handlers
import os
from typing import Optional

from app.config import get_settings


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup logger with both console and file handlers
    
    Args:
        name: Logger name
        log_file: Log file path
        level: Log level
        max_bytes: Maximum file size before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get logger with default configuration from settings
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger
    """
    settings = get_settings()
    
    return setup_logger(
        name=name,
        log_file=settings.log_file,
        level=settings.log_level,
        max_bytes=int(settings.log_max_size.replace('MB', '')) * 1024 * 1024,
        backup_count=settings.log_backup_count
    )


# Application loggers
main_logger = get_logger("fx_trading.main")
trading_logger = get_logger("fx_trading.trading")
analysis_logger = get_logger("fx_trading.analysis")
mt5_logger = get_logger("fx_trading.mt5")
db_logger = get_logger("fx_trading.db")