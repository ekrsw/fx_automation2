"""
Configuration management for FX Auto Trading System
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # Database
    database_url: str = "sqlite:///./data/fx_trading.db"
    
    # FastAPI
    app_name: str = "FX Auto Trading System"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # MT5 Configuration (loaded from environment variables)
    mt5_login: Optional[int] = None
    mt5_password: Optional[str] = None
    mt5_server: Optional[str] = None
    mt5_timeout: int = 30
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load MT5 credentials from environment variables
        if os.getenv('MT5_LOGIN'):
            self.mt5_login = int(os.getenv('MT5_LOGIN'))
        if os.getenv('MT5_PASSWORD'):
            self.mt5_password = os.getenv('MT5_PASSWORD')
        if os.getenv('MT5_SERVER'):
            self.mt5_server = os.getenv('MT5_SERVER')
    
    # Trading Configuration
    trading_symbols: List[str] = ["USDJPY", "EURJPY", "GBPJPY"]
    default_volume: float = 0.1
    max_positions: int = 3
    risk_per_trade: float = 0.02
    
    # Strategy Configuration
    dow_atr_multiplier: float = 0.5
    dow_swing_period: int = 5
    dow_trend_confirmation_bars: int = 3
    
    elliott_fibonacci_tolerance: float = 0.1
    elliott_min_wave_size: float = 0.001
    elliott_max_waves_to_analyze: int = 20
    
    # Risk Management
    max_daily_loss: float = 0.05
    max_drawdown: float = 0.15
    position_sizing_method: str = "fixed_risk"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/trading.log"
    log_max_size: str = "10MB"
    log_backup_count: int = 5
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get settings instance"""
    return settings