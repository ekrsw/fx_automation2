"""
Dashboard API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime
from typing import Dict, Any

from app.dependencies import get_settings_cached
from app.config import Settings
from app.mt5.connection import get_mt5_connection
from app.mt5.data_fetcher import get_data_fetcher
from app.utils.logger import main_logger

router = APIRouter()


@router.get("/status")
async def get_system_status(settings: Settings = Depends(get_settings_cached)) -> Dict[str, Any]:
    """
    Get overall system status
    
    Returns:
        System status information
    """
    try:
        mt5_connection = get_mt5_connection()
        data_fetcher = get_data_fetcher()
        
        # Check MT5 connection
        mt5_healthy = await mt5_connection.health_check()
        mt5_info = mt5_connection.get_connection_info()
        
        # Test data fetching
        data_fetch_test = False
        try:
            test_data = await data_fetcher.get_live_prices(["USDJPY"])
            data_fetch_test = len(test_data) > 0
        except Exception:
            pass
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy" if mt5_healthy and data_fetch_test else "unhealthy",
            "components": {
                "mt5_connection": {
                    "status": "healthy" if mt5_healthy else "unhealthy",
                    "details": mt5_info
                },
                "data_fetcher": {
                    "status": "healthy" if data_fetch_test else "unhealthy"
                },
                "database": {
                    "status": "healthy",  # Assume healthy if no errors
                    "url": settings.database_url
                }
            },
            "configuration": {
                "trading_symbols": settings.trading_symbols,
                "max_positions": settings.max_positions,
                "risk_per_trade": settings.risk_per_trade
            }
        }
        
        main_logger.info("System status check completed")
        return status
        
    except Exception as e:
        main_logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Simple health check endpoint
    
    Returns:
        Health status
    """
    try:
        mt5_connection = get_mt5_connection()
        mt5_healthy = await mt5_connection.health_check()
        
        return {
            "status": "healthy" if mt5_healthy else "degraded",
            "timestamp": datetime.now().isoformat(),
            "mt5_connected": mt5_connection.is_connected()
        }
        
    except Exception as e:
        main_logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


@router.get("/performance")
async def get_performance_metrics() -> Dict[str, Any]:
    """
    Get performance metrics (placeholder)
    
    Returns:
        Performance metrics
    """
    # This will be implemented in later phases
    return {
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "profit_loss": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0
        },
        "note": "Performance tracking will be implemented in Phase 2"
    }


@router.get("/positions")
async def get_current_positions() -> Dict[str, Any]:
    """
    Get current trading positions (placeholder)
    
    Returns:
        Current positions
    """
    # This will be implemented in later phases
    return {
        "timestamp": datetime.now().isoformat(),
        "positions": [],
        "total_positions": 0,
        "total_exposure": 0.0,
        "note": "Position tracking will be implemented in Phase 2"
    }


@router.get("/recent-trades")
async def get_recent_trades() -> Dict[str, Any]:
    """
    Get recent trade history (placeholder)
    
    Returns:
        Recent trades
    """
    # This will be implemented in later phases
    return {
        "timestamp": datetime.now().isoformat(),
        "trades": [],
        "total_trades": 0,
        "note": "Trade history will be implemented in Phase 2"
    }