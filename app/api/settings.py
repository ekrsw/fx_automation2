"""
Settings Management API endpoints
Phase 7.1 Implementation - System configuration and parameter management
"""

from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import asyncio

from app.dependencies import get_settings_cached
from app.config import Settings
from app.trading.trading_engine import trading_engine
from app.trading.risk_manager import risk_manager
from app.analysis.strategy_engine.unified_analyzer import unified_analyzer
from app.db.crud.settings import settings_crud
from app.utils.logger import main_logger

router = APIRouter()


# Request/Response Models
class TradingSettingsUpdate(BaseModel):
    trading_symbols: Optional[List[str]] = None
    max_positions: Optional[int] = Field(None, ge=1, le=100)
    risk_per_trade: Optional[float] = Field(None, ge=0.001, le=0.1)
    enable_trading: Optional[bool] = None

class RiskSettingsUpdate(BaseModel):
    max_risk_per_trade: Optional[float] = Field(None, ge=0.001, le=0.1)
    max_portfolio_risk: Optional[float] = Field(None, ge=0.01, le=0.5)
    max_drawdown_limit: Optional[float] = Field(None, ge=0.05, le=0.5)
    emergency_stop_loss: Optional[float] = Field(None, ge=0.05, le=0.3)
    position_size_method: Optional[str] = None

class AnalysisSettingsUpdate(BaseModel):
    dow_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    elliott_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_confidence: Optional[float] = Field(None, ge=0.1, le=1.0)
    signal_validity_hours: Optional[int] = Field(None, ge=1, le=24)

class SystemSettingUpdate(BaseModel):
    key: str
    value: Any
    description: Optional[str] = None


# Current Settings
@router.get("/current")
async def get_current_settings(settings: Settings = Depends(get_settings_cached)) -> Dict[str, Any]:
    """
    Get current system settings
    
    Returns:
        Current configuration settings
    """
    try:
        # Get trading engine status
        trading_status = await trading_engine.get_trading_status() if trading_engine else None
        
        # Get risk manager configuration
        risk_config = risk_manager.config if risk_manager else {}
        
        # Get analysis configuration
        unified_config = unified_analyzer.config if unified_analyzer else {}
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "trading_settings": {
                "trading_symbols": settings.trading_symbols,
                "max_positions": settings.max_positions,
                "risk_per_trade": settings.risk_per_trade,
                "enable_trading": getattr(settings, 'enable_trading', True),
                "trading_session_active": bool(trading_status and trading_status.get('session', {}).get('status') == 'active')
            },
            "risk_management": {
                "max_risk_per_trade": risk_config.get('max_risk_per_trade', 0.02),
                "max_portfolio_risk": risk_config.get('max_portfolio_risk', 0.10),
                "max_drawdown_limit": risk_config.get('max_drawdown_limit', 0.20),
                "emergency_stop_active": risk_manager.emergency_stop_active if risk_manager else False,
                "position_size_method": risk_config.get('position_size_method', 'fixed_percentage')
            },
            "analysis_settings": {
                "dow_weight": unified_config.get('dow_weight', 0.4),
                "elliott_weight": unified_config.get('elliott_weight', 0.6),
                "min_confidence": unified_config.get('min_confidence', 0.6),
                "signal_validity_hours": unified_config.get('signal_validity_hours', 4)
            },
            "database_settings": {
                "database_url": settings.database_url,
                "connection_pool_size": getattr(settings, 'connection_pool_size', 5)
            },
            "mt5_settings": {
                "mt5_enabled": getattr(settings, 'mt5_enabled', True),
                "mt5_timeout": getattr(settings, 'mt5_timeout', 30000)
            }
        }
        
        main_logger.info("Current settings retrieved")
        return result
        
    except Exception as e:
        main_logger.error(f"Error getting current settings: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# Trading Settings
@router.put("/trading")
async def update_trading_settings(
    request: TradingSettingsUpdate,
    settings: Settings = Depends(get_settings_cached)
) -> Dict[str, Any]:
    """
    Update trading settings
    
    Args:
        request: Trading settings to update
        
    Returns:
        Update result
    """
    try:
        updated_fields = []
        
        # Update trading symbols
        if request.trading_symbols is not None:
            # Validate symbols
            valid_symbols = ["USDJPY", "EURJPY", "GBPJPY", "EURUSD", "GBPUSD", "USDCHF", "AUDUSD", "NZDUSD"]
            invalid_symbols = [s for s in request.trading_symbols if s not in valid_symbols]
            if invalid_symbols:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid symbols: {invalid_symbols}. Valid symbols: {valid_symbols}"
                )
            
            await settings_crud.create_or_update_setting(
                "trading_symbols", 
                ",".join(request.trading_symbols),
                "Active trading symbols"
            )
            updated_fields.append("trading_symbols")
        
        # Update max positions
        if request.max_positions is not None:
            await settings_crud.create_or_update_setting(
                "max_positions",
                str(request.max_positions),
                "Maximum concurrent positions"
            )
            updated_fields.append("max_positions")
        
        # Update risk per trade
        if request.risk_per_trade is not None:
            await settings_crud.create_or_update_setting(
                "risk_per_trade",
                str(request.risk_per_trade),
                "Risk percentage per trade"
            )
            updated_fields.append("risk_per_trade")
        
        # Update trading enabled status
        if request.enable_trading is not None:
            await settings_crud.create_or_update_setting(
                "enable_trading",
                str(request.enable_trading),
                "Enable/disable trading"
            )
            updated_fields.append("enable_trading")
        
        result = {
            "success": True,
            "updated_fields": updated_fields,
            "updated_at": datetime.now().isoformat(),
            "message": f"Trading settings updated: {', '.join(updated_fields)}"
        }
        
        main_logger.info(f"Trading settings updated: {updated_fields}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        main_logger.error(f"Error updating trading settings: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# Risk Management Settings
@router.put("/risk")
async def update_risk_settings(request: RiskSettingsUpdate) -> Dict[str, Any]:
    """
    Update risk management settings
    
    Args:
        request: Risk settings to update
        
    Returns:
        Update result
    """
    try:
        if not risk_manager:
            raise HTTPException(
                status_code=503,
                detail="Risk manager not available"
            )
        
        updated_fields = []
        
        # Update risk manager configuration
        risk_updates = {}
        
        if request.max_risk_per_trade is not None:
            risk_updates['max_risk_per_trade'] = request.max_risk_per_trade
            updated_fields.append("max_risk_per_trade")
        
        if request.max_portfolio_risk is not None:
            risk_updates['max_portfolio_risk'] = request.max_portfolio_risk
            updated_fields.append("max_portfolio_risk")
        
        if request.max_drawdown_limit is not None:
            risk_updates['max_drawdown_limit'] = request.max_drawdown_limit
            updated_fields.append("max_drawdown_limit")
        
        if request.position_size_method is not None:
            if request.position_size_method not in ["fixed_percentage", "kelly_criterion", "volatility_adjusted"]:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid position size method"
                )
            risk_updates['position_size_method'] = request.position_size_method
            updated_fields.append("position_size_method")
        
        # Update risk manager config
        if risk_updates:
            risk_manager.config.update(risk_updates)
        
        # Store in database
        for field in updated_fields:
            await settings_crud.create_or_update_setting(
                f"risk_{field}",
                str(risk_updates[field]),
                f"Risk management: {field}"
            )
        
        result = {
            "success": True,
            "updated_fields": updated_fields,
            "updated_at": datetime.now().isoformat(),
            "current_risk_config": risk_manager.config,
            "message": f"Risk settings updated: {', '.join(updated_fields)}"
        }
        
        main_logger.info(f"Risk settings updated: {updated_fields}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        main_logger.error(f"Error updating risk settings: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# Analysis Settings
@router.put("/analysis")
async def update_analysis_settings(request: AnalysisSettingsUpdate) -> Dict[str, Any]:
    """
    Update analysis settings
    
    Args:
        request: Analysis settings to update
        
    Returns:
        Update result
    """
    try:
        if not unified_analyzer:
            raise HTTPException(
                status_code=503,
                detail="Unified analyzer not available"
            )
        
        updated_fields = []
        analysis_updates = {}
        
        # Validate weight sum
        current_dow_weight = unified_analyzer.config.get('dow_weight', 0.4)
        current_elliott_weight = unified_analyzer.config.get('elliott_weight', 0.6)
        
        new_dow_weight = request.dow_weight if request.dow_weight is not None else current_dow_weight
        new_elliott_weight = request.elliott_weight if request.elliott_weight is not None else current_elliott_weight
        
        if abs(new_dow_weight + new_elliott_weight - 1.0) > 0.01:
            raise HTTPException(
                status_code=400,
                detail="Dow weight + Elliott weight must equal 1.0"
            )
        
        # Update weights
        if request.dow_weight is not None:
            analysis_updates['dow_weight'] = request.dow_weight
            updated_fields.append("dow_weight")
        
        if request.elliott_weight is not None:
            analysis_updates['elliott_weight'] = request.elliott_weight
            updated_fields.append("elliott_weight")
        
        if request.min_confidence is not None:
            analysis_updates['min_confidence'] = request.min_confidence
            updated_fields.append("min_confidence")
        
        if request.signal_validity_hours is not None:
            analysis_updates['signal_validity_hours'] = request.signal_validity_hours
            updated_fields.append("signal_validity_hours")
        
        # Update analyzer config
        if analysis_updates:
            unified_analyzer.config.update(analysis_updates)
        
        # Store in database
        for field in updated_fields:
            await settings_crud.create_or_update_setting(
                f"analysis_{field}",
                str(analysis_updates[field]),
                f"Analysis setting: {field}"
            )
        
        result = {
            "success": True,
            "updated_fields": updated_fields,
            "updated_at": datetime.now().isoformat(),
            "current_analysis_config": unified_analyzer.config,
            "message": f"Analysis settings updated: {', '.join(updated_fields)}"
        }
        
        main_logger.info(f"Analysis settings updated: {updated_fields}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        main_logger.error(f"Error updating analysis settings: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# System Settings (Generic)
@router.get("/system")
async def get_system_settings() -> Dict[str, Any]:
    """
    Get all system settings from database
    
    Returns:
        All stored system settings
    """
    try:
        # Get all settings from database
        all_settings = await settings_crud.get_all_settings()
        
        # Group by category
        categorized_settings = {
            "trading": {},
            "risk": {},
            "analysis": {},
            "system": {},
            "other": {}
        }
        
        for setting in all_settings:
            key = setting.key
            value = setting.value
            
            if key.startswith("trading_"):
                categorized_settings["trading"][key[8:]] = {
                    "value": value,
                    "description": setting.description,
                    "updated_at": setting.updated_at.isoformat() if setting.updated_at else None
                }
            elif key.startswith("risk_"):
                categorized_settings["risk"][key[5:]] = {
                    "value": value,
                    "description": setting.description,
                    "updated_at": setting.updated_at.isoformat() if setting.updated_at else None
                }
            elif key.startswith("analysis_"):
                categorized_settings["analysis"][key[9:]] = {
                    "value": value,
                    "description": setting.description,
                    "updated_at": setting.updated_at.isoformat() if setting.updated_at else None
                }
            elif key.startswith("system_"):
                categorized_settings["system"][key[7:]] = {
                    "value": value,
                    "description": setting.description,
                    "updated_at": setting.updated_at.isoformat() if setting.updated_at else None
                }
            else:
                categorized_settings["other"][key] = {
                    "value": value,
                    "description": setting.description,
                    "updated_at": setting.updated_at.isoformat() if setting.updated_at else None
                }
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "total_settings": len(all_settings),
            "categories": categorized_settings
        }
        
        main_logger.info(f"Retrieved {len(all_settings)} system settings")
        return result
        
    except Exception as e:
        main_logger.error(f"Error getting system settings: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.put("/system/{key}")
async def update_system_setting(key: str, request: SystemSettingUpdate) -> Dict[str, Any]:
    """
    Update a specific system setting
    
    Args:
        key: Setting key
        request: Setting update data
        
    Returns:
        Update result
    """
    try:
        # Validate key matches URL parameter
        if request.key != key:
            raise HTTPException(
                status_code=400,
                detail="Key in URL must match key in request body"
            )
        
        # Update setting in database
        await settings_crud.create_or_update_setting(
            key=request.key,
            value=str(request.value),
            description=request.description
        )
        
        result = {
            "success": True,
            "key": request.key,
            "new_value": request.value,
            "description": request.description,
            "updated_at": datetime.now().isoformat(),
            "message": f"System setting '{request.key}' updated successfully"
        }
        
        main_logger.info(f"System setting updated: {request.key}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        main_logger.error(f"Error updating system setting {key}: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.delete("/system/{key}")
async def delete_system_setting(key: str) -> Dict[str, Any]:
    """
    Delete a system setting
    
    Args:
        key: Setting key to delete
        
    Returns:
        Deletion result
    """
    try:
        # Check if setting exists
        setting = await settings_crud.get_setting(key)
        if not setting:
            raise HTTPException(
                status_code=404,
                detail=f"Setting '{key}' not found"
            )
        
        # Delete setting
        await settings_crud.delete_setting(key)
        
        result = {
            "success": True,
            "deleted_key": key,
            "deleted_at": datetime.now().isoformat(),
            "message": f"System setting '{key}' deleted successfully"
        }
        
        main_logger.info(f"System setting deleted: {key}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        main_logger.error(f"Error deleting system setting {key}: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# Configuration Reset
@router.post("/reset")
async def reset_to_defaults() -> Dict[str, Any]:
    """
    Reset all settings to default values
    
    Returns:
        Reset result
    """
    try:
        # This would reset all configurations to defaults
        # For safety, we'll just return the operation info without actually resetting
        
        result = {
            "success": False,  # Set to False for safety
            "message": "Configuration reset is disabled for safety. Please update settings individually.",
            "timestamp": datetime.now().isoformat(),
            "note": "To reset settings, delete individual settings and restart the application"
        }
        
        main_logger.warning("Configuration reset attempted but blocked for safety")
        return result
        
    except Exception as e:
        main_logger.error(f"Error in reset operation: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# Configuration Export/Import
@router.get("/export")
async def export_configuration() -> Dict[str, Any]:
    """
    Export current configuration
    
    Returns:
        Exportable configuration data
    """
    try:
        # Get current settings
        current_settings = await get_current_settings()
        
        # Add export metadata
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "export_version": "1.0",
            "system_info": {
                "phase": "7.1",
                "components": ["trading_engine", "risk_manager", "unified_analyzer"]
            },
            "configuration": current_settings
        }
        
        main_logger.info("Configuration exported")
        return export_data
        
    except Exception as e:
        main_logger.error(f"Error exporting configuration: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )