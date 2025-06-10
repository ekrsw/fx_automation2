"""
MT5 Connection Control API endpoints
"""

from fastapi import APIRouter, HTTPException, status
from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel

from app.mt5.connection import get_mt5_connection
from app.config import get_settings
from app.utils.logger import mt5_logger

router = APIRouter()


class MT5ConnectRequest(BaseModel):
    login: Optional[int] = None
    password: Optional[str] = None
    server: Optional[str] = None


@router.post("/connect")
async def connect_mt5(request: MT5ConnectRequest = None) -> Dict[str, Any]:
    """
    Connect to MT5 server
    
    Args:
        request: Optional connection parameters
        
    Returns:
        Connection result
    """
    try:
        mt5_connection = get_mt5_connection()
        settings = get_settings()
        
        # Use provided credentials or fall back to settings
        login = request.login if request and request.login else settings.mt5_login
        password = request.password if request and request.password else settings.mt5_password
        server = request.server if request and request.server else settings.mt5_server
        
        if not all([login, password, server]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing MT5 credentials. Please provide login, password, and server."
            )
        
        # Attempt connection
        success = await mt5_connection.connect(login, password, server)
        
        if success:
            connection_status = mt5_connection.get_connection_info()
            terminal_info = mt5_connection.get_terminal_info()
            account_info = mt5_connection.get_account_info()
            
            result = {
                "success": True,
                "message": "Successfully connected to MT5",
                "connection_status": connection_status,
                "terminal_info": terminal_info,
                "account_info": account_info,
                "connected_at": datetime.now().isoformat()
            }
            
            mt5_logger.info(f"MT5 connection established via API - Login: {login}, Server: {server}")
            return result
        else:
            status_info = mt5_connection.get_connection_info()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to connect to MT5: {status_info.get('error_message', 'Unknown error')}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        mt5_logger.error(f"Error in MT5 connect API: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}"
        )


@router.post("/disconnect")
async def disconnect_mt5() -> Dict[str, Any]:
    """
    Disconnect from MT5 server
    
    Returns:
        Disconnection result
    """
    try:
        mt5_connection = get_mt5_connection()
        
        success = await mt5_connection.disconnect()
        
        if success:
            result = {
                "success": True,
                "message": "Successfully disconnected from MT5",
                "disconnected_at": datetime.now().isoformat()
            }
            mt5_logger.info("MT5 disconnected via API")
            return result
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to disconnect from MT5"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        mt5_logger.error(f"Error in MT5 disconnect API: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}"
        )


@router.get("/status")
async def get_mt5_status() -> Dict[str, Any]:
    """
    Get current MT5 connection status
    
    Returns:
        MT5 connection status
    """
    try:
        mt5_connection = get_mt5_connection()
        
        connection_status = mt5_connection.get_connection_info()
        
        result = {
            "connection_status": connection_status,
            "is_connected": mt5_connection.is_connected(),
            "checked_at": datetime.now().isoformat()
        }
        
        # Add terminal and account info if connected
        if mt5_connection.is_connected():
            terminal_info = mt5_connection.get_terminal_info()
            account_info = mt5_connection.get_account_info()
            
            if terminal_info:
                result["terminal_info"] = terminal_info
            if account_info:
                result["account_info"] = account_info
        
        return result
        
    except Exception as e:
        mt5_logger.error(f"Error getting MT5 status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}"
        )


@router.post("/reconnect")
async def reconnect_mt5() -> Dict[str, Any]:
    """
    Reconnect to MT5 server
    
    Returns:
        Reconnection result
    """
    try:
        mt5_connection = get_mt5_connection()
        
        success = await mt5_connection.reconnect()
        
        if success:
            connection_status = mt5_connection.get_connection_info()
            result = {
                "success": True,
                "message": "Successfully reconnected to MT5",
                "connection_status": connection_status,
                "reconnected_at": datetime.now().isoformat()
            }
            mt5_logger.info("MT5 reconnected via API")
            return result
        else:
            status_info = mt5_connection.get_connection_info()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to reconnect to MT5: {status_info.get('error_message', 'Unknown error')}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        mt5_logger.error(f"Error in MT5 reconnect API: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}"
        )


@router.post("/health-check")
async def health_check_mt5() -> Dict[str, Any]:
    """
    Perform MT5 health check
    
    Returns:
        Health check result
    """
    try:
        mt5_connection = get_mt5_connection()
        
        health_ok = await mt5_connection.health_check()
        
        result = {
            "health_ok": health_ok,
            "message": "MT5 connection is healthy" if health_ok else "MT5 connection has issues",
            "checked_at": datetime.now().isoformat(),
            "connection_status": mt5_connection.get_connection_info()
        }
        
        return result
        
    except Exception as e:
        mt5_logger.error(f"Error in MT5 health check: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}"
        )