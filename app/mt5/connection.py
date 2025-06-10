"""
MetaTrader 5 connection management
"""

import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass

from app.config import get_settings
from app.utils.logger import mt5_logger


@dataclass
class MT5ConnectionStatus:
    """MT5 connection status"""
    connected: bool = False
    login: Optional[int] = None
    server: Optional[str] = None
    last_heartbeat: Optional[datetime] = None
    connection_retry_count: int = 0
    error_message: Optional[str] = None


class MT5Connection:
    """
    MetaTrader 5 connection manager
    
    Note: This is a mock implementation since MetaTrader5 package
    is not available on macOS. In a Windows environment, replace
    this with actual MT5 API calls.
    """
    
    def __init__(self):
        self.status = MT5ConnectionStatus()
        self.settings = get_settings()
        self._connection_lock = asyncio.Lock()
    
    def get_status(self) -> MT5ConnectionStatus:
        """Get current connection status"""
        return self.status
        
    async def connect(
        self, 
        login: Optional[int] = None, 
        password: Optional[str] = None, 
        server: Optional[str] = None
    ) -> bool:
        """
        Establish connection to MT5
        
        Args:
            login: MT5 login number
            password: MT5 password
            server: MT5 server name
            
        Returns:
            True if connection successful, False otherwise
        """
        async with self._connection_lock:
            try:
                # Use provided credentials or fall back to settings
                login = login or self.settings.mt5_login
                password = password or self.settings.mt5_password
                server = server or self.settings.mt5_server
                
                if not all([login, password, server]):
                    self.status.error_message = "Missing MT5 credentials"
                    mt5_logger.error("Missing MT5 credentials in configuration")
                    return False
                
                mt5_logger.info(f"Attempting to connect to MT5 server: {server}")
                
                # Mock implementation - replace with actual MT5 connection
                # import MetaTrader5 as mt5
                # 
                # if not mt5.initialize():
                #     self.status.error_message = "Failed to initialize MT5"
                #     mt5_logger.error("Failed to initialize MT5")
                #     return False
                # 
                # if not mt5.login(login, password, server):
                #     error_code = mt5.last_error()
                #     self.status.error_message = f"Login failed: {error_code}"
                #     mt5_logger.error(f"MT5 login failed: {error_code}")
                #     return False
                
                # Mock successful connection
                await asyncio.sleep(0.1)  # Simulate connection delay
                
                self.status.connected = True
                self.status.login = login
                self.status.server = server
                self.status.last_heartbeat = datetime.now()
                self.status.connection_retry_count = 0
                self.status.error_message = None
                
                mt5_logger.info(f"Successfully connected to MT5 - Login: {login}, Server: {server}")
                return True
                
            except Exception as e:
                self.status.error_message = str(e)
                self.status.connection_retry_count += 1
                mt5_logger.error(f"MT5 connection error: {e}")
                return False
    
    async def disconnect(self) -> bool:
        """
        Disconnect from MT5
        
        Returns:
            True if disconnection successful
        """
        async with self._connection_lock:
            try:
                if not self.status.connected:
                    return True
                
                # Mock implementation - replace with actual MT5 disconnection
                # import MetaTrader5 as mt5
                # mt5.shutdown()
                
                self.status.connected = False
                self.status.login = None
                self.status.server = None
                self.status.last_heartbeat = None
                self.status.error_message = None
                
                mt5_logger.info("Disconnected from MT5")
                return True
                
            except Exception as e:
                mt5_logger.error(f"MT5 disconnection error: {e}")
                return False
    
    async def health_check(self) -> bool:
        """
        Check connection health
        
        Returns:
            True if connection is healthy
        """
        try:
            if not self.status.connected:
                return False
            
            # Mock implementation - replace with actual MT5 health check
            # import MetaTrader5 as mt5
            # 
            # # Try to get account info as health check
            # account_info = mt5.account_info()
            # if account_info is None:
            #     return False
            
            # Mock successful health check
            await asyncio.sleep(0.05)  # Simulate check delay
            
            self.status.last_heartbeat = datetime.now()
            return True
            
        except Exception as e:
            mt5_logger.error(f"MT5 health check error: {e}")
            self.status.error_message = str(e)
            return False
    
    async def reconnect(self) -> bool:
        """
        Attempt to reconnect to MT5
        
        Returns:
            True if reconnection successful
        """
        mt5_logger.info("Attempting MT5 reconnection...")
        
        # Disconnect first
        await self.disconnect()
        
        # Wait before reconnecting
        await asyncio.sleep(1)
        
        # Attempt to reconnect
        return await self.connect()
    
    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get current connection information
        
        Returns:
            Dictionary with connection details
        """
        return {
            "connected": self.status.connected,
            "login": self.status.login,
            "server": self.status.server,
            "last_heartbeat": self.status.last_heartbeat.isoformat() if self.status.last_heartbeat else None,
            "retry_count": self.status.connection_retry_count,
            "error_message": self.status.error_message
        }
    
    def is_connected(self) -> bool:
        """
        Check if currently connected
        
        Returns:
            True if connected
        """
        return self.status.connected
    
    def get_terminal_info(self) -> Optional[Dict[str, Any]]:
        """
        Get MT5 terminal information
        
        Returns:
            Terminal information dict or None
        """
        if not self.status.connected:
            return None
        
        # Mock terminal info for development
        return {
            "name": "MetaTrader 5",
            "build": "3390",
            "path": "/path/to/terminal",
            "company": "Demo Terminal",
            "language": "English",
            "connected": True
        }
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        Get MT5 account information
        
        Returns:
            Account information dict or None
        """
        if not self.status.connected:
            return None
        
        # Mock account info for development
        return {
            "login": self.status.login,
            "balance": 10000.00,
            "currency": "USD",
            "leverage": 100,
            "equity": 10000.00,
            "margin": 0.00,
            "free_margin": 10000.00,
            "company": "Demo Broker",
            "name": "Demo Account",
            "server": self.status.server
        }


# Global connection instance
_mt5_connection: Optional[MT5Connection] = None


def get_mt5_connection() -> MT5Connection:
    """
    Get global MT5 connection instance
    
    Returns:
        MT5 connection instance
    """
    global _mt5_connection
    if _mt5_connection is None:
        _mt5_connection = MT5Connection()
    return _mt5_connection