"""
Real MetaTrader 5 connection implementation for Windows

This file contains the actual MT5 implementation that should be used
when running on Windows with MetaTrader5 package installed.

To use this implementation:
1. Install MetaTrader5: pip install MetaTrader5
2. Replace the mock implementation in connection.py with this code
3. Configure actual MT5 credentials in settings
"""

import asyncio
from datetime import datetime
from typing import Optional, Dict, Any

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    # Create mock mt5 object for development on macOS
    class MockMT5:
        @staticmethod
        def initialize(): return False
        @staticmethod
        def login(*args): return False
        @staticmethod
        def shutdown(): pass
        @staticmethod
        def last_error(): return (0, "MT5 not available")
        @staticmethod
        def terminal_info(): return None
        @staticmethod
        def account_info(): return None
    mt5 = MockMT5()

from app.config import get_settings
from app.utils.logger import mt5_logger
from app.mt5.connection import MT5ConnectionStatus


class RealMT5Connection:
    """
    Real MetaTrader 5 connection implementation
    
    This class provides actual MT5 connectivity when running on Windows
    with the MetaTrader5 package installed.
    """
    
    def __init__(self):
        self.status = MT5ConnectionStatus()
        self.settings = get_settings()
        self._connection_lock = asyncio.Lock()
        self._initialized = False
        
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
        Establish connection to real MT5
        
        Args:
            login: MT5 login number
            password: MT5 password
            server: MT5 server name
            
        Returns:
            True if connection successful, False otherwise
        """
        if not MT5_AVAILABLE:
            self.status.error_message = "MetaTrader5 package not available"
            mt5_logger.error("MetaTrader5 package not installed")
            return False
        
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
                
                mt5_logger.info(f"Attempting to connect to real MT5 server: {server}")
                
                # Initialize MT5
                if not self._initialized:
                    if not mt5.initialize():
                        error_code = mt5.last_error()
                        self.status.error_message = f"Failed to initialize MT5: {error_code}"
                        mt5_logger.error(f"Failed to initialize MT5: {error_code}")
                        return False
                    self._initialized = True
                
                # Login to MT5
                if not mt5.login(login, password, server):
                    error_code = mt5.last_error()
                    self.status.error_message = f"Login failed: {error_code}"
                    mt5_logger.error(f"MT5 login failed: {error_code}")
                    return False
                
                # Update status
                self.status.connected = True
                self.status.login = login
                self.status.server = server
                self.status.last_heartbeat = datetime.now()
                self.status.connection_retry_count = 0
                self.status.error_message = None
                
                mt5_logger.info(f"Successfully connected to real MT5 - Login: {login}, Server: {server}")
                
                # Log terminal and account info
                terminal_info = mt5.terminal_info()
                account_info = mt5.account_info()
                
                if terminal_info:
                    mt5_logger.info(f"Terminal: {terminal_info.name}, Build: {terminal_info.build}")
                
                if account_info:
                    mt5_logger.info(f"Account: {account_info.login}, Balance: {account_info.balance}")
                
                return True
                
            except Exception as e:
                self.status.error_message = str(e)
                self.status.connection_retry_count += 1
                mt5_logger.error(f"Real MT5 connection error: {e}")
                return False
    
    async def disconnect(self) -> bool:
        """Disconnect from MT5"""
        try:
            if self._initialized and MT5_AVAILABLE:
                mt5.shutdown()
                self._initialized = False
            
            self.status.connected = False
            self.status.last_heartbeat = None
            mt5_logger.info("Disconnected from MT5")
            return True
            
        except Exception as e:
            mt5_logger.error(f"Error disconnecting from MT5: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if currently connected to MT5"""
        if not MT5_AVAILABLE or not self._initialized:
            return False
        
        # Check if still logged in
        account_info = mt5.account_info()
        connected = account_info is not None
        
        self.status.connected = connected
        if connected:
            self.status.last_heartbeat = datetime.now()
        
        return connected
    
    async def reconnect(self) -> bool:
        """Attempt to reconnect to MT5"""
        mt5_logger.info("Attempting to reconnect to MT5...")
        
        # First disconnect
        await self.disconnect()
        
        # Small delay
        await asyncio.sleep(1)
        
        # Reconnect
        return await self.connect()
    
    async def health_check(self) -> bool:
        """Perform health check on MT5 connection"""
        try:
            if not self.is_connected():
                return False
            
            # Try to get account info as a health check
            account_info = mt5.account_info()
            if account_info is None:
                mt5_logger.warning("Health check failed - cannot get account info")
                return False
            
            # Update heartbeat
            self.status.last_heartbeat = datetime.now()
            return True
            
        except Exception as e:
            mt5_logger.error(f"Health check error: {e}")
            return False
    
    def get_terminal_info(self) -> Optional[Dict[str, Any]]:
        """Get MT5 terminal information"""
        if not MT5_AVAILABLE or not self.is_connected():
            return None
        
        try:
            terminal_info = mt5.terminal_info()
            if terminal_info:
                return terminal_info._asdict()
            return None
            
        except Exception as e:
            mt5_logger.error(f"Error getting terminal info: {e}")
            return None
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get MT5 account information"""
        if not MT5_AVAILABLE or not self.is_connected():
            return None
        
        try:
            account_info = mt5.account_info()
            if account_info:
                return account_info._asdict()
            return None
            
        except Exception as e:
            mt5_logger.error(f"Error getting account info: {e}")
            return None


# Instructions for integration:
"""
To integrate this real MT5 implementation:

1. On Windows, install MetaTrader5:
   pip install MetaTrader5

2. Replace the mock implementation in app/mt5/connection.py:
   # Replace MT5Connection class with RealMT5Connection
   # Or use conditional import based on platform
   
   import sys
   if sys.platform == 'win32':
       from app.mt5.real_connection import RealMT5Connection as MT5Connection
   else:
       # Keep existing mock implementation
       pass

3. Configure real credentials in app/config.py:
   mt5_login: Optional[int] = int(os.getenv('MT5_LOGIN'))
   mt5_password: Optional[str] = os.getenv('MT5_PASSWORD')
   mt5_server: Optional[str] = os.getenv('MT5_SERVER')

4. Create .env file with your credentials:
   MT5_LOGIN=your_actual_login
   MT5_PASSWORD=your_actual_password
   MT5_SERVER=your_actual_server

5. Test the connection:
   python scripts/test_mt5_connection.py
"""