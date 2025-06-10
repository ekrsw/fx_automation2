#!/usr/bin/env python3
"""
MT5 Demo Account Setup and Connection Test
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.mt5.connection import MT5Connection
from app.utils.logger import mt5_logger


async def test_mt5_demo_connection():
    """Test MT5 demo account connection"""
    
    print("=" * 60)
    print("MT5 Demo Account Connection Test")
    print("=" * 60)
    
    # Get settings
    settings = get_settings()
    
    # Display current configuration
    print(f"\nConfiguration:")
    print(f"MT5 Login: {settings.mt5_login}")
    print(f"MT5 Server: {settings.mt5_server}")
    print(f"Password Set: {'Yes' if settings.mt5_password else 'No'}")
    
    if not all([settings.mt5_login, settings.mt5_password, settings.mt5_server]):
        print("\n❌ ERROR: Missing MT5 credentials!")
        print("Please update the .env file with your demo account details:")
        print("1. Get a demo account from MetaQuotes or any broker")
        print("2. Update MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER in .env file")
        return False
    
    # Initialize MT5 connection
    print(f"\n🔌 Attempting to connect to MT5...")
    mt5_connection = MT5Connection()
    
    # Try to connect
    success = await mt5_connection.connect(
        login=settings.mt5_login,
        password=settings.mt5_password,
        server=settings.mt5_server
    )
    
    if success:
        print("✅ Successfully connected to MT5!")
        
        # Get connection status
        status = mt5_connection.get_status()
        print(f"\nConnection Details:")
        print(f"Connected: {status.connected}")
        print(f"Login: {status.login}")
        print(f"Server: {status.server}")
        print(f"Last Heartbeat: {status.last_heartbeat}")
        
        # Test health check
        print(f"\n🏥 Performing health check...")
        health_ok = await mt5_connection.health_check()
        if health_ok:
            print("✅ Health check passed")
        else:
            print("❌ Health check failed")
        
        # Get terminal info if available
        try:
            terminal_info = mt5_connection.get_terminal_info()
            if terminal_info:
                print(f"\nTerminal Information:")
                print(f"Name: {terminal_info.get('name', 'N/A')}")
                print(f"Build: {terminal_info.get('build', 'N/A')}")
                print(f"Path: {terminal_info.get('path', 'N/A')}")
        except Exception as e:
            print(f"Terminal info not available: {e}")
        
        # Get account info if available
        try:
            account_info = mt5_connection.get_account_info()
            if account_info:
                print(f"\nAccount Information:")
                print(f"Login: {account_info.get('login', 'N/A')}")
                print(f"Balance: {account_info.get('balance', 'N/A')}")
                print(f"Currency: {account_info.get('currency', 'N/A')}")
                print(f"Leverage: {account_info.get('leverage', 'N/A')}")
                print(f"Company: {account_info.get('company', 'N/A')}")
        except Exception as e:
            print(f"Account info not available: {e}")
        
        # Disconnect
        print(f"\n🔌 Disconnecting...")
        await mt5_connection.disconnect()
        print("✅ Disconnected successfully")
        
        return True
    else:
        print("❌ Failed to connect to MT5")
        status = mt5_connection.get_status()
        if status.error_message:
            print(f"Error: {status.error_message}")
        
        print(f"\n🔍 Troubleshooting Tips:")
        print("1. Verify your demo account credentials are correct")
        print("2. Check if MetaTrader5 terminal is installed (Windows only)")
        print("3. Ensure the server name is correct (e.g., 'MetaQuotes-Demo')")
        print("4. Try different demo servers:")
        print("   - MetaQuotes-Demo")
        print("   - Deriv-Demo") 
        print("   - FTMO-Demo")
        print("   - ICMarkets-Demo")
        print("5. On macOS/Linux: This will use mock connection for development")
        
        return False


def get_demo_account_instructions():
    """Provide instructions for getting a demo account"""
    
    print("\n" + "=" * 60)
    print("How to Get an MT5 Demo Account")
    print("=" * 60)
    
    instructions = """
1. MetaQuotes Demo Account (Recommended):
   - Go to: https://www.mql5.com/en/articles/5096
   - Download MetaTrader 5
   - Open MT5 and select "File" > "Open an Account"
   - Choose "Demo Account" and select a server
   - Fill in your details and get login credentials

2. Broker Demo Accounts:
   - IC Markets: https://www.icmarkets.com/
   - FTMO: https://ftmo.com/
   - Deriv: https://deriv.com/
   - XM: https://www.xm.com/

3. Update Configuration:
   After getting demo credentials, update the .env file:
   
   MT5_LOGIN=your_demo_login_number
   MT5_PASSWORD=your_demo_password
   MT5_SERVER=your_demo_server
   
4. Common Demo Servers:
   - MetaQuotes-Demo
   - ICMarkets-MT5-Demo
   - FTMO-Demo
   - Deriv-Demo
   
5. Test Connection:
   Run this script again to test the connection.
"""
    
    print(instructions)


async def main():
    """Main function"""
    
    try:
        # Test connection
        success = await test_mt5_demo_connection()
        
        if not success:
            get_demo_account_instructions()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Starting MT5 Demo Connection Test...")
    asyncio.run(main())