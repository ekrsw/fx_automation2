#!/usr/bin/env python3
"""
MT5 Connection Test Script

This script can be run to test MT5 connectivity either in mock mode (macOS)
or real mode (Windows with MT5 installed).

Usage:
    python scripts/test_mt5_connection.py
"""

import sys
import os
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.mt5.test_connection import MT5ConnectionTester


def print_instructions():
    """Print setup instructions for different platforms"""
    print("""
MT5 CONNECTION TESTING INSTRUCTIONS
===================================

CURRENT PLATFORM: {platform}

For macOS (Current):
-------------------
- This will run in MOCK mode
- Tests the application logic and mock data generation
- Useful for development and testing without real MT5

For Windows (Real MT5 Testing):
------------------------------
1. Install MetaTrader 5 terminal
2. Create a demo account or use existing account
3. Install Python MT5 package:
   pip install MetaTrader5
4. Run this script on Windows machine
5. Ensure MT5 terminal is running and logged in

Demo Account Setup:
------------------
- Go to https://www.metaquotes.net/en/metatrader5
- Download and install MT5
- Open demo account with any broker (e.g., MetaQuotes Demo)
- Note: Login, Password, Server for configuration

Next Steps:
----------
1. Run this test to verify mock functionality
2. Set up Windows environment for real MT5 testing
3. Configure actual MT5 credentials in environment variables:
   - MT5_LOGIN=your_login
   - MT5_PASSWORD=your_password  
   - MT5_SERVER=your_server
""".format(platform=sys.platform))


async def main():
    print_instructions()
    
    response = input("\nProceed with testing? (y/n): ").lower().strip()
    if response != 'y':
        print("Testing cancelled.")
        return
    
    print("\nStarting MT5 connection tests...")
    print("-" * 50)
    
    try:
        tester = MT5ConnectionTester()
        results = await tester.run_connection_tests()
        
        # Print results
        report = tester.generate_test_report(results)
        print(report)
        
        # Recommendations based on results
        print("\nRECOMMENDATIONS:")
        print("-" * 20)
        
        if sys.platform == 'darwin':  # macOS
            print("✓ Mock testing completed on macOS")
            print("➤ Next: Set up Windows environment for real MT5 testing")
            print("➤ Create demo account and configure credentials")
        
        elif sys.platform == 'win32':  # Windows
            summary = results.get('summary', {})
            if summary.get('overall_status') == 'PASS':
                print("✓ All tests passed! MT5 connection is working")
                print("➤ Ready to proceed with live trading development")
            else:
                print("⚠ Some tests failed. Check the following:")
                print("  - MT5 terminal is running and logged in")
                print("  - Account credentials are correct")
                print("  - Internet connection is stable")
                print("  - MetaTrader5 Python package is installed")
        
        else:
            print("⚠ Unsupported platform for real MT5 testing")
            print("➤ Use Windows for real MT5 integration")
        
        # Save detailed results
        import json
        with open('mt5_test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: mt5_test_results.json")
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())