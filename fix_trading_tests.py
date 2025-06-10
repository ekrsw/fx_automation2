#!/usr/bin/env python3
"""
Script to fix trading API tests by correcting async method mocking
"""

import re

def fix_trading_tests():
    """Fix trading API test file by replacing incorrect mock patterns"""
    
    file_path = 'tests/test_api/test_trading_api.py'
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Define replacement patterns for trading engine methods
    replacements = [
        (r'mock_trading_engine\.start_session\.return_value', 'mock_trading_engine.start_trading_session = AsyncMock(return_value'),
        (r'mock_trading_engine\.start_session\.side_effect', 'mock_trading_engine.start_trading_session = AsyncMock(side_effect'),
        (r'mock_trading_engine\.stop_session\.return_value', 'mock_trading_engine.stop_trading_session = AsyncMock(return_value'),
        (r'mock_trading_engine\.pause_session\.return_value', 'mock_trading_engine.pause_trading = AsyncMock(return_value'),
        (r'mock_trading_engine\.resume_session\.return_value', 'mock_trading_engine.resume_trading = AsyncMock(return_value'),
        (r'mock_trading_engine\.get_session_status\.return_value', 'mock_trading_engine.get_trading_status = AsyncMock(return_value'),
        (r'mock_trading_engine\.process_signal\.return_value', 'mock_trading_engine.process_signal = AsyncMock(return_value'),
        
        # Order manager fixes  
        (r'mock_order_manager\.create_order\.return_value', 'mock_order_manager.place_order = AsyncMock(return_value'),
        (r'mock_order_manager\.cancel_order\.return_value', 'mock_order_manager.cancel_order = AsyncMock(return_value'),
        (r'mock_order_manager\.cancel_order\.side_effect', 'mock_order_manager.cancel_order = AsyncMock(side_effect'),
        
        # Position manager fixes
        (r'mock_position_manager\.modify_position\.return_value', 'mock_position_manager.modify_position = AsyncMock(return_value'),
        (r'mock_position_manager\.close_position\.return_value', 'mock_position_manager.close_position = AsyncMock(return_value'),
        (r'mock_position_manager\.close_all_positions\.return_value', 'mock_position_manager.close_all_positions = AsyncMock(return_value'),
        
        # Risk manager fixes
        (r'mock_risk_manager\.activate_emergency_stop\.return_value', 'mock_risk_manager.emergency_stop = AsyncMock(return_value'),
        (r'mock_risk_manager\.deactivate_emergency_stop\.return_value', 'mock_risk_manager.deactivate_emergency_stop = AsyncMock(return_value'),
        (r'mock_risk_manager\.get_risk_status\.return_value', 'mock_risk_manager.get_risk_report = AsyncMock(return_value'),
        
        # Signal generator fixes
        (r'mock_signal_generator\.process_signal\.return_value', 'mock_signal_generator.process_signal = AsyncMock(return_value'),
    ]
    
    # Apply replacements
    for pattern, replacement in replacements:
        content = re.sub(pattern + r'\s*=', replacement + '=', content)
    
    # Fix specific method names that need closing parentheses
    content = re.sub(r'= AsyncMock\(return_value=([^)]+)\)$', r'= AsyncMock(return_value=\1)', content, flags=re.MULTILINE)
    
    # Write back to file
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("Trading API tests fixed successfully!")

if __name__ == "__main__":
    fix_trading_tests()