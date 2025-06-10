"""
Simple Strategy Test

Quick test to verify strategy testing framework is operational.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add app to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.analysis.strategy_engine.signal_tester import SignalTester, TestCase, TestType, MarketCondition
from app.utils.logger import analysis_logger


def generate_simple_price_data(symbol: str, periods: int = 100) -> pd.DataFrame:
    """Generate simple sample price data for testing"""
    
    # Start with base price
    base_price = 1.1000 if 'USD' in symbol else 150.0 if 'JPY' in symbol else 1.3000
    
    # Generate time index
    start_date = datetime.now() - timedelta(hours=periods)
    time_index = pd.date_range(start=start_date, periods=periods, freq='h')
    
    # Generate simple uptrend
    trend = np.linspace(0, 0.02, periods)  # 2% uptrend
    volatility = 0.001  # 0.1% hourly volatility
    
    # Generate random walks
    random_walks = np.random.normal(0, volatility, periods)
    price_changes = trend + random_walks
    
    # Calculate prices
    prices = base_price * np.exp(np.cumsum(price_changes))
    
    # Generate OHLC data
    data = []
    for i, price in enumerate(prices):
        open_price = prices[i-1] if i > 0 else price
        close_price = price
        high_price = max(open_price, close_price) * 1.0003
        low_price = min(open_price, close_price) * 0.9997
        
        data.append({
            'timestamp': time_index[i],
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': np.random.randint(1000, 5000)
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df


async def run_simple_test():
    """Run a simple signal generation test"""
    print("=" * 50)
    print("SIMPLE STRATEGY TEST")
    print("=" * 50)
    
    signal_tester = SignalTester()
    
    # Create one simple test case
    price_data = generate_simple_price_data('EURUSD', 100)
    
    test_case = TestCase(
        test_id="simple_test_1",
        test_type=TestType.ACCURACY_TEST,
        market_condition=MarketCondition.TRENDING_UP,
        symbol='EURUSD',
        timeframe="H1",
        price_data=price_data,
        expected_signal="BUY",
        description="Simple accuracy test for EURUSD uptrend"
    )
    
    # Run test suite with single test
    try:
        suite_result = await signal_tester.run_test_suite("Simple Test Suite", [test_case])
        result = suite_result.test_results[0] if suite_result.test_results else None
        
        if not result:
            print("No test results generated")
            return False
        
        print(f"\nTest Results:")
        print(f"Test ID: {result.test_id}")
        print(f"Passed: {result.passed}")
        print(f"Execution Time: {result.execution_time_ms:.1f}ms")
        print(f"Signal Generated: {result.signal_generated}")
        print(f"Accuracy: {result.signal_accuracy:.2f}")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Quality Score: {result.quality_score:.2f}")
        print(f"Direction Correct: {result.direction_correct}")
        print(f"Timing Accuracy: {result.timing_accuracy:.2f}")
        
        if result.error_message:
            print(f"Error: {result.error_message}")
        
        print("\nStrategy testing framework operational!")
        return True
        
    except Exception as e:
        print(f"Error in simple test: {e}")
        analysis_logger.error(f"Error in simple strategy test: {e}")
        return False


async def main():
    """Run simple strategy test"""
    print("Starting Simple Strategy Test")
    
    try:
        success = await run_simple_test()
        
        if success:
            print("\n✅ Simple strategy test completed successfully!")
            print("Strategy engine is ready for comprehensive testing.")
        else:
            print("\n❌ Simple strategy test failed!")
            print("Check logs for details.")
            
        return success
        
    except Exception as e:
        print(f"Error in main: {e}")
        analysis_logger.error(f"Error in simple strategy test main: {e}")
        return False


if __name__ == "__main__":
    # Run the simple strategy test
    result = asyncio.run(main())
    
    if result:
        print("\n✅ Test completed successfully!")
        exit(0)
    else:
        print("\n❌ Test failed!")
        exit(1)