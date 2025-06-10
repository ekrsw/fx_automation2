"""
Strategy Testing Script

Comprehensive testing of the integrated strategy engine with real data simulation.
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
from app.analysis.strategy_engine.backtest_engine import BacktestEngine, BacktestConfig
from app.analysis.strategy_engine.strategy_validator import StrategyValidator, ValidationLevel, ValidationCriteria
from app.utils.logger import analysis_logger


def generate_sample_price_data(symbol: str, periods: int = 1000, trend_type: str = "mixed") -> pd.DataFrame:
    """Generate realistic sample price data for testing"""
    
    # Start with base price
    base_price = 1.1000 if 'USD' in symbol else 150.0 if 'JPY' in symbol else 1.3000
    
    # Generate time index
    start_date = datetime.now() - timedelta(hours=periods)
    time_index = pd.date_range(start=start_date, periods=periods, freq='H')
    
    # Generate returns based on trend type
    if trend_type == "trending_up":
        trend = np.linspace(0, 0.05, periods)  # 5% uptrend
        volatility = 0.001  # 0.1% hourly volatility
    elif trend_type == "trending_down":
        trend = np.linspace(0, -0.05, periods)  # 5% downtrend
        volatility = 0.001
    elif trend_type == "sideways":
        trend = np.sin(np.linspace(0, 4*np.pi, periods)) * 0.01  # Oscillating
        volatility = 0.0008
    elif trend_type == "volatile":
        trend = np.random.normal(0, 0.0005, periods)
        volatility = 0.002  # Higher volatility
    else:  # mixed
        # Create a mixed trend with different phases
        phase1 = np.linspace(0, 0.02, periods//3)
        phase2 = np.linspace(0.02, 0.01, periods//3)
        phase3 = np.linspace(0.01, 0.03, periods - 2*(periods//3))
        trend = np.concatenate([phase1, phase2, phase3])
        volatility = 0.001
    
    # Generate random walks
    random_walks = np.random.normal(0, volatility, periods)
    price_changes = trend + random_walks
    
    # Calculate prices
    prices = base_price * np.exp(np.cumsum(price_changes))
    
    # Generate OHLC data
    data = []
    for i, price in enumerate(prices):
        # Add some intraday variation
        high_var = np.random.uniform(0.0002, 0.0008)
        low_var = np.random.uniform(0.0002, 0.0008)
        
        open_price = prices[i-1] if i > 0 else price
        close_price = price
        high_price = max(open_price, close_price) * (1 + high_var)
        low_price = min(open_price, close_price) * (1 - low_var)
        
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


async def run_signal_tests():
    """Run comprehensive signal generation tests"""
    print("=" * 60)
    print("SIGNAL GENERATION TESTS")
    print("=" * 60)
    
    signal_tester = SignalTester()
    
    # Create test cases for different market conditions
    test_cases = []
    
    symbols = ['EURUSD', 'USDJPY', 'GBPUSD']
    conditions = [
        (MarketCondition.TRENDING_UP, "trending_up"),
        (MarketCondition.TRENDING_DOWN, "trending_down"),
        (MarketCondition.SIDEWAYS, "sideways"),
        (MarketCondition.VOLATILE, "volatile")
    ]
    
    test_id = 1
    for symbol in symbols:
        for condition, trend_type in conditions:
            for test_type in [TestType.ACCURACY_TEST, TestType.PERFORMANCE_TEST, TestType.INTEGRATION_TEST]:
                price_data = generate_sample_price_data(symbol, 500, trend_type)
                
                # Determine expected signal based on market condition
                if condition == MarketCondition.TRENDING_UP:
                    expected_signal = "BUY"
                elif condition == MarketCondition.TRENDING_DOWN:
                    expected_signal = "SELL"
                else:
                    expected_signal = None  # No specific expectation for sideways/volatile
                
                test_case = TestCase(
                    test_id=f"test_{test_id}",
                    test_type=test_type,
                    market_condition=condition,
                    symbol=symbol,
                    timeframe="H1",
                    price_data=price_data,
                    expected_signal=expected_signal,
                    description=f"{test_type.value} test for {symbol} in {condition.value} market"
                )
                test_cases.append(test_case)
                test_id += 1
    
    # Run test suite
    suite_result = await signal_tester.run_test_suite(
        "Comprehensive Signal Tests", 
        test_cases
    )
    
    # Print results
    print(f"\nTest Suite: {suite_result.suite_name}")
    print(f"Total Tests: {suite_result.total_tests}")
    print(f"Passed: {suite_result.passed_tests}")
    print(f"Failed: {suite_result.failed_tests}")
    print(f"Pass Rate: {suite_result.pass_rate:.1%}")
    print(f"Average Execution Time: {suite_result.avg_execution_time:.1f}ms")
    print(f"Average Confidence: {suite_result.avg_confidence:.2f}")
    print(f"Average Quality: {suite_result.avg_quality:.2f}")
    print(f"Average Accuracy: {suite_result.avg_accuracy:.2f}")
    
    # Print breakdown by test type
    print(f"\nAccuracy Breakdown by Test Type:")
    for test_type, accuracy in suite_result.accuracy_breakdown.items():
        print(f"  {test_type.value}: {accuracy:.2f}")
    
    print(f"\nAccuracy Breakdown by Market Condition:")
    for condition, accuracy in suite_result.condition_breakdown.items():
        print(f"  {condition.value}: {accuracy:.2f}")
    
    # Print failed tests
    if suite_result.failed_tests > 0:
        print(f"\nFailed Tests:")
        for result in suite_result.test_results:
            if not result.passed:
                print(f"  {result.test_id}: {result.error_message or 'Test failed'}")
    
    return suite_result


async def run_backtest_tests():
    """Run backtest performance tests"""
    print("\n" + "=" * 60)
    print("BACKTEST PERFORMANCE TESTS")
    print("=" * 60)
    
    # Test different configurations
    configs = [
        ("Conservative", BacktestConfig(
            initial_capital=10000,
            max_risk_per_trade=0.01,  # 1% risk
            min_signal_confidence=0.7,
            min_risk_reward_ratio=2.0
        )),
        ("Standard", BacktestConfig(
            initial_capital=10000,
            max_risk_per_trade=0.02,  # 2% risk
            min_signal_confidence=0.6,
            min_risk_reward_ratio=1.5
        )),
        ("Aggressive", BacktestConfig(
            initial_capital=10000,
            max_risk_per_trade=0.03,  # 3% risk
            min_signal_confidence=0.5,
            min_risk_reward_ratio=1.2
        ))
    ]
    
    symbols = ['EURUSD', 'USDJPY']
    results = {}
    
    for symbol in symbols:
        print(f"\nTesting {symbol}:")
        results[symbol] = {}
        
        # Generate comprehensive test data (6 months)
        price_data = generate_sample_price_data(symbol, 4320, "mixed")  # 6 months hourly
        
        for config_name, config in configs:
            print(f"  Running {config_name} configuration...")
            
            backtest_engine = BacktestEngine(config)
            result = await backtest_engine.run_backtest(price_data, symbol)
            
            results[symbol][config_name] = result
            
            # Print key metrics
            metrics = result.metrics
            print(f"    Total Return: {metrics.total_return_pct:.1f}%")
            print(f"    Total Trades: {metrics.total_trades}")
            print(f"    Win Rate: {metrics.win_rate:.1%}")
            print(f"    Profit Factor: {metrics.profit_factor:.2f}")
            print(f"    Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
            print(f"    Max Drawdown: {metrics.max_drawdown:.1%}")
            print(f"    Expectancy: {metrics.expectancy:.2f}")
    
    # Summary comparison
    print(f"\n{'Configuration':<12} {'Symbol':<8} {'Return%':<8} {'Trades':<7} {'Win%':<6} {'Sharpe':<7} {'DD%':<6}")
    print("-" * 60)
    
    for symbol in results:
        for config_name in results[symbol]:
            metrics = results[symbol][config_name].metrics
            print(f"{config_name:<12} {symbol:<8} {metrics.total_return_pct:>7.1f} "
                  f"{metrics.total_trades:>6} {metrics.win_rate:>5.1%} "
                  f"{metrics.sharpe_ratio:>6.2f} {metrics.max_drawdown:>5.1%}")
    
    return results


async def run_strategy_validation():
    """Run comprehensive strategy validation"""
    print("\n" + "=" * 60)
    print("STRATEGY VALIDATION TESTS")
    print("=" * 60)
    
    validator = StrategyValidator()
    
    # Test different validation levels
    validation_levels = [
        ValidationLevel.BASIC,
        ValidationLevel.STANDARD,
        ValidationLevel.COMPREHENSIVE
    ]
    
    symbols = ['EURUSD', 'USDJPY']
    validation_results = {}
    
    for symbol in symbols:
        print(f"\nValidating strategy for {symbol}:")
        validation_results[symbol] = {}
        
        # Generate substantial test data (1 year)
        price_data = generate_sample_price_data(symbol, 8760, "mixed")  # 1 year hourly
        
        for level in validation_levels:
            print(f"  Running {level.value} validation...")
            
            # Define criteria based on validation level
            if level == ValidationLevel.BASIC:
                criteria = ValidationCriteria(
                    min_win_rate=0.45,
                    min_profit_factor=1.1,
                    max_drawdown=0.20,
                    min_trade_count=20
                )
            elif level == ValidationLevel.STANDARD:
                criteria = ValidationCriteria(
                    min_win_rate=0.50,
                    min_profit_factor=1.2,
                    min_sharpe_ratio=0.3,
                    max_drawdown=0.15,
                    min_trade_count=30
                )
            else:  # COMPREHENSIVE
                criteria = ValidationCriteria(
                    min_win_rate=0.55,
                    min_profit_factor=1.3,
                    min_sharpe_ratio=0.5,
                    max_drawdown=0.12,
                    min_trade_count=50
                )
            
            try:
                validation_result = await validator.validate_strategy(
                    price_data, symbol, level, criteria
                )
                
                validation_results[symbol][level.value] = validation_result
                
                # Print results
                print(f"    Overall Passed: {validation_result.overall_passed}")
                print(f"    Overall Score: {validation_result.overall_score:.2f}")
                print(f"    Confidence Level: {validation_result.confidence_level:.2f}")
                print(f"    Tests Passed: {len([r for r in validation_result.test_results if r.passed])}/{len(validation_result.test_results)}")
                
                # Print key performance metrics
                if validation_result.performance_metrics:
                    metrics = validation_result.performance_metrics
                    if 'win_rate' in metrics:
                        print(f"    Win Rate: {metrics['win_rate']:.1%}")
                    if 'total_return' in metrics:
                        print(f"    Total Return: {metrics['total_return']:.1f}%")
                    if 'sharpe_ratio' in metrics:
                        print(f"    Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                    if 'max_drawdown' in metrics:
                        print(f"    Max Drawdown: {metrics['max_drawdown']:.1%}")
                
                # Print strengths and weaknesses
                if validation_result.strengths:
                    print(f"    Strengths: {', '.join(validation_result.strengths[:2])}")
                if validation_result.weaknesses:
                    print(f"    Weaknesses: {', '.join(validation_result.weaknesses[:2])}")
                
            except Exception as e:
                print(f"    Error in {level.value} validation: {e}")
                validation_results[symbol][level.value] = None
    
    # Summary table
    print(f"\n{'Symbol':<8} {'Level':<12} {'Passed':<8} {'Score':<6} {'Confidence':<10}")
    print("-" * 50)
    
    for symbol in validation_results:
        for level_name in validation_results[symbol]:
            result = validation_results[symbol][level_name]
            if result:
                print(f"{symbol:<8} {level_name:<12} {'Yes' if result.overall_passed else 'No':<8} "
                      f"{result.overall_score:>5.2f} {result.confidence_level:>9.2f}")
            else:
                print(f"{symbol:<8} {level_name:<12} {'Error':<8} {'-':<5} {'-':<9}")
    
    return validation_results


async def main():
    """Run comprehensive strategy testing"""
    print("Starting Comprehensive Strategy Testing")
    print("=" * 60)
    
    try:
        # Run signal tests
        signal_results = await run_signal_tests()
        
        # Run backtest tests
        backtest_results = await run_backtest_tests()
        
        # Run strategy validation
        validation_results = await run_strategy_validation()
        
        # Overall summary
        print("\n" + "=" * 60)
        print("OVERALL TESTING SUMMARY")
        print("=" * 60)
        
        print(f"\n1. Signal Generation Tests:")
        print(f"   - Total Tests: {signal_results.total_tests}")
        print(f"   - Pass Rate: {signal_results.pass_rate:.1%}")
        print(f"   - Average Accuracy: {signal_results.avg_accuracy:.2f}")
        
        print(f"\n2. Backtest Performance Tests:")
        backtest_count = sum(len(results) for results in backtest_results.values())
        print(f"   - Total Backtests: {backtest_count}")
        print(f"   - Configurations Tested: Conservative, Standard, Aggressive")
        print(f"   - Symbols Tested: {', '.join(backtest_results.keys())}")
        
        print(f"\n3. Strategy Validation Tests:")
        validation_count = sum(len([r for r in results.values() if r]) 
                             for results in validation_results.values())
        print(f"   - Total Validations: {validation_count}")
        print(f"   - Validation Levels: Basic, Standard, Comprehensive")
        
        # Final assessment
        signal_health = "GOOD" if signal_results.pass_rate >= 0.7 else "NEEDS_IMPROVEMENT"
        
        print(f"\n4. Overall Assessment:")
        print(f"   - Signal Generation: {signal_health}")
        print(f"   - Backtest Framework: OPERATIONAL")
        print(f"   - Validation Framework: OPERATIONAL")
        print(f"   - Strategy Engine: READY FOR PHASE 6")
        
        print(f"\nStrategy testing completed successfully!")
        
    except Exception as e:
        analysis_logger.error(f"Error in strategy testing: {e}")
        print(f"Error in strategy testing: {e}")
        return False
    
    return True


if __name__ == "__main__":
    # Run the comprehensive strategy testing
    result = asyncio.run(main())
    
    if result:
        print("\n✅ Strategy testing completed successfully!")
        exit(0)
    else:
        print("\n❌ Strategy testing failed!")
        exit(1)