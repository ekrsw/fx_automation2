"""
MT5 connection testing utilities
"""

import asyncio
import sys
from typing import Dict, Any, Optional
from datetime import datetime

from app.mt5.connection import get_mt5_connection
from app.mt5.data_fetcher import get_data_fetcher
from app.utils.logger import mt5_logger


class MT5ConnectionTester:
    """Test MT5 connection and functionality"""
    
    def __init__(self):
        self.connection = get_mt5_connection()
        self.data_fetcher = get_data_fetcher()
    
    async def run_connection_tests(self) -> Dict[str, Any]:
        """Run comprehensive connection tests"""
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'platform': sys.platform,
            'tests': {}
        }
        
        # Test 1: Basic connection
        mt5_logger.info("Testing basic MT5 connection...")
        try:
            connection_result = await self.connection.connect()
            results['tests']['basic_connection'] = {
                'success': connection_result,
                'status': self.connection.get_status().__dict__
            }
        except Exception as e:
            results['tests']['basic_connection'] = {
                'success': False,
                'error': str(e)
            }
        
        # Test 2: Health check
        mt5_logger.info("Testing health check...")
        try:
            health_result = await self.connection.health_check()
            results['tests']['health_check'] = {
                'success': health_result,
                'details': 'Health check completed'
            }
        except Exception as e:
            results['tests']['health_check'] = {
                'success': False,
                'error': str(e)
            }
        
        # Test 3: Symbol information
        mt5_logger.info("Testing symbol information retrieval...")
        try:
            symbols = ['USDJPY', 'EURJPY', 'GBPJPY']
            symbol_info = {}
            
            for symbol in symbols:
                info = await self.data_fetcher.get_symbol_info(symbol)
                symbol_info[symbol] = info
            
            results['tests']['symbol_info'] = {
                'success': True,
                'symbols': symbol_info
            }
        except Exception as e:
            results['tests']['symbol_info'] = {
                'success': False,
                'error': str(e)
            }
        
        # Test 4: Live price data
        mt5_logger.info("Testing live price data...")
        try:
            symbols = ['USDJPY', 'EURJPY']
            live_prices = await self.data_fetcher.get_live_prices(symbols)
            
            results['tests']['live_prices'] = {
                'success': len(live_prices) > 0,
                'data_count': len(live_prices),
                'symbols': list(live_prices.keys()) if live_prices else []
            }
        except Exception as e:
            results['tests']['live_prices'] = {
                'success': False,
                'error': str(e)
            }
        
        # Test 5: Historical data
        mt5_logger.info("Testing historical data retrieval...")
        try:
            from datetime import timedelta
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=5)
            
            hist_data = await self.data_fetcher.get_historical_data(
                'USDJPY', 'H1', start_date, end_date
            )
            
            results['tests']['historical_data'] = {
                'success': not hist_data.empty if hasattr(hist_data, 'empty') else len(hist_data) > 0,
                'data_points': len(hist_data) if hasattr(hist_data, '__len__') else 0,
                'columns': list(hist_data.columns) if hasattr(hist_data, 'columns') else []
            }
        except Exception as e:
            results['tests']['historical_data'] = {
                'success': False,
                'error': str(e)
            }
        
        # Test 6: OHLC data
        mt5_logger.info("Testing OHLC data retrieval...")
        try:
            ohlc_data = await self.data_fetcher.get_ohlc_data('USDJPY', 'H1', 10)
            
            results['tests']['ohlc_data'] = {
                'success': not ohlc_data.empty if hasattr(ohlc_data, 'empty') else len(ohlc_data) > 0,
                'data_points': len(ohlc_data) if hasattr(ohlc_data, '__len__') else 0,
                'columns': list(ohlc_data.columns) if hasattr(ohlc_data, 'columns') else []
            }
        except Exception as e:
            results['tests']['ohlc_data'] = {
                'success': False,
                'error': str(e)
            }
        
        # Summary
        successful_tests = sum(1 for test in results['tests'].values() if test.get('success', False))
        total_tests = len(results['tests'])
        
        results['summary'] = {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
            'overall_status': 'PASS' if successful_tests == total_tests else 'PARTIAL' if successful_tests > 0 else 'FAIL'
        }
        
        return results
    
    async def test_real_mt5_availability(self) -> Dict[str, Any]:
        """Test if real MT5 package is available"""
        result = {
            'mt5_package_available': False,
            'platform': sys.platform,
            'can_import': False,
            'error': None
        }
        
        try:
            import MetaTrader5 as mt5
            result['mt5_package_available'] = True
            result['can_import'] = True
            
            # Try to initialize
            if hasattr(mt5, 'initialize'):
                init_result = mt5.initialize()
                result['can_initialize'] = init_result
                
                if init_result:
                    # Get terminal info
                    terminal_info = mt5.terminal_info()
                    account_info = mt5.account_info()
                    
                    result['terminal_info'] = terminal_info._asdict() if terminal_info else None
                    result['account_info'] = account_info._asdict() if account_info else None
                    
                    # Shutdown
                    mt5.shutdown()
                else:
                    result['error'] = 'Failed to initialize MT5'
            
        except ImportError as e:
            result['error'] = f'MT5 package not available: {str(e)}'
        except Exception as e:
            result['error'] = f'MT5 test error: {str(e)}'
        
        return result
    
    def generate_test_report(self, results: Dict[str, Any]) -> str:
        """Generate formatted test report"""
        report = []
        report.append("=" * 60)
        report.append("MT5 CONNECTION TEST REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {results['timestamp']}")
        report.append(f"Platform: {results['platform']}")
        report.append("")
        
        # Summary
        summary = results.get('summary', {})
        report.append("SUMMARY:")
        report.append(f"  Total Tests: {summary.get('total_tests', 0)}")
        report.append(f"  Successful: {summary.get('successful_tests', 0)}")
        report.append(f"  Success Rate: {summary.get('success_rate', 0):.1%}")
        report.append(f"  Overall Status: {summary.get('overall_status', 'UNKNOWN')}")
        report.append("")
        
        # Individual test results
        report.append("DETAILED RESULTS:")
        for test_name, test_result in results.get('tests', {}).items():
            status = "✅ PASS" if test_result.get('success', False) else "❌ FAIL"
            report.append(f"  {test_name}: {status}")
            
            if not test_result.get('success', False) and 'error' in test_result:
                report.append(f"    Error: {test_result['error']}")
            
            if 'data_count' in test_result:
                report.append(f"    Data Count: {test_result['data_count']}")
            
            if 'data_points' in test_result:
                report.append(f"    Data Points: {test_result['data_points']}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


async def main():
    """Run MT5 connection tests"""
    print("Starting MT5 connection tests...")
    
    tester = MT5ConnectionTester()
    
    # Test real MT5 availability first
    print("\n1. Testing real MT5 package availability...")
    mt5_availability = await tester.test_real_mt5_availability()
    
    print(f"MT5 Package Available: {mt5_availability['mt5_package_available']}")
    print(f"Platform: {mt5_availability['platform']}")
    
    if mt5_availability.get('error'):
        print(f"Error: {mt5_availability['error']}")
    
    # Run connection tests
    print("\n2. Running connection tests...")
    test_results = await tester.run_connection_tests()
    
    # Generate and print report
    report = tester.generate_test_report(test_results)
    print("\n" + report)
    
    # Save report to file
    with open('mt5_test_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\nTest report saved to: mt5_test_report.txt")
    
    return test_results


if __name__ == "__main__":
    asyncio.run(main())