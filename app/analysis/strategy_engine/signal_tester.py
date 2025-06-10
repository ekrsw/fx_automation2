"""
Signal Generation Testing Module

Comprehensive testing framework for validating signal generation quality,
accuracy, and performance across different market conditions.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio

from app.analysis.strategy_engine.unified_analyzer import UnifiedAnalyzer
from app.analysis.strategy_engine.multi_timeframe_analyzer import MultiTimeframeAnalyzer
from app.analysis.strategy_engine.signal_generator import SignalGenerator, TradingSignal
from app.analysis.strategy_engine.confidence_calculator import ConfidenceCalculator
from app.analysis.strategy_engine.entry_exit_engine import EntryExitEngine
from app.analysis.strategy_engine.stop_loss_calculator import StopLossCalculator
from app.analysis.strategy_engine.take_profit_calculator import TakeProfitCalculator
from app.analysis.strategy_engine.risk_reward_calculator import RiskRewardCalculator
from app.utils.logger import analysis_logger


class TestType(Enum):
    """Types of signal tests"""
    ACCURACY_TEST = "accuracy_test"
    PERFORMANCE_TEST = "performance_test"
    STRESS_TEST = "stress_test"
    ROBUSTNESS_TEST = "robustness_test"
    CONSISTENCY_TEST = "consistency_test"
    INTEGRATION_TEST = "integration_test"


class MarketCondition(Enum):
    """Market condition types for testing"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"


@dataclass
class TestCase:
    """Individual test case"""
    test_id: str
    test_type: TestType
    market_condition: MarketCondition
    symbol: str
    timeframe: str
    price_data: pd.DataFrame
    expected_signal: Optional[str] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Individual test result"""
    test_id: str
    test_type: TestType
    passed: bool
    signal_generated: Optional[TradingSignal]
    
    # Performance metrics
    execution_time_ms: float
    confidence_score: float
    quality_score: float
    
    # Accuracy metrics
    signal_accuracy: float
    direction_correct: bool
    timing_accuracy: float
    
    # Details
    error_message: Optional[str] = None
    analysis_details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class TestSuite:
    """Complete test suite results"""
    suite_name: str
    test_cases: List[TestCase]
    test_results: List[TestResult]
    
    # Summary metrics
    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_rate: float
    
    # Performance summary
    avg_execution_time: float
    avg_confidence: float
    avg_quality: float
    avg_accuracy: float
    
    # Test breakdown
    accuracy_breakdown: Dict[TestType, float]
    condition_breakdown: Dict[MarketCondition, float]
    
    # Execution info
    execution_timestamp: datetime
    total_execution_time: float


class SignalTester:
    """
    Comprehensive signal testing framework
    
    Tests signal generation accuracy, performance, and robustness
    across various market conditions and scenarios.
    """
    
    def __init__(self):
        """Initialize signal tester"""
        
        # Initialize all strategy components
        self.unified_analyzer = UnifiedAnalyzer()
        self.mtf_analyzer = MultiTimeframeAnalyzer()
        self.signal_generator = SignalGenerator()
        self.confidence_calculator = ConfidenceCalculator()
        self.entry_exit_engine = EntryExitEngine()
        self.stop_loss_calculator = StopLossCalculator()
        self.take_profit_calculator = TakeProfitCalculator()
        self.risk_reward_calculator = RiskRewardCalculator()
        
        # Test configuration
        self.config = {
            'performance_thresholds': {
                'max_execution_time_ms': 1000,  # 1 second max
                'min_confidence': 0.5,
                'min_quality': 0.4,
                'min_accuracy': 0.6
            },
            'stress_test_params': {
                'max_data_points': 10000,
                'concurrent_tests': 10,
                'memory_limit_mb': 500
            },
            'accuracy_tolerances': {
                'signal_direction': 0.1,  # 10% tolerance
                'confidence_variance': 0.2,  # 20% tolerance
                'timing_variance': 5  # 5 bars tolerance
            }
        }
        
        # Test data storage
        self.test_suites: List[TestSuite] = []
        self.benchmark_data: Dict[str, Any] = {}
    
    async def run_test_suite(self, suite_name: str, test_cases: List[TestCase]) -> TestSuite:
        """
        Run a complete test suite
        
        Args:
            suite_name: Name of the test suite
            test_cases: List of test cases to execute
            
        Returns:
            Complete test suite results
        """
        try:
            start_time = datetime.utcnow()
            test_results = []
            
            analysis_logger.info(f"Starting test suite: {suite_name} with {len(test_cases)} test cases")
            
            # Run all test cases
            for test_case in test_cases:
                try:
                    result = await self._run_single_test(test_case)
                    test_results.append(result)
                    
                    # Log progress
                    if len(test_results) % 10 == 0:
                        analysis_logger.info(f"Completed {len(test_results)}/{len(test_cases)} tests")
                        
                except Exception as e:
                    # Create failed test result
                    failed_result = TestResult(
                        test_id=test_case.test_id,
                        test_type=test_case.test_type,
                        passed=False,
                        signal_generated=None,
                        execution_time_ms=0.0,
                        confidence_score=0.0,
                        quality_score=0.0,
                        signal_accuracy=0.0,
                        direction_correct=False,
                        timing_accuracy=0.0,
                        error_message=str(e)
                    )
                    test_results.append(failed_result)
                    analysis_logger.error(f"Test {test_case.test_id} failed: {e}")
            
            # Calculate summary metrics
            total_tests = len(test_results)
            passed_tests = sum(1 for r in test_results if r.passed)
            failed_tests = total_tests - passed_tests
            pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0
            
            # Calculate averages
            valid_results = [r for r in test_results if r.passed]
            
            avg_execution_time = np.mean([r.execution_time_ms for r in valid_results]) if valid_results else 0.0
            avg_confidence = np.mean([r.confidence_score for r in valid_results]) if valid_results else 0.0
            avg_quality = np.mean([r.quality_score for r in valid_results]) if valid_results else 0.0
            avg_accuracy = np.mean([r.signal_accuracy for r in valid_results]) if valid_results else 0.0
            
            # Calculate breakdowns
            accuracy_breakdown = self._calculate_accuracy_breakdown(test_results)
            condition_breakdown = self._calculate_condition_breakdown(test_results, test_cases)
            
            end_time = datetime.utcnow()
            total_execution_time = (end_time - start_time).total_seconds()
            
            # Create test suite
            test_suite = TestSuite(
                suite_name=suite_name,
                test_cases=test_cases,
                test_results=test_results,
                total_tests=total_tests,
                passed_tests=passed_tests,
                failed_tests=failed_tests,
                pass_rate=pass_rate,
                avg_execution_time=avg_execution_time,
                avg_confidence=avg_confidence,
                avg_quality=avg_quality,
                avg_accuracy=avg_accuracy,
                accuracy_breakdown=accuracy_breakdown,
                condition_breakdown=condition_breakdown,
                execution_timestamp=start_time,
                total_execution_time=total_execution_time
            )
            
            self.test_suites.append(test_suite)
            
            analysis_logger.info(
                f"Test suite completed: {suite_name} - "
                f"Pass rate: {pass_rate:.1%}, "
                f"Avg accuracy: {avg_accuracy:.2f}, "
                f"Total time: {total_execution_time:.1f}s"
            )
            
            return test_suite
            
        except Exception as e:
            analysis_logger.error(f"Error running test suite {suite_name}: {e}")
            raise
    
    async def _run_single_test(self, test_case: TestCase) -> TestResult:
        """Run a single test case"""
        start_time = datetime.utcnow()
        
        try:
            # Run signal generation based on test type
            if test_case.test_type == TestType.ACCURACY_TEST:
                result = await self._run_accuracy_test(test_case)
            elif test_case.test_type == TestType.PERFORMANCE_TEST:
                result = await self._run_performance_test(test_case)
            elif test_case.test_type == TestType.STRESS_TEST:
                result = await self._run_stress_test(test_case)
            elif test_case.test_type == TestType.ROBUSTNESS_TEST:
                result = await self._run_robustness_test(test_case)
            elif test_case.test_type == TestType.CONSISTENCY_TEST:
                result = await self._run_consistency_test(test_case)
            elif test_case.test_type == TestType.INTEGRATION_TEST:
                result = await self._run_integration_test(test_case)
            else:
                raise ValueError(f"Unknown test type: {test_case.test_type}")
            
            # Calculate execution time
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds() * 1000  # Convert to ms
            result.execution_time_ms = execution_time
            
            return result
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return TestResult(
                test_id=test_case.test_id,
                test_type=test_case.test_type,
                passed=False,
                signal_generated=None,
                execution_time_ms=execution_time,
                confidence_score=0.0,
                quality_score=0.0,
                signal_accuracy=0.0,
                direction_correct=False,
                timing_accuracy=0.0,
                error_message=str(e)
            )
    
    async def _run_accuracy_test(self, test_case: TestCase) -> TestResult:
        """Run accuracy test for signal generation"""
        try:
            # Generate unified analysis
            unified_result = self.unified_analyzer.analyze(
                test_case.price_data, test_case.symbol, test_case.timeframe
            )
            
            # Generate signal
            current_price = test_case.price_data['close'].iloc[-1]
            signal = self.signal_generator.generate_signal_from_unified(
                unified_result, current_price
            )
            
            # Calculate confidence and quality
            confidence_score = self.confidence_calculator.calculate_unified_confidence(
                unified_result, signal
            )
            
            # Evaluate accuracy against expected signal
            signal_accuracy = self._calculate_signal_accuracy(
                signal, test_case.expected_signal, unified_result
            )
            
            direction_correct = self._check_direction_accuracy(
                signal, test_case.expected_signal
            )
            
            timing_accuracy = self._calculate_timing_accuracy(
                signal, test_case.price_data
            )
            
            # Determine if test passed
            passed = (
                signal is not None and
                signal_accuracy >= self.config['performance_thresholds']['min_accuracy'] and
                confidence_score.overall_score >= self.config['performance_thresholds']['min_confidence']
            )
            
            return TestResult(
                test_id=test_case.test_id,
                test_type=test_case.test_type,
                passed=passed,
                signal_generated=signal,
                execution_time_ms=0.0,  # Will be set by caller
                confidence_score=confidence_score.overall_score,
                quality_score=signal.quality_score if signal else 0.0,
                signal_accuracy=signal_accuracy,
                direction_correct=direction_correct,
                timing_accuracy=timing_accuracy,
                analysis_details={
                    'unified_result': {
                        'combined_signal': unified_result.combined_signal,
                        'combined_confidence': unified_result.combined_confidence,
                        'agreement_score': unified_result.agreement_score
                    },
                    'confidence_breakdown': confidence_score.factors if confidence_score else []
                }
            )
            
        except Exception as e:
            return TestResult(
                test_id=test_case.test_id,
                test_type=test_case.test_type,
                passed=False,
                signal_generated=None,
                execution_time_ms=0.0,
                confidence_score=0.0,
                quality_score=0.0,
                signal_accuracy=0.0,
                direction_correct=False,
                timing_accuracy=0.0,
                error_message=str(e)
            )
    
    async def _run_performance_test(self, test_case: TestCase) -> TestResult:
        """Run performance test for signal generation"""
        # Similar to accuracy test but focuses on execution time
        start_time = datetime.utcnow()
        
        try:
            # Generate analysis and signal
            unified_result = self.unified_analyzer.analyze(
                test_case.price_data, test_case.symbol, test_case.timeframe
            )
            
            current_price = test_case.price_data['close'].iloc[-1]
            signal = self.signal_generator.generate_signal_from_unified(
                unified_result, current_price
            )
            
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds() * 1000
            
            # Check performance thresholds
            passed = (
                execution_time <= self.config['performance_thresholds']['max_execution_time_ms'] and
                signal is not None
            )
            
            return TestResult(
                test_id=test_case.test_id,
                test_type=test_case.test_type,
                passed=passed,
                signal_generated=signal,
                execution_time_ms=execution_time,
                confidence_score=unified_result.combined_confidence,
                quality_score=signal.quality_score if signal else 0.0,
                signal_accuracy=1.0 if signal else 0.0,
                direction_correct=True if signal else False,
                timing_accuracy=1.0 if signal else 0.0,
                analysis_details={
                    'performance_metrics': {
                        'execution_time_ms': execution_time,
                        'threshold_ms': self.config['performance_thresholds']['max_execution_time_ms'],
                        'within_threshold': execution_time <= self.config['performance_thresholds']['max_execution_time_ms']
                    }
                }
            )
            
        except Exception as e:
            return TestResult(
                test_id=test_case.test_id,
                test_type=test_case.test_type,
                passed=False,
                signal_generated=None,
                execution_time_ms=0.0,
                confidence_score=0.0,
                quality_score=0.0,
                signal_accuracy=0.0,
                direction_correct=False,
                timing_accuracy=0.0,
                error_message=str(e)
            )
    
    async def _run_stress_test(self, test_case: TestCase) -> TestResult:
        """Run stress test with large datasets or concurrent operations"""
        try:
            # Test with different data sizes
            data_sizes = [100, 500, 1000, 5000]
            results = []
            
            for size in data_sizes:
                if len(test_case.price_data) >= size:
                    test_data = test_case.price_data.tail(size)
                    start_time = datetime.utcnow()
                    
                    unified_result = self.unified_analyzer.analyze(
                        test_data, test_case.symbol, test_case.timeframe
                    )
                    
                    current_price = test_data['close'].iloc[-1]
                    signal = self.signal_generator.generate_signal_from_unified(
                        unified_result, current_price
                    )
                    
                    end_time = datetime.utcnow()
                    execution_time = (end_time - start_time).total_seconds() * 1000
                    
                    results.append({
                        'data_size': size,
                        'execution_time': execution_time,
                        'signal_generated': signal is not None,
                        'confidence': unified_result.combined_confidence
                    })
            
            # Check if all sizes passed performance requirements
            max_time = max(r['execution_time'] for r in results)
            all_generated = all(r['signal_generated'] for r in results)
            
            passed = (
                max_time <= self.config['stress_test_params']['max_data_points'] and
                all_generated
            )
            
            # Use results from largest dataset
            largest_result = results[-1] if results else None
            
            return TestResult(
                test_id=test_case.test_id,
                test_type=test_case.test_type,
                passed=passed,
                signal_generated=None,  # Not storing actual signal for stress test
                execution_time_ms=max_time,
                confidence_score=largest_result['confidence'] if largest_result else 0.0,
                quality_score=0.8 if passed else 0.3,
                signal_accuracy=1.0 if all_generated else 0.0,
                direction_correct=True if all_generated else False,
                timing_accuracy=1.0 if passed else 0.0,
                analysis_details={
                    'stress_test_results': results,
                    'max_execution_time': max_time,
                    'all_sizes_passed': all_generated
                }
            )
            
        except Exception as e:
            return TestResult(
                test_id=test_case.test_id,
                test_type=test_case.test_type,
                passed=False,
                signal_generated=None,
                execution_time_ms=0.0,
                confidence_score=0.0,
                quality_score=0.0,
                signal_accuracy=0.0,
                direction_correct=False,
                timing_accuracy=0.0,
                error_message=str(e)
            )
    
    async def _run_robustness_test(self, test_case: TestCase) -> TestResult:
        """Run robustness test with data variations"""
        try:
            # Test with various data modifications
            modifications = [
                'original',
                'add_noise',
                'remove_outliers', 
                'interpolate_gaps',
                'shift_prices'
            ]
            
            results = []
            
            for modification in modifications:
                modified_data = self._modify_test_data(test_case.price_data, modification)
                
                unified_result = self.unified_analyzer.analyze(
                    modified_data, test_case.symbol, test_case.timeframe
                )
                
                current_price = modified_data['close'].iloc[-1]
                signal = self.signal_generator.generate_signal_from_unified(
                    unified_result, current_price
                )
                
                results.append({
                    'modification': modification,
                    'signal': unified_result.combined_signal,
                    'confidence': unified_result.combined_confidence,
                    'signal_generated': signal is not None
                })
            
            # Check consistency across modifications
            signals = [r['signal'] for r in results if r['signal'] != 'HOLD']
            consistency = len(set(signals)) <= 1 if signals else False
            
            avg_confidence = np.mean([r['confidence'] for r in results])
            generation_rate = sum(r['signal_generated'] for r in results) / len(results)
            
            passed = (
                consistency and
                avg_confidence >= 0.4 and
                generation_rate >= 0.6
            )
            
            return TestResult(
                test_id=test_case.test_id,
                test_type=test_case.test_type,
                passed=passed,
                signal_generated=None,
                execution_time_ms=0.0,
                confidence_score=avg_confidence,
                quality_score=0.8 if consistency else 0.4,
                signal_accuracy=1.0 if consistency else 0.0,
                direction_correct=consistency,
                timing_accuracy=generation_rate,
                analysis_details={
                    'robustness_results': results,
                    'signal_consistency': consistency,
                    'avg_confidence': avg_confidence,
                    'generation_rate': generation_rate
                }
            )
            
        except Exception as e:
            return TestResult(
                test_id=test_case.test_id,
                test_type=test_case.test_type,
                passed=False,
                signal_generated=None,
                execution_time_ms=0.0,
                confidence_score=0.0,
                quality_score=0.0,
                signal_accuracy=0.0,
                direction_correct=False,
                timing_accuracy=0.0,
                error_message=str(e)
            )
    
    async def _run_consistency_test(self, test_case: TestCase) -> TestResult:
        """Run consistency test with multiple executions"""
        try:
            # Run the same analysis multiple times
            num_runs = 5
            results = []
            
            for i in range(num_runs):
                unified_result = self.unified_analyzer.analyze(
                    test_case.price_data, test_case.symbol, test_case.timeframe
                )
                
                current_price = test_case.price_data['close'].iloc[-1]
                signal = self.signal_generator.generate_signal_from_unified(
                    unified_result, current_price
                )
                
                results.append({
                    'run': i + 1,
                    'signal': unified_result.combined_signal,
                    'confidence': unified_result.combined_confidence,
                    'agreement_score': unified_result.agreement_score,
                    'signal_generated': signal is not None
                })
            
            # Check consistency
            signals = [r['signal'] for r in results]
            confidences = [r['confidence'] for r in results]
            
            signal_consistency = len(set(signals)) == 1
            confidence_variance = np.std(confidences)
            
            passed = (
                signal_consistency and
                confidence_variance <= self.config['accuracy_tolerances']['confidence_variance']
            )
            
            return TestResult(
                test_id=test_case.test_id,
                test_type=test_case.test_type,
                passed=passed,
                signal_generated=None,
                execution_time_ms=0.0,
                confidence_score=np.mean(confidences),
                quality_score=0.9 if signal_consistency else 0.3,
                signal_accuracy=1.0 if signal_consistency else 0.0,
                direction_correct=signal_consistency,
                timing_accuracy=1.0 if confidence_variance <= 0.1 else 0.5,
                analysis_details={
                    'consistency_results': results,
                    'signal_consistency': signal_consistency,
                    'confidence_variance': confidence_variance,
                    'unique_signals': list(set(signals))
                }
            )
            
        except Exception as e:
            return TestResult(
                test_id=test_case.test_id,
                test_type=test_case.test_type,
                passed=False,
                signal_generated=None,
                execution_time_ms=0.0,
                confidence_score=0.0,
                quality_score=0.0,
                signal_accuracy=0.0,
                direction_correct=False,
                timing_accuracy=0.0,
                error_message=str(e)
            )
    
    async def _run_integration_test(self, test_case: TestCase) -> TestResult:
        """Run full integration test with all components"""
        try:
            # Full pipeline test
            unified_result = self.unified_analyzer.analyze(
                test_case.price_data, test_case.symbol, test_case.timeframe
            )
            
            current_price = test_case.price_data['close'].iloc[-1]
            signal = self.signal_generator.generate_signal_from_unified(
                unified_result, current_price
            )
            
            if signal:
                # Test all downstream components
                confidence_score = self.confidence_calculator.calculate_unified_confidence(
                    unified_result, signal
                )
                
                entry_decision = self.entry_exit_engine.evaluate_entry_conditions(
                    signal, unified_result
                )
                
                stop_loss_rec = self.stop_loss_calculator.calculate_stop_loss(
                    signal, unified_result, test_case.price_data
                )
                
                take_profit_rec = self.take_profit_calculator.calculate_take_profit(
                    signal, unified_result, test_case.price_data, stop_loss_rec.primary_stop
                )
                
                risk_reward_analysis = self.risk_reward_calculator.calculate_risk_reward(
                    signal, stop_loss_rec, take_profit_rec
                )
                
                # Check all components worked
                components_passed = all([
                    signal is not None,
                    confidence_score is not None,
                    entry_decision is not None,
                    stop_loss_rec is not None,
                    take_profit_rec is not None,
                    risk_reward_analysis is not None
                ])
                
                # Check quality thresholds
                quality_passed = (
                    confidence_score.overall_score >= 0.4 and
                    entry_decision.weighted_score >= 0.4 and
                    risk_reward_analysis.primary_ratio >= 1.0
                )
                
                passed = components_passed and quality_passed
                
                return TestResult(
                    test_id=test_case.test_id,
                    test_type=test_case.test_type,
                    passed=passed,
                    signal_generated=signal,
                    execution_time_ms=0.0,
                    confidence_score=confidence_score.overall_score,
                    quality_score=risk_reward_analysis.trade_quality_score,
                    signal_accuracy=1.0 if components_passed else 0.0,
                    direction_correct=True,
                    timing_accuracy=1.0 if quality_passed else 0.5,
                    analysis_details={
                        'integration_results': {
                            'components_passed': components_passed,
                            'quality_passed': quality_passed,
                            'confidence_score': confidence_score.overall_score,
                            'entry_score': entry_decision.weighted_score,
                            'risk_reward_ratio': risk_reward_analysis.primary_ratio,
                            'recommendation': risk_reward_analysis.recommendation
                        }
                    }
                )
            else:
                return TestResult(
                    test_id=test_case.test_id,
                    test_type=test_case.test_type,
                    passed=False,
                    signal_generated=None,
                    execution_time_ms=0.0,
                    confidence_score=0.0,
                    quality_score=0.0,
                    signal_accuracy=0.0,
                    direction_correct=False,
                    timing_accuracy=0.0,
                    error_message="No signal generated"
                )
                
        except Exception as e:
            return TestResult(
                test_id=test_case.test_id,
                test_type=test_case.test_type,
                passed=False,
                signal_generated=None,
                execution_time_ms=0.0,
                confidence_score=0.0,
                quality_score=0.0,
                signal_accuracy=0.0,
                direction_correct=False,
                timing_accuracy=0.0,
                error_message=str(e)
            )
    
    def _calculate_signal_accuracy(self, signal: Optional[TradingSignal], 
                                 expected_signal: Optional[str],
                                 unified_result) -> float:
        """Calculate signal accuracy score"""
        if not signal or not expected_signal:
            return 0.0
        
        # Compare signal direction
        signal_direction = signal.signal_type.value
        if ('BUY' in signal_direction and expected_signal.upper() == 'BUY') or \
           ('SELL' in signal_direction and expected_signal.upper() == 'SELL'):
            direction_score = 1.0
        else:
            direction_score = 0.0
        
        # Factor in confidence alignment
        confidence_score = signal.confidence
        
        # Factor in analysis agreement
        agreement_score = unified_result.agreement_score
        
        # Weighted accuracy
        accuracy = (direction_score * 0.6 + confidence_score * 0.2 + agreement_score * 0.2)
        return accuracy
    
    def _check_direction_accuracy(self, signal: Optional[TradingSignal], 
                                expected_signal: Optional[str]) -> bool:
        """Check if signal direction matches expected"""
        if not signal or not expected_signal:
            return False
        
        signal_direction = signal.signal_type.value
        return ('BUY' in signal_direction and expected_signal.upper() == 'BUY') or \
               ('SELL' in signal_direction and expected_signal.upper() == 'SELL')
    
    def _calculate_timing_accuracy(self, signal: Optional[TradingSignal], 
                                 price_data: pd.DataFrame) -> float:
        """Calculate timing accuracy based on signal urgency vs market conditions"""
        if not signal:
            return 0.0
        
        # Simple timing accuracy based on urgency appropriateness
        # In real implementation, would compare against actual market timing
        
        # Calculate recent volatility
        returns = price_data['close'].pct_change().tail(20)
        volatility = returns.std()
        
        # High urgency should match high volatility periods
        if signal.urgency.value == 'IMMEDIATE' and volatility > 0.02:
            return 1.0
        elif signal.urgency.value == 'HIGH' and volatility > 0.015:
            return 0.8
        elif signal.urgency.value == 'MEDIUM' and 0.005 < volatility < 0.015:
            return 0.9
        elif signal.urgency.value == 'LOW' and volatility < 0.01:
            return 0.7
        else:
            return 0.5
    
    def _modify_test_data(self, data: pd.DataFrame, modification: str) -> pd.DataFrame:
        """Modify test data for robustness testing"""
        modified_data = data.copy()
        
        if modification == 'add_noise':
            # Add small random noise
            noise_factor = 0.001  # 0.1%
            for col in ['open', 'high', 'low', 'close']:
                noise = np.random.normal(0, noise_factor, len(modified_data))
                modified_data[col] *= (1 + noise)
        
        elif modification == 'remove_outliers':
            # Remove extreme values
            for col in ['open', 'high', 'low', 'close']:
                q99 = modified_data[col].quantile(0.99)
                q01 = modified_data[col].quantile(0.01)
                modified_data[col] = modified_data[col].clip(q01, q99)
        
        elif modification == 'interpolate_gaps':
            # Simulate and fill gaps
            if len(modified_data) > 10:
                # Remove some random rows to create gaps
                drop_indices = np.random.choice(modified_data.index[5:-5], size=3, replace=False)
                modified_data = modified_data.drop(drop_indices)
                # Interpolate
                modified_data = modified_data.interpolate()
        
        elif modification == 'shift_prices':
            # Shift all prices by small amount
            shift_factor = 0.002  # 0.2%
            for col in ['open', 'high', 'low', 'close']:
                modified_data[col] *= (1 + shift_factor)
        
        return modified_data
    
    def _calculate_accuracy_breakdown(self, test_results: List[TestResult]) -> Dict[TestType, float]:
        """Calculate accuracy breakdown by test type"""
        breakdown = {}
        
        for test_type in TestType:
            type_results = [r for r in test_results if r.test_type == test_type]
            if type_results:
                accuracy = np.mean([r.signal_accuracy for r in type_results])
                breakdown[test_type] = accuracy
            else:
                breakdown[test_type] = 0.0
        
        return breakdown
    
    def _calculate_condition_breakdown(self, test_results: List[TestResult], 
                                     test_cases: List[TestCase]) -> Dict[MarketCondition, float]:
        """Calculate accuracy breakdown by market condition"""
        breakdown = {}
        
        # Map test results to test cases
        result_map = {r.test_id: r for r in test_results}
        
        for condition in MarketCondition:
            condition_cases = [c for c in test_cases if c.market_condition == condition]
            if condition_cases:
                condition_results = [result_map[c.test_id] for c in condition_cases if c.test_id in result_map]
                if condition_results:
                    accuracy = np.mean([r.signal_accuracy for r in condition_results])
                    breakdown[condition] = accuracy
                else:
                    breakdown[condition] = 0.0
            else:
                breakdown[condition] = 0.0
        
        return breakdown
    
    def generate_test_report(self, test_suite: TestSuite) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        return {
            'suite_summary': {
                'name': test_suite.suite_name,
                'execution_time': test_suite.execution_timestamp.isoformat(),
                'total_duration': test_suite.total_execution_time,
                'total_tests': test_suite.total_tests,
                'passed_tests': test_suite.passed_tests,
                'failed_tests': test_suite.failed_tests,
                'pass_rate': test_suite.pass_rate
            },
            'performance_metrics': {
                'avg_execution_time_ms': test_suite.avg_execution_time,
                'avg_confidence': test_suite.avg_confidence,
                'avg_quality': test_suite.avg_quality,
                'avg_accuracy': test_suite.avg_accuracy
            },
            'accuracy_breakdown': {k.value: v for k, v in test_suite.accuracy_breakdown.items()},
            'condition_breakdown': {k.value: v for k, v in test_suite.condition_breakdown.items()},
            'failed_tests': [
                {
                    'test_id': r.test_id,
                    'test_type': r.test_type.value,
                    'error_message': r.error_message,
                    'execution_time': r.execution_time_ms
                }
                for r in test_suite.test_results if not r.passed
            ],
            'performance_analysis': {
                'within_time_threshold': sum(1 for r in test_suite.test_results 
                                           if r.execution_time_ms <= self.config['performance_thresholds']['max_execution_time_ms']),
                'above_confidence_threshold': sum(1 for r in test_suite.test_results 
                                                if r.confidence_score >= self.config['performance_thresholds']['min_confidence']),
                'above_quality_threshold': sum(1 for r in test_suite.test_results 
                                             if r.quality_score >= self.config['performance_thresholds']['min_quality'])
            }
        }


# Create default signal tester instance
signal_tester = SignalTester()