"""
Strategy Validation Module

Comprehensive validation framework for testing strategy components
and overall system performance with statistical analysis.
"""

from typing import List, Dict, Any, Optional, Tuple, Callable
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from scipy import stats

from app.analysis.strategy_engine.signal_tester import SignalTester, TestCase, TestType, MarketCondition
from app.analysis.strategy_engine.backtest_engine import BacktestEngine, BacktestConfig
from app.utils.logger import analysis_logger


class ValidationLevel(Enum):
    """Validation level types"""
    BASIC = "basic"           # Basic functionality validation
    STANDARD = "standard"     # Standard performance validation
    COMPREHENSIVE = "comprehensive"  # Full statistical validation
    STRESS = "stress"         # Stress testing validation


class ValidationMetric(Enum):
    """Validation metrics"""
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    EXPECTANCY = "expectancy"
    CONSISTENCY = "consistency"
    ROBUSTNESS = "robustness"


@dataclass
class ValidationCriteria:
    """Validation criteria for strategy performance"""
    min_win_rate: float = 0.5          # 50% minimum win rate
    min_profit_factor: float = 1.2      # 1.2 minimum profit factor
    min_sharpe_ratio: float = 0.5       # 0.5 minimum Sharpe ratio
    max_drawdown: float = 0.15          # 15% maximum drawdown
    min_expectancy: float = 0.0         # Positive expectancy
    min_trade_count: int = 30           # Minimum trades for statistical significance
    min_consistency_score: float = 0.6  # 60% minimum consistency
    min_robustness_score: float = 0.5   # 50% minimum robustness


@dataclass
class ValidationTest:
    """Individual validation test"""
    test_id: str
    test_name: str
    validation_level: ValidationLevel
    criteria: ValidationCriteria
    test_function: Callable
    weight: float = 1.0
    description: str = ""


@dataclass
class ValidationResult:
    """Individual validation result"""
    test_id: str
    test_name: str
    passed: bool
    score: float              # 0.0 to 1.0
    actual_value: float
    expected_value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    validation_id: str
    strategy_name: str
    validation_level: ValidationLevel
    
    # Overall results
    overall_passed: bool
    overall_score: float      # 0.0 to 1.0
    confidence_level: float   # Statistical confidence
    
    # Individual test results
    test_results: List[ValidationResult]
    
    # Performance summary
    performance_metrics: Dict[str, float]
    statistical_significance: Dict[str, float]
    
    # Analysis breakdown
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    
    # Execution info
    validation_timestamp: datetime
    total_execution_time: float
    data_period: Tuple[datetime, datetime]
    sample_size: int


class StrategyValidator:
    """
    Comprehensive strategy validation framework
    
    Validates strategy performance using statistical analysis,
    robustness testing, and comprehensive backtesting.
    """
    
    def __init__(self):
        """Initialize strategy validator"""
        
        self.signal_tester = SignalTester()
        self.backtest_engine = BacktestEngine()
        
        # Validation configuration
        self.config = {
            'validation_levels': {
                ValidationLevel.BASIC: {
                    'min_backtest_days': 30,
                    'min_signals': 10,
                    'statistical_tests': ['basic_performance']
                },
                ValidationLevel.STANDARD: {
                    'min_backtest_days': 90,
                    'min_signals': 30,
                    'statistical_tests': ['performance', 'consistency', 'drawdown']
                },
                ValidationLevel.COMPREHENSIVE: {
                    'min_backtest_days': 365,
                    'min_signals': 100,
                    'statistical_tests': ['performance', 'consistency', 'robustness', 'walk_forward', 'monte_carlo']
                },
                ValidationLevel.STRESS: {
                    'min_backtest_days': 730,
                    'min_signals': 200,
                    'statistical_tests': ['all', 'stress_conditions', 'regime_changes']
                }
            },
            'confidence_levels': {
                'high': 0.95,
                'medium': 0.85,
                'low': 0.70
            },
            'monte_carlo_runs': 1000,
            'walk_forward_periods': 12,
            'bootstrap_samples': 500
        }
        
        # Define validation tests
        self.validation_tests = self._setup_validation_tests()
    
    async def validate_strategy(self,
                              price_data: pd.DataFrame,
                              symbol: str,
                              validation_level: ValidationLevel = ValidationLevel.STANDARD,
                              custom_criteria: Optional[ValidationCriteria] = None) -> ValidationReport:
        """
        Run comprehensive strategy validation
        
        Args:
            price_data: Historical price data for validation
            symbol: Trading symbol
            validation_level: Level of validation to perform
            custom_criteria: Custom validation criteria
            
        Returns:
            Comprehensive validation report
        """
        try:
            start_time = datetime.utcnow()
            validation_id = f"{symbol}_{validation_level.value}_{int(start_time.timestamp())}"
            
            analysis_logger.info(
                f"Starting strategy validation: {validation_id} - "
                f"Level: {validation_level.value}, "
                f"Data: {len(price_data)} bars"
            )
            
            # Prepare validation data
            validation_data = self._prepare_validation_data(price_data, validation_level)
            
            # Set validation criteria
            criteria = custom_criteria or ValidationCriteria()
            
            # Get applicable tests for this validation level
            applicable_tests = self._get_applicable_tests(validation_level)
            
            # Run validation tests
            test_results = []
            for test in applicable_tests:
                try:
                    result = await self._run_validation_test(
                        test, validation_data, symbol, criteria
                    )
                    test_results.append(result)
                    
                    analysis_logger.debug(
                        f"Completed test {test.test_name}: "
                        f"Passed: {result.passed}, Score: {result.score:.2f}"
                    )
                    
                except Exception as e:
                    analysis_logger.error(f"Error in validation test {test.test_name}: {e}")
                    # Create failed test result
                    failed_result = ValidationResult(
                        test_id=test.test_id,
                        test_name=test.test_name,
                        passed=False,
                        score=0.0,
                        actual_value=0.0,
                        expected_value=0.0,
                        details={'error': str(e)}
                    )
                    test_results.append(failed_result)
            
            # Calculate overall results
            overall_score = self._calculate_overall_score(test_results, applicable_tests)
            overall_passed = self._determine_overall_pass(test_results, criteria)
            confidence_level = self._calculate_confidence_level(test_results)
            
            # Generate performance metrics
            performance_metrics = self._extract_performance_metrics(test_results)
            statistical_significance = self._calculate_statistical_significance(test_results)
            
            # Generate analysis
            strengths, weaknesses, recommendations = self._generate_analysis(
                test_results, performance_metrics
            )
            
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            
            # Create validation report
            report = ValidationReport(
                validation_id=validation_id,
                strategy_name="Integrated Strategy Engine",
                validation_level=validation_level,
                overall_passed=overall_passed,
                overall_score=overall_score,
                confidence_level=confidence_level,
                test_results=test_results,
                performance_metrics=performance_metrics,
                statistical_significance=statistical_significance,
                strengths=strengths,
                weaknesses=weaknesses,
                recommendations=recommendations,
                validation_timestamp=start_time,
                total_execution_time=execution_time,
                data_period=(price_data.index[0], price_data.index[-1]),
                sample_size=len(price_data)
            )
            
            analysis_logger.info(
                f"Strategy validation completed: {validation_id} - "
                f"Overall passed: {overall_passed}, "
                f"Score: {overall_score:.2f}, "
                f"Confidence: {confidence_level:.2f}, "
                f"Execution time: {execution_time:.1f}s"
            )
            
            return report
            
        except Exception as e:
            analysis_logger.error(f"Error in strategy validation: {e}")
            raise
    
    def _setup_validation_tests(self) -> List[ValidationTest]:
        """Setup validation test definitions"""
        tests = []
        
        # Basic performance tests
        tests.append(ValidationTest(
            test_id="basic_backtest",
            test_name="Basic Backtest Performance",
            validation_level=ValidationLevel.BASIC,
            criteria=ValidationCriteria(),
            test_function=self._test_basic_backtest_performance,
            weight=1.0,
            description="Basic backtest to verify strategy generates positive returns"
        ))
        
        # Standard performance tests
        tests.append(ValidationTest(
            test_id="win_rate_test",
            test_name="Win Rate Validation",
            validation_level=ValidationLevel.STANDARD,
            criteria=ValidationCriteria(),
            test_function=self._test_win_rate,
            weight=1.0,
            description="Validate strategy win rate meets minimum requirements"
        ))
        
        tests.append(ValidationTest(
            test_id="profit_factor_test",
            test_name="Profit Factor Validation",
            validation_level=ValidationLevel.STANDARD,
            criteria=ValidationCriteria(),
            test_function=self._test_profit_factor,
            weight=1.0,
            description="Validate strategy profit factor meets minimum requirements"
        ))
        
        tests.append(ValidationTest(
            test_id="sharpe_ratio_test",
            test_name="Sharpe Ratio Validation",
            validation_level=ValidationLevel.STANDARD,
            criteria=ValidationCriteria(),
            test_function=self._test_sharpe_ratio,
            weight=0.8,
            description="Validate strategy risk-adjusted returns"
        ))
        
        tests.append(ValidationTest(
            test_id="drawdown_test",
            test_name="Maximum Drawdown Validation",
            validation_level=ValidationLevel.STANDARD,
            criteria=ValidationCriteria(),
            test_function=self._test_max_drawdown,
            weight=1.2,
            description="Validate strategy maximum drawdown stays within limits"
        ))
        
        # Comprehensive tests
        tests.append(ValidationTest(
            test_id="consistency_test",
            test_name="Performance Consistency",
            validation_level=ValidationLevel.COMPREHENSIVE,
            criteria=ValidationCriteria(),
            test_function=self._test_consistency,
            weight=1.0,
            description="Test strategy performance consistency across time periods"
        ))
        
        tests.append(ValidationTest(
            test_id="robustness_test",
            test_name="Parameter Robustness",
            validation_level=ValidationLevel.COMPREHENSIVE,
            criteria=ValidationCriteria(),
            test_function=self._test_robustness,
            weight=1.0,
            description="Test strategy robustness to parameter changes"
        ))
        
        tests.append(ValidationTest(
            test_id="walk_forward_test",
            test_name="Walk Forward Analysis",
            validation_level=ValidationLevel.COMPREHENSIVE,
            criteria=ValidationCriteria(),
            test_function=self._test_walk_forward,
            weight=1.5,
            description="Test strategy performance in out-of-sample periods"
        ))
        
        # Stress tests
        tests.append(ValidationTest(
            test_id="monte_carlo_test",
            test_name="Monte Carlo Simulation",
            validation_level=ValidationLevel.STRESS,
            criteria=ValidationCriteria(),
            test_function=self._test_monte_carlo,
            weight=1.0,
            description="Monte Carlo simulation of strategy performance"
        ))
        
        tests.append(ValidationTest(
            test_id="market_regime_test",
            test_name="Market Regime Analysis",
            validation_level=ValidationLevel.STRESS,
            criteria=ValidationCriteria(),
            test_function=self._test_market_regimes,
            weight=1.0,
            description="Test strategy performance across different market regimes"
        ))
        
        return tests
    
    def _get_applicable_tests(self, validation_level: ValidationLevel) -> List[ValidationTest]:
        """Get tests applicable to validation level"""
        applicable_tests = []
        
        for test in self.validation_tests:
            # Include tests for current and lower levels
            if (validation_level == ValidationLevel.BASIC and test.validation_level == ValidationLevel.BASIC) or \
               (validation_level == ValidationLevel.STANDARD and test.validation_level in [ValidationLevel.BASIC, ValidationLevel.STANDARD]) or \
               (validation_level == ValidationLevel.COMPREHENSIVE and test.validation_level in [ValidationLevel.BASIC, ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE]) or \
               (validation_level == ValidationLevel.STRESS):  # Include all tests for stress level
                applicable_tests.append(test)
        
        return applicable_tests
    
    async def _run_validation_test(self,
                                 test: ValidationTest,
                                 data: pd.DataFrame,
                                 symbol: str,
                                 criteria: ValidationCriteria) -> ValidationResult:
        """Run individual validation test"""
        try:
            result = await test.test_function(data, symbol, criteria)
            return result
        except Exception as e:
            return ValidationResult(
                test_id=test.test_id,
                test_name=test.test_name,
                passed=False,
                score=0.0,
                actual_value=0.0,
                expected_value=0.0,
                details={'error': str(e)}
            )
    
    async def _test_basic_backtest_performance(self,
                                             data: pd.DataFrame,
                                             symbol: str,
                                             criteria: ValidationCriteria) -> ValidationResult:
        """Test basic backtest performance"""
        # Run basic backtest
        config = BacktestConfig(
            initial_capital=10000,
            max_risk_per_trade=0.02,
            min_signal_confidence=0.6
        )
        
        backtest_engine = BacktestEngine(config)
        result = await backtest_engine.run_backtest(data, symbol)
        
        # Check if strategy generated any trades and positive returns
        total_return = result.metrics.total_return_pct
        trade_count = result.metrics.total_trades
        
        passed = total_return > 0 and trade_count >= criteria.min_trade_count
        score = min(1.0, max(0.0, (total_return / 10.0) + 0.5))  # Normalize return to score
        
        return ValidationResult(
            test_id="basic_backtest",
            test_name="Basic Backtest Performance",
            passed=passed,
            score=score,
            actual_value=total_return,
            expected_value=0.0,
            details={
                'total_trades': trade_count,
                'total_return_pct': total_return,
                'win_rate': result.metrics.win_rate,
                'backtest_summary': {
                    'start_date': result.metrics.start_date.isoformat(),
                    'end_date': result.metrics.end_date.isoformat(),
                    'duration_days': result.metrics.total_duration_days
                }
            }
        )
    
    async def _test_win_rate(self,
                           data: pd.DataFrame,
                           symbol: str,
                           criteria: ValidationCriteria) -> ValidationResult:
        """Test win rate validation"""
        config = BacktestConfig()
        backtest_engine = BacktestEngine(config)
        result = await backtest_engine.run_backtest(data, symbol)
        
        win_rate = result.metrics.win_rate
        passed = win_rate >= criteria.min_win_rate
        
        # Calculate confidence interval for win rate
        n = result.metrics.total_trades
        if n > 0:
            se = np.sqrt(win_rate * (1 - win_rate) / n)
            ci_lower = win_rate - 1.96 * se
            ci_upper = win_rate + 1.96 * se
            confidence_interval = (ci_lower, ci_upper)
            
            # Calculate p-value for hypothesis test (H0: win_rate <= 0.5)
            z_score = (win_rate - 0.5) / se if se > 0 else 0
            p_value = 1 - stats.norm.cdf(z_score)
        else:
            confidence_interval = None
            p_value = None
        
        score = min(1.0, win_rate / criteria.min_win_rate) if criteria.min_win_rate > 0 else 0.0
        
        return ValidationResult(
            test_id="win_rate_test",
            test_name="Win Rate Validation",
            passed=passed,
            score=score,
            actual_value=win_rate,
            expected_value=criteria.min_win_rate,
            confidence_interval=confidence_interval,
            p_value=p_value,
            details={
                'winning_trades': result.metrics.winning_trades,
                'total_trades': result.metrics.total_trades,
                'statistical_significance': p_value < 0.05 if p_value else False
            }
        )
    
    async def _test_profit_factor(self,
                                data: pd.DataFrame,
                                symbol: str,
                                criteria: ValidationCriteria) -> ValidationResult:
        """Test profit factor validation"""
        config = BacktestConfig()
        backtest_engine = BacktestEngine(config)
        result = await backtest_engine.run_backtest(data, symbol)
        
        profit_factor = result.metrics.profit_factor
        passed = profit_factor >= criteria.min_profit_factor
        score = min(1.0, profit_factor / criteria.min_profit_factor) if criteria.min_profit_factor > 0 else 0.0
        
        return ValidationResult(
            test_id="profit_factor_test",
            test_name="Profit Factor Validation",
            passed=passed,
            score=score,
            actual_value=profit_factor,
            expected_value=criteria.min_profit_factor,
            details={
                'avg_win': result.metrics.avg_win,
                'avg_loss': result.metrics.avg_loss,
                'largest_win': result.metrics.largest_win,
                'largest_loss': result.metrics.largest_loss
            }
        )
    
    async def _test_sharpe_ratio(self,
                               data: pd.DataFrame,
                               symbol: str,
                               criteria: ValidationCriteria) -> ValidationResult:
        """Test Sharpe ratio validation"""
        config = BacktestConfig()
        backtest_engine = BacktestEngine(config)
        result = await backtest_engine.run_backtest(data, symbol)
        
        sharpe_ratio = result.metrics.sharpe_ratio
        passed = sharpe_ratio >= criteria.min_sharpe_ratio
        score = min(1.0, max(0.0, sharpe_ratio / criteria.min_sharpe_ratio)) if criteria.min_sharpe_ratio > 0 else 0.0
        
        return ValidationResult(
            test_id="sharpe_ratio_test",
            test_name="Sharpe Ratio Validation",
            passed=passed,
            score=score,
            actual_value=sharpe_ratio,
            expected_value=criteria.min_sharpe_ratio,
            details={
                'volatility': result.metrics.volatility,
                'sortino_ratio': result.metrics.sortino_ratio,
                'calmar_ratio': result.metrics.calmar_ratio
            }
        )
    
    async def _test_max_drawdown(self,
                               data: pd.DataFrame,
                               symbol: str,
                               criteria: ValidationCriteria) -> ValidationResult:
        """Test maximum drawdown validation"""
        config = BacktestConfig()
        backtest_engine = BacktestEngine(config)
        result = await backtest_engine.run_backtest(data, symbol)
        
        max_drawdown = result.metrics.max_drawdown
        passed = max_drawdown <= criteria.max_drawdown
        score = min(1.0, max(0.0, (criteria.max_drawdown - max_drawdown) / criteria.max_drawdown))
        
        return ValidationResult(
            test_id="drawdown_test",
            test_name="Maximum Drawdown Validation",
            passed=passed,
            score=score,
            actual_value=max_drawdown,
            expected_value=criteria.max_drawdown,
            details={
                'max_drawdown_duration': result.metrics.max_drawdown_duration,
                'recovery_factor': result.metrics.recovery_factor,
                'var_95': result.metrics.var_95,
                'cvar_95': result.metrics.cvar_95
            }
        )
    
    async def _test_consistency(self,
                              data: pd.DataFrame,
                              symbol: str,
                              criteria: ValidationCriteria) -> ValidationResult:
        """Test performance consistency across periods"""
        # Split data into periods and test consistency
        period_length = len(data) // 4  # Quarterly periods
        period_returns = []
        
        for i in range(4):
            start_idx = i * period_length
            end_idx = (i + 1) * period_length if i < 3 else len(data)
            period_data = data.iloc[start_idx:end_idx]
            
            if len(period_data) > 50:  # Minimum data for meaningful backtest
                config = BacktestConfig()
                backtest_engine = BacktestEngine(config)
                result = await backtest_engine.run_backtest(period_data, symbol)
                period_returns.append(result.metrics.total_return_pct)
        
        if len(period_returns) >= 3:
            # Calculate consistency metrics
            returns_std = np.std(period_returns)
            returns_mean = np.mean(period_returns)
            
            # Consistency score based on coefficient of variation
            cv = returns_std / abs(returns_mean) if returns_mean != 0 else float('inf')
            consistency_score = max(0.0, 1.0 - cv / 2.0)  # Normalize CV to score
            
            passed = consistency_score >= criteria.min_consistency_score
            
            return ValidationResult(
                test_id="consistency_test",
                test_name="Performance Consistency",
                passed=passed,
                score=consistency_score,
                actual_value=consistency_score,
                expected_value=criteria.min_consistency_score,
                details={
                    'period_returns': period_returns,
                    'returns_std': returns_std,
                    'returns_mean': returns_mean,
                    'coefficient_of_variation': cv
                }
            )
        else:
            return ValidationResult(
                test_id="consistency_test",
                test_name="Performance Consistency",
                passed=False,
                score=0.0,
                actual_value=0.0,
                expected_value=criteria.min_consistency_score,
                details={'error': 'Insufficient data for consistency test'}
            )
    
    async def _test_robustness(self,
                             data: pd.DataFrame,
                             symbol: str,
                             criteria: ValidationCriteria) -> ValidationResult:
        """Test parameter robustness"""
        # Test strategy with different parameter settings
        base_config = BacktestConfig()
        
        # Test different risk levels
        risk_levels = [0.01, 0.02, 0.03]  # 1%, 2%, 3%
        confidence_levels = [0.5, 0.6, 0.7]  # Different confidence thresholds
        
        results = []
        
        for risk in risk_levels:
            for confidence in confidence_levels:
                config = BacktestConfig(
                    max_risk_per_trade=risk,
                    min_signal_confidence=confidence
                )
                
                backtest_engine = BacktestEngine(config)
                result = await backtest_engine.run_backtest(data, symbol)
                
                results.append({
                    'risk_level': risk,
                    'confidence_level': confidence,
                    'total_return': result.metrics.total_return_pct,
                    'win_rate': result.metrics.win_rate,
                    'max_drawdown': result.metrics.max_drawdown
                })
        
        if results:
            # Calculate robustness metrics
            returns = [r['total_return'] for r in results]
            win_rates = [r['win_rate'] for r in results]
            
            # Robustness based on stability of results
            returns_cv = np.std(returns) / abs(np.mean(returns)) if np.mean(returns) != 0 else float('inf')
            win_rates_cv = np.std(win_rates) / np.mean(win_rates) if np.mean(win_rates) > 0 else float('inf')
            
            robustness_score = max(0.0, 1.0 - (returns_cv + win_rates_cv) / 4.0)
            passed = robustness_score >= criteria.min_robustness_score
            
            return ValidationResult(
                test_id="robustness_test",
                test_name="Parameter Robustness",
                passed=passed,
                score=robustness_score,
                actual_value=robustness_score,
                expected_value=criteria.min_robustness_score,
                details={
                    'parameter_results': results,
                    'returns_coefficient_of_variation': returns_cv,
                    'win_rates_coefficient_of_variation': win_rates_cv,
                    'avg_return': np.mean(returns),
                    'avg_win_rate': np.mean(win_rates)
                }
            )
        else:
            return ValidationResult(
                test_id="robustness_test",
                test_name="Parameter Robustness",
                passed=False,
                score=0.0,
                actual_value=0.0,
                expected_value=criteria.min_robustness_score,
                details={'error': 'No results from robustness test'}
            )
    
    async def _test_walk_forward(self,
                               data: pd.DataFrame,
                               symbol: str,
                               criteria: ValidationCriteria) -> ValidationResult:
        """Test walk forward analysis"""
        # Implement walk-forward analysis
        n_periods = self.config['walk_forward_periods']
        period_size = len(data) // n_periods
        
        out_of_sample_returns = []
        
        for i in range(n_periods - 1):  # Leave last period for out-of-sample
            # In-sample period (for optimization - simplified here)
            in_sample_start = i * period_size
            in_sample_end = (i + 1) * period_size
            
            # Out-of-sample period
            oos_start = in_sample_end
            oos_end = min(oos_start + period_size, len(data))
            
            if oos_end > oos_start + 20:  # Minimum out-of-sample size
                oos_data = data.iloc[oos_start:oos_end]
                
                config = BacktestConfig()
                backtest_engine = BacktestEngine(config)
                result = await backtest_engine.run_backtest(oos_data, symbol)
                
                out_of_sample_returns.append(result.metrics.total_return_pct)
        
        if out_of_sample_returns:
            avg_oos_return = np.mean(out_of_sample_returns)
            oos_consistency = 1.0 - np.std(out_of_sample_returns) / (abs(avg_oos_return) + 1e-6)
            
            # Walk forward score based on average return and consistency
            wf_score = max(0.0, min(1.0, (avg_oos_return / 10.0 + oos_consistency) / 2.0))
            passed = wf_score >= 0.4  # 40% minimum for walk-forward
            
            return ValidationResult(
                test_id="walk_forward_test",
                test_name="Walk Forward Analysis",
                passed=passed,
                score=wf_score,
                actual_value=avg_oos_return,
                expected_value=0.0,
                details={
                    'out_of_sample_returns': out_of_sample_returns,
                    'avg_oos_return': avg_oos_return,
                    'oos_consistency': oos_consistency,
                    'periods_tested': len(out_of_sample_returns)
                }
            )
        else:
            return ValidationResult(
                test_id="walk_forward_test",
                test_name="Walk Forward Analysis",
                passed=False,
                score=0.0,
                actual_value=0.0,
                expected_value=0.0,
                details={'error': 'Insufficient data for walk-forward analysis'}
            )
    
    async def _test_monte_carlo(self,
                              data: pd.DataFrame,
                              symbol: str,
                              criteria: ValidationCriteria) -> ValidationResult:
        """Test Monte Carlo simulation"""
        # Simplified Monte Carlo test
        n_runs = min(100, self.config['monte_carlo_runs'])  # Reduced for demo
        returns = []
        
        for run in range(n_runs):
            # Bootstrap sample the data
            sample_indices = np.random.choice(len(data), size=len(data), replace=True)
            sample_data = data.iloc[sample_indices].copy()
            sample_data.index = data.index  # Preserve time index structure
            
            config = BacktestConfig()
            backtest_engine = BacktestEngine(config)
            result = await backtest_engine.run_backtest(sample_data, symbol)
            
            returns.append(result.metrics.total_return_pct)
        
        if returns:
            avg_return = np.mean(returns)
            return_std = np.std(returns)
            
            # Calculate confidence intervals
            ci_95_lower = np.percentile(returns, 2.5)
            ci_95_upper = np.percentile(returns, 97.5)
            
            # Probability of positive returns
            prob_positive = sum(1 for r in returns if r > 0) / len(returns)
            
            # Monte Carlo score based on average return and probability of success
            mc_score = max(0.0, min(1.0, (avg_return / 10.0 + prob_positive) / 2.0))
            passed = prob_positive >= 0.6  # 60% probability of positive returns
            
            return ValidationResult(
                test_id="monte_carlo_test",
                test_name="Monte Carlo Simulation",
                passed=passed,
                score=mc_score,
                actual_value=prob_positive,
                expected_value=0.6,
                confidence_interval=(ci_95_lower, ci_95_upper),
                details={
                    'monte_carlo_runs': n_runs,
                    'avg_return': avg_return,
                    'return_std': return_std,
                    'prob_positive': prob_positive,
                    'ci_95_lower': ci_95_lower,
                    'ci_95_upper': ci_95_upper,
                    'worst_case': min(returns),
                    'best_case': max(returns)
                }
            )
        else:
            return ValidationResult(
                test_id="monte_carlo_test",
                test_name="Monte Carlo Simulation",
                passed=False,
                score=0.0,
                actual_value=0.0,
                expected_value=0.6,
                details={'error': 'Monte Carlo simulation failed'}
            )
    
    async def _test_market_regimes(self,
                                 data: pd.DataFrame,
                                 symbol: str,
                                 criteria: ValidationCriteria) -> ValidationResult:
        """Test performance across market regimes"""
        # Identify market regimes based on volatility and trend
        returns = data['close'].pct_change().dropna()
        volatility = returns.rolling(20).std()
        
        # Define regimes
        high_vol_threshold = volatility.quantile(0.7)
        low_vol_threshold = volatility.quantile(0.3)
        
        regime_results = {}
        
        # High volatility periods
        high_vol_mask = volatility > high_vol_threshold
        if high_vol_mask.sum() > 50:
            high_vol_data = data[high_vol_mask]
            config = BacktestConfig()
            backtest_engine = BacktestEngine(config)
            result = await backtest_engine.run_backtest(high_vol_data, symbol)
            regime_results['high_volatility'] = result.metrics.total_return_pct
        
        # Low volatility periods
        low_vol_mask = volatility < low_vol_threshold
        if low_vol_mask.sum() > 50:
            low_vol_data = data[low_vol_mask]
            config = BacktestConfig()
            backtest_engine = BacktestEngine(config)
            result = await backtest_engine.run_backtest(low_vol_data, symbol)
            regime_results['low_volatility'] = result.metrics.total_return_pct
        
        if len(regime_results) >= 2:
            # Calculate regime consistency
            regime_returns = list(regime_results.values())
            regime_consistency = 1.0 - np.std(regime_returns) / (abs(np.mean(regime_returns)) + 1e-6)
            
            passed = regime_consistency >= 0.5 and all(r >= 0 for r in regime_returns)
            score = max(0.0, min(1.0, regime_consistency))
            
            return ValidationResult(
                test_id="market_regime_test",
                test_name="Market Regime Analysis",
                passed=passed,
                score=score,
                actual_value=regime_consistency,
                expected_value=0.5,
                details={
                    'regime_results': regime_results,
                    'regime_consistency': regime_consistency,
                    'high_vol_threshold': high_vol_threshold,
                    'low_vol_threshold': low_vol_threshold
                }
            )
        else:
            return ValidationResult(
                test_id="market_regime_test",
                test_name="Market Regime Analysis",
                passed=False,
                score=0.0,
                actual_value=0.0,
                expected_value=0.5,
                details={'error': 'Insufficient data for regime analysis'}
            )
    
    def _prepare_validation_data(self,
                               price_data: pd.DataFrame,
                               validation_level: ValidationLevel) -> pd.DataFrame:
        """Prepare data for validation based on level"""
        level_config = self.config['validation_levels'][validation_level]
        min_days = level_config['min_backtest_days']
        
        # Ensure we have enough data
        if len(price_data) < min_days:
            analysis_logger.warning(
                f"Insufficient data for {validation_level.value} validation: "
                f"{len(price_data)} bars < {min_days} required"
            )
        
        return price_data.copy()
    
    def _calculate_overall_score(self,
                               test_results: List[ValidationResult],
                               tests: List[ValidationTest]) -> float:
        """Calculate weighted overall score"""
        if not test_results:
            return 0.0
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        test_weights = {test.test_id: test.weight for test in tests}
        
        for result in test_results:
            weight = test_weights.get(result.test_id, 1.0)
            total_weighted_score += result.score * weight
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_overall_pass(self,
                              test_results: List[ValidationResult],
                              criteria: ValidationCriteria) -> bool:
        """Determine if overall validation passes"""
        if not test_results:
            return False
        
        # All critical tests must pass
        critical_tests = ['basic_backtest', 'win_rate_test', 'drawdown_test']
        
        for result in test_results:
            if result.test_id in critical_tests and not result.passed:
                return False
        
        # At least 70% of all tests must pass
        pass_rate = sum(1 for r in test_results if r.passed) / len(test_results)
        return pass_rate >= 0.7
    
    def _calculate_confidence_level(self, test_results: List[ValidationResult]) -> float:
        """Calculate statistical confidence level"""
        # Average p-values where available
        p_values = [r.p_value for r in test_results if r.p_value is not None]
        
        if p_values:
            avg_p_value = np.mean(p_values)
            confidence = 1.0 - avg_p_value
        else:
            # Use score-based confidence
            avg_score = np.mean([r.score for r in test_results])
            confidence = avg_score
        
        return min(1.0, max(0.0, confidence))
    
    def _extract_performance_metrics(self, test_results: List[ValidationResult]) -> Dict[str, float]:
        """Extract key performance metrics from test results"""
        metrics = {}
        
        for result in test_results:
            if result.test_id == 'win_rate_test':
                metrics['win_rate'] = result.actual_value
            elif result.test_id == 'profit_factor_test':
                metrics['profit_factor'] = result.actual_value
            elif result.test_id == 'sharpe_ratio_test':
                metrics['sharpe_ratio'] = result.actual_value
            elif result.test_id == 'drawdown_test':
                metrics['max_drawdown'] = result.actual_value
            elif result.test_id == 'basic_backtest':
                metrics['total_return'] = result.actual_value
        
        return metrics
    
    def _calculate_statistical_significance(self, test_results: List[ValidationResult]) -> Dict[str, float]:
        """Calculate statistical significance metrics"""
        significance = {}
        
        for result in test_results:
            if result.p_value is not None:
                significance[result.test_id] = {
                    'p_value': result.p_value,
                    'significant': result.p_value < 0.05
                }
        
        return significance
    
    def _generate_analysis(self,
                         test_results: List[ValidationResult],
                         performance_metrics: Dict[str, float]) -> Tuple[List[str], List[str], List[str]]:
        """Generate strengths, weaknesses, and recommendations"""
        strengths = []
        weaknesses = []
        recommendations = []
        
        # Analyze test results
        passed_tests = [r for r in test_results if r.passed]
        failed_tests = [r for r in test_results if not r.passed]
        
        # Identify strengths
        if len(passed_tests) >= len(test_results) * 0.8:
            strengths.append("Strong overall validation performance")
        
        if 'win_rate' in performance_metrics and performance_metrics['win_rate'] > 0.6:
            strengths.append("High win rate indicates good signal quality")
        
        if 'sharpe_ratio' in performance_metrics and performance_metrics['sharpe_ratio'] > 1.0:
            strengths.append("Excellent risk-adjusted returns")
        
        # Identify weaknesses
        if len(failed_tests) > len(test_results) * 0.3:
            weaknesses.append("Multiple validation tests failed")
        
        if 'max_drawdown' in performance_metrics and performance_metrics['max_drawdown'] > 0.2:
            weaknesses.append("High maximum drawdown indicates risk management issues")
        
        if 'profit_factor' in performance_metrics and performance_metrics['profit_factor'] < 1.2:
            weaknesses.append("Low profit factor suggests insufficient edge")
        
        # Generate recommendations
        for result in failed_tests:
            if result.test_id == 'win_rate_test':
                recommendations.append("Consider improving signal quality or entry criteria")
            elif result.test_id == 'drawdown_test':
                recommendations.append("Implement stricter risk management and position sizing")
            elif result.test_id == 'consistency_test':
                recommendations.append("Review strategy parameters for better consistency across periods")
            elif result.test_id == 'robustness_test':
                recommendations.append("Optimize parameters for better robustness to market changes")
        
        if not strengths:
            strengths.append("Strategy requires significant improvement")
        
        if not recommendations:
            recommendations.append("Continue monitoring strategy performance")
        
        return strengths, weaknesses, recommendations


# Create default strategy validator
strategy_validator = StrategyValidator()