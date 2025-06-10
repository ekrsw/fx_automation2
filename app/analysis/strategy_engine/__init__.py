"""
Integrated Strategy Engine

Combines Dow Theory and Elliott Wave analysis for unified signal generation
with comprehensive entry/exit management and risk/reward calculations.
"""

from .unified_analyzer import UnifiedAnalyzer, unified_analyzer
from .multi_timeframe_analyzer import MultiTimeframeAnalyzer, multi_timeframe_analyzer
from .signal_generator import SignalGenerator, signal_generator
from .confidence_calculator import ConfidenceCalculator, confidence_calculator
from .entry_exit_engine import EntryExitEngine, entry_exit_engine
from .stop_loss_calculator import StopLossCalculator, stop_loss_calculator
from .take_profit_calculator import TakeProfitCalculator, take_profit_calculator
from .risk_reward_calculator import RiskRewardCalculator, risk_reward_calculator
from .signal_tester import SignalTester, signal_tester
from .backtest_engine import BacktestEngine, backtest_engine
from .strategy_validator import StrategyValidator, strategy_validator

__all__ = [
    # Core Analysis
    'UnifiedAnalyzer', 'unified_analyzer',
    'MultiTimeframeAnalyzer', 'multi_timeframe_analyzer',
    'SignalGenerator', 'signal_generator', 
    'ConfidenceCalculator', 'confidence_calculator',
    
    # Entry/Exit Management
    'EntryExitEngine', 'entry_exit_engine',
    'StopLossCalculator', 'stop_loss_calculator',
    'TakeProfitCalculator', 'take_profit_calculator',
    'RiskRewardCalculator', 'risk_reward_calculator',
    
    # Testing and Validation
    'SignalTester', 'signal_tester',
    'BacktestEngine', 'backtest_engine',
    'StrategyValidator', 'strategy_validator'
]