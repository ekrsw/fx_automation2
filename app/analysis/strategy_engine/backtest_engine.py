"""
Backtest Engine

Comprehensive backtesting framework for evaluating strategy performance
across historical data with detailed metrics and analysis.
"""

from typing import List, Dict, Any, Optional, Tuple, Callable
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio

from app.analysis.strategy_engine.unified_analyzer import UnifiedAnalyzer
from app.analysis.strategy_engine.signal_generator import SignalGenerator, TradingSignal, SignalType
from app.analysis.strategy_engine.confidence_calculator import ConfidenceCalculator
from app.analysis.strategy_engine.entry_exit_engine import EntryExitEngine
from app.analysis.strategy_engine.stop_loss_calculator import StopLossCalculator
from app.analysis.strategy_engine.take_profit_calculator import TakeProfitCalculator
from app.analysis.strategy_engine.risk_reward_calculator import RiskRewardCalculator
from app.utils.logger import analysis_logger


class TradeStatus(Enum):
    """Trade status types"""
    OPEN = "open"
    CLOSED_PROFIT = "closed_profit"
    CLOSED_LOSS = "closed_loss"
    CLOSED_BREAKEVEN = "closed_breakeven"
    EXPIRED = "expired"


class BacktestMode(Enum):
    """Backtest execution modes"""
    SEQUENTIAL = "sequential"      # Process data sequentially
    VECTORIZED = "vectorized"      # Vectorized operations
    EVENT_DRIVEN = "event_driven"  # Event-driven simulation


@dataclass
class BacktestTrade:
    """Individual trade record"""
    trade_id: str
    symbol: str
    signal: TradingSignal
    
    # Entry details
    entry_time: datetime
    entry_price: float
    position_size: float
    
    # Exit details
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    
    # Risk management
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    # Performance
    pnl: float = 0.0
    pnl_pct: float = 0.0
    duration_hours: float = 0.0
    
    # Metrics
    max_adverse_excursion: float = 0.0  # MAE
    max_favorable_excursion: float = 0.0  # MFE
    
    status: TradeStatus = TradeStatus.OPEN
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestConfig:
    """Backtest configuration"""
    # Capital management
    initial_capital: float = 10000.0
    max_risk_per_trade: float = 0.02  # 2%
    max_concurrent_trades: int = 3
    
    # Execution settings
    slippage_pct: float = 0.001  # 0.1% slippage
    commission_pct: float = 0.0002  # 0.02% commission
    
    # Strategy settings
    min_signal_confidence: float = 0.6
    min_risk_reward_ratio: float = 1.5
    max_trade_duration_hours: int = 168  # 1 week
    
    # Analysis settings
    lookback_periods: int = 100  # Periods for analysis
    rebalance_frequency: str = "H1"  # Rebalance frequency
    
    # Risk management
    max_drawdown_limit: float = 0.15  # 15% max drawdown
    daily_loss_limit: float = 0.05  # 5% daily loss limit


@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics"""
    # Basic metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # PnL metrics
    total_pnl: float
    total_return_pct: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    
    # Risk metrics
    max_drawdown: float
    max_drawdown_duration: int
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Trade metrics
    avg_trade_duration: float
    profit_factor: float
    expectancy: float
    
    # Advanced metrics
    var_95: float  # Value at Risk
    cvar_95: float  # Conditional Value at Risk
    recovery_factor: float
    
    # Execution metrics
    total_commission: float
    total_slippage: float
    
    # Timing metrics
    start_date: datetime
    end_date: datetime
    total_duration_days: int
    
    # Additional analysis
    monthly_returns: List[float] = field(default_factory=list)
    drawdown_periods: List[Dict[str, Any]] = field(default_factory=list)
    trade_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Complete backtest result"""
    backtest_id: str
    config: BacktestConfig
    
    # Trade records
    trades: List[BacktestTrade]
    equity_curve: pd.DataFrame
    
    # Performance metrics
    metrics: BacktestMetrics
    
    # Analysis details
    signal_analysis: Dict[str, Any]
    risk_analysis: Dict[str, Any]
    
    # Execution info
    execution_time: float
    completion_timestamp: datetime


class BacktestEngine:
    """
    Comprehensive backtesting engine
    
    Simulates strategy performance across historical data with
    realistic execution modeling and detailed analysis.
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """Initialize backtest engine"""
        
        self.config = config or BacktestConfig()
        
        # Initialize strategy components
        self.unified_analyzer = UnifiedAnalyzer()
        self.signal_generator = SignalGenerator()
        self.confidence_calculator = ConfidenceCalculator()
        self.entry_exit_engine = EntryExitEngine()
        self.stop_loss_calculator = StopLossCalculator()
        self.take_profit_calculator = TakeProfitCalculator()
        self.risk_reward_calculator = RiskRewardCalculator()
        
        # Backtest state
        self.current_capital = self.config.initial_capital
        self.open_trades: List[BacktestTrade] = []
        self.closed_trades: List[BacktestTrade] = []
        self.equity_history = []
        self.daily_pnl = []
        
        # Performance tracking
        self.peak_equity = self.config.initial_capital
        self.max_drawdown_start = None
        self.max_drawdown_end = None
        
    async def run_backtest(self,
                          price_data: pd.DataFrame,
                          symbol: str,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          mode: BacktestMode = BacktestMode.SEQUENTIAL) -> BacktestResult:
        """
        Run comprehensive backtest
        
        Args:
            price_data: Historical price data
            symbol: Trading symbol
            start_date: Backtest start date
            end_date: Backtest end date
            mode: Backtest execution mode
            
        Returns:
            Complete backtest result
        """
        try:
            start_time = datetime.utcnow()
            
            # Prepare data
            backtest_data = self._prepare_backtest_data(
                price_data, start_date, end_date
            )
            
            backtest_id = f"{symbol}_{int(start_time.timestamp())}"
            
            analysis_logger.info(
                f"Starting backtest: {backtest_id} - "
                f"Data: {len(backtest_data)} bars, "
                f"Period: {backtest_data.index[0]} to {backtest_data.index[-1]}"
            )
            
            # Reset state
            self._reset_backtest_state()
            
            # Run backtest based on mode
            if mode == BacktestMode.SEQUENTIAL:
                await self._run_sequential_backtest(backtest_data, symbol)
            elif mode == BacktestMode.VECTORIZED:
                await self._run_vectorized_backtest(backtest_data, symbol)
            elif mode == BacktestMode.EVENT_DRIVEN:
                await self._run_event_driven_backtest(backtest_data, symbol)
            else:
                raise ValueError(f"Unknown backtest mode: {mode}")
            
            # Calculate final metrics
            metrics = self._calculate_metrics(backtest_data)
            
            # Create equity curve
            equity_curve = self._create_equity_curve(backtest_data)
            
            # Analyze results
            signal_analysis = self._analyze_signals()
            risk_analysis = self._analyze_risk()
            
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            
            result = BacktestResult(
                backtest_id=backtest_id,
                config=self.config,
                trades=self.closed_trades + self.open_trades,
                equity_curve=equity_curve,
                metrics=metrics,
                signal_analysis=signal_analysis,
                risk_analysis=risk_analysis,
                execution_time=execution_time,
                completion_timestamp=end_time
            )
            
            analysis_logger.info(
                f"Backtest completed: {backtest_id} - "
                f"Total trades: {metrics.total_trades}, "
                f"Win rate: {metrics.win_rate:.1%}, "
                f"Total return: {metrics.total_return_pct:.1%}, "
                f"Execution time: {execution_time:.1f}s"
            )
            
            return result
            
        except Exception as e:
            analysis_logger.error(f"Error running backtest: {e}")
            raise
    
    async def _run_sequential_backtest(self, data: pd.DataFrame, symbol: str):
        """Run sequential backtest processing each bar"""
        
        for i in range(self.config.lookback_periods, len(data)):
            current_time = data.index[i]
            current_bar = data.iloc[i]
            
            # Get lookback data for analysis
            lookback_data = data.iloc[i-self.config.lookback_periods:i+1]
            
            # Update open trades
            await self._update_open_trades(current_bar, current_time)
            
            # Check for new signals
            if len(self.open_trades) < self.config.max_concurrent_trades:
                await self._check_for_new_signals(
                    lookback_data, symbol, current_bar, current_time
                )
            
            # Update equity tracking
            self._update_equity_tracking(current_time)
            
            # Check risk limits
            if not self._check_risk_limits():
                analysis_logger.warning(f"Risk limits breached at {current_time}")
                break
    
    async def _run_vectorized_backtest(self, data: pd.DataFrame, symbol: str):
        """Run vectorized backtest for performance"""
        # Simplified vectorized approach
        # Generate all signals first, then simulate execution
        
        signals = []
        
        # Generate signals for entire dataset
        for i in range(self.config.lookback_periods, len(data), 24):  # Every 24 bars
            lookback_data = data.iloc[i-self.config.lookback_periods:i+1]
            current_time = data.index[i]
            current_bar = data.iloc[i]
            
            signal = await self._generate_signal(lookback_data, symbol, current_bar)
            if signal:
                signals.append((current_time, signal))
        
        # Simulate trade execution
        for signal_time, signal in signals:
            if len(self.open_trades) < self.config.max_concurrent_trades:
                await self._execute_signal(signal, signal_time, data)
        
        # Close remaining open trades
        final_time = data.index[-1]
        final_price = data['close'].iloc[-1]
        for trade in self.open_trades[:]:
            await self._close_trade(trade, final_time, final_price, "End of backtest")
    
    async def _run_event_driven_backtest(self, data: pd.DataFrame, symbol: str):
        """Run event-driven backtest simulation"""
        # Event-driven approach with market events
        
        events = self._generate_market_events(data)
        
        for event in events:
            event_time = event['timestamp']
            event_type = event['type']
            event_data = event['data']
            
            if event_type == 'price_update':
                await self._handle_price_update_event(event_data, event_time)
            elif event_type == 'signal_check':
                await self._handle_signal_check_event(event_data, symbol, event_time)
            elif event_type == 'trade_management':
                await self._handle_trade_management_event(event_data, event_time)
    
    async def _update_open_trades(self, current_bar: pd.Series, current_time: datetime):
        """Update all open trades"""
        current_price = current_bar['close']
        high_price = current_bar['high']
        low_price = current_bar['low']
        
        for trade in self.open_trades[:]:  # Copy list to avoid modification during iteration
            # Update MAE and MFE
            if trade.signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
                # Long position
                adverse_move = trade.entry_price - low_price
                favorable_move = high_price - trade.entry_price
            else:
                # Short position
                adverse_move = high_price - trade.entry_price
                favorable_move = trade.entry_price - low_price
            
            trade.max_adverse_excursion = max(trade.max_adverse_excursion, adverse_move)
            trade.max_favorable_excursion = max(trade.max_favorable_excursion, favorable_move)
            
            # Check exit conditions
            exit_triggered = False
            exit_price = current_price
            exit_reason = ""
            
            # Check stop loss
            if trade.signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
                if low_price <= trade.stop_loss:
                    exit_triggered = True
                    exit_price = trade.stop_loss
                    exit_reason = "Stop Loss"
                elif high_price >= trade.take_profit:
                    exit_triggered = True
                    exit_price = trade.take_profit
                    exit_reason = "Take Profit"
            else:
                if high_price >= trade.stop_loss:
                    exit_triggered = True
                    exit_price = trade.stop_loss
                    exit_reason = "Stop Loss"
                elif low_price <= trade.take_profit:
                    exit_triggered = True
                    exit_price = trade.take_profit
                    exit_reason = "Take Profit"
            
            # Check time-based exit
            duration = (current_time - trade.entry_time).total_seconds() / 3600
            if duration >= self.config.max_trade_duration_hours:
                exit_triggered = True
                exit_reason = "Time Limit"
            
            if exit_triggered:
                await self._close_trade(trade, current_time, exit_price, exit_reason)
    
    async def _check_for_new_signals(self,
                                   lookback_data: pd.DataFrame,
                                   symbol: str,
                                   current_bar: pd.Series,
                                   current_time: datetime):
        """Check for new trading signals"""
        
        signal = await self._generate_signal(lookback_data, symbol, current_bar)
        
        if signal and self._validate_signal(signal):
            await self._execute_signal(signal, current_time, lookback_data)
    
    async def _generate_signal(self,
                             price_data: pd.DataFrame,
                             symbol: str,
                             current_bar: pd.Series) -> Optional[TradingSignal]:
        """Generate trading signal from current market data"""
        try:
            # Run unified analysis
            unified_result = self.unified_analyzer.analyze(
                price_data, symbol, self.config.rebalance_frequency
            )
            
            # Check minimum confidence
            if unified_result.combined_confidence < self.config.min_signal_confidence:
                return None
            
            # Generate signal
            current_price = current_bar['close']
            signal = self.signal_generator.generate_signal_from_unified(
                unified_result, current_price
            )
            
            if not signal:
                return None
            
            # Calculate risk management levels
            stop_loss_rec = self.stop_loss_calculator.calculate_stop_loss(
                signal, unified_result, price_data, self.current_capital
            )
            
            take_profit_rec = self.take_profit_calculator.calculate_take_profit(
                signal, unified_result, price_data, stop_loss_rec.primary_stop
            )
            
            # Update signal with calculated levels
            signal.stop_loss = stop_loss_rec.primary_stop
            signal.take_profit_1 = take_profit_rec.primary_targets[0] if take_profit_rec.primary_targets else None
            
            # Calculate risk/reward
            if signal.stop_loss and signal.take_profit_1:
                risk_reward_analysis = self.risk_reward_calculator.calculate_risk_reward(
                    signal, stop_loss_rec, take_profit_rec, self.current_capital
                )
                
                signal.risk_reward_ratio = risk_reward_analysis.primary_ratio
                
                # Check minimum risk/reward ratio
                if signal.risk_reward_ratio < self.config.min_risk_reward_ratio:
                    return None
            
            return signal
            
        except Exception as e:
            analysis_logger.error(f"Error generating signal: {e}")
            return None
    
    def _validate_signal(self, signal: TradingSignal) -> bool:
        """Validate signal meets requirements"""
        if not signal:
            return False
        
        # Check required fields
        if not signal.stop_loss or not signal.take_profit_1:
            return False
        
        # Check confidence
        if signal.confidence < self.config.min_signal_confidence:
            return False
        
        # Check risk/reward
        if signal.risk_reward_ratio and signal.risk_reward_ratio < self.config.min_risk_reward_ratio:
            return False
        
        return True
    
    async def _execute_signal(self,
                            signal: TradingSignal,
                            entry_time: datetime,
                            price_data: pd.DataFrame):
        """Execute trading signal"""
        try:
            # Calculate position size
            risk_amount = self.current_capital * self.config.max_risk_per_trade
            
            if signal.stop_loss:
                risk_per_unit = abs(signal.entry_price - signal.stop_loss)
                position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
            else:
                position_size = 0
            
            if position_size <= 0:
                return
            
            # Apply slippage
            if signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
                entry_price = signal.entry_price * (1 + self.config.slippage_pct)
            else:
                entry_price = signal.entry_price * (1 - self.config.slippage_pct)
            
            # Calculate commission
            commission = position_size * entry_price * self.config.commission_pct
            
            # Create trade record
            trade = BacktestTrade(
                trade_id=f"{signal.symbol}_{len(self.closed_trades) + len(self.open_trades)}",
                symbol=signal.symbol,
                signal=signal,
                entry_time=entry_time,
                entry_price=entry_price,
                position_size=position_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit_1 or 0.0,
                metadata={'commission': commission}
            )
            
            self.open_trades.append(trade)
            self.current_capital -= commission
            
            analysis_logger.debug(
                f"Executed signal: {signal.symbol} {signal.signal_type.value} "
                f"@ {entry_price:.5f}, Size: {position_size:.2f}, "
                f"SL: {signal.stop_loss:.5f}, TP: {signal.take_profit_1:.5f}"
            )
            
        except Exception as e:
            analysis_logger.error(f"Error executing signal: {e}")
    
    async def _close_trade(self,
                         trade: BacktestTrade,
                         exit_time: datetime,
                         exit_price: float,
                         exit_reason: str):
        """Close an open trade"""
        try:
            # Apply slippage to exit
            if trade.signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
                adjusted_exit_price = exit_price * (1 - self.config.slippage_pct)
            else:
                adjusted_exit_price = exit_price * (1 + self.config.slippage_pct)
            
            # Calculate PnL
            if trade.signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
                pnl = (adjusted_exit_price - trade.entry_price) * trade.position_size
            else:
                pnl = (trade.entry_price - adjusted_exit_price) * trade.position_size
            
            # Calculate commission
            exit_commission = trade.position_size * adjusted_exit_price * self.config.commission_pct
            pnl -= exit_commission
            pnl -= trade.metadata.get('commission', 0)  # Entry commission
            
            # Update trade record
            trade.exit_time = exit_time
            trade.exit_price = adjusted_exit_price
            trade.exit_reason = exit_reason
            trade.pnl = pnl
            trade.pnl_pct = (pnl / (trade.entry_price * trade.position_size)) * 100
            trade.duration_hours = (exit_time - trade.entry_time).total_seconds() / 3600
            
            # Determine trade status
            if pnl > 0:
                trade.status = TradeStatus.CLOSED_PROFIT
            elif pnl < 0:
                trade.status = TradeStatus.CLOSED_LOSS
            else:
                trade.status = TradeStatus.CLOSED_BREAKEVEN
            
            # Update capital
            self.current_capital += pnl
            
            # Move to closed trades
            self.open_trades.remove(trade)
            self.closed_trades.append(trade)
            
            analysis_logger.debug(
                f"Closed trade: {trade.trade_id} - "
                f"PnL: {pnl:.2f} ({trade.pnl_pct:.1f}%), "
                f"Duration: {trade.duration_hours:.1f}h, "
                f"Reason: {exit_reason}"
            )
            
        except Exception as e:
            analysis_logger.error(f"Error closing trade: {e}")
    
    def _update_equity_tracking(self, current_time: datetime):
        """Update equity curve and drawdown tracking"""
        # Calculate current equity (capital + unrealized PnL)
        unrealized_pnl = 0.0
        for trade in self.open_trades:
            # Simplified unrealized PnL calculation
            pass  # Would need current price to calculate
        
        current_equity = self.current_capital + unrealized_pnl
        
        # Track equity history
        self.equity_history.append({
            'timestamp': current_time,
            'equity': current_equity,
            'capital': self.current_capital,
            'unrealized_pnl': unrealized_pnl,
            'open_trades': len(self.open_trades)
        })
        
        # Update peak equity and drawdown tracking
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            self.max_drawdown_start = None
        elif current_equity < self.peak_equity:
            if not self.max_drawdown_start:
                self.max_drawdown_start = current_time
            self.max_drawdown_end = current_time
    
    def _check_risk_limits(self) -> bool:
        """Check if risk limits are breached"""
        # Check max drawdown
        if self.peak_equity > 0:
            current_drawdown = (self.peak_equity - self.current_capital) / self.peak_equity
            if current_drawdown > self.config.max_drawdown_limit:
                return False
        
        # Check daily loss limit (simplified)
        if len(self.equity_history) > 24:  # Assuming hourly data
            day_start_equity = self.equity_history[-24]['equity']
            daily_return = (self.current_capital - day_start_equity) / day_start_equity
            if daily_return < -self.config.daily_loss_limit:
                return False
        
        return True
    
    def _prepare_backtest_data(self,
                             price_data: pd.DataFrame,
                             start_date: Optional[datetime],
                             end_date: Optional[datetime]) -> pd.DataFrame:
        """Prepare data for backtesting"""
        data = price_data.copy()
        
        # Filter by date range
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        # Ensure required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                if col == 'volume':
                    data[col] = 1000  # Default volume
                else:
                    raise ValueError(f"Required column {col} not found in data")
        
        return data
    
    def _reset_backtest_state(self):
        """Reset backtest state for new run"""
        self.current_capital = self.config.initial_capital
        self.open_trades = []
        self.closed_trades = []
        self.equity_history = []
        self.daily_pnl = []
        self.peak_equity = self.config.initial_capital
        self.max_drawdown_start = None
        self.max_drawdown_end = None
    
    def _generate_market_events(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate market events for event-driven backtest"""
        events = []
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            # Price update event
            events.append({
                'timestamp': timestamp,
                'type': 'price_update',
                'data': row
            })
            
            # Signal check event (every few bars)
            if i % 4 == 0 and i >= self.config.lookback_periods:
                events.append({
                    'timestamp': timestamp,
                    'type': 'signal_check',
                    'data': data.iloc[max(0, i-self.config.lookback_periods):i+1]
                })
            
            # Trade management event
            if len(self.open_trades) > 0:
                events.append({
                    'timestamp': timestamp,
                    'type': 'trade_management',
                    'data': row
                })
        
        return events
    
    async def _handle_price_update_event(self, price_data: pd.Series, timestamp: datetime):
        """Handle price update event"""
        await self._update_open_trades(price_data, timestamp)
        self._update_equity_tracking(timestamp)
    
    async def _handle_signal_check_event(self,
                                       price_data: pd.DataFrame,
                                       symbol: str,
                                       timestamp: datetime):
        """Handle signal check event"""
        if len(self.open_trades) < self.config.max_concurrent_trades:
            current_bar = price_data.iloc[-1]
            await self._check_for_new_signals(price_data, symbol, current_bar, timestamp)
    
    async def _handle_trade_management_event(self, price_data: pd.Series, timestamp: datetime):
        """Handle trade management event"""
        # Additional trade management logic could go here
        pass
    
    def _close_trade_sync(self,
                         trade: BacktestTrade,
                         exit_time: datetime,
                         exit_price: float,
                         exit_reason: str):
        """Close an open trade (synchronous version)"""
        try:
            # Apply slippage to exit
            if trade.signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
                adjusted_exit_price = exit_price * (1 - self.config.slippage_pct)
            else:
                adjusted_exit_price = exit_price * (1 + self.config.slippage_pct)
            
            # Calculate PnL
            if trade.signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
                pnl = (adjusted_exit_price - trade.entry_price) * trade.position_size
            else:
                pnl = (trade.entry_price - adjusted_exit_price) * trade.position_size
            
            # Calculate commission
            exit_commission = trade.position_size * adjusted_exit_price * self.config.commission_pct
            pnl -= exit_commission
            pnl -= trade.metadata.get('commission', 0)  # Entry commission
            
            # Update trade record
            trade.exit_time = exit_time
            trade.exit_price = adjusted_exit_price
            trade.exit_reason = exit_reason
            trade.pnl = pnl
            trade.pnl_pct = (pnl / (trade.entry_price * trade.position_size)) * 100
            trade.duration_hours = (exit_time - trade.entry_time).total_seconds() / 3600
            
            # Determine trade status
            if pnl > 0:
                trade.status = TradeStatus.CLOSED_PROFIT
            elif pnl < 0:
                trade.status = TradeStatus.CLOSED_LOSS
            else:
                trade.status = TradeStatus.CLOSED_BREAKEVEN
            
            # Update capital
            self.current_capital += pnl
            
            # Move to closed trades
            self.open_trades.remove(trade)
            self.closed_trades.append(trade)
            
        except Exception as e:
            analysis_logger.error(f"Error closing trade: {e}")
    
    def _calculate_metrics(self, price_data: pd.DataFrame) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics"""
        try:
            # Close any remaining open trades
            if self.open_trades:
                final_time = price_data.index[-1]
                final_price = price_data['close'].iloc[-1]
                for trade in self.open_trades[:]:
                    self._close_trade_sync(trade, final_time, final_price, "End of backtest")
            
            if not self.closed_trades:
                return self._create_empty_metrics(price_data)
            
            # Basic trade metrics
            total_trades = len(self.closed_trades)
            winning_trades = len([t for t in self.closed_trades if t.pnl > 0])
            losing_trades = len([t for t in self.closed_trades if t.pnl < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            # PnL metrics
            pnls = [t.pnl for t in self.closed_trades]
            total_pnl = sum(pnls)
            total_return_pct = (total_pnl / self.config.initial_capital) * 100
            
            winning_pnls = [t.pnl for t in self.closed_trades if t.pnl > 0]
            losing_pnls = [t.pnl for t in self.closed_trades if t.pnl < 0]
            
            avg_win = np.mean(winning_pnls) if winning_pnls else 0.0
            avg_loss = np.mean(losing_pnls) if losing_pnls else 0.0
            largest_win = max(pnls) if pnls else 0.0
            largest_loss = min(pnls) if pnls else 0.0
            
            # Risk metrics
            returns = [t.pnl / self.config.initial_capital for t in self.closed_trades]
            
            # Calculate drawdown from equity curve
            equity_values = [e['equity'] for e in self.equity_history]
            if equity_values:
                peak = np.maximum.accumulate(equity_values)
                drawdown = (peak - equity_values) / peak
                max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
                
                # Drawdown duration
                max_dd_idx = np.argmax(drawdown)
                max_drawdown_duration = max_dd_idx  # Simplified
            else:
                max_drawdown = 0.0
                max_drawdown_duration = 0
            
            # Volatility and ratios
            volatility = np.std(returns) if returns else 0.0
            mean_return = np.mean(returns) if returns else 0.0
            
            # Sharpe ratio (assuming 0% risk-free rate)
            sharpe_ratio = mean_return / volatility if volatility > 0 else 0.0
            
            # Sortino ratio
            negative_returns = [r for r in returns if r < 0]
            downside_vol = np.std(negative_returns) if negative_returns else 0.0
            sortino_ratio = mean_return / downside_vol if downside_vol > 0 else 0.0
            
            # Calmar ratio
            calmar_ratio = mean_return / max_drawdown if max_drawdown > 0 else 0.0
            
            # Trade duration
            durations = [t.duration_hours for t in self.closed_trades]
            avg_trade_duration = np.mean(durations) if durations else 0.0
            
            # Profit factor
            gross_profit = sum(winning_pnls) if winning_pnls else 0.0
            gross_loss = abs(sum(losing_pnls)) if losing_pnls else 0.0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
            
            # Expectancy
            expectancy = (avg_win * win_rate) - (abs(avg_loss) * (1 - win_rate))
            
            # Value at Risk (95%)
            var_95 = np.percentile(returns, 5) if returns else 0.0
            
            # Conditional Value at Risk
            var_returns = [r for r in returns if r <= var_95]
            cvar_95 = np.mean(var_returns) if var_returns else 0.0
            
            # Recovery factor
            recovery_factor = total_return_pct / (max_drawdown * 100) if max_drawdown > 0 else 0.0
            
            # Execution costs
            total_commission = sum(t.metadata.get('commission', 0) for t in self.closed_trades)
            total_slippage = 0.0  # Would need to calculate from execution prices
            
            # Dates
            start_date = self.closed_trades[0].entry_time if self.closed_trades else price_data.index[0]
            end_date = self.closed_trades[-1].exit_time if self.closed_trades else price_data.index[-1]
            total_duration_days = (end_date - start_date).days
            
            return BacktestMetrics(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                total_return_pct=total_return_pct,
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                max_drawdown=max_drawdown,
                max_drawdown_duration=max_drawdown_duration,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                avg_trade_duration=avg_trade_duration,
                profit_factor=profit_factor,
                expectancy=expectancy,
                var_95=var_95,
                cvar_95=cvar_95,
                recovery_factor=recovery_factor,
                total_commission=total_commission,
                total_slippage=total_slippage,
                start_date=start_date,
                end_date=end_date,
                total_duration_days=total_duration_days
            )
            
        except Exception as e:
            analysis_logger.error(f"Error calculating metrics: {e}")
            return self._create_empty_metrics(price_data)
    
    def _create_empty_metrics(self, price_data: pd.DataFrame) -> BacktestMetrics:
        """Create empty metrics for when no trades occurred"""
        return BacktestMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            total_return_pct=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            avg_trade_duration=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            var_95=0.0,
            cvar_95=0.0,
            recovery_factor=0.0,
            total_commission=0.0,
            total_slippage=0.0,
            start_date=price_data.index[0],
            end_date=price_data.index[-1],
            total_duration_days=(price_data.index[-1] - price_data.index[0]).days
        )
    
    def _create_equity_curve(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Create equity curve DataFrame"""
        if not self.equity_history:
            return pd.DataFrame()
        
        equity_df = pd.DataFrame(self.equity_history)
        equity_df.set_index('timestamp', inplace=True)
        
        # Add additional metrics
        equity_df['returns'] = equity_df['equity'].pct_change()
        equity_df['cumulative_returns'] = (1 + equity_df['returns']).cumprod() - 1
        
        # Calculate running drawdown
        peak = equity_df['equity'].expanding().max()
        equity_df['drawdown'] = (peak - equity_df['equity']) / peak
        
        return equity_df
    
    def _analyze_signals(self) -> Dict[str, Any]:
        """Analyze signal generation patterns"""
        if not self.closed_trades:
            return {}
        
        signal_types = [t.signal.signal_type.value for t in self.closed_trades]
        confidence_scores = [t.signal.confidence for t in self.closed_trades]
        
        return {
            'signal_distribution': {signal: signal_types.count(signal) for signal in set(signal_types)},
            'avg_signal_confidence': np.mean(confidence_scores),
            'confidence_vs_performance': self._analyze_confidence_performance(),
            'signal_timing_analysis': self._analyze_signal_timing()
        }
    
    def _analyze_risk(self) -> Dict[str, Any]:
        """Analyze risk management effectiveness"""
        if not self.closed_trades:
            return {}
        
        stop_losses_hit = len([t for t in self.closed_trades if t.exit_reason == "Stop Loss"])
        take_profits_hit = len([t for t in self.closed_trades if t.exit_reason == "Take Profit"])
        
        return {
            'stop_loss_hit_rate': stop_losses_hit / len(self.closed_trades),
            'take_profit_hit_rate': take_profits_hit / len(self.closed_trades),
            'risk_reward_distribution': self._analyze_risk_reward_distribution(),
            'mae_mfe_analysis': self._analyze_mae_mfe()
        }
    
    def _analyze_confidence_performance(self) -> Dict[str, float]:
        """Analyze relationship between confidence and performance"""
        if not self.closed_trades:
            return {}
        
        high_conf_trades = [t for t in self.closed_trades if t.signal.confidence >= 0.8]
        med_conf_trades = [t for t in self.closed_trades if 0.6 <= t.signal.confidence < 0.8]
        low_conf_trades = [t for t in self.closed_trades if t.signal.confidence < 0.6]
        
        return {
            'high_confidence_win_rate': len([t for t in high_conf_trades if t.pnl > 0]) / len(high_conf_trades) if high_conf_trades else 0.0,
            'med_confidence_win_rate': len([t for t in med_conf_trades if t.pnl > 0]) / len(med_conf_trades) if med_conf_trades else 0.0,
            'low_confidence_win_rate': len([t for t in low_conf_trades if t.pnl > 0]) / len(low_conf_trades) if low_conf_trades else 0.0
        }
    
    def _analyze_signal_timing(self) -> Dict[str, Any]:
        """Analyze signal timing effectiveness"""
        if not self.closed_trades:
            return {}
        
        urgency_performance = {}
        for urgency in ['IMMEDIATE', 'HIGH', 'MEDIUM', 'LOW']:
            urgency_trades = [t for t in self.closed_trades if t.signal.urgency.value == urgency]
            if urgency_trades:
                win_rate = len([t for t in urgency_trades if t.pnl > 0]) / len(urgency_trades)
                urgency_performance[urgency] = win_rate
        
        return {'urgency_performance': urgency_performance}
    
    def _analyze_risk_reward_distribution(self) -> Dict[str, int]:
        """Analyze distribution of risk/reward ratios"""
        if not self.closed_trades:
            return {}
        
        rr_ratios = [t.signal.risk_reward_ratio for t in self.closed_trades if t.signal.risk_reward_ratio]
        
        distribution = {
            '< 1.5': len([r for r in rr_ratios if r < 1.5]),
            '1.5-2.0': len([r for r in rr_ratios if 1.5 <= r < 2.0]),
            '2.0-3.0': len([r for r in rr_ratios if 2.0 <= r < 3.0]),
            '> 3.0': len([r for r in rr_ratios if r >= 3.0])
        }
        
        return distribution
    
    def _analyze_mae_mfe(self) -> Dict[str, float]:
        """Analyze Maximum Adverse/Favorable Excursion"""
        if not self.closed_trades:
            return {}
        
        maes = [t.max_adverse_excursion for t in self.closed_trades]
        mfes = [t.max_favorable_excursion for t in self.closed_trades]
        
        return {
            'avg_mae': np.mean(maes) if maes else 0.0,
            'avg_mfe': np.mean(mfes) if mfes else 0.0,
            'mae_efficiency': len([t for t in self.closed_trades if t.max_adverse_excursion < abs(t.pnl)]) / len(self.closed_trades)
        }


# Create default backtest engine
backtest_engine = BacktestEngine()