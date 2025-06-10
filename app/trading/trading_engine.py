"""
Trading Engine

Main trading engine that orchestrates all trading components including
signal processing, order execution, position management, and risk control.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio

from app.trading.order_manager import OrderManager, order_manager
from app.trading.execution_engine import ExecutionEngine, ExecutionRequest, ExecutionMode, ExecutionPriority
from app.trading.position_manager import PositionManager, position_manager
from app.trading.risk_manager import RiskManager, risk_manager
from app.analysis.strategy_engine.signal_generator import TradingSignal
from app.analysis.strategy_engine.unified_analyzer import unified_analyzer
from app.analysis.strategy_engine.stop_loss_calculator import stop_loss_calculator
from app.analysis.strategy_engine.take_profit_calculator import take_profit_calculator
from app.analysis.strategy_engine.risk_reward_calculator import risk_reward_calculator
from app.mt5.connection import MT5Connection
from app.utils.logger import analysis_logger


class TradingMode(Enum):
    """Trading mode enumeration"""
    LIVE = "live"
    DEMO = "demo"
    SIMULATION = "simulation"
    BACKTEST = "backtest"


class TradingStatus(Enum):
    """Trading status enumeration"""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class TradingConfig:
    """Trading engine configuration"""
    trading_mode: TradingMode = TradingMode.DEMO
    auto_trading_enabled: bool = True
    signal_processing_enabled: bool = True
    risk_management_enabled: bool = True
    position_management_enabled: bool = True
    
    # Timing configuration
    signal_check_interval_seconds: int = 60
    position_update_interval_seconds: int = 30
    risk_monitoring_interval_seconds: int = 30
    
    # Trading limits
    max_concurrent_signals: int = 5
    signal_timeout_minutes: int = 60
    enable_partial_fills: bool = True
    
    # Strategy configuration
    default_symbols: List[str] = field(default_factory=lambda: ['EURUSD', 'USDJPY', 'GBPUSD'])
    default_timeframe: str = 'H1'
    min_signal_confidence: float = 0.6
    
    # Execution configuration
    default_execution_mode: ExecutionMode = ExecutionMode.IMMEDIATE
    default_priority: ExecutionPriority = ExecutionPriority.MEDIUM


@dataclass
class TradingSession:
    """Trading session information"""
    session_id: str
    start_time: datetime
    mode: TradingMode
    status: TradingStatus
    
    # Session statistics
    signals_processed: int = 0
    orders_placed: int = 0
    positions_opened: int = 0
    positions_closed: int = 0
    total_pnl: float = 0.0
    
    # Timing
    last_signal_check: Optional[datetime] = None
    last_position_update: Optional[datetime] = None
    last_risk_check: Optional[datetime] = None
    
    end_time: Optional[datetime] = None


class TradingEngine:
    """
    Main trading engine orchestrating all trading operations
    
    Coordinates:
    - Signal generation and processing
    - Order execution and management
    - Position tracking and management
    - Risk monitoring and control
    - Performance tracking and reporting
    """
    
    def __init__(self, mt5_connection: Optional[MT5Connection] = None):
        """Initialize trading engine"""
        
        self.mt5_connection = mt5_connection
        
        # Initialize components
        self.order_manager = order_manager
        self.execution_engine = ExecutionEngine(self.order_manager, mt5_connection)
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        
        # Set up component dependencies
        self.position_manager.order_manager = self.order_manager
        self.risk_manager.position_manager = self.position_manager
        
        # Configuration
        self.config = TradingConfig()
        
        # Session management
        self.current_session: Optional[TradingSession] = None
        self.session_history: List[TradingSession] = []
        
        # Signal processing
        self.pending_signals: Dict[str, TradingSignal] = {}
        self.processed_signals: List[TradingSignal] = []
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.running = False
        
        # Statistics
        self.stats = {
            'total_sessions': 0,
            'total_signals_processed': 0,
            'total_orders_placed': 0,
            'total_positions_opened': 0,
            'total_pnl': 0.0,
            'avg_session_duration_hours': 0.0,
            'success_rate': 0.0
        }
        
        analysis_logger.info("Trading Engine initialized")
    
    async def start_trading_session(self, mode: TradingMode = TradingMode.DEMO) -> str:
        """Start new trading session"""
        
        try:
            if self.current_session and self.current_session.status == TradingStatus.ACTIVE:
                analysis_logger.warning("Trading session already active")
                return self.current_session.session_id
            
            # Create new session
            session_id = f"session_{int(datetime.now().timestamp())}"
            self.current_session = TradingSession(
                session_id=session_id,
                start_time=datetime.now(),
                mode=mode,
                status=TradingStatus.ACTIVE
            )
            
            # Update configuration
            self.config.trading_mode = mode
            
            # Start background tasks
            if not self.running:
                await self._start_background_tasks()
            
            # Update statistics
            self.stats['total_sessions'] += 1
            
            analysis_logger.info(f"Trading session started: {session_id} (Mode: {mode.value})")
            
            return session_id
            
        except Exception as e:
            analysis_logger.error(f"Error starting trading session: {e}")
            raise
    
    async def stop_trading_session(self):
        """Stop current trading session"""
        
        try:
            if not self.current_session:
                analysis_logger.warning("No active trading session")
                return
            
            # Update session status
            self.current_session.status = TradingStatus.STOPPED
            self.current_session.end_time = datetime.now()
            
            # Stop background tasks
            await self._stop_background_tasks()
            
            # Calculate session statistics
            duration = (self.current_session.end_time - self.current_session.start_time).total_seconds() / 3600
            self.current_session.total_pnl = await self._calculate_session_pnl()
            
            # Move to history
            self.session_history.append(self.current_session)
            
            # Update global statistics
            self.stats['total_signals_processed'] += self.current_session.signals_processed
            self.stats['total_orders_placed'] += self.current_session.orders_placed
            self.stats['total_positions_opened'] += self.current_session.positions_opened
            self.stats['total_pnl'] += self.current_session.total_pnl
            
            # Update average duration
            n = len(self.session_history)
            self.stats['avg_session_duration_hours'] = (
                (self.stats['avg_session_duration_hours'] * (n - 1) + duration) / n
            )
            
            analysis_logger.info(
                f"Trading session stopped: {self.current_session.session_id} - "
                f"Duration: {duration:.1f}h, PnL: {self.current_session.total_pnl:.2f}"
            )
            
            self.current_session = None
            
        except Exception as e:
            analysis_logger.error(f"Error stopping trading session: {e}")
    
    async def pause_trading(self):
        """Pause trading operations"""
        
        try:
            if not self.current_session:
                analysis_logger.warning("No active trading session")
                return
            
            self.current_session.status = TradingStatus.PAUSED
            
            analysis_logger.info("Trading paused")
            
        except Exception as e:
            analysis_logger.error(f"Error pausing trading: {e}")
    
    async def resume_trading(self):
        """Resume trading operations"""
        
        try:
            if not self.current_session:
                analysis_logger.warning("No active trading session")
                return
            
            if self.current_session.status == TradingStatus.PAUSED:
                self.current_session.status = TradingStatus.ACTIVE
                
                analysis_logger.info("Trading resumed")
            
        except Exception as e:
            analysis_logger.error(f"Error resuming trading: {e}")
    
    async def emergency_stop(self):
        """Emergency stop all trading operations"""
        
        try:
            # Activate risk manager emergency stop
            await self.risk_manager.emergency_stop("Manual emergency stop")
            
            # Update session status
            if self.current_session:
                self.current_session.status = TradingStatus.EMERGENCY_STOP
            
            # Stop all background tasks
            await self._stop_background_tasks()
            
            analysis_logger.critical("Emergency stop activated")
            
        except Exception as e:
            analysis_logger.error(f"Error in emergency stop: {e}")
    
    async def process_signal(self, signal: TradingSignal) -> bool:
        """Process incoming trading signal"""
        
        try:
            if not self._can_process_signals():
                return False
            
            # Validate signal
            if not self._validate_signal(signal):
                return False
            
            # Add to pending signals
            self.pending_signals[signal.signal_id] = signal
            
            # Process signal asynchronously
            asyncio.create_task(self._process_single_signal(signal))
            
            # Update session statistics
            if self.current_session:
                self.current_session.signals_processed += 1
                self.current_session.last_signal_check = datetime.now()
            
            analysis_logger.info(f"Signal queued for processing: {signal.signal_id}")
            
            return True
            
        except Exception as e:
            analysis_logger.error(f"Error processing signal: {e}")
            return False
    
    async def get_trading_status(self) -> Dict[str, Any]:
        """Get current trading status"""
        
        try:
            # Get component statuses
            order_stats = self.order_manager.get_statistics()
            execution_stats = self.execution_engine.get_execution_statistics()
            position_stats = self.position_manager.get_statistics()
            risk_stats = self.risk_manager.get_statistics()
            
            # Get portfolio summary
            portfolio_summary = await self.position_manager.get_portfolio_summary()
            
            # Get risk report
            risk_report = await self.risk_manager.get_risk_report()
            
            return {
                'session': {
                    'session_id': self.current_session.session_id if self.current_session else None,
                    'status': self.current_session.status.value if self.current_session else 'stopped',
                    'mode': self.current_session.mode.value if self.current_session else None,
                    'start_time': self.current_session.start_time if self.current_session else None,
                    'signals_processed': self.current_session.signals_processed if self.current_session else 0,
                    'orders_placed': self.current_session.orders_placed if self.current_session else 0,
                    'positions_opened': self.current_session.positions_opened if self.current_session else 0
                },
                'components': {
                    'order_manager': order_stats,
                    'execution_engine': execution_stats,
                    'position_manager': position_stats,
                    'risk_manager': risk_stats
                },
                'portfolio': portfolio_summary,
                'risk': risk_report['risk_metrics'] if risk_report else {},
                'pending_signals': len(self.pending_signals),
                'background_tasks_running': self.running
            }
            
        except Exception as e:
            analysis_logger.error(f"Error getting trading status: {e}")
            return {}
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        
        try:
            # Session performance
            session_performance = []
            for session in self.session_history[-10:]:  # Last 10 sessions
                duration = 0.0
                if session.end_time:
                    duration = (session.end_time - session.start_time).total_seconds() / 3600
                
                session_performance.append({
                    'session_id': session.session_id,
                    'start_time': session.start_time,
                    'end_time': session.end_time,
                    'duration_hours': duration,
                    'mode': session.mode.value,
                    'signals_processed': session.signals_processed,
                    'orders_placed': session.orders_placed,
                    'positions_opened': session.positions_opened,
                    'total_pnl': session.total_pnl
                })
            
            # Component performance
            component_stats = {
                'order_manager': self.order_manager.get_statistics(),
                'execution_engine': self.execution_engine.get_execution_statistics(),
                'position_manager': self.position_manager.get_statistics(),
                'risk_manager': self.risk_manager.get_statistics()
            }
            
            # Portfolio metrics
            portfolio_summary = await self.position_manager.get_portfolio_summary()
            
            return {
                'overall_statistics': self.stats,
                'session_performance': session_performance,
                'component_statistics': component_stats,
                'portfolio_summary': portfolio_summary,
                'current_session': {
                    'session_id': self.current_session.session_id if self.current_session else None,
                    'status': self.current_session.status.value if self.current_session else 'stopped',
                    'signals_processed': self.current_session.signals_processed if self.current_session else 0
                }
            }
            
        except Exception as e:
            analysis_logger.error(f"Error generating performance report: {e}")
            return {}
    
    async def _start_background_tasks(self):
        """Start background monitoring tasks"""
        
        try:
            if self.running:
                return
            
            self.running = True
            
            # Start monitoring tasks
            self.background_tasks = [
                asyncio.create_task(self._signal_processing_loop()),
                asyncio.create_task(self._position_monitoring_loop()),
                asyncio.create_task(self._risk_monitoring_loop()),
                asyncio.create_task(self._order_status_loop())
            ]
            
            analysis_logger.info("Background tasks started")
            
        except Exception as e:
            analysis_logger.error(f"Error starting background tasks: {e}")
    
    async def _stop_background_tasks(self):
        """Stop background monitoring tasks"""
        
        try:
            self.running = False
            
            # Cancel all background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            self.background_tasks.clear()
            
            analysis_logger.info("Background tasks stopped")
            
        except Exception as e:
            analysis_logger.error(f"Error stopping background tasks: {e}")
    
    async def _signal_processing_loop(self):
        """Background signal processing loop"""
        
        try:
            while self.running:
                if self.config.signal_processing_enabled and self._can_process_signals():
                    # Generate signals for configured symbols
                    for symbol in self.config.default_symbols:
                        await self._check_for_new_signals(symbol)
                
                await asyncio.sleep(self.config.signal_check_interval_seconds)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            analysis_logger.error(f"Error in signal processing loop: {e}")
    
    async def _position_monitoring_loop(self):
        """Background position monitoring loop"""
        
        try:
            while self.running:
                if self.config.position_management_enabled:
                    # Update position PnL
                    await self.position_manager.update_positions_pnl()
                    
                    # Update session last position update
                    if self.current_session:
                        self.current_session.last_position_update = datetime.now()
                
                await asyncio.sleep(self.config.position_update_interval_seconds)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            analysis_logger.error(f"Error in position monitoring loop: {e}")
    
    async def _risk_monitoring_loop(self):
        """Background risk monitoring loop"""
        
        try:
            while self.running:
                if self.config.risk_management_enabled:
                    # Monitor risk limits
                    await self.risk_manager.monitor_risk_limits()
                    
                    # Update session last risk check
                    if self.current_session:
                        self.current_session.last_risk_check = datetime.now()
                
                await asyncio.sleep(self.config.risk_monitoring_interval_seconds)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            analysis_logger.error(f"Error in risk monitoring loop: {e}")
    
    async def _order_status_loop(self):
        """Background order status monitoring loop"""
        
        try:
            while self.running:
                # Update order statuses
                await self.order_manager.update_orders_status()
                
                # Process execution queue
                await self.execution_engine.process_execution_queue()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            analysis_logger.error(f"Error in order status loop: {e}")
    
    async def _process_single_signal(self, signal: TradingSignal):
        """Process a single trading signal"""
        
        try:
            # Calculate risk management components
            stop_loss_rec = await stop_loss_calculator.calculate_stop_loss(signal, None, None)
            take_profit_rec = await take_profit_calculator.calculate_take_profit(signal, None, None)
            risk_analysis = await risk_reward_calculator.calculate_risk_reward(signal, stop_loss_rec, take_profit_rec)
            
            # Validate with risk manager
            validation = await self.risk_manager.validate_new_position(signal, risk_analysis)
            
            if not validation['approved']:
                analysis_logger.warning(f"Signal rejected by risk manager: {signal.signal_id} - {validation['errors']}")
                self.pending_signals.pop(signal.signal_id, None)
                return
            
            # Adjust position size if needed
            if validation['adjusted_size']:
                risk_analysis.recommended_position_size = validation['adjusted_size']
            
            # Create execution request
            execution_request = ExecutionRequest(
                signal=signal,
                stop_loss_rec=stop_loss_rec,
                take_profit_rec=take_profit_rec,
                risk_analysis=risk_analysis,
                execution_mode=self.config.default_execution_mode,
                priority=self.config.default_priority
            )
            
            # Execute signal
            execution_result = await self.execution_engine.execute_signal(execution_request)
            
            if execution_result.success:
                # Create position from successful execution
                for order_execution in execution_result.orders_placed:
                    if order_execution.success:
                        position = await self.position_manager.open_position_from_execution(order_execution, signal)
                        
                        if position:
                            # Update session statistics
                            if self.current_session:
                                self.current_session.orders_placed += 1
                                self.current_session.positions_opened += 1
                
                analysis_logger.info(f"Signal executed successfully: {signal.signal_id}")
            else:
                analysis_logger.warning(f"Signal execution failed: {signal.signal_id} - {execution_result.error_message}")
            
            # Remove from pending
            self.pending_signals.pop(signal.signal_id, None)
            self.processed_signals.append(signal)
            
        except Exception as e:
            analysis_logger.error(f"Error processing signal {signal.signal_id}: {e}")
            self.pending_signals.pop(signal.signal_id, None)
    
    async def _check_for_new_signals(self, symbol: str):
        """Check for new trading signals for symbol"""
        
        try:
            # Mock signal generation - in real system would use strategy engine
            # This is just for demonstration
            pass
            
        except Exception as e:
            analysis_logger.error(f"Error checking for signals on {symbol}: {e}")
    
    def _can_process_signals(self) -> bool:
        """Check if signals can be processed"""
        
        if not self.current_session:
            return False
        
        if self.current_session.status != TradingStatus.ACTIVE:
            return False
        
        if not self.config.auto_trading_enabled:
            return False
        
        if self.risk_manager.emergency_stop_active:
            return False
        
        return True
    
    def _validate_signal(self, signal: TradingSignal) -> bool:
        """Validate incoming signal"""
        
        try:
            # Check signal confidence
            if signal.confidence < self.config.min_signal_confidence:
                analysis_logger.warning(f"Signal confidence too low: {signal.confidence}")
                return False
            
            # Check if signal already processed
            if signal.signal_id in self.pending_signals:
                analysis_logger.warning(f"Signal already pending: {signal.signal_id}")
                return False
            
            # Check signal age
            if signal.timestamp and (datetime.now() - signal.timestamp).total_seconds() > self.config.signal_timeout_minutes * 60:
                analysis_logger.warning(f"Signal too old: {signal.signal_id}")
                return False
            
            # Check pending signals limit
            if len(self.pending_signals) >= self.config.max_concurrent_signals:
                analysis_logger.warning("Too many pending signals")
                return False
            
            return True
            
        except Exception as e:
            analysis_logger.error(f"Error validating signal: {e}")
            return False
    
    async def _calculate_session_pnl(self) -> float:
        """Calculate total PnL for current session"""
        
        try:
            portfolio_summary = await self.position_manager.get_portfolio_summary()
            return portfolio_summary.get('total_realized_pnl', 0.0) + portfolio_summary.get('total_unrealized_pnl', 0.0)
            
        except Exception as e:
            analysis_logger.error(f"Error calculating session PnL: {e}")
            return 0.0


# Global instance
trading_engine = TradingEngine()