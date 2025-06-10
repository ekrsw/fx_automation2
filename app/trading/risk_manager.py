"""
Risk Management System

Comprehensive risk management system for FX trading including position sizing,
maximum loss limits, drawdown monitoring, and emergency stop functionality.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio

from app.trading.position_manager import PositionManager, Position, PositionType, PositionCloseRequest
from app.analysis.strategy_engine.signal_generator import TradingSignal
from app.analysis.strategy_engine.risk_reward_calculator import RiskRewardAnalysis
from app.utils.logger import analysis_logger


class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Risk alert types"""
    POSITION_SIZE = "position_size"
    DAILY_LOSS = "daily_loss"
    DRAWDOWN = "drawdown"
    CONCENTRATION = "concentration"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"
    MARGIN = "margin"


@dataclass
class RiskLimits:
    """Risk limit configuration"""
    max_position_size: float = 1.0  # Maximum position size
    max_risk_per_trade: float = 0.02  # 2% per trade
    max_daily_loss: float = 0.05  # 5% daily loss limit
    max_weekly_loss: float = 0.10  # 10% weekly loss limit
    max_drawdown: float = 0.15  # 15% maximum drawdown
    max_positions: int = 10  # Maximum concurrent positions
    max_correlation: float = 0.8  # Maximum correlation between positions
    max_concentration: float = 0.3  # Maximum concentration in single symbol
    
    # Emergency stops
    emergency_stop_loss: float = 0.20  # 20% emergency stop
    force_close_threshold: float = 0.25  # 25% force close all
    
    # Margin requirements
    margin_call_level: float = 0.50  # 50% margin call
    stop_out_level: float = 0.20  # 20% stop out


@dataclass
class RiskAlert:
    """Risk alert structure"""
    alert_id: str
    alert_type: AlertType
    risk_level: RiskLevel
    message: str
    value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskMetrics:
    """Current risk metrics"""
    # Position metrics
    total_positions: int = 0
    total_exposure: float = 0.0
    largest_position: float = 0.0
    concentration_risk: float = 0.0
    
    # PnL metrics
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    
    # Risk ratios
    risk_per_trade: float = 0.0
    portfolio_risk: float = 0.0
    correlation_risk: float = 0.0
    
    # Account metrics
    account_balance: float = 10000.0  # Mock balance
    used_margin: float = 0.0
    free_margin: float = 10000.0
    margin_level: float = 1.0
    
    # Risk score (0-100)
    overall_risk_score: float = 0.0


class RiskManager:
    """
    Comprehensive risk management system
    
    Monitors and manages trading risk including:
    - Position sizing and limits
    - Daily, weekly, and drawdown limits
    - Portfolio concentration and correlation
    - Emergency stop mechanisms
    - Real-time risk monitoring and alerts
    """
    
    def __init__(self, position_manager: PositionManager):
        """Initialize risk manager"""
        
        self.position_manager = position_manager
        
        # Risk configuration
        self.risk_limits = RiskLimits()
        
        # Risk tracking
        self.alerts: List[RiskAlert] = []
        self.active_alerts: Dict[str, RiskAlert] = {}
        self.risk_metrics = RiskMetrics()
        
        # Historical tracking
        self.daily_pnl_history: List[Tuple[datetime, float]] = []
        self.drawdown_history: List[Tuple[datetime, float]] = []
        self.high_water_mark: float = 10000.0  # Starting balance
        
        # Emergency state
        self.emergency_stop_active: bool = False
        self.force_close_pending: bool = False
        
        # Configuration
        self.config = {
            'enable_position_sizing': True,
            'enable_daily_limits': True,
            'enable_drawdown_monitoring': True,
            'enable_correlation_monitoring': True,
            'enable_emergency_stops': True,
            'alert_cooldown_minutes': 5,
            'risk_update_interval_seconds': 30,
            'max_alerts_per_type': 10,
            'enable_force_close': True
        }
        
        # Statistics
        self.stats = {
            'total_alerts': 0,
            'critical_alerts': 0,
            'emergency_stops': 0,
            'force_closes': 0,
            'positions_rejected': 0,
            'avg_position_risk': 0.0,
            'max_portfolio_risk': 0.0,
            'max_drawdown_reached': 0.0
        }
        
        analysis_logger.info("Risk Manager initialized")
    
    async def validate_new_position(self, signal: TradingSignal, risk_analysis: Optional[RiskRewardAnalysis] = None) -> Dict[str, Any]:
        """Validate if new position meets risk criteria"""
        
        try:
            validation_result = {
                'approved': True,
                'adjusted_size': None,
                'warnings': [],
                'errors': []
            }
            
            # Check if emergency stop is active
            if self.emergency_stop_active:
                validation_result['approved'] = False
                validation_result['errors'].append("Emergency stop is active")
                return validation_result
            
            # Get proposed position size
            if risk_analysis and risk_analysis.recommended_position_size:
                position_size = risk_analysis.recommended_position_size
            else:
                position_size = 0.1  # Default
            
            # Check maximum position size
            if position_size > self.risk_limits.max_position_size:
                if self.config['enable_position_sizing']:
                    validation_result['adjusted_size'] = self.risk_limits.max_position_size
                    validation_result['warnings'].append(f"Position size reduced from {position_size} to {self.risk_limits.max_position_size}")
                else:
                    validation_result['approved'] = False
                    validation_result['errors'].append(f"Position size {position_size} exceeds limit {self.risk_limits.max_position_size}")
            
            # Check maximum positions
            open_positions = await self.position_manager.get_open_positions()
            if len(open_positions) >= self.risk_limits.max_positions:
                validation_result['approved'] = False
                validation_result['errors'].append(f"Maximum positions limit reached: {len(open_positions)}")
            
            # Check risk per trade
            if risk_analysis:
                risk_percentage = abs(risk_analysis.max_loss_percentage) if risk_analysis.max_loss_percentage else 0.02
                if risk_percentage > self.risk_limits.max_risk_per_trade:
                    validation_result['approved'] = False
                    validation_result['errors'].append(f"Risk per trade {risk_percentage:.1%} exceeds limit {self.risk_limits.max_risk_per_trade:.1%}")
            
            # Check daily loss limit
            if self.config['enable_daily_limits']:
                current_daily_pnl = await self._calculate_daily_pnl()
                if current_daily_pnl < -self.risk_limits.max_daily_loss:
                    validation_result['approved'] = False
                    validation_result['errors'].append(f"Daily loss limit reached: {current_daily_pnl:.1%}")
            
            # Check drawdown limit
            if self.config['enable_drawdown_monitoring']:
                current_drawdown = await self._calculate_current_drawdown()
                if current_drawdown > self.risk_limits.max_drawdown:
                    validation_result['approved'] = False
                    validation_result['errors'].append(f"Drawdown limit exceeded: {current_drawdown:.1%}")
            
            # Check concentration risk
            symbol_concentration = await self._calculate_symbol_concentration(signal.symbol)
            if symbol_concentration > self.risk_limits.max_concentration:
                validation_result['warnings'].append(f"High concentration in {signal.symbol}: {symbol_concentration:.1%}")
            
            # Update statistics
            if not validation_result['approved']:
                self.stats['positions_rejected'] += 1
            
            return validation_result
            
        except Exception as e:
            analysis_logger.error(f"Error validating new position: {e}")
            return {
                'approved': False,
                'errors': [str(e)]
            }
    
    async def calculate_position_size(self, signal: TradingSignal, account_balance: float = 10000.0) -> float:
        """Calculate optimal position size based on risk management"""
        
        try:
            # Base position size calculation
            risk_amount = account_balance * self.risk_limits.max_risk_per_trade
            
            # Calculate distance to stop loss
            stop_distance = 0.001  # Default 10 pips
            if signal.stop_loss and signal.entry_price:
                stop_distance = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
            
            # Calculate position size
            position_size = risk_amount / (stop_distance * signal.entry_price) if signal.entry_price else 0.1
            
            # Apply limits
            position_size = min(position_size, self.risk_limits.max_position_size)
            position_size = max(position_size, 0.01)  # Minimum size
            
            # Round to 2 decimal places
            position_size = round(position_size, 2)
            
            analysis_logger.info(
                f"Calculated position size: {position_size} for {signal.symbol} "
                f"(Risk: {self.risk_limits.max_risk_per_trade:.1%}, "
                f"Stop distance: {stop_distance:.4f})"
            )
            
            return position_size
            
        except Exception as e:
            analysis_logger.error(f"Error calculating position size: {e}")
            return 0.1  # Default safe size
    
    async def monitor_risk_limits(self):
        """Monitor all risk limits and generate alerts"""
        
        try:
            # Update risk metrics
            await self._update_risk_metrics()
            
            # Check various risk limits
            await self._check_position_limits()
            await self._check_loss_limits()
            await self._check_drawdown_limits()
            await self._check_concentration_limits()
            await self._check_margin_limits()
            
            # Handle emergency situations
            await self._handle_emergency_conditions()
            
        except Exception as e:
            analysis_logger.error(f"Error monitoring risk limits: {e}")
    
    async def emergency_stop(self, reason: str = "Manual emergency stop"):
        """Activate emergency stop - halt all trading"""
        
        try:
            if self.emergency_stop_active:
                analysis_logger.warning("Emergency stop already active")
                return
            
            self.emergency_stop_active = True
            
            # Create critical alert
            alert = RiskAlert(
                alert_id=f"emergency_{int(datetime.now().timestamp())}",
                alert_type=AlertType.DAILY_LOSS,
                risk_level=RiskLevel.CRITICAL,
                message=f"EMERGENCY STOP ACTIVATED: {reason}",
                value=0.0,
                threshold=0.0
            )
            
            await self._add_alert(alert)
            
            # Update statistics
            self.stats['emergency_stops'] += 1
            
            analysis_logger.critical(f"Emergency stop activated: {reason}")
            
        except Exception as e:
            analysis_logger.error(f"Error activating emergency stop: {e}")
    
    async def force_close_all_positions(self, reason: str = "Risk limit exceeded"):
        """Force close all open positions"""
        
        try:
            if self.force_close_pending:
                analysis_logger.warning("Force close already pending")
                return
            
            self.force_close_pending = True
            
            # Get all open positions
            open_positions = await self.position_manager.get_open_positions()
            
            if not open_positions:
                analysis_logger.info("No open positions to force close")
                self.force_close_pending = False
                return
            
            # Close all positions
            results = await self.position_manager.close_all_positions()
            
            # Create critical alert
            alert = RiskAlert(
                alert_id=f"force_close_{int(datetime.now().timestamp())}",
                alert_type=AlertType.DAILY_LOSS,
                risk_level=RiskLevel.CRITICAL,
                message=f"FORCE CLOSED ALL POSITIONS: {reason} ({len(results)} positions)",
                value=len(results),
                threshold=0.0
            )
            
            await self._add_alert(alert)
            
            # Update statistics
            self.stats['force_closes'] += 1
            
            self.force_close_pending = False
            
            analysis_logger.critical(f"Force closed {len(results)} positions: {reason}")
            
        except Exception as e:
            analysis_logger.error(f"Error force closing positions: {e}")
            self.force_close_pending = False
    
    async def deactivate_emergency_stop(self):
        """Deactivate emergency stop"""
        
        try:
            if not self.emergency_stop_active:
                analysis_logger.warning("Emergency stop not active")
                return
            
            self.emergency_stop_active = False
            
            analysis_logger.info("Emergency stop deactivated")
            
        except Exception as e:
            analysis_logger.error(f"Error deactivating emergency stop: {e}")
    
    async def get_risk_report(self) -> Dict[str, Any]:
        """Get comprehensive risk report"""
        
        try:
            await self._update_risk_metrics()
            
            # Get recent alerts
            recent_alerts = [alert for alert in self.alerts if (datetime.now() - alert.timestamp).hours < 24]
            
            # Get position breakdown
            open_positions = await self.position_manager.get_open_positions()
            portfolio_summary = await self.position_manager.get_portfolio_summary()
            
            return {
                'risk_metrics': {
                    'total_positions': self.risk_metrics.total_positions,
                    'total_exposure': self.risk_metrics.total_exposure,
                    'unrealized_pnl': self.risk_metrics.unrealized_pnl,
                    'daily_pnl': self.risk_metrics.daily_pnl,
                    'current_drawdown': self.risk_metrics.current_drawdown,
                    'max_drawdown': self.risk_metrics.max_drawdown,
                    'overall_risk_score': self.risk_metrics.overall_risk_score
                },
                'risk_limits': {
                    'max_position_size': self.risk_limits.max_position_size,
                    'max_risk_per_trade': self.risk_limits.max_risk_per_trade,
                    'max_daily_loss': self.risk_limits.max_daily_loss,
                    'max_drawdown': self.risk_limits.max_drawdown
                },
                'current_status': {
                    'emergency_stop_active': self.emergency_stop_active,
                    'force_close_pending': self.force_close_pending,
                    'active_alerts_count': len(self.active_alerts),
                    'recent_alerts_count': len(recent_alerts)
                },
                'recent_alerts': [
                    {
                        'type': alert.alert_type.value,
                        'level': alert.risk_level.value,
                        'message': alert.message,
                        'timestamp': alert.timestamp
                    }
                    for alert in recent_alerts[-10:]  # Last 10 alerts
                ],
                'portfolio_summary': portfolio_summary,
                'statistics': self.stats
            }
            
        except Exception as e:
            analysis_logger.error(f"Error generating risk report: {e}")
            return {}
    
    async def _update_risk_metrics(self):
        """Update current risk metrics"""
        
        try:
            # Get portfolio data
            open_positions = await self.position_manager.get_open_positions()
            portfolio_summary = await self.position_manager.get_portfolio_summary()
            
            # Update position metrics
            self.risk_metrics.total_positions = len(open_positions)
            self.risk_metrics.total_exposure = sum(pos.current_volume for pos in open_positions)
            self.risk_metrics.largest_position = max([pos.current_volume for pos in open_positions], default=0.0)
            
            # Update PnL metrics
            self.risk_metrics.unrealized_pnl = portfolio_summary.get('total_unrealized_pnl', 0.0)
            self.risk_metrics.daily_pnl = await self._calculate_daily_pnl()
            self.risk_metrics.current_drawdown = await self._calculate_current_drawdown()
            
            # Calculate overall risk score (0-100)
            risk_score = 0.0
            
            # Position risk (0-30 points)
            position_risk = min(self.risk_metrics.total_positions / self.risk_limits.max_positions * 30, 30)
            risk_score += position_risk
            
            # Drawdown risk (0-40 points)
            drawdown_risk = min(self.risk_metrics.current_drawdown / self.risk_limits.max_drawdown * 40, 40)
            risk_score += drawdown_risk
            
            # Daily loss risk (0-30 points)
            daily_loss_risk = min(abs(self.risk_metrics.daily_pnl) / self.risk_limits.max_daily_loss * 30, 30)
            risk_score += daily_loss_risk
            
            self.risk_metrics.overall_risk_score = risk_score
            
        except Exception as e:
            analysis_logger.error(f"Error updating risk metrics: {e}")
    
    async def _check_position_limits(self):
        """Check position-related limits"""
        
        try:
            open_positions = await self.position_manager.get_open_positions()
            
            # Check maximum positions
            if len(open_positions) >= self.risk_limits.max_positions * 0.9:  # 90% threshold
                alert = RiskAlert(
                    alert_id=f"max_positions_{int(datetime.now().timestamp())}",
                    alert_type=AlertType.POSITION_SIZE,
                    risk_level=RiskLevel.HIGH if len(open_positions) >= self.risk_limits.max_positions else RiskLevel.MEDIUM,
                    message=f"Approaching maximum positions limit: {len(open_positions)}/{self.risk_limits.max_positions}",
                    value=len(open_positions),
                    threshold=self.risk_limits.max_positions
                )
                await self._add_alert(alert)
            
        except Exception as e:
            analysis_logger.error(f"Error checking position limits: {e}")
    
    async def _check_loss_limits(self):
        """Check loss limits"""
        
        try:
            daily_pnl = await self._calculate_daily_pnl()
            
            # Check daily loss limit
            if daily_pnl < -self.risk_limits.max_daily_loss * 0.8:  # 80% threshold
                alert = RiskAlert(
                    alert_id=f"daily_loss_{int(datetime.now().timestamp())}",
                    alert_type=AlertType.DAILY_LOSS,
                    risk_level=RiskLevel.CRITICAL if daily_pnl < -self.risk_limits.max_daily_loss else RiskLevel.HIGH,
                    message=f"Daily loss approaching limit: {daily_pnl:.1%}",
                    value=daily_pnl,
                    threshold=-self.risk_limits.max_daily_loss
                )
                await self._add_alert(alert)
            
        except Exception as e:
            analysis_logger.error(f"Error checking loss limits: {e}")
    
    async def _check_drawdown_limits(self):
        """Check drawdown limits"""
        
        try:
            current_drawdown = await self._calculate_current_drawdown()
            
            if current_drawdown > self.risk_limits.max_drawdown * 0.8:  # 80% threshold
                alert = RiskAlert(
                    alert_id=f"drawdown_{int(datetime.now().timestamp())}",
                    alert_type=AlertType.DRAWDOWN,
                    risk_level=RiskLevel.CRITICAL if current_drawdown > self.risk_limits.max_drawdown else RiskLevel.HIGH,
                    message=f"Drawdown approaching limit: {current_drawdown:.1%}",
                    value=current_drawdown,
                    threshold=self.risk_limits.max_drawdown
                )
                await self._add_alert(alert)
            
        except Exception as e:
            analysis_logger.error(f"Error checking drawdown limits: {e}")
    
    async def _check_concentration_limits(self):
        """Check concentration limits"""
        
        try:
            open_positions = await self.position_manager.get_open_positions()
            
            # Calculate concentration by symbol
            symbol_volumes = {}
            total_volume = 0.0
            
            for position in open_positions:
                symbol = position.symbol
                symbol_volumes[symbol] = symbol_volumes.get(symbol, 0.0) + position.current_volume
                total_volume += position.current_volume
            
            # Check each symbol concentration
            for symbol, volume in symbol_volumes.items():
                concentration = volume / max(total_volume, 1.0)
                
                if concentration > self.risk_limits.max_concentration * 0.8:  # 80% threshold
                    alert = RiskAlert(
                        alert_id=f"concentration_{symbol}_{int(datetime.now().timestamp())}",
                        alert_type=AlertType.CONCENTRATION,
                        risk_level=RiskLevel.HIGH if concentration > self.risk_limits.max_concentration else RiskLevel.MEDIUM,
                        message=f"High concentration in {symbol}: {concentration:.1%}",
                        value=concentration,
                        threshold=self.risk_limits.max_concentration
                    )
                    await self._add_alert(alert)
            
        except Exception as e:
            analysis_logger.error(f"Error checking concentration limits: {e}")
    
    async def _check_margin_limits(self):
        """Check margin limits"""
        
        try:
            # Mock margin calculation
            # In real implementation, this would get actual margin from MT5
            margin_level = self.risk_metrics.margin_level
            
            if margin_level < self.risk_limits.margin_call_level:
                alert = RiskAlert(
                    alert_id=f"margin_{int(datetime.now().timestamp())}",
                    alert_type=AlertType.MARGIN,
                    risk_level=RiskLevel.CRITICAL if margin_level < self.risk_limits.stop_out_level else RiskLevel.HIGH,
                    message=f"Low margin level: {margin_level:.1%}",
                    value=margin_level,
                    threshold=self.risk_limits.margin_call_level
                )
                await self._add_alert(alert)
            
        except Exception as e:
            analysis_logger.error(f"Error checking margin limits: {e}")
    
    async def _handle_emergency_conditions(self):
        """Handle emergency risk conditions"""
        
        try:
            # Check for force close conditions
            if (self.risk_metrics.current_drawdown > self.risk_limits.force_close_threshold or
                self.risk_metrics.daily_pnl < -self.risk_limits.force_close_threshold):
                
                if self.config['enable_force_close'] and not self.force_close_pending:
                    await self.force_close_all_positions("Critical risk threshold exceeded")
            
            # Check for emergency stop conditions
            if (self.risk_metrics.current_drawdown > self.risk_limits.emergency_stop_loss or
                self.risk_metrics.daily_pnl < -self.risk_limits.emergency_stop_loss):
                
                if self.config['enable_emergency_stops'] and not self.emergency_stop_active:
                    await self.emergency_stop("Emergency risk threshold exceeded")
            
        except Exception as e:
            analysis_logger.error(f"Error handling emergency conditions: {e}")
    
    async def _add_alert(self, alert: RiskAlert):
        """Add risk alert"""
        
        try:
            # Check cooldown for similar alerts
            similar_alerts = [a for a in self.alerts 
                            if a.alert_type == alert.alert_type and 
                            (datetime.now() - a.timestamp).total_seconds() < self.config['alert_cooldown_minutes'] * 60]
            
            if similar_alerts:
                return  # Skip duplicate alert
            
            # Add alert
            self.alerts.append(alert)
            self.active_alerts[alert.alert_id] = alert
            
            # Update statistics
            self.stats['total_alerts'] += 1
            if alert.risk_level == RiskLevel.CRITICAL:
                self.stats['critical_alerts'] += 1
            
            # Trim old alerts
            if len(self.alerts) > 1000:
                self.alerts = self.alerts[-500:]  # Keep last 500
            
            analysis_logger.warning(f"Risk alert: {alert.message}")
            
        except Exception as e:
            analysis_logger.error(f"Error adding alert: {e}")
    
    async def _calculate_daily_pnl(self) -> float:
        """Calculate daily PnL"""
        
        try:
            # Mock calculation - in real system would get from account history
            portfolio_summary = await self.position_manager.get_portfolio_summary()
            return portfolio_summary.get('total_unrealized_pnl', 0.0) * 0.1  # Mock daily portion
            
        except Exception as e:
            analysis_logger.error(f"Error calculating daily PnL: {e}")
            return 0.0
    
    async def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from high water mark"""
        
        try:
            # Mock calculation
            current_balance = self.high_water_mark + await self._calculate_daily_pnl()
            drawdown = max(0.0, (self.high_water_mark - current_balance) / self.high_water_mark)
            
            # Update high water mark
            if current_balance > self.high_water_mark:
                self.high_water_mark = current_balance
                drawdown = 0.0
            
            return drawdown
            
        except Exception as e:
            analysis_logger.error(f"Error calculating drawdown: {e}")
            return 0.0
    
    async def _calculate_symbol_concentration(self, symbol: str) -> float:
        """Calculate concentration risk for specific symbol"""
        
        try:
            open_positions = await self.position_manager.get_open_positions()
            
            symbol_volume = sum(pos.current_volume for pos in open_positions if pos.symbol == symbol)
            total_volume = sum(pos.current_volume for pos in open_positions)
            
            return symbol_volume / max(total_volume, 1.0)
            
        except Exception as e:
            analysis_logger.error(f"Error calculating symbol concentration: {e}")
            return 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get risk manager statistics"""
        
        return {
            **self.stats,
            'emergency_stop_active': self.emergency_stop_active,
            'force_close_pending': self.force_close_pending,
            'active_alerts_count': len(self.active_alerts),
            'total_alerts_count': len(self.alerts),
            'overall_risk_score': self.risk_metrics.overall_risk_score
        }


# Global instance
risk_manager = RiskManager(position_manager=None)  # Will be initialized with position_manager