"""
Position Management System

Comprehensive position management for FX trading including position tracking,
modification, partial closing, and portfolio management.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio

from app.trading.order_manager import OrderManager, OrderExecution, OrderType
from app.analysis.strategy_engine.signal_generator import TradingSignal, SignalType
from app.mt5.connection import MT5Connection
from app.utils.logger import analysis_logger


class PositionType(Enum):
    """Position types"""
    LONG = "long"
    SHORT = "short"


class PositionStatus(Enum):
    """Position status enumeration"""
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    PARTIALLY_CLOSED = "partially_closed"


@dataclass
class Position:
    """Position tracking structure"""
    position_id: str
    symbol: str
    position_type: PositionType
    volume: float
    entry_price: float
    entry_time: datetime
    
    # Current state
    current_volume: float = 0.0
    current_price: float = 0.0
    status: PositionStatus = PositionStatus.OPEN
    
    # Stop loss and take profit
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Financial metrics
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    
    # Strategy attribution
    strategy_id: Optional[str] = None
    signal_id: Optional[str] = None
    
    # MT5 integration
    mt5_ticket: Optional[int] = None
    
    # Tracking
    last_update: datetime = field(default_factory=datetime.now)
    close_time: Optional[datetime] = None
    close_price: Optional[float] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.current_volume == 0.0:
            self.current_volume = self.volume


@dataclass
class PositionModification:
    """Position modification request"""
    position_id: str
    modification_type: str  # 'stop_loss', 'take_profit', 'both'
    new_stop_loss: Optional[float] = None
    new_take_profit: Optional[float] = None
    reason: str = ""


@dataclass
class PositionCloseRequest:
    """Position closing request"""
    position_id: str
    volume: Optional[float] = None  # None for full close
    close_price: Optional[float] = None  # None for market price
    reason: str = ""


@dataclass
class PositionCloseResult:
    """Position close result"""
    success: bool
    position_id: str
    closed_volume: float = 0.0
    close_price: Optional[float] = None
    realized_pnl: float = 0.0
    commission: float = 0.0
    error_message: Optional[str] = None
    close_time: Optional[datetime] = None


class PositionManager:
    """
    Comprehensive position management system
    
    Manages position lifecycle including:
    - Position tracking and updates
    - Stop loss and take profit modification
    - Partial and full position closing
    - Portfolio-level position management
    - Risk monitoring and alerts
    """
    
    def __init__(self, order_manager: OrderManager, mt5_connection: Optional[MT5Connection] = None):
        """Initialize position manager"""
        
        self.order_manager = order_manager
        self.mt5_connection = mt5_connection
        
        # Position storage
        self.positions: Dict[str, Position] = {}
        self.open_positions: Dict[str, Position] = {}
        self.closed_positions: Dict[str, Position] = {}
        
        # Configuration
        self.config = {
            'max_positions': 20,
            'max_positions_per_symbol': 5,
            'position_update_interval_seconds': 30,
            'auto_update_pnl': True,
            'enable_trailing_stops': True,
            'default_trailing_distance': 0.001,  # 0.1%
            'risk_monitoring_enabled': True,
            'max_position_risk': 0.02,  # 2% per position
            'max_portfolio_risk': 0.10,  # 10% total
        }
        
        # Market data cache for PnL calculation
        self.market_prices: Dict[str, Dict[str, float]] = {}
        
        # Statistics
        self.stats = {
            'total_positions': 0,
            'winning_positions': 0,
            'losing_positions': 0,
            'total_realized_pnl': 0.0,
            'total_unrealized_pnl': 0.0,
            'total_commission': 0.0,
            'total_swap': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'avg_position_duration_hours': 0.0
        }
        
        analysis_logger.info("Position Manager initialized")
    
    async def open_position_from_execution(self, execution: OrderExecution, signal: TradingSignal) -> Optional[Position]:
        """Create position from successful order execution"""
        
        try:
            if not execution.success or execution.executed_volume <= 0:
                return None
            
            # Determine position type
            position_type = PositionType.LONG if execution.order_id.endswith('BUY') or 'buy' in execution.order_id.lower() else PositionType.SHORT
            
            # Create position
            position = Position(
                position_id=f"pos_{execution.mt5_ticket or int(datetime.now().timestamp())}",
                symbol=signal.symbol,
                position_type=position_type,
                volume=execution.executed_volume,
                entry_price=execution.execution_price or signal.entry_price,
                entry_time=execution.execution_time or datetime.now(),
                current_volume=execution.executed_volume,
                current_price=execution.execution_price or signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=getattr(signal, 'take_profit_1', None),
                commission=execution.commission,
                strategy_id=signal.metadata.get('strategy_id'),
                signal_id=signal.signal_id,
                mt5_ticket=execution.mt5_ticket
            )
            
            # Store position
            self.positions[position.position_id] = position
            self.open_positions[position.position_id] = position
            
            # Update statistics
            self.stats['total_positions'] += 1
            
            analysis_logger.info(
                f"Position opened: {position.position_id} - "
                f"{position.position_type.value} {position.volume} {position.symbol} @ {position.entry_price}"
            )
            
            return position
            
        except Exception as e:
            analysis_logger.error(f"Error opening position from execution: {e}")
            return None
    
    async def modify_position(self, modification: PositionModification) -> bool:
        """Modify position stop loss and/or take profit"""
        
        try:
            if modification.position_id not in self.open_positions:
                analysis_logger.warning(f"Position not found for modification: {modification.position_id}")
                return False
            
            position = self.open_positions[modification.position_id]
            
            # Validate modification
            if not self._validate_position_modification(position, modification):
                return False
            
            # Update stop loss
            if modification.modification_type in ['stop_loss', 'both'] and modification.new_stop_loss is not None:
                old_sl = position.stop_loss
                position.stop_loss = modification.new_stop_loss
                
                analysis_logger.info(
                    f"Stop loss modified: {modification.position_id} - "
                    f"{old_sl} -> {modification.new_stop_loss}"
                )
            
            # Update take profit
            if modification.modification_type in ['take_profit', 'both'] and modification.new_take_profit is not None:
                old_tp = position.take_profit
                position.take_profit = modification.new_take_profit
                
                analysis_logger.info(
                    f"Take profit modified: {modification.position_id} - "
                    f"{old_tp} -> {modification.new_take_profit}"
                )
            
            # Update through MT5 if connected
            if self.mt5_connection and position.mt5_ticket:
                mt5_result = await self._modify_position_mt5(position, modification)
                if not mt5_result:
                    analysis_logger.warning(f"MT5 modification failed for position: {modification.position_id}")
                    return False
            
            position.last_update = datetime.now()
            
            return True
            
        except Exception as e:
            analysis_logger.error(f"Error modifying position: {e}")
            return False
    
    async def close_position(self, close_request: PositionCloseRequest) -> PositionCloseResult:
        """Close position (partial or full)"""
        
        try:
            if close_request.position_id not in self.open_positions:
                return PositionCloseResult(
                    success=False,
                    position_id=close_request.position_id,
                    error_message="Position not found"
                )
            
            position = self.open_positions[close_request.position_id]
            
            # Determine close volume
            close_volume = close_request.volume or position.current_volume
            close_volume = min(close_volume, position.current_volume)
            
            if close_volume <= 0:
                return PositionCloseResult(
                    success=False,
                    position_id=close_request.position_id,
                    error_message="Invalid close volume"
                )
            
            # Get current price
            current_price = close_request.close_price
            if not current_price:
                market_data = await self._get_current_market_price(position.symbol)
                if not market_data:
                    return PositionCloseResult(
                        success=False,
                        position_id=close_request.position_id,
                        error_message="Cannot get current market price"
                    )
                
                # Use bid for long positions, ask for short positions
                current_price = market_data['bid'] if position.position_type == PositionType.LONG else market_data['ask']
            
            # Calculate realized PnL
            realized_pnl = self._calculate_realized_pnl(position, close_volume, current_price)
            
            # Execute close order through MT5
            close_result = await self._close_position_mt5(position, close_volume, current_price)
            
            if not close_result['success']:
                return PositionCloseResult(
                    success=False,
                    position_id=close_request.position_id,
                    error_message=close_result.get('error', 'MT5 close failed')
                )
            
            # Update position
            position.current_volume -= close_volume
            position.realized_pnl += realized_pnl
            position.commission += close_result.get('commission', 0.0)
            position.last_update = datetime.now()
            
            # Handle full vs partial close
            if position.current_volume <= 0.001:  # Essentially zero
                position.status = PositionStatus.CLOSED
                position.close_time = datetime.now()
                position.close_price = current_price
                
                # Move to closed positions
                self.closed_positions[position.position_id] = position
                self.open_positions.pop(position.position_id, None)
                
                # Update statistics
                if realized_pnl > 0:
                    self.stats['winning_positions'] += 1
                    self.stats['largest_win'] = max(self.stats['largest_win'], realized_pnl)
                else:
                    self.stats['losing_positions'] += 1
                    self.stats['largest_loss'] = min(self.stats['largest_loss'], realized_pnl)
                
                self.stats['total_realized_pnl'] += realized_pnl
                self.stats['total_commission'] += close_result.get('commission', 0.0)
                
                # Calculate duration
                duration_hours = (position.close_time - position.entry_time).total_seconds() / 3600
                n = self.stats['winning_positions'] + self.stats['losing_positions']
                self.stats['avg_position_duration_hours'] = (
                    (self.stats['avg_position_duration_hours'] * (n - 1) + duration_hours) / n
                )
                
                analysis_logger.info(
                    f"Position fully closed: {position.position_id} - "
                    f"PnL: {realized_pnl:.2f}, Duration: {duration_hours:.1f}h"
                )
            else:
                position.status = PositionStatus.PARTIALLY_CLOSED
                
                analysis_logger.info(
                    f"Position partially closed: {position.position_id} - "
                    f"Closed: {close_volume}, Remaining: {position.current_volume}, "
                    f"PnL: {realized_pnl:.2f}"
                )
            
            return PositionCloseResult(
                success=True,
                position_id=close_request.position_id,
                closed_volume=close_volume,
                close_price=current_price,
                realized_pnl=realized_pnl,
                commission=close_result.get('commission', 0.0),
                close_time=datetime.now()
            )
            
        except Exception as e:
            analysis_logger.error(f"Error closing position: {e}")
            return PositionCloseResult(
                success=False,
                position_id=close_request.position_id,
                error_message=str(e)
            )
    
    async def close_all_positions(self, symbol: Optional[str] = None) -> List[PositionCloseResult]:
        """Close all open positions, optionally filtered by symbol"""
        
        results = []
        
        try:
            positions_to_close = []
            
            for position in self.open_positions.values():
                if symbol is None or position.symbol == symbol:
                    positions_to_close.append(position)
            
            for position in positions_to_close:
                close_request = PositionCloseRequest(
                    position_id=position.position_id,
                    reason="Close all positions"
                )
                
                result = await self.close_position(close_request)
                results.append(result)
            
            analysis_logger.info(f"Closed {len(results)} positions" + (f" for {symbol}" if symbol else ""))
            
        except Exception as e:
            analysis_logger.error(f"Error closing all positions: {e}")
        
        return results
    
    async def update_positions_pnl(self):
        """Update unrealized PnL for all open positions"""
        
        try:
            if not self.config['auto_update_pnl']:
                return
            
            for position in self.open_positions.values():
                await self._update_position_pnl(position)
                
                # Check for automatic stops
                if self._should_trigger_stop_loss(position) or self._should_trigger_take_profit(position):
                    close_request = PositionCloseRequest(
                        position_id=position.position_id,
                        reason="Stop triggered"
                    )
                    await self.close_position(close_request)
            
        except Exception as e:
            analysis_logger.error(f"Error updating positions PnL: {e}")
    
    async def get_position(self, position_id: str) -> Optional[Position]:
        """Get position by ID"""
        return self.positions.get(position_id)
    
    async def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get all open positions, optionally filtered by symbol"""
        
        positions = list(self.open_positions.values())
        
        if symbol:
            positions = [pos for pos in positions if pos.symbol == symbol]
        
        return positions
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary with key metrics"""
        
        try:
            open_positions = list(self.open_positions.values())
            
            # Calculate totals
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in open_positions)
            total_positions = len(open_positions)
            total_volume = sum(pos.current_volume for pos in open_positions)
            
            # Calculate by symbol
            symbol_breakdown = {}
            for position in open_positions:
                symbol = position.symbol
                if symbol not in symbol_breakdown:
                    symbol_breakdown[symbol] = {
                        'positions': 0,
                        'volume': 0.0,
                        'unrealized_pnl': 0.0,
                        'net_exposure': 0.0
                    }
                
                symbol_breakdown[symbol]['positions'] += 1
                symbol_breakdown[symbol]['volume'] += position.current_volume
                symbol_breakdown[symbol]['unrealized_pnl'] += position.unrealized_pnl
                
                # Net exposure (long positive, short negative)
                exposure = position.current_volume if position.position_type == PositionType.LONG else -position.current_volume
                symbol_breakdown[symbol]['net_exposure'] += exposure
            
            # Win rate
            total_closed = self.stats['winning_positions'] + self.stats['losing_positions']
            win_rate = self.stats['winning_positions'] / max(total_closed, 1)
            
            return {
                'total_open_positions': total_positions,
                'total_volume': total_volume,
                'total_unrealized_pnl': total_unrealized_pnl,
                'total_realized_pnl': self.stats['total_realized_pnl'],
                'total_commission': self.stats['total_commission'],
                'win_rate': win_rate,
                'symbol_breakdown': symbol_breakdown,
                'avg_position_duration_hours': self.stats['avg_position_duration_hours'],
                'largest_win': self.stats['largest_win'],
                'largest_loss': self.stats['largest_loss']
            }
            
        except Exception as e:
            analysis_logger.error(f"Error calculating portfolio summary: {e}")
            return {}
    
    def _validate_position_modification(self, position: Position, modification: PositionModification) -> bool:
        """Validate position modification request"""
        
        try:
            # Check if position is open
            if position.status != PositionStatus.OPEN:
                analysis_logger.warning(f"Cannot modify closed position: {position.position_id}")
                return False
            
            # Validate stop loss level
            if modification.new_stop_loss is not None:
                if position.position_type == PositionType.LONG:
                    if modification.new_stop_loss >= position.current_price:
                        analysis_logger.warning("Stop loss must be below current price for long positions")
                        return False
                else:  # SHORT
                    if modification.new_stop_loss <= position.current_price:
                        analysis_logger.warning("Stop loss must be above current price for short positions")
                        return False
            
            # Validate take profit level
            if modification.new_take_profit is not None:
                if position.position_type == PositionType.LONG:
                    if modification.new_take_profit <= position.current_price:
                        analysis_logger.warning("Take profit must be above current price for long positions")
                        return False
                else:  # SHORT
                    if modification.new_take_profit >= position.current_price:
                        analysis_logger.warning("Take profit must be below current price for short positions")
                        return False
            
            return True
            
        except Exception as e:
            analysis_logger.error(f"Error validating position modification: {e}")
            return False
    
    def _calculate_realized_pnl(self, position: Position, close_volume: float, close_price: float) -> float:
        """Calculate realized PnL for closing volume"""
        
        try:
            if position.position_type == PositionType.LONG:
                pnl = (close_price - position.entry_price) * close_volume
            else:  # SHORT
                pnl = (position.entry_price - close_price) * close_volume
            
            return pnl
            
        except Exception as e:
            analysis_logger.error(f"Error calculating realized PnL: {e}")
            return 0.0
    
    async def _update_position_pnl(self, position: Position):
        """Update unrealized PnL for position"""
        
        try:
            # Get current market price
            market_data = await self._get_current_market_price(position.symbol)
            if not market_data:
                return
            
            # Use bid for long positions, ask for short positions
            current_price = market_data['bid'] if position.position_type == PositionType.LONG else market_data['ask']
            
            position.current_price = current_price
            
            # Calculate unrealized PnL
            if position.position_type == PositionType.LONG:
                position.unrealized_pnl = (current_price - position.entry_price) * position.current_volume
            else:  # SHORT
                position.unrealized_pnl = (position.entry_price - current_price) * position.current_volume
            
            position.last_update = datetime.now()
            
        except Exception as e:
            analysis_logger.error(f"Error updating position PnL: {e}")
    
    async def _get_current_market_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get current market price for symbol"""
        
        try:
            # Mock implementation - in real system would fetch from MT5
            base_price = 1.1000 if 'USD' in symbol else 150.0 if 'JPY' in symbol else 1.3000
            spread = 0.0001 if 'USD' in symbol else 0.01 if 'JPY' in symbol else 0.0002
            
            return {
                'bid': base_price - spread / 2,
                'ask': base_price + spread / 2,
                'mid': base_price
            }
            
        except Exception as e:
            analysis_logger.error(f"Error getting market price: {e}")
            return None
    
    def _should_trigger_stop_loss(self, position: Position) -> bool:
        """Check if stop loss should be triggered"""
        
        if not position.stop_loss:
            return False
        
        if position.position_type == PositionType.LONG:
            return position.current_price <= position.stop_loss
        else:  # SHORT
            return position.current_price >= position.stop_loss
    
    def _should_trigger_take_profit(self, position: Position) -> bool:
        """Check if take profit should be triggered"""
        
        if not position.take_profit:
            return False
        
        if position.position_type == PositionType.LONG:
            return position.current_price >= position.take_profit
        else:  # SHORT
            return position.current_price <= position.take_profit
    
    async def _modify_position_mt5(self, position: Position, modification: PositionModification) -> bool:
        """Modify position in MT5"""
        
        try:
            # Mock implementation
            if self.mt5_connection and position.mt5_ticket:
                # Real MT5 modification would go here
                pass
            
            return True
            
        except Exception as e:
            analysis_logger.error(f"Error modifying position in MT5: {e}")
            return False
    
    async def _close_position_mt5(self, position: Position, volume: float, price: float) -> Dict[str, Any]:
        """Close position in MT5"""
        
        try:
            # Mock implementation
            if self.mt5_connection and position.mt5_ticket:
                # Real MT5 close would go here
                pass
            
            return {
                'success': True,
                'commission': volume * 0.00001,  # Mock commission
                'swap': 0.0
            }
            
        except Exception as e:
            analysis_logger.error(f"Error closing position in MT5: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get position manager statistics"""
        
        total_closed = self.stats['winning_positions'] + self.stats['losing_positions']
        win_rate = self.stats['winning_positions'] / max(total_closed, 1)
        
        profit_factor = 0.0
        if self.stats['largest_loss'] < 0:
            total_wins = self.stats['largest_win'] * self.stats['winning_positions']
            total_losses = abs(self.stats['largest_loss']) * self.stats['losing_positions']
            profit_factor = total_wins / max(total_losses, 1)
        
        return {
            **self.stats,
            'open_positions_count': len(self.open_positions),
            'closed_positions_count': len(self.closed_positions),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': self.stats['largest_win'] / max(self.stats['winning_positions'], 1),
            'avg_loss': self.stats['largest_loss'] / max(self.stats['losing_positions'], 1)
        }


# Global instance
position_manager = PositionManager(order_manager=None)  # Will be initialized with order_manager