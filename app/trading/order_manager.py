"""
Order Management System

Comprehensive order management system for FX trading with support for
market orders, pending orders, order modification, and execution tracking.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import uuid

from app.analysis.strategy_engine.signal_generator import TradingSignal, SignalType
from app.analysis.strategy_engine.stop_loss_calculator import StopLossRecommendation
from app.analysis.strategy_engine.take_profit_calculator import TakeProfitRecommendation
from app.analysis.strategy_engine.risk_reward_calculator import RiskRewardAnalysis
from app.mt5.connection import MT5Connection
from app.utils.logger import analysis_logger


class OrderType(Enum):
    """Order types supported by the system"""
    MARKET_BUY = "market_buy"
    MARKET_SELL = "market_sell"
    BUY_LIMIT = "buy_limit"
    SELL_LIMIT = "sell_limit"
    BUY_STOP = "buy_stop"
    SELL_STOP = "sell_stop"
    BUY_STOP_LIMIT = "buy_stop_limit"
    SELL_STOP_LIMIT = "sell_stop_limit"


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    PLACED = "placed"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderTimeInForce(Enum):
    """Order time in force options"""
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    DAY = "day"  # Day order
    GTD = "gtd"  # Good Till Date


@dataclass
class OrderRequest:
    """Order request structure"""
    symbol: str
    order_type: OrderType
    volume: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    time_in_force: OrderTimeInForce = OrderTimeInForce.GTC
    expiry_time: Optional[datetime] = None
    comment: str = ""
    magic: int = 0
    deviation: int = 10  # Price deviation in points
    
    # Strategy attribution
    strategy_id: Optional[str] = None
    signal_id: Optional[str] = None
    confidence_score: Optional[float] = None
    
    # Risk management
    max_slippage: float = 0.0005  # 0.05%
    min_profit_points: Optional[float] = None


@dataclass 
class Order:
    """Order tracking structure"""
    order_id: str
    request: OrderRequest
    status: OrderStatus
    created_time: datetime
    
    # Execution details
    mt5_ticket: Optional[int] = None
    filled_volume: float = 0.0
    remaining_volume: float = 0.0
    avg_fill_price: Optional[float] = None
    execution_time: Optional[datetime] = None
    
    # Tracking
    last_update: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None
    execution_details: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.remaining_volume = self.request.volume


@dataclass
class OrderExecution:
    """Order execution result"""
    success: bool
    order_id: str
    mt5_ticket: Optional[int] = None
    executed_volume: float = 0.0
    execution_price: Optional[float] = None
    commission: float = 0.0
    swap: float = 0.0
    error_message: Optional[str] = None
    execution_time: Optional[datetime] = None
    slippage: float = 0.0


class OrderManager:
    """
    Comprehensive order management system
    
    Manages order lifecycle from creation to execution, including:
    - Order validation and risk checks
    - MT5 integration for order placement
    - Order status tracking and updates
    - Partial fill handling
    - Order modification and cancellation
    """
    
    def __init__(self, mt5_connection: Optional[MT5Connection] = None):
        """Initialize order manager"""
        
        self.mt5_connection = mt5_connection
        self.orders: Dict[str, Order] = {}
        self.active_orders: Dict[str, Order] = {}
        
        # Configuration
        self.config = {
            'max_pending_orders': 50,
            'max_orders_per_symbol': 10,
            'default_deviation': 10,  # points
            'order_timeout_minutes': 1440,  # 24 hours
            'retry_attempts': 3,
            'retry_delay_seconds': 1.0,
            'enable_partial_fills': True,
            'min_order_size': 0.01,
            'max_order_size': 100.0
        }
        
        # Statistics
        self.stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'cancelled_orders': 0,
            'partial_fills': 0,
            'total_volume': 0.0,
            'total_commission': 0.0
        }
        
        analysis_logger.info("Order Manager initialized")
    
    async def create_order_from_signal(self, 
                                     signal: TradingSignal,
                                     stop_loss_rec: Optional[StopLossRecommendation] = None,
                                     take_profit_rec: Optional[TakeProfitRecommendation] = None,
                                     risk_analysis: Optional[RiskRewardAnalysis] = None) -> OrderRequest:
        """Create order request from trading signal"""
        
        try:
            # Determine order type from signal
            if signal.signal_type == SignalType.MARKET_BUY:
                order_type = OrderType.MARKET_BUY
            elif signal.signal_type == SignalType.MARKET_SELL:
                order_type = OrderType.MARKET_SELL
            elif signal.signal_type == SignalType.LIMIT_BUY:
                order_type = OrderType.BUY_LIMIT
            elif signal.signal_type == SignalType.LIMIT_SELL:
                order_type = OrderType.SELL_LIMIT
            else:
                raise ValueError(f"Unsupported signal type: {signal.signal_type}")
            
            # Calculate position size
            position_size = self._calculate_position_size(signal, risk_analysis)
            
            # Get stop loss and take profit levels
            stop_loss = None
            take_profit = None
            
            if stop_loss_rec and stop_loss_rec.recommended_levels:
                stop_loss = stop_loss_rec.recommended_levels[0].price
            
            if take_profit_rec and take_profit_rec.recommended_levels:
                take_profit = take_profit_rec.recommended_levels[0].price
            elif hasattr(signal, 'take_profit_1') and signal.take_profit_1:
                take_profit = signal.take_profit_1
            
            # Create order request
            order_request = OrderRequest(
                symbol=signal.symbol,
                order_type=order_type,
                volume=position_size,
                price=signal.entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy_id=signal.metadata.get('strategy_id'),
                signal_id=signal.signal_id,
                confidence_score=signal.confidence,
                comment=f"Signal {signal.signal_id} - {signal.signal_type.value}",
                magic=signal.metadata.get('magic', 12345)
            )
            
            analysis_logger.info(
                f"Created order request from signal: {signal.signal_id} - "
                f"{order_type.value} {position_size} {signal.symbol} @ {signal.entry_price}"
            )
            
            return order_request
            
        except Exception as e:
            analysis_logger.error(f"Error creating order from signal: {e}")
            raise
    
    async def place_order(self, order_request: OrderRequest) -> OrderExecution:
        """Place order through MT5"""
        
        order_id = str(uuid.uuid4())
        
        try:
            # Validate order request
            validation_result = await self._validate_order_request(order_request)
            if not validation_result['valid']:
                return OrderExecution(
                    success=False,
                    order_id=order_id,
                    error_message=validation_result['error']
                )
            
            # Create order object
            order = Order(
                order_id=order_id,
                request=order_request,
                status=OrderStatus.PENDING,
                created_time=datetime.now()
            )
            
            # Store order
            self.orders[order_id] = order
            self.active_orders[order_id] = order
            
            # Execute order through MT5
            execution_result = await self._execute_order_mt5(order)
            
            # Update order status
            if execution_result.success:
                order.status = OrderStatus.FILLED if execution_result.executed_volume == order_request.volume else OrderStatus.PARTIALLY_FILLED
                order.mt5_ticket = execution_result.mt5_ticket
                order.filled_volume = execution_result.executed_volume
                order.remaining_volume = order_request.volume - execution_result.executed_volume
                order.avg_fill_price = execution_result.execution_price
                order.execution_time = execution_result.execution_time
                
                # Update statistics
                self.stats['successful_orders'] += 1
                self.stats['total_volume'] += execution_result.executed_volume
                self.stats['total_commission'] += execution_result.commission
                
                if order.status == OrderStatus.PARTIALLY_FILLED:
                    self.stats['partial_fills'] += 1
                
                analysis_logger.info(
                    f"Order executed successfully: {order_id} - "
                    f"Ticket: {execution_result.mt5_ticket}, "
                    f"Volume: {execution_result.executed_volume}, "
                    f"Price: {execution_result.execution_price}"
                )
            else:
                order.status = OrderStatus.REJECTED
                order.error_message = execution_result.error_message
                
                # Remove from active orders
                self.active_orders.pop(order_id, None)
                
                # Update statistics
                self.stats['failed_orders'] += 1
                
                analysis_logger.error(
                    f"Order execution failed: {order_id} - {execution_result.error_message}"
                )
            
            order.last_update = datetime.now()
            self.stats['total_orders'] += 1
            
            return execution_result
            
        except Exception as e:
            analysis_logger.error(f"Error placing order: {e}")
            return OrderExecution(
                success=False,
                order_id=order_id,
                error_message=str(e)
            )
    
    async def modify_order(self, order_id: str, modifications: Dict[str, Any]) -> bool:
        """Modify existing order"""
        
        try:
            if order_id not in self.active_orders:
                analysis_logger.warning(f"Order not found for modification: {order_id}")
                return False
            
            order = self.active_orders[order_id]
            
            # Only pending orders can be modified
            if order.status != OrderStatus.PENDING:
                analysis_logger.warning(f"Cannot modify order in status: {order.status}")
                return False
            
            # Modify through MT5 if ticket exists
            if order.mt5_ticket and self.mt5_connection:
                mt5_result = await self._modify_order_mt5(order, modifications)
                if not mt5_result:
                    return False
            
            # Update order request
            for key, value in modifications.items():
                if hasattr(order.request, key):
                    setattr(order.request, key, value)
            
            order.last_update = datetime.now()
            
            analysis_logger.info(f"Order modified successfully: {order_id}")
            return True
            
        except Exception as e:
            analysis_logger.error(f"Error modifying order: {e}")
            return False
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        
        try:
            if order_id not in self.active_orders:
                analysis_logger.warning(f"Order not found for cancellation: {order_id}")
                return False
            
            order = self.active_orders[order_id]
            
            # Only pending orders can be cancelled
            if order.status not in [OrderStatus.PENDING, OrderStatus.PLACED]:
                analysis_logger.warning(f"Cannot cancel order in status: {order.status}")
                return False
            
            # Cancel through MT5 if ticket exists
            if order.mt5_ticket and self.mt5_connection:
                mt5_result = await self._cancel_order_mt5(order)
                if not mt5_result:
                    return False
            
            # Update order status
            order.status = OrderStatus.CANCELLED
            order.last_update = datetime.now()
            
            # Remove from active orders
            self.active_orders.pop(order_id, None)
            
            # Update statistics
            self.stats['cancelled_orders'] += 1
            
            analysis_logger.info(f"Order cancelled successfully: {order_id}")
            return True
            
        except Exception as e:
            analysis_logger.error(f"Error cancelling order: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get current order status"""
        return self.orders.get(order_id)
    
    async def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all active orders, optionally filtered by symbol"""
        
        orders = list(self.active_orders.values())
        
        if symbol:
            orders = [order for order in orders if order.request.symbol == symbol]
        
        return orders
    
    async def update_orders_status(self):
        """Update status of all active orders from MT5"""
        
        try:
            if not self.mt5_connection:
                return
            
            for order_id, order in list(self.active_orders.items()):
                if order.mt5_ticket:
                    # Check order status in MT5
                    mt5_status = await self._check_order_status_mt5(order)
                    
                    if mt5_status and mt5_status != order.status:
                        old_status = order.status
                        order.status = mt5_status
                        order.last_update = datetime.now()
                        
                        # Remove from active if no longer active
                        if mt5_status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                            self.active_orders.pop(order_id, None)
                        
                        analysis_logger.info(
                            f"Order status updated: {order_id} - {old_status.value} -> {mt5_status.value}"
                        )
        
        except Exception as e:
            analysis_logger.error(f"Error updating order status: {e}")
    
    def _calculate_position_size(self, signal: TradingSignal, risk_analysis: Optional[RiskRewardAnalysis]) -> float:
        """Calculate position size based on risk management"""
        
        try:
            # Default position size
            default_size = 0.1
            
            if not risk_analysis or not risk_analysis.recommended_position_size:
                return default_size
            
            # Use recommended position size from risk analysis
            position_size = risk_analysis.recommended_position_size
            
            # Apply limits
            position_size = max(self.config['min_order_size'], position_size)
            position_size = min(self.config['max_order_size'], position_size)
            
            return round(position_size, 2)
            
        except Exception as e:
            analysis_logger.warning(f"Error calculating position size: {e}")
            return 0.1
    
    async def _validate_order_request(self, order_request: OrderRequest) -> Dict[str, Any]:
        """Validate order request"""
        
        try:
            # Check volume limits
            if order_request.volume < self.config['min_order_size']:
                return {'valid': False, 'error': f"Volume too small: {order_request.volume}"}
            
            if order_request.volume > self.config['max_order_size']:
                return {'valid': False, 'error': f"Volume too large: {order_request.volume}"}
            
            # Check pending order limits
            if len(self.active_orders) >= self.config['max_pending_orders']:
                return {'valid': False, 'error': "Too many pending orders"}
            
            # Check symbol-specific limits
            symbol_orders = [o for o in self.active_orders.values() if o.request.symbol == order_request.symbol]
            if len(symbol_orders) >= self.config['max_orders_per_symbol']:
                return {'valid': False, 'error': f"Too many orders for symbol: {order_request.symbol}"}
            
            # Validate price levels
            if order_request.order_type in [OrderType.BUY_LIMIT, OrderType.SELL_LIMIT] and not order_request.price:
                return {'valid': False, 'error': "Limit orders require price"}
            
            return {'valid': True, 'error': None}
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    async def _execute_order_mt5(self, order: Order) -> OrderExecution:
        """Execute order through MT5"""
        
        # Mock implementation for now
        # In real implementation, this would use MT5 API
        
        try:
            execution_time = datetime.now()
            
            # Simulate order execution
            if self.mt5_connection:
                # Real MT5 execution would go here
                pass
            
            # Mock successful execution
            mt5_ticket = int(datetime.now().timestamp() * 1000) % 1000000
            executed_volume = order.request.volume
            execution_price = order.request.price or 1.1000  # Mock price
            commission = executed_volume * 0.00001  # Mock commission
            
            return OrderExecution(
                success=True,
                order_id=order.order_id,
                mt5_ticket=mt5_ticket,
                executed_volume=executed_volume,
                execution_price=execution_price,
                commission=commission,
                execution_time=execution_time
            )
            
        except Exception as e:
            return OrderExecution(
                success=False,
                order_id=order.order_id,
                error_message=str(e)
            )
    
    async def _modify_order_mt5(self, order: Order, modifications: Dict[str, Any]) -> bool:
        """Modify order in MT5"""
        
        try:
            # Mock implementation
            if self.mt5_connection:
                # Real MT5 modification would go here
                pass
            
            return True
            
        except Exception as e:
            analysis_logger.error(f"Error modifying order in MT5: {e}")
            return False
    
    async def _cancel_order_mt5(self, order: Order) -> bool:
        """Cancel order in MT5"""
        
        try:
            # Mock implementation
            if self.mt5_connection:
                # Real MT5 cancellation would go here
                pass
            
            return True
            
        except Exception as e:
            analysis_logger.error(f"Error cancelling order in MT5: {e}")
            return False
    
    async def _check_order_status_mt5(self, order: Order) -> Optional[OrderStatus]:
        """Check order status in MT5"""
        
        try:
            # Mock implementation
            if self.mt5_connection:
                # Real MT5 status check would go here
                pass
            
            return order.status
            
        except Exception as e:
            analysis_logger.error(f"Error checking order status in MT5: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get order manager statistics"""
        
        success_rate = 0.0
        if self.stats['total_orders'] > 0:
            success_rate = self.stats['successful_orders'] / self.stats['total_orders']
        
        return {
            **self.stats,
            'success_rate': success_rate,
            'active_orders_count': len(self.active_orders),
            'avg_commission': self.stats['total_commission'] / max(self.stats['successful_orders'], 1)
        }


# Global instance
order_manager = OrderManager()