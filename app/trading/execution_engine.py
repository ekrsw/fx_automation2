"""
Execution Engine

Real-time trade execution engine for FX trading with market order processing,
slippage management, and execution optimization.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio

from app.trading.order_manager import OrderManager, OrderRequest, OrderExecution, OrderType, OrderStatus
from app.analysis.strategy_engine.signal_generator import TradingSignal, SignalType
from app.analysis.strategy_engine.stop_loss_calculator import StopLossRecommendation
from app.analysis.strategy_engine.take_profit_calculator import TakeProfitRecommendation
from app.analysis.strategy_engine.risk_reward_calculator import RiskRewardAnalysis
from app.mt5.connection import MT5Connection
from app.utils.logger import analysis_logger


class ExecutionMode(Enum):
    """Execution mode options"""
    IMMEDIATE = "immediate"
    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"
    ICEBERG = "iceberg"
    TWAP = "twap"  # Time Weighted Average Price


class ExecutionPriority(Enum):
    """Execution priority levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ExecutionRequest:
    """Execution request structure"""
    signal: TradingSignal
    stop_loss_rec: Optional[StopLossRecommendation] = None
    take_profit_rec: Optional[TakeProfitRecommendation] = None
    risk_analysis: Optional[RiskRewardAnalysis] = None
    
    # Execution parameters
    execution_mode: ExecutionMode = ExecutionMode.IMMEDIATE
    priority: ExecutionPriority = ExecutionPriority.MEDIUM
    max_slippage: float = 0.0005  # 0.05%
    timeout_seconds: int = 30
    
    # Advanced options
    split_orders: bool = False
    max_order_size: float = 1.0
    execution_delay: float = 0.0
    
    # Metadata
    request_id: str = field(default_factory=lambda: str(datetime.now().timestamp()))
    created_time: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionResult:
    """Execution result structure"""
    request_id: str
    success: bool
    
    # Order information
    orders_placed: List[OrderExecution] = field(default_factory=list)
    total_volume: float = 0.0
    avg_execution_price: Optional[float] = None
    total_commission: float = 0.0
    
    # Performance metrics
    execution_time_ms: float = 0.0
    slippage: float = 0.0
    slippage_cost: float = 0.0
    
    # Status
    error_message: Optional[str] = None
    execution_details: Dict[str, Any] = field(default_factory=dict)
    completed_time: Optional[datetime] = None


class ExecutionEngine:
    """
    Real-time trade execution engine
    
    Handles the execution of trading signals with:
    - Market order processing
    - Slippage management and optimization
    - Order splitting and aggregation
    - Execution timing optimization
    - Performance tracking and analysis
    """
    
    def __init__(self, order_manager: OrderManager, mt5_connection: Optional[MT5Connection] = None):
        """Initialize execution engine"""
        
        self.order_manager = order_manager
        self.mt5_connection = mt5_connection
        
        # Execution queue
        self.execution_queue: List[ExecutionRequest] = []
        self.processing_requests: Dict[str, ExecutionRequest] = {}
        
        # Configuration
        self.config = {
            'max_queue_size': 100,
            'max_concurrent_executions': 5,
            'default_timeout_seconds': 30,
            'max_slippage_threshold': 0.001,  # 0.1%
            'min_execution_interval_ms': 100,
            'execution_retry_attempts': 3,
            'price_update_interval_ms': 500,
            'market_impact_threshold': 0.0002,  # 0.02%
        }
        
        # Market data cache
        self.market_data: Dict[str, Dict[str, Any]] = {}
        self.last_price_update: Dict[str, datetime] = {}
        
        # Execution statistics
        self.stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'avg_execution_time_ms': 0.0,
            'avg_slippage': 0.0,
            'total_slippage_cost': 0.0,
            'total_volume_executed': 0.0,
            'total_commission': 0.0
        }
        
        analysis_logger.info("Execution Engine initialized")
    
    async def execute_signal(self, execution_request: ExecutionRequest) -> ExecutionResult:
        """Execute trading signal"""
        
        start_time = datetime.now()
        
        try:
            # Validate execution request
            validation_result = await self._validate_execution_request(execution_request)
            if not validation_result['valid']:
                return ExecutionResult(
                    request_id=execution_request.request_id,
                    success=False,
                    error_message=validation_result['error'],
                    completed_time=datetime.now()
                )
            
            # Add to processing
            self.processing_requests[execution_request.request_id] = execution_request
            
            # Update market data for symbol
            await self._update_market_data(execution_request.signal.symbol)
            
            # Execute based on mode
            if execution_request.execution_mode == ExecutionMode.IMMEDIATE:
                result = await self._execute_immediate(execution_request)
            elif execution_request.execution_mode == ExecutionMode.AGGRESSIVE:
                result = await self._execute_aggressive(execution_request)
            elif execution_request.execution_mode == ExecutionMode.PASSIVE:
                result = await self._execute_passive(execution_request)
            elif execution_request.execution_mode == ExecutionMode.ICEBERG:
                result = await self._execute_iceberg(execution_request)
            elif execution_request.execution_mode == ExecutionMode.TWAP:
                result = await self._execute_twap(execution_request)
            else:
                result = await self._execute_immediate(execution_request)
            
            # Calculate execution metrics
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            result.execution_time_ms = execution_time
            result.completed_time = datetime.now()
            
            # Calculate slippage
            if result.avg_execution_price and execution_request.signal.entry_price:
                result.slippage = abs(result.avg_execution_price - execution_request.signal.entry_price) / execution_request.signal.entry_price
                result.slippage_cost = result.slippage * result.total_volume * result.avg_execution_price
            
            # Update statistics
            self._update_execution_stats(result)
            
            # Remove from processing
            self.processing_requests.pop(execution_request.request_id, None)
            
            analysis_logger.info(
                f"Signal execution completed: {execution_request.request_id} - "
                f"Success: {result.success}, "
                f"Volume: {result.total_volume}, "
                f"Price: {result.avg_execution_price}, "
                f"Time: {result.execution_time_ms:.1f}ms"
            )
            
            return result
            
        except Exception as e:
            analysis_logger.error(f"Error executing signal: {e}")
            
            # Remove from processing
            self.processing_requests.pop(execution_request.request_id, None)
            
            return ExecutionResult(
                request_id=execution_request.request_id,
                success=False,
                error_message=str(e),
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                completed_time=datetime.now()
            )
    
    async def queue_execution(self, execution_request: ExecutionRequest) -> bool:
        """Add execution request to queue"""
        
        try:
            if len(self.execution_queue) >= self.config['max_queue_size']:
                analysis_logger.warning("Execution queue is full")
                return False
            
            self.execution_queue.append(execution_request)
            
            analysis_logger.info(f"Execution request queued: {execution_request.request_id}")
            return True
            
        except Exception as e:
            analysis_logger.error(f"Error queuing execution: {e}")
            return False
    
    async def process_execution_queue(self):
        """Process queued execution requests"""
        
        try:
            while self.execution_queue and len(self.processing_requests) < self.config['max_concurrent_executions']:
                # Get highest priority request
                request = self._get_next_priority_request()
                if not request:
                    break
                
                # Execute asynchronously
                asyncio.create_task(self.execute_signal(request))
                
                # Delay between executions
                if self.config['min_execution_interval_ms'] > 0:
                    await asyncio.sleep(self.config['min_execution_interval_ms'] / 1000)
            
        except Exception as e:
            analysis_logger.error(f"Error processing execution queue: {e}")
    
    async def cancel_execution(self, request_id: str) -> bool:
        """Cancel pending execution"""
        
        try:
            # Remove from queue
            for i, request in enumerate(self.execution_queue):
                if request.request_id == request_id:
                    self.execution_queue.pop(i)
                    analysis_logger.info(f"Execution request cancelled from queue: {request_id}")
                    return True
            
            # Cannot cancel if already processing
            if request_id in self.processing_requests:
                analysis_logger.warning(f"Cannot cancel processing execution: {request_id}")
                return False
            
            analysis_logger.warning(f"Execution request not found: {request_id}")
            return False
            
        except Exception as e:
            analysis_logger.error(f"Error cancelling execution: {e}")
            return False
    
    async def _execute_immediate(self, execution_request: ExecutionRequest) -> ExecutionResult:
        """Execute immediately as market order"""
        
        try:
            # Create order request from signal
            order_request = await self.order_manager.create_order_from_signal(
                execution_request.signal,
                execution_request.stop_loss_rec,
                execution_request.take_profit_rec,
                execution_request.risk_analysis
            )
            
            # Force market order for immediate execution
            if order_request.order_type == OrderType.BUY_LIMIT:
                order_request.order_type = OrderType.MARKET_BUY
            elif order_request.order_type == OrderType.SELL_LIMIT:
                order_request.order_type = OrderType.MARKET_SELL
            
            # Set execution parameters
            order_request.max_slippage = execution_request.max_slippage
            order_request.deviation = int(execution_request.max_slippage * 100000)  # Convert to points
            
            # Execute order
            execution = await self.order_manager.place_order(order_request)
            
            # Create result
            result = ExecutionResult(
                request_id=execution_request.request_id,
                success=execution.success,
                orders_placed=[execution],
                total_volume=execution.executed_volume,
                avg_execution_price=execution.execution_price,
                total_commission=execution.commission,
                error_message=execution.error_message
            )
            
            return result
            
        except Exception as e:
            return ExecutionResult(
                request_id=execution_request.request_id,
                success=False,
                error_message=str(e)
            )
    
    async def _execute_aggressive(self, execution_request: ExecutionRequest) -> ExecutionResult:
        """Execute with aggressive market taking"""
        
        try:
            # Similar to immediate but with higher slippage tolerance
            execution_request.max_slippage *= 1.5  # Allow 50% more slippage
            
            return await self._execute_immediate(execution_request)
            
        except Exception as e:
            return ExecutionResult(
                request_id=execution_request.request_id,
                success=False,
                error_message=str(e)
            )
    
    async def _execute_passive(self, execution_request: ExecutionRequest) -> ExecutionResult:
        """Execute with passive limit orders"""
        
        try:
            # Create order request from signal
            order_request = await self.order_manager.create_order_from_signal(
                execution_request.signal,
                execution_request.stop_loss_rec,
                execution_request.take_profit_rec,
                execution_request.risk_analysis
            )
            
            # Ensure limit order with conservative pricing
            current_price = await self._get_current_price(execution_request.signal.symbol)
            if not current_price:
                raise ValueError("Cannot get current price for passive execution")
            
            # Adjust price for better fill probability
            price_adjustment = 0.0001  # 1 pip improvement
            
            if execution_request.signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
                order_request.order_type = OrderType.BUY_LIMIT
                order_request.price = current_price['ask'] + price_adjustment
            else:
                order_request.order_type = OrderType.SELL_LIMIT
                order_request.price = current_price['bid'] - price_adjustment
            
            # Execute order
            execution = await self.order_manager.place_order(order_request)
            
            # Create result
            result = ExecutionResult(
                request_id=execution_request.request_id,
                success=execution.success,
                orders_placed=[execution],
                total_volume=execution.executed_volume,
                avg_execution_price=execution.execution_price,
                total_commission=execution.commission,
                error_message=execution.error_message
            )
            
            return result
            
        except Exception as e:
            return ExecutionResult(
                request_id=execution_request.request_id,
                success=False,
                error_message=str(e)
            )
    
    async def _execute_iceberg(self, execution_request: ExecutionRequest) -> ExecutionResult:
        """Execute with iceberg order splitting"""
        
        try:
            total_volume = 0.0
            if execution_request.risk_analysis and execution_request.risk_analysis.recommended_position_size:
                total_volume = execution_request.risk_analysis.recommended_position_size
            else:
                total_volume = 0.1  # Default
            
            # Split into smaller orders
            max_order_size = execution_request.max_order_size
            num_orders = int(np.ceil(total_volume / max_order_size))
            
            executions = []
            total_executed_volume = 0.0
            total_commission = 0.0
            
            for i in range(num_orders):
                # Calculate order size
                remaining_volume = total_volume - total_executed_volume
                order_size = min(max_order_size, remaining_volume)
                
                if order_size <= 0:
                    break
                
                # Create modified signal with smaller volume
                modified_signal = execution_request.signal
                
                # Create order request
                order_request = await self.order_manager.create_order_from_signal(
                    modified_signal,
                    execution_request.stop_loss_rec,
                    execution_request.take_profit_rec,
                    execution_request.risk_analysis
                )
                
                order_request.volume = order_size
                
                # Execute order
                execution = await self.order_manager.place_order(order_request)
                executions.append(execution)
                
                if execution.success:
                    total_executed_volume += execution.executed_volume
                    total_commission += execution.commission
                
                # Small delay between orders
                await asyncio.sleep(0.1)
            
            # Calculate average execution price
            avg_price = None
            if executions:
                successful_executions = [e for e in executions if e.success and e.execution_price]
                if successful_executions:
                    weighted_sum = sum(e.execution_price * e.executed_volume for e in successful_executions)
                    total_volume_sum = sum(e.executed_volume for e in successful_executions)
                    avg_price = weighted_sum / total_volume_sum if total_volume_sum > 0 else None
            
            # Create result
            result = ExecutionResult(
                request_id=execution_request.request_id,
                success=any(e.success for e in executions),
                orders_placed=executions,
                total_volume=total_executed_volume,
                avg_execution_price=avg_price,
                total_commission=total_commission
            )
            
            return result
            
        except Exception as e:
            return ExecutionResult(
                request_id=execution_request.request_id,
                success=False,
                error_message=str(e)
            )
    
    async def _execute_twap(self, execution_request: ExecutionRequest) -> ExecutionResult:
        """Execute with Time Weighted Average Price strategy"""
        
        try:
            # TWAP execution over time intervals
            execution_duration = 300  # 5 minutes
            num_intervals = 10
            interval_duration = execution_duration / num_intervals
            
            total_volume = 0.0
            if execution_request.risk_analysis and execution_request.risk_analysis.recommended_position_size:
                total_volume = execution_request.risk_analysis.recommended_position_size
            else:
                total_volume = 0.1
            
            volume_per_interval = total_volume / num_intervals
            
            executions = []
            total_executed_volume = 0.0
            total_commission = 0.0
            
            for i in range(num_intervals):
                # Create order request
                order_request = await self.order_manager.create_order_from_signal(
                    execution_request.signal,
                    execution_request.stop_loss_rec,
                    execution_request.take_profit_rec,
                    execution_request.risk_analysis
                )
                
                order_request.volume = volume_per_interval
                
                # Execute order
                execution = await self.order_manager.place_order(order_request)
                executions.append(execution)
                
                if execution.success:
                    total_executed_volume += execution.executed_volume
                    total_commission += execution.commission
                
                # Wait for next interval
                if i < num_intervals - 1:
                    await asyncio.sleep(interval_duration)
            
            # Calculate average execution price
            avg_price = None
            if executions:
                successful_executions = [e for e in executions if e.success and e.execution_price]
                if successful_executions:
                    weighted_sum = sum(e.execution_price * e.executed_volume for e in successful_executions)
                    total_volume_sum = sum(e.executed_volume for e in successful_executions)
                    avg_price = weighted_sum / total_volume_sum if total_volume_sum > 0 else None
            
            # Create result
            result = ExecutionResult(
                request_id=execution_request.request_id,
                success=any(e.success for e in executions),
                orders_placed=executions,
                total_volume=total_executed_volume,
                avg_execution_price=avg_price,
                total_commission=total_commission
            )
            
            return result
            
        except Exception as e:
            return ExecutionResult(
                request_id=execution_request.request_id,
                success=False,
                error_message=str(e)
            )
    
    async def _validate_execution_request(self, execution_request: ExecutionRequest) -> Dict[str, Any]:
        """Validate execution request"""
        
        try:
            # Check signal validity
            if not execution_request.signal:
                return {'valid': False, 'error': 'No trading signal provided'}
            
            # Check symbol
            if not execution_request.signal.symbol:
                return {'valid': False, 'error': 'No symbol specified'}
            
            # Check signal type
            if execution_request.signal.signal_type not in [SignalType.MARKET_BUY, SignalType.MARKET_SELL, SignalType.LIMIT_BUY, SignalType.LIMIT_SELL]:
                return {'valid': False, 'error': f'Unsupported signal type: {execution_request.signal.signal_type}'}
            
            # Check confidence
            if execution_request.signal.confidence < 0.1:
                return {'valid': False, 'error': 'Signal confidence too low'}
            
            # Check timeout
            if execution_request.timeout_seconds <= 0:
                return {'valid': False, 'error': 'Invalid timeout'}
            
            return {'valid': True, 'error': None}
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    async def _update_market_data(self, symbol: str):
        """Update market data for symbol"""
        
        try:
            current_time = datetime.now()
            
            # Check if update needed
            last_update = self.last_price_update.get(symbol)
            if last_update and (current_time - last_update).total_seconds() < self.config['price_update_interval_ms'] / 1000:
                return
            
            # Mock market data
            # In real implementation, this would fetch from MT5
            base_price = 1.1000 if 'USD' in symbol else 150.0 if 'JPY' in symbol else 1.3000
            spread = 0.0001 if 'USD' in symbol else 0.01 if 'JPY' in symbol else 0.0002
            
            self.market_data[symbol] = {
                'bid': base_price - spread / 2,
                'ask': base_price + spread / 2,
                'spread': spread,
                'timestamp': current_time
            }
            
            self.last_price_update[symbol] = current_time
            
        except Exception as e:
            analysis_logger.error(f"Error updating market data: {e}")
    
    async def _get_current_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get current market price for symbol"""
        
        await self._update_market_data(symbol)
        return self.market_data.get(symbol)
    
    def _get_next_priority_request(self) -> Optional[ExecutionRequest]:
        """Get next priority execution request from queue"""
        
        if not self.execution_queue:
            return None
        
        # Sort by priority and creation time
        sorted_queue = sorted(
            self.execution_queue,
            key=lambda r: (r.priority.value, r.created_time)
        )
        
        # Remove and return highest priority request
        request = sorted_queue[0]
        self.execution_queue.remove(request)
        
        return request
    
    def _update_execution_stats(self, result: ExecutionResult):
        """Update execution statistics"""
        
        self.stats['total_executions'] += 1
        
        if result.success:
            self.stats['successful_executions'] += 1
            self.stats['total_volume_executed'] += result.total_volume
            self.stats['total_commission'] += result.total_commission
            
            # Update averages
            n = self.stats['successful_executions']
            self.stats['avg_execution_time_ms'] = (
                (self.stats['avg_execution_time_ms'] * (n - 1) + result.execution_time_ms) / n
            )
            self.stats['avg_slippage'] = (
                (self.stats['avg_slippage'] * (n - 1) + result.slippage) / n
            )
            self.stats['total_slippage_cost'] += result.slippage_cost
        else:
            self.stats['failed_executions'] += 1
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution engine statistics"""
        
        success_rate = 0.0
        if self.stats['total_executions'] > 0:
            success_rate = self.stats['successful_executions'] / self.stats['total_executions']
        
        return {
            **self.stats,
            'success_rate': success_rate,
            'queue_size': len(self.execution_queue),
            'processing_count': len(self.processing_requests),
            'avg_slippage_bps': self.stats['avg_slippage'] * 10000,  # Basis points
            'avg_commission': self.stats['total_commission'] / max(self.stats['successful_executions'], 1)
        }


# Global instance
execution_engine = ExecutionEngine(order_manager=None)  # Will be initialized with order_manager