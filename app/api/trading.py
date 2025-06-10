"""
Trading Control API endpoints
Phase 7.1 Implementation - Trading system control and management
"""

from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import asyncio

from app.dependencies import get_settings_cached
from app.config import Settings
from app.trading.trading_engine import TradingEngine, TradingMode, trading_engine
from app.trading.position_manager import position_manager, PositionModification, PositionCloseRequest
from app.trading.order_manager import order_manager, OrderRequest, OrderType, OrderTimeInForce
from app.trading.risk_manager import risk_manager
from app.analysis.strategy_engine.signal_generator import TradingSignal, SignalType, SignalUrgency
from app.integrations.websocket_integration import (
    notify_signal_generated, notify_order_event, notify_position_event,
    notify_session_event, notify_risk_alert, notify_emergency_stop
)
from app.utils.logger import main_logger

router = APIRouter()


# Request/Response Models
class TradingSessionRequest(BaseModel):
    mode: str  # "demo", "live"
    
class SignalRequest(BaseModel):
    symbol: str
    signal_type: str
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.5
    urgency: str = "MEDIUM"

class OrderCreateRequest(BaseModel):
    symbol: str
    order_type: str
    volume: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    comment: str = ""

class PositionModifyRequest(BaseModel):
    position_id: str
    modification_type: str  # "stop_loss", "take_profit", "both"
    new_stop_loss: Optional[float] = None
    new_take_profit: Optional[float] = None
    reason: str = ""

class PositionCloseRequestModel(BaseModel):
    position_id: str
    volume: Optional[float] = None  # None for full close
    reason: str = ""

class EmergencyStopRequest(BaseModel):
    reason: str


# Trading Session Control
@router.post("/session/start")
async def start_trading_session(request: TradingSessionRequest) -> Dict[str, Any]:
    """
    Start a new trading session
    
    Args:
        request: Trading session configuration
        
    Returns:
        Session start result
    """
    try:
        if not trading_engine:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Trading engine not available"
            )
        
        # Map string mode to enum
        mode_map = {
            "demo": TradingMode.DEMO,
            "live": TradingMode.LIVE,
            "simulation": TradingMode.SIMULATION
        }
        
        if request.mode.lower() not in mode_map:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid trading mode: {request.mode}"
            )
        
        trading_mode = mode_map[request.mode.lower()]
        
        # Start trading session
        session_id = await trading_engine.start_trading_session(trading_mode)
        
        result = {
            "success": True,
            "session_id": session_id,
            "mode": request.mode,
            "started_at": datetime.now().isoformat(),
            "message": f"Trading session started in {request.mode} mode"
        }
        
        # Notify WebSocket subscribers
        await notify_session_event("started", result)
        
        main_logger.info(f"Trading session started: {session_id} in {request.mode} mode")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        main_logger.error(f"Error starting trading session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/session/stop")
async def stop_trading_session() -> Dict[str, Any]:
    """
    Stop the current trading session
    
    Returns:
        Session stop result
    """
    try:
        if not trading_engine:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Trading engine not available"
            )
        
        # Stop trading session
        await trading_engine.stop_trading_session()
        
        result = {
            "success": True,
            "stopped_at": datetime.now().isoformat(),
            "message": "Trading session stopped"
        }
        
        # Notify WebSocket subscribers
        await notify_session_event("stopped", result)
        
        main_logger.info("Trading session stopped")
        return result
        
    except Exception as e:
        main_logger.error(f"Error stopping trading session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/session/pause")
async def pause_trading() -> Dict[str, Any]:
    """
    Pause trading (keeps session active but stops processing)
    
    Returns:
        Pause result
    """
    try:
        if not trading_engine:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Trading engine not available"
            )
        
        await trading_engine.pause_trading()
        
        result = {
            "success": True,
            "paused_at": datetime.now().isoformat(),
            "message": "Trading paused"
        }
        
        # Notify WebSocket subscribers
        await notify_session_event("paused", result)
        
        main_logger.info("Trading paused")
        return result
        
    except Exception as e:
        main_logger.error(f"Error pausing trading: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/session/resume")
async def resume_trading() -> Dict[str, Any]:
    """
    Resume trading after pause
    
    Returns:
        Resume result
    """
    try:
        if not trading_engine:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Trading engine not available"
            )
        
        await trading_engine.resume_trading()
        
        result = {
            "success": True,
            "resumed_at": datetime.now().isoformat(),
            "message": "Trading resumed"
        }
        
        # Notify WebSocket subscribers
        await notify_session_event("resumed", result)
        
        main_logger.info("Trading resumed")
        return result
        
    except Exception as e:
        main_logger.error(f"Error resuming trading: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Signal Processing
@router.post("/signals/process")
async def process_signal(request: SignalRequest) -> Dict[str, Any]:
    """
    Process a trading signal
    
    Args:
        request: Trading signal data
        
    Returns:
        Signal processing result
    """
    try:
        if not trading_engine:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Trading engine not available"
            )
        
        # Create TradingSignal object
        signal_type_map = {
            "MARKET_BUY": SignalType.MARKET_BUY,
            "MARKET_SELL": SignalType.MARKET_SELL,
            "LIMIT_BUY": SignalType.LIMIT_BUY,
            "LIMIT_SELL": SignalType.LIMIT_SELL
        }
        
        urgency_map = {
            "IMMEDIATE": SignalUrgency.IMMEDIATE,
            "HIGH": SignalUrgency.HIGH,
            "MEDIUM": SignalUrgency.MEDIUM,
            "LOW": SignalUrgency.LOW,
            "WATCH": SignalUrgency.WATCH
        }
        
        if request.signal_type not in signal_type_map:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid signal type: {request.signal_type}"
            )
        
        signal = TradingSignal(
            signal_id=f"api_signal_{int(datetime.now().timestamp())}",
            symbol=request.symbol,
            signal_type=signal_type_map[request.signal_type],
            urgency=urgency_map.get(request.urgency, SignalUrgency.MEDIUM),
            entry_price=request.entry_price,
            stop_loss=request.stop_loss,
            take_profit_1=request.take_profit,
            confidence=request.confidence
        )
        
        # Process signal
        success = await trading_engine.process_signal(signal)
        
        result = {
            "success": success,
            "signal_id": signal.signal_id,
            "processed_at": datetime.now().isoformat(),
            "message": "Signal processed successfully" if success else "Signal processing failed"
        }
        
        # Notify WebSocket subscribers
        if success:
            signal_data = {
                "signal_id": signal.signal_id,
                "symbol": signal.symbol,
                "signal_type": signal.signal_type.value,
                "entry_price": signal.entry_price,
                "confidence": signal.confidence,
                "processed_at": result["processed_at"]
            }
            await notify_signal_generated(signal_data)
        
        main_logger.info(f"Signal processed: {signal.signal_id}, success: {success}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        main_logger.error(f"Error processing signal: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Order Management
@router.post("/orders/create")
async def create_order(request: OrderCreateRequest) -> Dict[str, Any]:
    """
    Create a new order
    
    Args:
        request: Order creation data
        
    Returns:
        Order creation result
    """
    try:
        if not order_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Order manager not available"
            )
        
        # Map string order type to enum
        order_type_map = {
            "MARKET_BUY": OrderType.MARKET_BUY,
            "MARKET_SELL": OrderType.MARKET_SELL,
            "BUY_LIMIT": OrderType.BUY_LIMIT,
            "SELL_LIMIT": OrderType.SELL_LIMIT,
            "BUY_STOP": OrderType.BUY_STOP,
            "SELL_STOP": OrderType.SELL_STOP
        }
        
        if request.order_type not in order_type_map:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid order type: {request.order_type}"
            )
        
        # Create order request
        order_request = OrderRequest(
            symbol=request.symbol,
            order_type=order_type_map[request.order_type],
            volume=request.volume,
            price=request.price,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            comment=request.comment
        )
        
        # Place order
        execution = await order_manager.place_order(order_request)
        
        result = {
            "success": execution.success,
            "order_id": execution.order_id,
            "mt5_ticket": execution.mt5_ticket,
            "executed_volume": execution.executed_volume,
            "execution_price": execution.execution_price,
            "commission": execution.commission,
            "error_message": execution.error_message,
            "executed_at": execution.execution_time.isoformat() if execution.execution_time else None
        }
        
        # Notify WebSocket subscribers
        if execution.success:
            order_data = {
                "order_id": execution.order_id,
                "symbol": request.symbol,
                "order_type": request.order_type,
                "volume": execution.executed_volume,
                "price": execution.execution_price,
                "mt5_ticket": execution.mt5_ticket,
                "executed_at": result["executed_at"]
            }
            await notify_order_event("placed", order_data)
        
        main_logger.info(f"Order created: {execution.order_id}, success: {execution.success}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        main_logger.error(f"Error creating order: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/orders/{order_id}")
async def cancel_order(order_id: str) -> Dict[str, Any]:
    """
    Cancel an existing order
    
    Args:
        order_id: ID of order to cancel
        
    Returns:
        Cancellation result
    """
    try:
        if not order_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Order manager not available"
            )
        
        # Cancel order
        success = await order_manager.cancel_order(order_id)
        
        result = {
            "success": success,
            "order_id": order_id,
            "cancelled_at": datetime.now().isoformat(),
            "message": "Order cancelled successfully" if success else "Order cancellation failed"
        }
        
        # Notify WebSocket subscribers
        if success:
            order_data = {
                "order_id": order_id,
                "cancelled_at": result["cancelled_at"]
            }
            await notify_order_event("cancelled", order_data)
        
        main_logger.info(f"Order cancellation: {order_id}, success: {success}")
        return result
        
    except Exception as e:
        main_logger.error(f"Error cancelling order: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Position Management
@router.put("/positions/{position_id}/modify")
async def modify_position(position_id: str, request: PositionModifyRequest) -> Dict[str, Any]:
    """
    Modify an existing position
    
    Args:
        position_id: ID of position to modify
        request: Modification parameters
        
    Returns:
        Modification result
    """
    try:
        if not position_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Position manager not available"
            )
        
        # Create modification request
        modification = PositionModification(
            position_id=position_id,
            modification_type=request.modification_type,
            new_stop_loss=request.new_stop_loss,
            new_take_profit=request.new_take_profit,
            reason=request.reason
        )
        
        # Modify position
        success = await position_manager.modify_position(modification)
        
        result = {
            "success": success,
            "position_id": position_id,
            "modification_type": request.modification_type,
            "modified_at": datetime.now().isoformat(),
            "message": "Position modified successfully" if success else "Position modification failed"
        }
        
        # Notify WebSocket subscribers
        if success:
            position_data = {
                "position_id": position_id,
                "modification_type": request.modification_type,
                "new_stop_loss": request.new_stop_loss,
                "new_take_profit": request.new_take_profit,
                "modified_at": result["modified_at"]
            }
            await notify_position_event("modified", position_data)
        
        main_logger.info(f"Position modification: {position_id}, success: {success}")
        return result
        
    except Exception as e:
        main_logger.error(f"Error modifying position: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/positions/{position_id}/close")
async def close_position(position_id: str, request: PositionCloseRequestModel) -> Dict[str, Any]:
    """
    Close an existing position
    
    Args:
        position_id: ID of position to close
        request: Close parameters
        
    Returns:
        Close result
    """
    try:
        if not position_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Position manager not available"
            )
        
        # Create close request
        close_request = PositionCloseRequest(
            position_id=position_id,
            volume=request.volume,
            reason=request.reason
        )
        
        # Close position
        close_result = await position_manager.close_position(close_request)
        
        result = {
            "success": close_result.success,
            "position_id": position_id,
            "closed_volume": close_result.closed_volume,
            "close_price": close_result.close_price,
            "realized_pnl": close_result.realized_pnl,
            "commission": close_result.commission,
            "error_message": close_result.error_message,
            "closed_at": close_result.close_time.isoformat() if close_result.close_time else None
        }
        
        # Notify WebSocket subscribers
        if close_result.success:
            position_data = {
                "position_id": position_id,
                "closed_volume": close_result.closed_volume,
                "close_price": close_result.close_price,
                "realized_pnl": close_result.realized_pnl,
                "closed_at": result["closed_at"]
            }
            await notify_position_event("closed", position_data)
        
        main_logger.info(f"Position close: {position_id}, success: {close_result.success}")
        return result
        
    except Exception as e:
        main_logger.error(f"Error closing position: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/positions/close-all")
async def close_all_positions(symbol: Optional[str] = None) -> Dict[str, Any]:
    """
    Close all open positions
    
    Args:
        symbol: Optional symbol filter
        
    Returns:
        Close all result
    """
    try:
        if not position_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Position manager not available"
            )
        
        # Close all positions
        results = await position_manager.close_all_positions(symbol)
        
        successful_closes = len([r for r in results if r.success])
        total_pnl = sum(r.realized_pnl for r in results if r.success)
        
        result = {
            "success": len(results) > 0,
            "total_positions": len(results),
            "successful_closes": successful_closes,
            "failed_closes": len(results) - successful_closes,
            "total_realized_pnl": total_pnl,
            "symbol_filter": symbol,
            "closed_at": datetime.now().isoformat(),
            "results": [
                {
                    "position_id": r.position_id,
                    "success": r.success,
                    "realized_pnl": r.realized_pnl,
                    "error_message": r.error_message
                }
                for r in results
            ]
        }
        
        # Notify WebSocket subscribers
        if successful_closes > 0:
            close_all_data = {
                "total_positions": len(results),
                "successful_closes": successful_closes,
                "total_realized_pnl": total_pnl,
                "symbol_filter": symbol,
                "closed_at": result["closed_at"]
            }
            await notify_position_event("all_closed", close_all_data)
        
        main_logger.info(f"Close all positions: {successful_closes}/{len(results)} successful")
        return result
        
    except Exception as e:
        main_logger.error(f"Error closing all positions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Risk Management
@router.post("/risk/emergency-stop")
async def activate_emergency_stop(request: EmergencyStopRequest) -> Dict[str, Any]:
    """
    Activate emergency stop
    
    Args:
        request: Emergency stop reason
        
    Returns:
        Emergency stop result
    """
    try:
        if not risk_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Risk manager not available"
            )
        
        # Activate emergency stop
        await risk_manager.emergency_stop(request.reason)
        
        result = {
            "success": True,
            "reason": request.reason,
            "activated_at": datetime.now().isoformat(),
            "message": "Emergency stop activated"
        }
        
        # Notify WebSocket subscribers
        await notify_emergency_stop(request.reason)
        
        main_logger.warning(f"Emergency stop activated: {request.reason}")
        return result
        
    except Exception as e:
        main_logger.error(f"Error activating emergency stop: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/risk/emergency-stop/deactivate")
async def deactivate_emergency_stop() -> Dict[str, Any]:
    """
    Deactivate emergency stop
    
    Returns:
        Deactivation result
    """
    try:
        if not risk_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Risk manager not available"
            )
        
        # Deactivate emergency stop
        await risk_manager.deactivate_emergency_stop()
        
        result = {
            "success": True,
            "deactivated_at": datetime.now().isoformat(),
            "message": "Emergency stop deactivated"
        }
        
        main_logger.info("Emergency stop deactivated")
        return result
        
    except Exception as e:
        main_logger.error(f"Error deactivating emergency stop: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )