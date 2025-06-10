"""
Dashboard API endpoints
Phase 7.1 Implementation - Complete trading system dashboard
"""

from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import asyncio

from app.dependencies import get_settings_cached
from app.config import Settings
from app.mt5.connection import get_mt5_connection
from app.mt5.data_fetcher import get_data_fetcher
from app.trading.trading_engine import TradingEngine, trading_engine
from app.trading.position_manager import position_manager
from app.trading.order_manager import order_manager
from app.trading.risk_manager import risk_manager
from app.utils.logger import main_logger

router = APIRouter()


@router.get("/status")
async def get_system_status(settings: Settings = Depends(get_settings_cached)) -> Dict[str, Any]:
    """
    Get comprehensive system status including trading components
    
    Returns:
        Complete system status information
    """
    try:
        mt5_connection = get_mt5_connection()
        data_fetcher = get_data_fetcher()
        
        # Check MT5 connection
        mt5_healthy = await mt5_connection.health_check()
        mt5_info = mt5_connection.get_connection_info()
        
        # Test data fetching
        data_fetch_test = False
        try:
            test_data = await data_fetcher.get_live_prices(["USDJPY"])
            data_fetch_test = len(test_data) > 0
        except Exception:
            pass
        
        # Get trading system status
        trading_status = await trading_engine.get_trading_status() if trading_engine else None
        
        # Get component statistics
        order_stats = order_manager.get_statistics() if order_manager else {}
        position_stats = position_manager.get_statistics() if position_manager else {}
        risk_stats = risk_manager.get_statistics() if risk_manager else {}
        
        # Check trading system health
        trading_healthy = bool(trading_status and trading_status.get('session', {}).get('status') in ['active', 'paused'])
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy" if mt5_healthy and data_fetch_test and trading_healthy else "unhealthy",
            "components": {
                "mt5_connection": {
                    "status": "healthy" if mt5_healthy else "unhealthy",
                    "details": mt5_info
                },
                "data_fetcher": {
                    "status": "healthy" if data_fetch_test else "unhealthy"
                },
                "database": {
                    "status": "healthy",
                    "url": settings.database_url
                },
                "trading_engine": {
                    "status": "healthy" if trading_healthy else "inactive",
                    "session_status": trading_status.get('session', {}).get('status') if trading_status else "not_started",
                    "signals_processed": trading_status.get('session', {}).get('signals_processed', 0) if trading_status else 0
                },
                "order_manager": {
                    "status": "active",
                    "active_orders": order_stats.get('active_orders_count', 0),
                    "total_orders": order_stats.get('total_orders', 0),
                    "success_rate": f"{order_stats.get('success_rate', 0):.1%}"
                },
                "position_manager": {
                    "status": "active", 
                    "open_positions": position_stats.get('open_positions_count', 0),
                    "total_pnl": position_stats.get('total_realized_pnl', 0.0),
                    "win_rate": f"{position_stats.get('win_rate', 0):.1%}"
                },
                "risk_manager": {
                    "status": "active",
                    "emergency_stop": risk_stats.get('emergency_stop_active', False),
                    "risk_score": risk_stats.get('overall_risk_score', 0.0),
                    "alerts": risk_stats.get('total_alerts', 0)
                }
            },
            "configuration": {
                "trading_symbols": settings.trading_symbols,
                "max_positions": settings.max_positions,
                "risk_per_trade": settings.risk_per_trade
            }
        }
        
        main_logger.info("Comprehensive system status check completed")
        return status
        
    except Exception as e:
        main_logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Simple health check endpoint
    
    Returns:
        Health status
    """
    try:
        mt5_connection = get_mt5_connection()
        mt5_healthy = await mt5_connection.health_check()
        
        return {
            "status": "healthy" if mt5_healthy else "degraded",
            "timestamp": datetime.now().isoformat(),
            "mt5_connected": mt5_connection.is_connected()
        }
        
    except Exception as e:
        main_logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


@router.get("/performance")
async def get_performance_metrics() -> Dict[str, Any]:
    """
    Get comprehensive performance metrics from trading system
    
    Returns:
        Real-time performance metrics
    """
    try:
        # Get performance data from trading engine
        performance_report = await trading_engine.get_performance_report() if trading_engine else None
        
        # Get position manager statistics
        position_stats = position_manager.get_statistics() if position_manager else {}
        
        # Get order manager statistics  
        order_stats = order_manager.get_statistics() if order_manager else {}
        
        # Get portfolio summary
        portfolio_summary = await position_manager.get_portfolio_summary() if position_manager else {}
        
        # Calculate current metrics
        total_closed = position_stats.get('winning_positions', 0) + position_stats.get('losing_positions', 0)
        win_rate = position_stats.get('win_rate', 0.0)
        total_pnl = position_stats.get('total_realized_pnl', 0.0) + portfolio_summary.get('total_unrealized_pnl', 0.0)
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "trading_performance": {
                "total_trades": total_closed,
                "winning_trades": position_stats.get('winning_positions', 0),
                "losing_trades": position_stats.get('losing_positions', 0),
                "win_rate": win_rate,
                "profit_loss": total_pnl,
                "realized_pnl": position_stats.get('total_realized_pnl', 0.0),
                "unrealized_pnl": portfolio_summary.get('total_unrealized_pnl', 0.0),
                "largest_win": position_stats.get('largest_win', 0.0),
                "largest_loss": position_stats.get('largest_loss', 0.0),
                "avg_position_duration_hours": position_stats.get('avg_position_duration_hours', 0.0)
            },
            "order_performance": {
                "total_orders": order_stats.get('total_orders', 0),
                "successful_orders": order_stats.get('successful_orders', 0),
                "failed_orders": order_stats.get('failed_orders', 0),
                "success_rate": order_stats.get('success_rate', 0.0),
                "avg_commission": order_stats.get('avg_commission', 0.0),
                "total_volume": order_stats.get('total_volume', 0.0)
            },
            "portfolio_status": {
                "open_positions": portfolio_summary.get('total_open_positions', 0),
                "total_volume": portfolio_summary.get('total_volume', 0.0),
                "symbol_breakdown": portfolio_summary.get('symbol_breakdown', {})
            },
            "system_performance": performance_report.get('overall_statistics', {}) if performance_report else {}
        }
        
        main_logger.info("Performance metrics retrieved successfully")
        return metrics
        
    except Exception as e:
        main_logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions")
async def get_current_positions() -> Dict[str, Any]:
    """
    Get current trading positions with real-time data
    
    Returns:
        Current positions with PnL and details
    """
    try:
        # Get open positions
        open_positions = await position_manager.get_open_positions() if position_manager else []
        
        # Get portfolio summary
        portfolio_summary = await position_manager.get_portfolio_summary() if position_manager else {}
        
        # Format position data
        positions_data = []
        for position in open_positions:
            position_data = {
                "position_id": position.position_id,
                "symbol": position.symbol,
                "position_type": position.position_type.value,
                "volume": position.volume,
                "current_volume": position.current_volume,
                "entry_price": position.entry_price,
                "current_price": position.current_price,
                "entry_time": position.entry_time.isoformat(),
                "unrealized_pnl": position.unrealized_pnl,
                "realized_pnl": position.realized_pnl,
                "commission": position.commission,
                "stop_loss": position.stop_loss,
                "take_profit": position.take_profit,
                "status": position.status.value,
                "strategy_id": position.strategy_id,
                "mt5_ticket": position.mt5_ticket,
                "last_update": position.last_update.isoformat()
            }
            positions_data.append(position_data)
        
        # Calculate total exposure
        total_exposure = sum(pos.current_volume * pos.current_price for pos in open_positions)
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "positions": positions_data,
            "summary": {
                "total_positions": len(open_positions),
                "total_volume": portfolio_summary.get('total_volume', 0.0),
                "total_exposure": total_exposure,
                "total_unrealized_pnl": portfolio_summary.get('total_unrealized_pnl', 0.0),
                "total_realized_pnl": portfolio_summary.get('total_realized_pnl', 0.0),
                "symbol_breakdown": portfolio_summary.get('symbol_breakdown', {})
            }
        }
        
        main_logger.info(f"Retrieved {len(open_positions)} current positions")
        return result
        
    except Exception as e:
        main_logger.error(f"Error getting current positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recent-trades")
async def get_recent_trades(limit: int = 50) -> Dict[str, Any]:
    """
    Get recent trade history from position manager
    
    Args:
        limit: Maximum number of trades to return
        
    Returns:
        Recent closed positions (trades)
    """
    try:
        # Get recent closed positions from position manager
        if not position_manager:
            return {
                "timestamp": datetime.now().isoformat(),
                "trades": [],
                "total_trades": 0,
                "message": "Position manager not available"
            }
        
        # Get closed positions (recent trades)
        closed_positions = list(position_manager.closed_positions.values())
        
        # Sort by close time (most recent first)
        closed_positions.sort(key=lambda pos: pos.close_time or pos.entry_time, reverse=True)
        
        # Limit results
        recent_trades = closed_positions[:limit]
        
        # Format trade data
        trades_data = []
        for position in recent_trades:
            trade_data = {
                "trade_id": position.position_id,
                "symbol": position.symbol,
                "position_type": position.position_type.value,
                "volume": position.volume,
                "entry_price": position.entry_price,
                "close_price": position.close_price,
                "entry_time": position.entry_time.isoformat(),
                "close_time": position.close_time.isoformat() if position.close_time else None,
                "realized_pnl": position.realized_pnl,
                "commission": position.commission,
                "swap": position.swap,
                "strategy_id": position.strategy_id,
                "mt5_ticket": position.mt5_ticket,
                "duration_hours": (
                    (position.close_time - position.entry_time).total_seconds() / 3600
                    if position.close_time else 0
                ),
                "profit_loss": "profit" if position.realized_pnl > 0 else "loss"
            }
            trades_data.append(trade_data)
        
        # Calculate summary statistics
        total_trades = len(closed_positions)
        winning_trades = len([pos for pos in closed_positions if pos.realized_pnl > 0])
        losing_trades = len([pos for pos in closed_positions if pos.realized_pnl < 0])
        total_pnl = sum(pos.realized_pnl for pos in closed_positions)
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "trades": trades_data,
            "summary": {
                "total_trades": total_trades,
                "returned_trades": len(trades_data),
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": winning_trades / max(total_trades, 1),
                "total_realized_pnl": total_pnl,
                "avg_trade_pnl": total_pnl / max(total_trades, 1),
                "largest_win": max((pos.realized_pnl for pos in closed_positions if pos.realized_pnl > 0), default=0.0),
                "largest_loss": min((pos.realized_pnl for pos in closed_positions if pos.realized_pnl < 0), default=0.0)
            }
        }
        
        main_logger.info(f"Retrieved {len(trades_data)} recent trades out of {total_trades} total")
        return result
        
    except Exception as e:
        main_logger.error(f"Error getting recent trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orders")
async def get_active_orders() -> Dict[str, Any]:
    """
    Get active orders from order manager
    
    Returns:
        Current active orders
    """
    try:
        if not order_manager:
            return {
                "timestamp": datetime.now().isoformat(),
                "orders": [],
                "total_orders": 0,
                "message": "Order manager not available"
            }
        
        # Get active orders
        active_orders = await order_manager.get_active_orders()
        
        # Format order data
        orders_data = []
        for order in active_orders:
            order_data = {
                "order_id": order.order_id,
                "symbol": order.request.symbol,
                "order_type": order.request.order_type.value,
                "volume": order.request.volume,
                "price": order.request.price,
                "stop_loss": order.request.stop_loss,
                "take_profit": order.request.take_profit,
                "status": order.status.value,
                "created_time": order.created_time.isoformat(),
                "mt5_ticket": order.mt5_ticket,
                "filled_volume": order.filled_volume,
                "remaining_volume": order.remaining_volume,
                "strategy_id": order.request.strategy_id,
                "confidence_score": order.request.confidence_score
            }
            orders_data.append(order_data)
        
        # Get order statistics
        order_stats = order_manager.get_statistics()
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "orders": orders_data,
            "summary": {
                "active_orders": len(orders_data),
                "total_orders": order_stats.get('total_orders', 0),
                "successful_orders": order_stats.get('successful_orders', 0),
                "failed_orders": order_stats.get('failed_orders', 0),
                "success_rate": order_stats.get('success_rate', 0.0),
                "total_volume": order_stats.get('total_volume', 0.0)
            }
        }
        
        main_logger.info(f"Retrieved {len(orders_data)} active orders")
        return result
        
    except Exception as e:
        main_logger.error(f"Error getting active orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk-status")
async def get_risk_status() -> Dict[str, Any]:
    """
    Get current risk management status
    
    Returns:
        Risk management metrics and alerts
    """
    try:
        if not risk_manager:
            return {
                "timestamp": datetime.now().isoformat(),
                "risk_score": 0.0,
                "status": "unknown",
                "message": "Risk manager not available"
            }
        
        # Get risk report
        risk_report = await risk_manager.get_risk_report()
        
        # Get risk statistics
        risk_stats = risk_manager.get_statistics()
        
        # Get portfolio summary for risk calculation
        portfolio_summary = await position_manager.get_portfolio_summary() if position_manager else {}
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "risk_status": {
                "overall_risk_score": risk_stats.get('overall_risk_score', 0.0),
                "emergency_stop_active": risk_stats.get('emergency_stop_active', False),
                "total_alerts": risk_stats.get('total_alerts', 0),
                "risk_level": "low" if risk_stats.get('overall_risk_score', 0) < 0.3 else 
                             "medium" if risk_stats.get('overall_risk_score', 0) < 0.7 else "high"
            },
            "portfolio_risk": {
                "total_exposure": sum(
                    breakdown.get('volume', 0) * 1.1 
                    for breakdown in portfolio_summary.get('symbol_breakdown', {}).values()
                ),
                "position_count": portfolio_summary.get('total_open_positions', 0),
                "unrealized_pnl": portfolio_summary.get('total_unrealized_pnl', 0.0),
                "concentration_risk": len(portfolio_summary.get('symbol_breakdown', {}))
            },
            "limits": {
                "max_positions": risk_manager.config.get('max_positions', 0),
                "max_risk_per_trade": risk_manager.config.get('max_risk_per_trade', 0.02),
                "max_portfolio_risk": risk_manager.config.get('max_portfolio_risk', 0.10),
                "max_drawdown_limit": risk_manager.config.get('max_drawdown_limit', 0.20)
            },
            "detailed_metrics": risk_report.get('risk_metrics', {}) if risk_report else {}
        }
        
        main_logger.info("Risk status retrieved successfully")
        return result
        
    except Exception as e:
        main_logger.error(f"Error getting risk status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market-data")
async def get_market_data(symbols: Optional[str] = None) -> Dict[str, Any]:
    """
    Get current market data for trading symbols
    
    Args:
        symbols: Comma-separated list of symbols (optional)
        
    Returns:
        Current market prices and data
    """
    try:
        mt5_connection = get_mt5_connection()
        data_fetcher = get_data_fetcher()
        
        # Use provided symbols or default trading symbols
        if symbols:
            symbol_list = [s.strip() for s in symbols.split(',')]
        else:
            settings = get_settings_cached()
            symbol_list = settings.trading_symbols
        
        # Get live prices
        market_prices = {}
        for symbol in symbol_list:
            try:
                prices = await data_fetcher.get_live_prices([symbol])
                if prices:
                    market_prices[symbol] = prices[0]
            except Exception as e:
                main_logger.warning(f"Failed to get price for {symbol}: {e}")
                # Use mock data if real data fails
                market_prices[symbol] = {
                    "symbol": symbol,
                    "bid": 1.1000 if 'USD' in symbol else 150.0 if 'JPY' in symbol else 1.3000,
                    "ask": 1.1005 if 'USD' in symbol else 150.05 if 'JPY' in symbol else 1.3005,
                    "timestamp": datetime.now().isoformat(),
                    "source": "mock"
                }
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "market_data": market_prices,
            "symbols_count": len(market_prices),
            "connection_status": "connected" if await mt5_connection.health_check() else "disconnected"
        }
        
        main_logger.info(f"Retrieved market data for {len(market_prices)} symbols")
        return result
        
    except Exception as e:
        main_logger.error(f"Error getting market data: {e}")
        raise HTTPException(status_code=500, detail=str(e))