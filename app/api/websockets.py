"""
WebSocket Implementation
Phase 7.2 Implementation - Real-time data streaming via WebSocket
"""

from fastapi import WebSocket, WebSocketDisconnect, Depends
from typing import Dict, List, Any, Set
import asyncio
import json
from datetime import datetime
import uuid
from enum import Enum

from app.dependencies import get_settings_cached
from app.config import Settings
from app.mt5.data_fetcher import get_data_fetcher
from app.trading.trading_engine import trading_engine
from app.trading.position_manager import position_manager
from app.trading.order_manager import order_manager
from app.trading.risk_manager import risk_manager
from app.utils.logger import main_logger


class SubscriptionType(Enum):
    """WebSocket subscription types"""
    MARKET_DATA = "market_data"
    SIGNALS = "signals"
    TRADING_EVENTS = "trading_events"
    SYSTEM_STATUS = "system_status"
    POSITIONS = "positions"
    ORDERS = "orders"
    RISK_ALERTS = "risk_alerts"


class ConnectionManager:
    """Manages WebSocket connections and subscriptions"""
    
    def __init__(self):
        # Active connections
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Subscriptions by type
        self.subscriptions: Dict[SubscriptionType, Set[str]] = {
            subscription_type: set() for subscription_type in SubscriptionType
        }
        
        # Symbol-specific subscriptions
        self.symbol_subscriptions: Dict[str, Set[str]] = {}
        
        # Connection metadata
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Background tasks
        self.background_tasks: Dict[str, asyncio.Task] = {}
        
        main_logger.info("WebSocket Connection Manager initialized")
    
    async def connect(self, websocket: WebSocket, client_id: str = None) -> str:
        """Accept new WebSocket connection"""
        await websocket.accept()
        
        # Generate client ID if not provided
        if not client_id:
            client_id = str(uuid.uuid4())
        
        # Store connection
        self.active_connections[client_id] = websocket
        self.connection_metadata[client_id] = {
            "connected_at": datetime.now(),
            "subscriptions": [],
            "last_activity": datetime.now()
        }
        
        main_logger.info(f"WebSocket client connected: {client_id}")
        return client_id
    
    def disconnect(self, client_id: str):
        """Remove WebSocket connection"""
        if client_id in self.active_connections:
            # Remove from all subscriptions
            for subscription_type in SubscriptionType:
                self.subscriptions[subscription_type].discard(client_id)
            
            # Remove from symbol subscriptions
            for symbol_set in self.symbol_subscriptions.values():
                symbol_set.discard(client_id)
            
            # Cancel background tasks
            if client_id in self.background_tasks:
                self.background_tasks[client_id].cancel()
                del self.background_tasks[client_id]
            
            # Remove connection
            del self.active_connections[client_id]
            del self.connection_metadata[client_id]
            
            main_logger.info(f"WebSocket client disconnected: {client_id}")
    
    async def subscribe(self, client_id: str, subscription_type: SubscriptionType, symbols: List[str] = None):
        """Subscribe client to specific data type"""
        if client_id not in self.active_connections:
            return False
        
        # Add to general subscription
        self.subscriptions[subscription_type].add(client_id)
        
        # Add to symbol-specific subscriptions
        if symbols and subscription_type == SubscriptionType.MARKET_DATA:
            for symbol in symbols:
                if symbol not in self.symbol_subscriptions:
                    self.symbol_subscriptions[symbol] = set()
                self.symbol_subscriptions[symbol].add(client_id)
        
        # Update metadata
        self.connection_metadata[client_id]["subscriptions"].append({
            "type": subscription_type.value,
            "symbols": symbols or [],
            "subscribed_at": datetime.now()
        })
        
        main_logger.info(f"Client {client_id} subscribed to {subscription_type.value}")
        return True
    
    async def unsubscribe(self, client_id: str, subscription_type: SubscriptionType, symbols: List[str] = None):
        """Unsubscribe client from specific data type"""
        if client_id not in self.active_connections:
            return False
        
        # Remove from general subscription
        self.subscriptions[subscription_type].discard(client_id)
        
        # Remove from symbol-specific subscriptions
        if symbols and subscription_type == SubscriptionType.MARKET_DATA:
            for symbol in symbols:
                if symbol in self.symbol_subscriptions:
                    self.symbol_subscriptions[symbol].discard(client_id)
        
        main_logger.info(f"Client {client_id} unsubscribed from {subscription_type.value}")
        return True
    
    async def send_personal_message(self, message: dict, client_id: str):
        """Send message to specific client"""
        if client_id in self.active_connections:
            try:
                websocket = self.active_connections[client_id]
                await websocket.send_text(json.dumps(message))
                self.connection_metadata[client_id]["last_activity"] = datetime.now()
            except Exception as e:
                main_logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast_to_subscribers(self, message: dict, subscription_type: SubscriptionType, symbols: List[str] = None):
        """Broadcast message to all subscribers of a type"""
        
        # Determine target clients
        target_clients = set()
        
        if symbols and subscription_type == SubscriptionType.MARKET_DATA:
            # For market data, send to symbol-specific subscribers
            for symbol in symbols:
                if symbol in self.symbol_subscriptions:
                    target_clients.update(self.symbol_subscriptions[symbol])
        else:
            # For other types, send to all subscribers
            target_clients = self.subscriptions[subscription_type].copy()
        
        # Send to all target clients
        disconnected_clients = []
        for client_id in target_clients:
            try:
                if client_id in self.active_connections:
                    await self.send_personal_message(message, client_id)
            except Exception as e:
                main_logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        total_connections = len(self.active_connections)
        subscription_stats = {
            sub_type.value: len(subscribers) 
            for sub_type, subscribers in self.subscriptions.items()
        }
        
        return {
            "total_connections": total_connections,
            "subscription_stats": subscription_stats,
            "symbol_subscriptions": {
                symbol: len(subscribers) 
                for symbol, subscribers in self.symbol_subscriptions.items()
            },
            "timestamp": datetime.now().isoformat()
        }


# Global connection manager
manager = ConnectionManager()


# WebSocket Data Streaming Classes
class MarketDataStreamer:
    """Streams real-time market data"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.manager = connection_manager
        self.data_fetcher = get_data_fetcher()
        self.streaming_task = None
        self.streaming_interval = 1.0  # seconds
        
    async def start_streaming(self):
        """Start market data streaming"""
        if self.streaming_task and not self.streaming_task.done():
            return
        
        self.streaming_task = asyncio.create_task(self._stream_market_data())
        main_logger.info("Market data streaming started")
    
    async def stop_streaming(self):
        """Stop market data streaming"""
        if self.streaming_task and not self.streaming_task.done():
            self.streaming_task.cancel()
        main_logger.info("Market data streaming stopped")
    
    async def _stream_market_data(self):
        """Background task for streaming market data"""
        try:
            while True:
                # Get all symbols with active subscriptions
                active_symbols = list(self.manager.symbol_subscriptions.keys())
                
                if active_symbols:
                    try:
                        # Fetch current prices for all active symbols
                        for symbol in active_symbols:
                            if symbol in self.manager.symbol_subscriptions and self.manager.symbol_subscriptions[symbol]:
                                price_data = await self._get_current_price(symbol)
                                if price_data:
                                    message = {
                                        "type": "market_data",
                                        "symbol": symbol,
                                        "data": price_data,
                                        "timestamp": datetime.now().isoformat()
                                    }
                                    await self.manager.broadcast_to_subscribers(
                                        message, SubscriptionType.MARKET_DATA, [symbol]
                                    )
                    except Exception as e:
                        main_logger.error(f"Error in market data streaming: {e}")
                
                await asyncio.sleep(self.streaming_interval)
                
        except asyncio.CancelledError:
            main_logger.info("Market data streaming cancelled")
    
    async def _get_current_price(self, symbol: str) -> Dict[str, Any]:
        """Get current price for symbol"""
        try:
            prices = await self.data_fetcher.get_live_prices([symbol])
            if prices:
                return prices[0]
            else:
                # Return mock data if real data unavailable
                base_price = 1.1000 if 'USD' in symbol else 150.0 if 'JPY' in symbol else 1.3000
                spread = 0.0001 if 'USD' in symbol else 0.01 if 'JPY' in symbol else 0.0002
                return {
                    "symbol": symbol,
                    "bid": base_price - spread / 2,
                    "ask": base_price + spread / 2,
                    "time": datetime.now().isoformat(),
                    "source": "mock"
                }
        except Exception as e:
            main_logger.error(f"Error getting price for {symbol}: {e}")
            return None


class SignalStreamer:
    """Streams trading signals"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.manager = connection_manager
        self.signal_queue = asyncio.Queue()
        
    async def add_signal(self, signal_data: Dict[str, Any]):
        """Add signal to streaming queue"""
        await self.signal_queue.put(signal_data)
    
    async def start_streaming(self):
        """Start signal streaming"""
        asyncio.create_task(self._stream_signals())
        main_logger.info("Signal streaming started")
    
    async def _stream_signals(self):
        """Background task for streaming signals"""
        try:
            while True:
                # Wait for signal
                signal_data = await self.signal_queue.get()
                
                # Broadcast signal to subscribers
                message = {
                    "type": "signal",
                    "data": signal_data,
                    "timestamp": datetime.now().isoformat()
                }
                
                await self.manager.broadcast_to_subscribers(
                    message, SubscriptionType.SIGNALS
                )
                
        except Exception as e:
            main_logger.error(f"Error in signal streaming: {e}")


class TradingEventStreamer:
    """Streams trading events (orders, positions, etc.)"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.manager = connection_manager
        self.event_queue = asyncio.Queue()
    
    async def add_trading_event(self, event_type: str, event_data: Dict[str, Any]):
        """Add trading event to streaming queue"""
        event = {
            "event_type": event_type,
            "data": event_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.event_queue.put(event)
    
    async def start_streaming(self):
        """Start trading event streaming"""
        asyncio.create_task(self._stream_trading_events())
        main_logger.info("Trading event streaming started")
    
    async def _stream_trading_events(self):
        """Background task for streaming trading events"""
        try:
            while True:
                # Wait for event
                event_data = await self.event_queue.get()
                
                # Broadcast event to subscribers
                message = {
                    "type": "trading_event",
                    "data": event_data,
                    "timestamp": datetime.now().isoformat()
                }
                
                await self.manager.broadcast_to_subscribers(
                    message, SubscriptionType.TRADING_EVENTS
                )
                
        except Exception as e:
            main_logger.error(f"Error in trading event streaming: {e}")


# Global streamers
market_data_streamer = MarketDataStreamer(manager)
signal_streamer = SignalStreamer(manager)
trading_event_streamer = TradingEventStreamer(manager)


# WebSocket endpoint handlers
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint"""
    client_id = None
    try:
        # Accept connection
        client_id = await manager.connect(websocket)
        
        # Start streamers if not already running
        await market_data_streamer.start_streaming()
        await signal_streamer.start_streaming()
        await trading_event_streamer.start_streaming()
        
        # Send welcome message
        welcome_message = {
            "type": "welcome",
            "client_id": client_id,
            "timestamp": datetime.now().isoformat(),
            "available_subscriptions": [sub_type.value for sub_type in SubscriptionType]
        }
        await manager.send_personal_message(welcome_message, client_id)
        
        # Listen for messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            await handle_websocket_message(client_id, message)
            
    except WebSocketDisconnect:
        if client_id:
            manager.disconnect(client_id)
    except Exception as e:
        main_logger.error(f"WebSocket error: {e}")
        if client_id:
            manager.disconnect(client_id)


async def handle_websocket_message(client_id: str, message: Dict[str, Any]):
    """Handle incoming WebSocket messages"""
    try:
        message_type = message.get("type")
        
        if message_type == "subscribe":
            subscription_type = SubscriptionType(message.get("subscription_type"))
            symbols = message.get("symbols", [])
            
            success = await manager.subscribe(client_id, subscription_type, symbols)
            
            response = {
                "type": "subscription_response",
                "subscription_type": subscription_type.value,
                "symbols": symbols,
                "success": success,
                "timestamp": datetime.now().isoformat()
            }
            await manager.send_personal_message(response, client_id)
            
        elif message_type == "unsubscribe":
            subscription_type = SubscriptionType(message.get("subscription_type"))
            symbols = message.get("symbols", [])
            
            success = await manager.unsubscribe(client_id, subscription_type, symbols)
            
            response = {
                "type": "unsubscription_response",
                "subscription_type": subscription_type.value,
                "symbols": symbols,
                "success": success,
                "timestamp": datetime.now().isoformat()
            }
            await manager.send_personal_message(response, client_id)
            
        elif message_type == "ping":
            pong_response = {
                "type": "pong",
                "timestamp": datetime.now().isoformat()
            }
            await manager.send_personal_message(pong_response, client_id)
            
        elif message_type == "get_status":
            status = manager.get_connection_stats()
            status_response = {
                "type": "status",
                "data": status,
                "timestamp": datetime.now().isoformat()
            }
            await manager.send_personal_message(status_response, client_id)
            
        else:
            error_response = {
                "type": "error",
                "message": f"Unknown message type: {message_type}",
                "timestamp": datetime.now().isoformat()
            }
            await manager.send_personal_message(error_response, client_id)
            
    except Exception as e:
        main_logger.error(f"Error handling WebSocket message: {e}")
        error_response = {
            "type": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        await manager.send_personal_message(error_response, client_id)


# Utility functions for integration with trading system
async def broadcast_signal(signal_data: Dict[str, Any]):
    """Broadcast trading signal to WebSocket subscribers"""
    await signal_streamer.add_signal(signal_data)


async def broadcast_trading_event(event_type: str, event_data: Dict[str, Any]):
    """Broadcast trading event to WebSocket subscribers"""
    await trading_event_streamer.add_trading_event(event_type, event_data)


async def broadcast_system_status(status_data: Dict[str, Any]):
    """Broadcast system status to WebSocket subscribers"""
    message = {
        "type": "system_status",
        "data": status_data,
        "timestamp": datetime.now().isoformat()
    }
    await manager.broadcast_to_subscribers(message, SubscriptionType.SYSTEM_STATUS)


async def broadcast_position_update(position_data: Dict[str, Any]):
    """Broadcast position update to WebSocket subscribers"""
    message = {
        "type": "position_update",
        "data": position_data,
        "timestamp": datetime.now().isoformat()
    }
    await manager.broadcast_to_subscribers(message, SubscriptionType.POSITIONS)


async def broadcast_order_update(order_data: Dict[str, Any]):
    """Broadcast order update to WebSocket subscribers"""
    message = {
        "type": "order_update",
        "data": order_data,
        "timestamp": datetime.now().isoformat()
    }
    await manager.broadcast_to_subscribers(message, SubscriptionType.ORDERS)


async def broadcast_risk_alert(alert_data: Dict[str, Any]):
    """Broadcast risk alert to WebSocket subscribers"""
    message = {
        "type": "risk_alert",
        "data": alert_data,
        "timestamp": datetime.now().isoformat()
    }
    await manager.broadcast_to_subscribers(message, SubscriptionType.RISK_ALERTS)