"""
WebSocket Integration with Trading System
Phase 7.2 Implementation - Real-time trading notifications
"""

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from app.utils.logger import analysis_logger


class WebSocketIntegration:
    """Integration layer between trading system and WebSocket"""
    
    def __init__(self):
        self.websocket_module = None
        self.enabled = False
        
    def initialize(self):
        """Initialize WebSocket integration"""
        try:
            # Import WebSocket module dynamically to avoid circular imports
            from app.api.websockets import (
                broadcast_signal, broadcast_trading_event,
                broadcast_position_update, broadcast_order_update,
                broadcast_risk_alert, broadcast_system_status
            )
            
            self.websocket_module = {
                'broadcast_signal': broadcast_signal,
                'broadcast_trading_event': broadcast_trading_event,
                'broadcast_position_update': broadcast_position_update,
                'broadcast_order_update': broadcast_order_update,
                'broadcast_risk_alert': broadcast_risk_alert,
                'broadcast_system_status': broadcast_system_status
            }
            
            self.enabled = True
            analysis_logger.info("WebSocket integration initialized")
            
        except Exception as e:
            analysis_logger.error(f"Failed to initialize WebSocket integration: {e}")
            self.enabled = False
    
    async def broadcast_signal(self, signal_data: Dict[str, Any]):
        """Broadcast trading signal via WebSocket"""
        if not self.enabled or not self.websocket_module:
            return
        
        try:
            await self.websocket_module['broadcast_signal'](signal_data)
        except Exception as e:
            analysis_logger.error(f"Error broadcasting signal: {e}")
    
    async def broadcast_order_execution(self, order_data: Dict[str, Any]):
        """Broadcast order execution event"""
        if not self.enabled or not self.websocket_module:
            return
        
        try:
            await self.websocket_module['broadcast_trading_event']("order_execution", order_data)
        except Exception as e:
            analysis_logger.error(f"Error broadcasting order execution: {e}")
    
    async def broadcast_position_opened(self, position_data: Dict[str, Any]):
        """Broadcast position opened event"""
        if not self.enabled or not self.websocket_module:
            return
        
        try:
            await self.websocket_module['broadcast_trading_event']("position_opened", position_data)
            await self.websocket_module['broadcast_position_update'](position_data)
        except Exception as e:
            analysis_logger.error(f"Error broadcasting position opened: {e}")
    
    async def broadcast_position_closed(self, position_data: Dict[str, Any]):
        """Broadcast position closed event"""
        if not self.enabled or not self.websocket_module:
            return
        
        try:
            await self.websocket_module['broadcast_trading_event']("position_closed", position_data)
            await self.websocket_module['broadcast_position_update'](position_data)
        except Exception as e:
            analysis_logger.error(f"Error broadcasting position closed: {e}")
    
    async def broadcast_position_modified(self, position_data: Dict[str, Any]):
        """Broadcast position modified event"""
        if not self.enabled or not self.websocket_module:
            return
        
        try:
            await self.websocket_module['broadcast_trading_event']("position_modified", position_data)
            await self.websocket_module['broadcast_position_update'](position_data)
        except Exception as e:
            analysis_logger.error(f"Error broadcasting position modified: {e}")
    
    async def broadcast_order_placed(self, order_data: Dict[str, Any]):
        """Broadcast order placed event"""
        if not self.enabled or not self.websocket_module:
            return
        
        try:
            await self.websocket_module['broadcast_trading_event']("order_placed", order_data)
            await self.websocket_module['broadcast_order_update'](order_data)
        except Exception as e:
            analysis_logger.error(f"Error broadcasting order placed: {e}")
    
    async def broadcast_order_cancelled(self, order_data: Dict[str, Any]):
        """Broadcast order cancelled event"""
        if not self.enabled or not self.websocket_module:
            return
        
        try:
            await self.websocket_module['broadcast_trading_event']("order_cancelled", order_data)
            await self.websocket_module['broadcast_order_update'](order_data)
        except Exception as e:
            analysis_logger.error(f"Error broadcasting order cancelled: {e}")
    
    async def broadcast_risk_alert(self, alert_type: str, alert_data: Dict[str, Any]):
        """Broadcast risk management alert"""
        if not self.enabled or not self.websocket_module:
            return
        
        try:
            risk_alert = {
                "alert_type": alert_type,
                "data": alert_data,
                "severity": self._determine_alert_severity(alert_type),
                "timestamp": datetime.now().isoformat()
            }
            await self.websocket_module['broadcast_risk_alert'](risk_alert)
        except Exception as e:
            analysis_logger.error(f"Error broadcasting risk alert: {e}")
    
    async def broadcast_emergency_stop(self, reason: str):
        """Broadcast emergency stop alert"""
        if not self.enabled or not self.websocket_module:
            return
        
        try:
            emergency_data = {
                "event_type": "emergency_stop",
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
                "severity": "critical"
            }
            await self.websocket_module['broadcast_risk_alert'](emergency_data)
        except Exception as e:
            analysis_logger.error(f"Error broadcasting emergency stop: {e}")
    
    async def broadcast_system_status(self, status_data: Dict[str, Any]):
        """Broadcast system status update"""
        if not self.enabled or not self.websocket_module:
            return
        
        try:
            await self.websocket_module['broadcast_system_status'](status_data)
        except Exception as e:
            analysis_logger.error(f"Error broadcasting system status: {e}")
    
    async def broadcast_session_event(self, event_type: str, session_data: Dict[str, Any]):
        """Broadcast trading session event"""
        if not self.enabled or not self.websocket_module:
            return
        
        try:
            await self.websocket_module['broadcast_trading_event'](f"session_{event_type}", session_data)
        except Exception as e:
            analysis_logger.error(f"Error broadcasting session event: {e}")
    
    def _determine_alert_severity(self, alert_type: str) -> str:
        """Determine alert severity based on type"""
        critical_alerts = ["emergency_stop", "max_drawdown_exceeded", "connection_lost"]
        high_alerts = ["position_limit_exceeded", "risk_limit_exceeded", "margin_call"]
        
        if alert_type in critical_alerts:
            return "critical"
        elif alert_type in high_alerts:
            return "high"
        else:
            return "medium"


# Global WebSocket integration instance
websocket_integration = WebSocketIntegration()


# Utility functions for easy integration
async def notify_signal_generated(signal_data: Dict[str, Any]):
    """Notify WebSocket subscribers of new signal"""
    await websocket_integration.broadcast_signal(signal_data)


async def notify_order_execution(order_data: Dict[str, Any]):
    """Notify WebSocket subscribers of order execution"""
    await websocket_integration.broadcast_order_execution(order_data)


async def notify_position_event(event_type: str, position_data: Dict[str, Any]):
    """Notify WebSocket subscribers of position events"""
    if event_type == "opened":
        await websocket_integration.broadcast_position_opened(position_data)
    elif event_type == "closed":
        await websocket_integration.broadcast_position_closed(position_data)
    elif event_type == "modified":
        await websocket_integration.broadcast_position_modified(position_data)


async def notify_order_event(event_type: str, order_data: Dict[str, Any]):
    """Notify WebSocket subscribers of order events"""
    if event_type == "placed":
        await websocket_integration.broadcast_order_placed(order_data)
    elif event_type == "cancelled":
        await websocket_integration.broadcast_order_cancelled(order_data)


async def notify_risk_alert(alert_type: str, alert_data: Dict[str, Any]):
    """Notify WebSocket subscribers of risk alerts"""
    await websocket_integration.broadcast_risk_alert(alert_type, alert_data)


async def notify_emergency_stop(reason: str):
    """Notify WebSocket subscribers of emergency stop"""
    await websocket_integration.broadcast_emergency_stop(reason)


async def notify_system_status(status_data: Dict[str, Any]):
    """Notify WebSocket subscribers of system status"""
    await websocket_integration.broadcast_system_status(status_data)


async def notify_session_event(event_type: str, session_data: Dict[str, Any]):
    """Notify WebSocket subscribers of session events"""
    await websocket_integration.broadcast_session_event(event_type, session_data)