"""
Comprehensive tests for Trading REST API endpoints
Testing all trading functionality with proper async mocking
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
import json

from app.main import app

client = TestClient(app)


class TestTradingAPI:
    """Comprehensive Trading API test suite"""
    
    @patch('app.api.trading.notify_session_event')
    @patch('app.api.trading.trading_engine')
    def test_session_start_success(self, mock_trading_engine, mock_notify):
        """Test successful trading session start"""
        mock_trading_engine.start_trading_session = AsyncMock(return_value="session_001")
        mock_notify = AsyncMock()
        
        response = client.post("/api/trading/session/start", json={
            "mode": "demo"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["session_id"] == "session_001"
        assert data["mode"] == "demo"
    
    @patch('app.api.trading.trading_engine')
    def test_session_start_invalid_mode(self, mock_trading_engine):
        """Test trading session start with invalid mode"""
        response = client.post("/api/trading/session/start", json={
            "mode": "invalid_mode"
        })
        
        assert response.status_code == 400  # Validation error
    
    @patch('app.api.trading.trading_engine')
    def test_session_start_engine_failure(self, mock_trading_engine):
        """Test trading session start with engine failure"""
        mock_trading_engine.start_trading_session = AsyncMock(side_effect=Exception("Engine failed to start"))
        
        response = client.post("/api/trading/session/start", json={
            "mode": "demo"
        })
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
    
    @patch('app.api.trading.trading_engine')
    def test_session_pause_success(self, mock_trading_engine):
        """Test successful trading session pause"""
        mock_trading_engine.pause_trading = AsyncMock(return_value={
            "success": True,
            "session_id": "session_001",
            "paused_at": datetime.now().isoformat()
        })
        
        response = client.post("/api/trading/session/pause")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
    
    @patch('app.api.trading.trading_engine')
    def test_session_resume_success(self, mock_trading_engine):
        """Test successful trading session resume"""
        mock_trading_engine.resume_trading = AsyncMock(return_value={
            "success": True,
            "session_id": "session_001",
            "resumed_at": datetime.now().isoformat()
        })
        
        response = client.post("/api/trading/session/resume")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
    
    @patch('app.api.trading.trading_engine')
    def test_session_stop_success(self, mock_trading_engine):
        """Test successful trading session stop"""
        mock_trading_engine.stop_trading_session = AsyncMock(return_value=None)
        
        response = client.post("/api/trading/session/stop")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "stopped_at" in data
        assert "message" in data
    
    # Note: session/status endpoint doesn't exist in current API
    # These tests are removed as the endpoint is not implemented
    
    @patch('app.api.trading.trading_engine')
    def test_process_signal_buy_success(self, mock_trading_engine):
        """Test successful buy signal processing"""
        # API returns boolean success, not dict
        mock_trading_engine.process_signal = AsyncMock(return_value=True)
        
        signal_data = {
            "symbol": "EURUSD",
            "signal_type": "MARKET_BUY",
            "confidence": 0.85,
            "entry_price": 1.0850,
            "stop_loss": 1.0800,
            "take_profit": 1.0950,
            "urgency": "MEDIUM"
        }
        
        response = client.post("/api/trading/signals/process", json=signal_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "signal_id" in data
        assert "processed_at" in data
        assert "message" in data
    
    @patch('app.api.trading.trading_engine')
    def test_process_signal_sell_success(self, mock_trading_engine):
        """Test successful sell signal processing"""
        # API returns boolean success, not dict
        mock_trading_engine.process_signal = AsyncMock(return_value=True)
        
        signal_data = {
            "symbol": "USDJPY",
            "signal_type": "MARKET_SELL",
            "confidence": 0.75,
            "entry_price": 149.50,
            "stop_loss": 150.00,
            "take_profit": 148.50,
            "urgency": "HIGH"
        }
        
        response = client.post("/api/trading/signals/process", json=signal_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "signal_id" in data
    
    def test_process_signal_invalid_data(self):
        """Test signal processing with invalid data"""
        invalid_signal = {
            "symbol": "INVALID",
            "signal_type": "INVALID_TYPE",
            "confidence": 2.0  # Invalid confidence > 1.0
        }
        
        response = client.post("/api/trading/signals/process", json=invalid_signal)
        
        assert response.status_code == 422  # Validation error
    
    @patch('app.api.trading.order_manager')
    def test_create_order_market_buy(self, mock_order_manager):
        """Test market buy order creation"""
        from unittest.mock import MagicMock
        from datetime import datetime
        
        # Create proper execution object mock
        mock_execution = MagicMock()
        mock_execution.success = True
        mock_execution.order_id = "ord_001"
        mock_execution.mt5_ticket = 12345
        mock_execution.executed_volume = 0.1
        mock_execution.execution_price = 1.0851
        mock_execution.commission = 2.5
        mock_execution.error_message = None
        mock_execution.execution_time = datetime.now()
        
        mock_order_manager.place_order = AsyncMock(return_value=mock_execution)
        
        order_data = {
            "symbol": "EURUSD",
            "order_type": "MARKET_BUY",
            "volume": 0.1,
            "stop_loss": 1.0800,
            "take_profit": 1.0950
        }
        
        response = client.post("/api/trading/orders/create", json=order_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["order_id"] == "ord_001"
        assert data["execution_price"] == 1.0851
        assert data["executed_volume"] == 0.1
    
    @patch('app.api.trading.order_manager')
    def test_create_order_limit_sell(self, mock_order_manager):
        """Test limit sell order creation"""
        from unittest.mock import MagicMock
        from datetime import datetime
        
        # Create proper execution object mock
        mock_execution = MagicMock()
        mock_execution.success = True
        mock_execution.order_id = "ord_002"
        mock_execution.mt5_ticket = 12346
        mock_execution.executed_volume = 0.05
        mock_execution.execution_price = 150.00
        mock_execution.commission = 1.25
        mock_execution.error_message = None
        mock_execution.execution_time = datetime.now()
        
        mock_order_manager.place_order = AsyncMock(return_value=mock_execution)
        
        order_data = {
            "symbol": "USDJPY",
            "order_type": "SELL_LIMIT",
            "volume": 0.05,
            "price": 150.00,
            "stop_loss": 150.50,
            "take_profit": 149.00
        }
        
        response = client.post("/api/trading/orders/create", json=order_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["order_id"] == "ord_002"
        assert data["execution_price"] == 150.00
        assert data["executed_volume"] == 0.05
    
    @patch('app.api.trading.order_manager')
    def test_cancel_order_success(self, mock_order_manager):
        """Test successful order cancellation"""
        # API returns boolean success, not dict
        mock_order_manager.cancel_order = AsyncMock(return_value=True)
        
        response = client.delete("/api/trading/orders/ord_001")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["order_id"] == "ord_001"
    
    @patch('app.api.trading.order_manager')
    def test_cancel_order_not_found(self, mock_order_manager):
        """Test order cancellation when order not found"""
        mock_order_manager.cancel_order = AsyncMock(side_effect=Exception("Order not found"))
        
        response = client.delete("/api/trading/orders/nonexistent")
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
    
    @patch('app.api.trading.position_manager')
    def test_modify_position_success(self, mock_position_manager):
        """Test successful position modification"""
        # API returns boolean success, not dict
        mock_position_manager.modify_position = AsyncMock(return_value=True)
        
        modification_data = {
            "position_id": "pos_001",
            "modification_type": "both",
            "new_stop_loss": 1.0820,
            "new_take_profit": 1.0970,
            "reason": "Trailing stop adjustment"
        }
        
        response = client.put("/api/trading/positions/pos_001/modify", json=modification_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["position_id"] == "pos_001"
    
    @patch('app.api.trading.position_manager')
    def test_close_position_success(self, mock_position_manager):
        """Test successful position closure"""
        from unittest.mock import MagicMock
        from datetime import datetime
        
        # Create proper close result object mock
        mock_close_result = MagicMock()
        mock_close_result.success = True
        mock_close_result.closed_volume = 0.1
        mock_close_result.close_price = 1.0875
        mock_close_result.realized_pnl = 25.0
        mock_close_result.commission = 2.5
        mock_close_result.error_message = None
        mock_close_result.close_time = datetime.now()
        
        mock_position_manager.close_position = AsyncMock(return_value=mock_close_result)
        
        close_data = {
            "position_id": "pos_001",
            "reason": "Take profit hit"
        }
        
        response = client.post("/api/trading/positions/pos_001/close", json=close_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["realized_pnl"] == 25.0
    
    @patch('app.api.trading.position_manager')
    def test_close_all_positions_success(self, mock_position_manager):
        """Test successful closure of all positions"""
        from unittest.mock import MagicMock
        from datetime import datetime
        
        # Create proper close result objects
        mock_result1 = MagicMock()
        mock_result1.success = True
        mock_result1.position_id = "pos_001"
        mock_result1.realized_pnl = 25.0
        mock_result1.error_message = None
        
        mock_result2 = MagicMock()
        mock_result2.success = True
        mock_result2.position_id = "pos_002"
        mock_result2.realized_pnl = 30.0
        mock_result2.error_message = None
        
        mock_result3 = MagicMock()
        mock_result3.success = True
        mock_result3.position_id = "pos_003"
        mock_result3.realized_pnl = 20.50
        mock_result3.error_message = None
        
        mock_position_manager.close_all_positions = AsyncMock(return_value=[mock_result1, mock_result2, mock_result3])
        
        response = client.post("/api/trading/positions/close-all")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["successful_closes"] == 3
        assert data["total_realized_pnl"] == 75.50
    
    @patch('app.api.trading.risk_manager')
    def test_emergency_stop_activation(self, mock_risk_manager):
        """Test emergency stop activation"""
        # API returns None for emergency_stop, not dict
        mock_risk_manager.emergency_stop = AsyncMock(return_value=None)
        
        emergency_data = {
            "reason": "High drawdown detected"
        }
        
        response = client.post("/api/trading/risk/emergency-stop", json=emergency_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["reason"] == "High drawdown detected"
    
    @patch('app.api.trading.risk_manager')
    def test_emergency_stop_deactivation(self, mock_risk_manager):
        """Test emergency stop deactivation"""
        # API returns None for deactivate_emergency_stop, not dict
        mock_risk_manager.deactivate_emergency_stop = AsyncMock(return_value=None)
        
        response = client.post("/api/trading/risk/emergency-stop/deactivate")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
    
    # Note: /api/trading/risk/status endpoint doesn't exist in current API
    # This test is commented out as the endpoint is not implemented
    # @patch('app.api.trading.risk_manager')
    # def test_risk_status_normal(self, mock_risk_manager):
    
    # Note: /api/trading/risk/status endpoint doesn't exist in current API
    # This test is commented out as the endpoint is not implemented
    # @patch('app.api.trading.risk_manager')
    # def test_risk_status_high_risk(self, mock_risk_manager):
    
    # Note: /api/trading/health endpoint doesn't exist in current API
    # This test is commented out as the endpoint is not implemented
    # def test_trading_api_health_check(self):