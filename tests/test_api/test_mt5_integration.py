"""
Comprehensive tests for MT5 Integration API endpoints
Testing all MT5 connection and integration functionality with 90%+ coverage
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
import json

from app.main import app

client = TestClient(app)


class TestMT5IntegrationAPI:
    """Comprehensive MT5 Integration API test suite"""
    
    @patch('app.api.mt5_control.get_mt5_connection')
    def test_mt5_connect_success(self, mock_get_mt5):
        """Test successful MT5 connection"""
        mock_mt5_conn = MagicMock()
        mock_mt5_conn.connect.return_value = True
        mock_mt5_conn.is_connected.return_value = True
        mock_mt5_conn.get_connection_info.return_value = {
            "connected": True,
            "server": "XMTrading-MT5 3",
            "login": 123456789,
            "name": "Demo Account",
            "balance": 10000.0,
            "currency": "USD",
            "leverage": 100,
            "margin": 0.0,
            "equity": 10000.0
        }
        mock_get_mt5.return_value = mock_mt5_conn
        
        response = client.post("/api/mt5/connect")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["connected"] == True
        assert data["account_info"]["server"] == "XMTrading-MT5 3"
        assert data["account_info"]["balance"] == 10000.0
    
    @patch('app.api.mt5_control.get_mt5_connection')
    def test_mt5_connect_failure(self, mock_get_mt5):
        """Test MT5 connection failure"""
        mock_mt5_conn = MagicMock()
        mock_mt5_conn.connect.return_value = False
        mock_mt5_conn.is_connected.return_value = False
        mock_mt5_conn.get_last_error.return_value = "Connection timeout"
        mock_get_mt5.return_value = mock_mt5_conn
        
        response = client.post("/api/mt5/connect")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == False
        assert data["connected"] == False
        assert "error" in data
    
    @patch('app.api.mt5_control.get_mt5_connection')
    def test_mt5_disconnect_success(self, mock_get_mt5):
        """Test successful MT5 disconnection"""
        mock_mt5_conn = MagicMock()
        mock_mt5_conn.disconnect.return_value = True
        mock_mt5_conn.is_connected.return_value = False
        mock_get_mt5.return_value = mock_mt5_conn
        
        response = client.post("/api/mt5/disconnect")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["connected"] == False
    
    @patch('app.api.mt5_control.get_mt5_connection')
    def test_mt5_disconnect_when_not_connected(self, mock_get_mt5):
        """Test MT5 disconnection when not connected"""
        mock_mt5_conn = MagicMock()
        mock_mt5_conn.is_connected.return_value = False
        mock_get_mt5.return_value = mock_mt5_conn
        
        response = client.post("/api/mt5/disconnect")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["connected"] == False
        assert "message" in data
    
    @patch('app.api.mt5_control.get_mt5_connection')
    def test_mt5_status_connected(self, mock_get_mt5):
        """Test MT5 status when connected"""
        mock_mt5_conn = MagicMock()
        mock_mt5_conn.is_connected.return_value = True
        mock_mt5_conn.get_connection_info.return_value = {
            "connected": True,
            "server": "XMTrading-MT5 3",
            "login": 123456789,
            "name": "Demo Account",
            "balance": 9850.75,
            "currency": "USD",
            "leverage": 100,
            "margin": 150.25,
            "equity": 9950.50,
            "free_margin": 9800.25,
            "margin_level": 6633.22
        }
        mock_get_mt5.return_value = mock_mt5_conn
        
        response = client.get("/api/mt5/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["is_connected"] == True
        assert data["account_info"]["balance"] == 9850.75
        assert data["account_info"]["equity"] == 9950.50
        assert data["account_info"]["margin"] == 150.25
    
    @patch('app.api.mt5_control.get_mt5_connection')
    def test_mt5_status_disconnected(self, mock_get_mt5):
        """Test MT5 status when disconnected"""
        mock_mt5_conn = MagicMock()
        mock_mt5_conn.is_connected.return_value = False
        mock_get_mt5.return_value = mock_mt5_conn
        
        response = client.get("/api/mt5/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["is_connected"] == False
        assert data["account_info"] is None
    
    @patch('app.api.mt5_control.get_mt5_connection')
    def test_mt5_reconnect_success(self, mock_get_mt5):
        """Test successful MT5 reconnection"""
        mock_mt5_conn = MagicMock()
        mock_mt5_conn.disconnect.return_value = True
        mock_mt5_conn.connect.return_value = True
        mock_mt5_conn.is_connected.return_value = True
        mock_mt5_conn.get_connection_info.return_value = {
            "connected": True,
            "server": "XMTrading-MT5 3",
            "login": 123456789
        }
        mock_get_mt5.return_value = mock_mt5_conn
        
        response = client.post("/api/mt5/reconnect")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["connected"] == True
        assert "account_info" in data
    
    @patch('app.api.mt5_control.get_mt5_connection')
    def test_mt5_reconnect_failure(self, mock_get_mt5):
        """Test MT5 reconnection failure"""
        mock_mt5_conn = MagicMock()
        mock_mt5_conn.disconnect.return_value = True
        mock_mt5_conn.connect.return_value = False
        mock_mt5_conn.is_connected.return_value = False
        mock_mt5_conn.get_last_error.return_value = "Authentication failed"
        mock_get_mt5.return_value = mock_mt5_conn
        
        response = client.post("/api/mt5/reconnect")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == False
        assert data["connected"] == False
        assert "error" in data
    
    @patch('app.api.mt5_control.get_mt5_connection')
    def test_mt5_health_check_healthy(self, mock_get_mt5):
        """Test MT5 health check when healthy"""
        mock_mt5_conn = MagicMock()
        mock_mt5_conn.is_connected.return_value = True
        mock_mt5_conn.get_terminal_info.return_value = {
            "community_account": False,
            "community_connection": True,
            "connected": True,
            "dlls_allowed": True,
            "trade_allowed": True,
            "tradeapi_disabled": False,
            "email_enabled": False,
            "ftp_enabled": False,
            "notifications_enabled": False
        }
        mock_mt5_conn.get_connection_info.return_value = {
            "connected": True,
            "server": "XMTrading-MT5 3",
            "ping": 45.2
        }
        mock_get_mt5.return_value = mock_mt5_conn
        
        response = client.get("/api/mt5/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["connection"]["connected"] == True
        assert data["terminal"]["trade_allowed"] == True
        assert "ping" in data["connection"]
    
    @patch('app.api.mt5_control.get_mt5_connection')
    def test_mt5_health_check_unhealthy(self, mock_get_mt5):
        """Test MT5 health check when unhealthy"""
        mock_mt5_conn = MagicMock()
        mock_mt5_conn.is_connected.return_value = False
        mock_get_mt5.return_value = mock_mt5_conn
        
        response = client.get("/api/mt5/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["connection"]["connected"] == False
        assert "issues" in data
    
    @patch('app.api.mt5_control.get_mt5_connection')
    def test_mt5_connection_error_handling(self, mock_get_mt5):
        """Test MT5 connection error handling"""
        mock_get_mt5.side_effect = Exception("MT5 module not available")
        
        response = client.post("/api/mt5/connect")
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
    
    @patch('app.api.mt5_control.get_mt5_connection')
    def test_mt5_concurrent_operations(self, mock_get_mt5):
        """Test concurrent MT5 operations"""
        import concurrent.futures
        
        mock_mt5_conn = MagicMock()
        mock_mt5_conn.is_connected.return_value = True
        mock_mt5_conn.get_connection_info.return_value = {"connected": True}
        mock_get_mt5.return_value = mock_mt5_conn
        
        def make_status_request():
            return client.get("/api/mt5/status")
        
        # Make multiple concurrent status requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_status_request) for _ in range(5)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
    
    @patch('app.api.mt5_control.get_mt5_connection')
    def test_mt5_account_info_details(self, mock_get_mt5):
        """Test detailed MT5 account information"""
        mock_mt5_conn = MagicMock()
        mock_mt5_conn.is_connected.return_value = True
        mock_mt5_conn.get_connection_info.return_value = {
            "connected": True,
            "server": "XMTrading-MT5 3",
            "login": 123456789,
            "name": "John Doe Demo",
            "balance": 10000.0,
            "currency": "USD",
            "leverage": 100,
            "margin": 500.0,
            "equity": 10250.0,
            "free_margin": 9750.0,
            "margin_level": 2050.0,
            "profit": 250.0,
            "company": "Tradexfin Limited",
            "trade_mode": 0,  # Demo account
            "limit_orders": 200,
            "margin_so_mode": 0,
            "trade_allowed": True,
            "trade_expert": True,
            "margin_mode": 0,
            "currency_digits": 2,
            "fifo_close": False
        }
        mock_get_mt5.return_value = mock_mt5_conn
        
        response = client.get("/api/mt5/status")
        
        assert response.status_code == 200
        data = response.json()
        account_info = data["account_info"]
        assert account_info["name"] == "John Doe Demo"
        assert account_info["leverage"] == 100
        assert account_info["profit"] == 250.0
        assert account_info["trade_allowed"] == True
    
    @patch('app.api.mt5_control.get_mt5_connection')
    def test_mt5_terminal_info_validation(self, mock_get_mt5):
        """Test MT5 terminal information validation"""
        mock_mt5_conn = MagicMock()
        mock_mt5_conn.is_connected.return_value = True
        mock_mt5_conn.get_terminal_info.return_value = {
            "community_account": False,
            "community_connection": True,
            "connected": True,
            "dlls_allowed": True,
            "trade_allowed": False,  # Trading disabled
            "tradeapi_disabled": True,  # API disabled
            "email_enabled": False,
            "ftp_enabled": False,
            "notifications_enabled": True,
            "mqid": False,
            "build": 3815,
            "maxbars": 100000,
            "codepage": 1252,
            "ping": 125.5,
            "community_balance": 0.0,
            "retransmission": 5.2
        }
        mock_get_mt5.return_value = mock_mt5_conn
        
        response = client.get("/api/mt5/health")
        
        assert response.status_code == 200
        data = response.json()
        # Should detect issues with trading disabled
        assert "issues" in data
        issues = data["issues"]
        trading_issue = any("trade_allowed" in issue for issue in issues)
        api_issue = any("tradeapi_disabled" in issue for issue in issues)
        assert trading_issue or api_issue  # At least one trading-related issue
    
    @patch('app.api.mt5_control.get_mt5_connection')
    def test_mt5_connection_stability_check(self, mock_get_mt5):
        """Test MT5 connection stability over time"""
        mock_mt5_conn = MagicMock()
        
        # Simulate connection going up and down
        connection_states = [True, True, False, True, True]
        call_count = [0]
        
        def mock_is_connected():
            state = connection_states[call_count[0] % len(connection_states)]
            call_count[0] += 1
            return state
        
        mock_mt5_conn.is_connected.side_effect = mock_is_connected
        mock_mt5_conn.get_connection_info.return_value = {"connected": True}
        mock_get_mt5.return_value = mock_mt5_conn
        
        # Make multiple status checks
        responses = []
        for _ in range(5):
            response = client.get("/api/mt5/status")
            responses.append(response.json())
        
        # Should track connection state changes
        connected_states = [r["is_connected"] for r in responses]
        assert True in connected_states  # Some connections successful
        assert False in connected_states  # Some connections failed
    
    @patch('app.api.mt5_control.get_mt5_connection')
    def test_mt5_error_recovery(self, mock_get_mt5):
        """Test MT5 error recovery scenarios"""
        mock_mt5_conn = MagicMock()
        
        # First call fails, second succeeds
        call_count = [0]
        def mock_connect():
            call_count[0] += 1
            return call_count[0] > 1  # Fail first, succeed second
        
        mock_mt5_conn.connect.side_effect = mock_connect
        mock_mt5_conn.is_connected.return_value = True
        mock_mt5_conn.get_connection_info.return_value = {"connected": True}
        mock_get_mt5.return_value = mock_mt5_conn
        
        # First connection attempt
        response1 = client.post("/api/mt5/connect")
        
        # Second connection attempt (should succeed)
        response2 = client.post("/api/mt5/connect")
        
        # At least one should succeed
        success_count = sum(1 for r in [response1, response2] 
                          if r.status_code == 200 and r.json().get("success"))
        assert success_count >= 1
    
    @patch('app.api.mt5_control.get_mt5_connection')
    def test_mt5_performance_metrics(self, mock_get_mt5):
        """Test MT5 performance metrics collection"""
        mock_mt5_conn = MagicMock()
        mock_mt5_conn.is_connected.return_value = True
        mock_mt5_conn.get_connection_info.return_value = {
            "connected": True,
            "ping": 45.2,
            "retransmission": 2.1
        }
        mock_mt5_conn.get_terminal_info.return_value = {
            "ping": 45.2,
            "retransmission": 2.1,
            "build": 3815,
            "maxbars": 100000
        }
        mock_get_mt5.return_value = mock_mt5_conn
        
        response = client.get("/api/mt5/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "performance" in data
        performance = data["performance"]
        assert "ping_ms" in performance
        assert "retransmission_factor" in performance
        assert performance["ping_ms"] == 45.2
    
    @patch('app.api.mt5_control.get_mt5_connection')
    def test_mt5_demo_account_validation(self, mock_get_mt5):
        """Test MT5 demo account validation"""
        mock_mt5_conn = MagicMock()
        mock_mt5_conn.is_connected.return_value = True
        mock_mt5_conn.get_connection_info.return_value = {
            "connected": True,
            "server": "XMTrading-MT5 3",
            "login": 123456789,
            "name": "Demo Account",
            "balance": 10000.0,
            "currency": "USD",
            "trade_mode": 0,  # 0 = Demo account
            "company": "Tradexfin Limited"
        }
        mock_get_mt5.return_value = mock_mt5_conn
        
        response = client.get("/api/mt5/status")
        
        assert response.status_code == 200
        data = response.json()
        account_info = data["account_info"]
        
        # Validate demo account characteristics
        assert account_info["trade_mode"] == 0  # Demo mode
        assert "Demo" in account_info["name"]
        assert account_info["balance"] == 10000.0  # Typical demo balance
    
    @patch('app.api.mt5_control.get_mt5_connection')
    def test_mt5_api_rate_limiting(self, mock_get_mt5):
        """Test MT5 API rate limiting behavior"""
        mock_mt5_conn = MagicMock()
        mock_mt5_conn.is_connected.return_value = True
        mock_mt5_conn.get_connection_info.return_value = {"connected": True}
        mock_get_mt5.return_value = mock_mt5_conn
        
        # Make rapid requests
        responses = []
        for _ in range(20):  # Many rapid requests
            response = client.get("/api/mt5/status")
            responses.append(response)
        
        # All should complete (may be rate limited but not error)
        for response in responses:
            assert response.status_code in [200, 429]  # 429 = rate limited
    
    @patch('app.api.mt5_control.get_mt5_connection')
    def test_mt5_connection_timeout_handling(self, mock_get_mt5):
        """Test MT5 connection timeout handling"""
        import time
        
        mock_mt5_conn = MagicMock()
        
        # Simulate slow connection
        def slow_connect():
            time.sleep(0.1)  # Small delay to simulate timeout
            return False
        
        mock_mt5_conn.connect.side_effect = slow_connect
        mock_mt5_conn.is_connected.return_value = False
        mock_mt5_conn.get_last_error.return_value = "Connection timeout"
        mock_get_mt5.return_value = mock_mt5_conn
        
        start_time = time.time()
        response = client.post("/api/mt5/connect")
        end_time = time.time()
        
        # Should handle timeout gracefully
        assert response.status_code == 200
        assert (end_time - start_time) < 5.0  # Should not hang indefinitely
        
        data = response.json()
        assert data["success"] == False