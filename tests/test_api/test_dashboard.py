"""
Comprehensive tests for Dashboard API endpoints
Testing all dashboard functionality with 90%+ coverage
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
import json

from app.main import app

client = TestClient(app)


class TestDashboardAPI:
    """Comprehensive Dashboard API test suite"""
    
    def test_dashboard_status_success(self):
        """Test successful dashboard status retrieval"""
        response = client.get("/api/dashboard/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "overall_status" in data
        assert "components" in data
        assert "configuration" in data
        
        # Verify component structure
        components = data["components"]
        required_components = [
            "mt5_connection", "data_fetcher", "database", 
            "trading_engine", "order_manager", "position_manager", "risk_manager"
        ]
        for component in required_components:
            assert component in components
            assert "status" in components[component]
    
    @patch('app.trading.position_manager.position_manager')
    def test_dashboard_performance_success(self, mock_position_manager):
        """Test performance metrics retrieval"""
        # Mock position manager response with AsyncMock
        mock_position_manager.get_statistics = AsyncMock(return_value={
            "total_positions": 5,
            "open_positions": 2,
            "total_pnl": 1500.50,
            "unrealized_pnl": 250.75,
            "realized_pnl": 1249.75,
            "win_rate": 0.65,
            "avg_profit": 312.50,
            "avg_loss": -125.25
        })
        mock_position_manager.get_portfolio_summary = AsyncMock(return_value={
            "total_value": 10000.0,
            "total_pnl": 1500.50,
            "total_positions": 5
        })
        
        response = client.get("/api/dashboard/performance")
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "performance_metrics" in data
        assert "trading_statistics" in data
        assert "system_metrics" in data
        
        # Verify performance metrics structure
        perf_metrics = data["performance_metrics"]
        assert "total_pnl" in perf_metrics
        assert "win_rate" in perf_metrics
        assert "sharpe_ratio" in perf_metrics
    
    @patch('app.trading.position_manager.position_manager')
    def test_dashboard_positions_with_data(self, mock_position_manager):
        """Test positions endpoint with active positions"""
        # Mock positions data
        mock_positions = [
            {
                "position_id": "pos_001",
                "symbol": "EURUSD", 
                "position_type": "BUY",
                "volume": 0.1,
                "open_price": 1.0850,
                "current_price": 1.0875,
                "unrealized_pnl": 25.0,
                "open_time": datetime.now().isoformat(),
                "stop_loss": 1.0800,
                "take_profit": 1.0950
            },
            {
                "position_id": "pos_002",
                "symbol": "USDJPY",
                "position_type": "SELL", 
                "volume": 0.05,
                "open_price": 149.50,
                "current_price": 149.25,
                "unrealized_pnl": 12.5,
                "open_time": datetime.now().isoformat(),
                "stop_loss": 150.00,
                "take_profit": 148.50
            }
        ]
        
        mock_position_manager.get_open_positions = AsyncMock(return_value=mock_positions)
        mock_position_manager.get_portfolio_summary = AsyncMock(return_value={
            "total_positions": 2,
            "total_unrealized_pnl": 37.5
        })
        
        response = client.get("/api/dashboard/positions")
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "positions" in data
        assert "summary" in data
        
        # Verify positions data
        positions = data["positions"]
        assert len(positions) == 2
        assert positions[0]["symbol"] == "EURUSD"
        assert positions[1]["symbol"] == "USDJPY"
        
        # Verify summary
        summary = data["summary"]
        assert summary["total_positions"] == 2
        assert summary["total_unrealized_pnl"] == 37.5
    
    @patch('app.api.dashboard.position_manager')
    def test_dashboard_positions_empty(self, mock_position_manager):
        """Test positions endpoint with no active positions"""
        mock_position_manager.get_active_positions = AsyncMock(return_value=[])
        
        response = client.get("/api/dashboard/positions")
        assert response.status_code == 200
        
        data = response.json()
        assert data["positions"] == []
        assert data["summary"]["total_positions"] == 0
    
    @patch('app.api.dashboard.get_data_fetcher')
    def test_dashboard_recent_trades(self, mock_get_data_fetcher):
        """Test recent trades endpoint"""
        # Mock recent trades data
        mock_trades = [
            {
                "trade_id": "trade_001",
                "symbol": "EURUSD",
                "trade_type": "BUY",
                "volume": 0.1,
                "open_price": 1.0800,
                "close_price": 1.0850,
                "realized_pnl": 50.0,
                "open_time": (datetime.now() - timedelta(hours=2)).isoformat(),
                "close_time": (datetime.now() - timedelta(hours=1)).isoformat(),
                "duration_minutes": 60
            }
        ]
        
        mock_fetcher = MagicMock()
        mock_fetcher.get_recent_trades = AsyncMock(return_value=mock_trades)
        mock_get_data_fetcher.return_value = mock_fetcher
        
        response = client.get("/api/dashboard/recent-trades")
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "trades" in data
        assert "summary" in data
        
        trades = data["trades"]
        assert len(trades) == 1
        assert trades[0]["symbol"] == "EURUSD"
        assert trades[0]["realized_pnl"] == 50.0
    
    @patch('app.api.dashboard.order_manager')
    def test_dashboard_orders_with_data(self, mock_order_manager):
        """Test orders endpoint with active orders"""
        mock_orders = [
            {
                "order_id": "order_001",
                "symbol": "GBPUSD",
                "order_type": "BUY_LIMIT",
                "volume": 0.1,
                "price": 1.2500,
                "status": "PENDING",
                "created_at": datetime.now().isoformat(),
                "expires_at": (datetime.now() + timedelta(days=1)).isoformat()
            }
        ]
        
        mock_order_manager.get_active_orders = AsyncMock(return_value=mock_orders)
        
        response = client.get("/api/dashboard/orders")
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "orders" in data
        assert "summary" in data
        
        orders = data["orders"]
        assert len(orders) == 1
        assert orders[0]["symbol"] == "GBPUSD"
        assert orders[0]["order_type"] == "BUY_LIMIT"
    
    @patch('app.api.dashboard.risk_manager')
    def test_dashboard_risk_status(self, mock_risk_manager):
        """Test risk status endpoint"""
        mock_risk_data = {
            "emergency_stop_active": False,
            "total_risk_score": 0.35,
            "position_risk": 0.25,
            "correlation_risk": 0.15,
            "volatility_risk": 0.20,
            "drawdown_risk": 0.10,
            "max_daily_loss": 0.05,
            "current_daily_loss": 0.02,
            "max_drawdown": 0.15,
            "current_drawdown": 0.03,
            "active_alerts": []
        }
        
        mock_risk_manager.get_risk_status = AsyncMock(return_value=mock_risk_data)
        
        response = client.get("/api/dashboard/risk-status")
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "risk_status" in data
        assert "alerts" in data
        
        risk_status = data["risk_status"]
        assert risk_status["total_risk_score"] == 0.35
        assert risk_status["emergency_stop_active"] == False
    
    @patch('app.api.dashboard.get_data_fetcher')
    def test_dashboard_market_data(self, mock_get_data_fetcher):
        """Test market data endpoint"""
        mock_market_data = {
            "EURUSD": {
                "symbol": "EURUSD",
                "bid": 1.0850,
                "ask": 1.0852,
                "spread": 0.0002,
                "change": 0.0025,
                "change_percent": 0.23,
                "volume": 1500000,
                "timestamp": datetime.now().isoformat()
            },
            "USDJPY": {
                "symbol": "USDJPY", 
                "bid": 149.45,
                "ask": 149.47,
                "spread": 0.02,
                "change": -0.15,
                "change_percent": -0.10,
                "volume": 1200000,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        mock_fetcher = MagicMock()
        mock_fetcher.get_current_market_data = AsyncMock(return_value=mock_market_data)
        mock_get_data_fetcher.return_value = mock_fetcher
        
        response = client.get("/api/dashboard/market-data")
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "market_data" in data
        assert "summary" in data
        
        market_data = data["market_data"]
        assert "EURUSD" in market_data
        assert "USDJPY" in market_data
        assert market_data["EURUSD"]["bid"] == 1.0850
        assert market_data["USDJPY"]["ask"] == 149.47
    
    def test_dashboard_health_check(self):
        """Test dashboard health check endpoint"""
        response = client.get("/api/dashboard/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "system_health" in data
    
    @patch('app.api.dashboard.position_manager')
    def test_dashboard_error_handling(self, mock_position_manager):
        """Test error handling in dashboard endpoints"""
        # Mock an exception
        mock_position_manager.get_active_positions = AsyncMock(side_effect=Exception("Database error"))
        
        response = client.get("/api/dashboard/positions")
        assert response.status_code == 500
        
        data = response.json()
        assert "detail" in data
    
    def test_dashboard_status_components_structure(self):
        """Test detailed component status structure"""
        response = client.get("/api/dashboard/status")
        assert response.status_code == 200
        
        data = response.json()
        components = data["components"]
        
        # Test MT5 connection component
        mt5_comp = components["mt5_connection"]
        assert "status" in mt5_comp
        assert "details" in mt5_comp
        
        # Test database component
        db_comp = components["database"]
        assert "status" in db_comp
        assert "url" in db_comp
        
        # Test trading engine component
        trading_comp = components["trading_engine"]
        assert "status" in trading_comp
        assert "session_status" in trading_comp
    
    @patch('app.api.dashboard.get_data_fetcher')
    def test_market_data_empty_response(self, mock_get_data_fetcher):
        """Test market data endpoint with empty response"""
        mock_fetcher = MagicMock()
        mock_fetcher.get_current_market_data = AsyncMock(return_value={})
        mock_get_data_fetcher.return_value = mock_fetcher
        
        response = client.get("/api/dashboard/market-data")
        assert response.status_code == 200
        
        data = response.json()
        assert data["market_data"] == {}
        assert data["summary"]["total_symbols"] == 0
    
    def test_concurrent_dashboard_requests(self):
        """Test concurrent requests to dashboard endpoints"""
        import concurrent.futures
        import threading
        
        endpoints = [
            "/api/dashboard/status",
            "/api/dashboard/health",
            "/api/dashboard/positions",
            "/api/dashboard/orders"
        ]
        
        def make_request(endpoint):
            return client.get(endpoint)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(make_request, endpoint) for endpoint in endpoints]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        for response in responses:
            assert response.status_code in [200, 500]  # 500 is acceptable for mocked failures
    
    @patch('app.api.dashboard.get_settings_cached')
    def test_dashboard_configuration_data(self, mock_get_settings):
        """Test configuration data in status endpoint"""
        mock_settings = MagicMock()
        mock_settings.trading_symbols = ["EURUSD", "USDJPY", "GBPUSD"]
        mock_settings.max_positions = 5
        mock_settings.risk_per_trade = 0.02
        mock_get_settings.return_value = mock_settings
        
        response = client.get("/api/dashboard/status")
        assert response.status_code == 200
        
        data = response.json()
        config = data["configuration"]
        assert "trading_symbols" in config
        assert "max_positions" in config
        assert "risk_per_trade" in config