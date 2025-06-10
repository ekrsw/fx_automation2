"""
Pytest configuration and shared fixtures for FX Trading System tests
"""

import pytest
import asyncio
from typing import Generator
from unittest.mock import MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_mt5_connection():
    """Mock MT5 connection for testing"""
    mock_conn = MagicMock()
    mock_conn.is_connected.return_value = False
    mock_conn.connect.return_value = True
    mock_conn.disconnect.return_value = True
    mock_conn.get_connection_info.return_value = {
        "connected": False,
        "server": "Test Server",
        "login": 123456,
        "balance": 10000.0
    }
    return mock_conn


@pytest.fixture
def mock_trading_engine():
    """Mock trading engine for testing"""
    mock_engine = MagicMock()
    mock_engine.is_running = False
    mock_engine.start_session.return_value = {"success": True, "session_id": "test_001"}
    mock_engine.stop_session.return_value = {"success": True}
    mock_engine.get_session_status.return_value = {"status": "stopped"}
    return mock_engine


@pytest.fixture
def mock_position_manager():
    """Mock position manager for testing"""
    mock_manager = MagicMock()
    mock_manager.get_active_positions.return_value = []
    mock_manager.get_statistics.return_value = {
        "total_positions": 0,
        "total_pnl": 0.0,
        "win_rate": 0.0
    }
    return mock_manager


@pytest.fixture
def mock_order_manager():
    """Mock order manager for testing"""
    mock_manager = MagicMock()
    mock_manager.get_active_orders.return_value = []
    mock_manager.create_order.return_value = {"success": True, "order_id": "test_order"}
    mock_manager.cancel_order.return_value = {"success": True}
    return mock_manager


@pytest.fixture
def mock_risk_manager():
    """Mock risk manager for testing"""
    mock_manager = MagicMock()
    mock_manager.get_risk_status.return_value = {
        "emergency_stop_active": False,
        "total_risk_score": 0.0,
        "active_alerts": []
    }
    return mock_manager


@pytest.fixture
def mock_data_fetcher():
    """Mock data fetcher for testing"""
    mock_fetcher = MagicMock()
    mock_fetcher.get_current_market_data.return_value = {}
    mock_fetcher.get_recent_trades.return_value = []
    return mock_fetcher


@pytest.fixture
def sample_market_data():
    """Sample market data for testing"""
    return {
        "EURUSD": {
            "symbol": "EURUSD",
            "bid": 1.0850,
            "ask": 1.0852,
            "spread": 0.0002,
            "change": 0.0025,
            "volume": 1500000
        },
        "USDJPY": {
            "symbol": "USDJPY",
            "bid": 149.45,
            "ask": 149.47,
            "spread": 0.02,
            "change": -0.15,
            "volume": 1200000
        }
    }


@pytest.fixture
def sample_signal_data():
    """Sample signal data for testing"""
    return {
        "signal_id": "test_signal_001",
        "symbol": "EURUSD",
        "signal_type": "BUY",
        "confidence": 0.85,
        "entry_price": 1.0850,
        "stop_loss": 1.0800,
        "take_profit": 1.0950,
        "risk_reward_ratio": 2.0
    }


@pytest.fixture
def sample_position_data():
    """Sample position data for testing"""
    return {
        "position_id": "test_pos_001",
        "symbol": "EURUSD",
        "position_type": "BUY",
        "volume": 0.1,
        "open_price": 1.0850,
        "current_price": 1.0875,
        "unrealized_pnl": 25.0,
        "stop_loss": 1.0800,
        "take_profit": 1.0950
    }


@pytest.fixture
def sample_order_data():
    """Sample order data for testing"""
    return {
        "order_id": "test_ord_001",
        "symbol": "EURUSD",
        "order_type": "MARKET_BUY",
        "volume": 0.1,
        "price": 1.0850,
        "status": "PENDING",
        "stop_loss": 1.0800,
        "take_profit": 1.0950
    }


# Test markers
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "api: mark test as an API test"
    )
    config.addinivalue_line(
        "markers", "websocket: mark test as a WebSocket test"
    )
    config.addinivalue_line(
        "markers", "mt5: mark test as an MT5 integration test"
    )
    config.addinivalue_line(
        "markers", "ui: mark test as a UI/frontend test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )


# Asyncio fixture for async tests
@pytest.fixture
def anyio_backend():
    """Use asyncio as the async backend"""
    return "asyncio"


# Mock settings for testing
@pytest.fixture
def mock_settings():
    """Mock application settings"""
    settings = MagicMock()
    settings.database_url = "sqlite:///test.db"
    settings.mt5_login = 123456
    settings.mt5_password = "test_password"
    settings.mt5_server = "Test-Server"
    settings.risk_per_trade = 0.02
    settings.max_positions = 5
    settings.trading_symbols = ["EURUSD", "USDJPY", "GBPUSD"]
    return settings