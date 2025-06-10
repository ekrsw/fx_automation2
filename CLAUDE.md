# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Application Startup
```bash
# Start development server
python main.py
# Alternative using uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Environment Setup
```bash
# Copy environment template and configure
cp .env.example .env
# Edit .env with your MT5 demo account credentials
# Required: MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

# Install dependencies
pip install -r requirements.txt

# Verify MT5 connection setup
python scripts/setup_mt5_demo.py
```

### Database Operations
```bash
# Apply migrations
alembic upgrade head
# Create new migration
alembic revision --autogenerate -m "description"
# Check migration status
alembic current
```

### Testing - Comprehensive Test Suite
```bash
# Run comprehensive test suite (322 tests across 17 files)
python -m pytest tests/ -v

# Test by category  
pytest tests/test_api/ -v                    # API endpoints (multiple files)
pytest tests/test_analysis/ -v               # Analysis engines
pytest tests/test_frontend/ -v               # UI functionality
pytest tests/test_integration/ -v            # System integration

# Specific test files
pytest tests/test_api/test_dashboard.py -v        # Dashboard API (14 tests)
pytest tests/test_api/test_trading_api.py -v      # Trading API (27 tests)
pytest tests/test_api/test_websockets.py -v       # WebSocket integration (20 tests)
pytest tests/test_api/test_mt5_integration.py -v  # MT5 integration (20 tests)
pytest tests/test_api/test_analysis_api.py -v     # Analysis API (20 tests)

# Analysis engine tests
pytest tests/test_analysis/test_dow_theory.py -v             # Dow Theory (16 tests)
pytest tests/test_analysis/test_elliott_wave.py -v           # Elliott Wave (28 tests)
pytest tests/test_analysis/test_fibonacci_accuracy.py -v     # Fibonacci precision (13 tests)

# Run single test with detailed output
pytest tests/test_api/test_analysis_api.py::TestAnalysisAPI::test_dow_theory_analysis_eurusd -xvs

# Run with coverage measurement
pytest tests/ --cov=app --cov-report=html

# Quick parallel test execution
pytest tests/ -n auto  # Requires pytest-xdist
```

### Strategy Testing
```bash
# Run comprehensive strategy tests (all components)
python scripts/run_strategy_tests.py
# Run demo strategy test (quick validation)
python scripts/demo_strategy_test.py
# Run simple strategy test (basic functionality check)
python scripts/simple_strategy_test.py
```

### MT5 Connection Setup
```bash
# Test MT5 demo connection setup
python scripts/setup_mt5_demo.py
# Test MT5 connection functionality
python scripts/test_mt5_connection.py
```

### Trading System Testing
```bash
# Run complete trading system integration test
python scripts/test_trading_system.py
```

### Code Quality
```bash
# Format code
black app/ tests/
isort app/ tests/
# Lint
flake8 app/ tests/
```

## Architecture Overview

This is a **sophisticated FX automated trading system** implementing **Dow Theory** and **Elliott Wave Theory** for algorithmic trading. The system combines modern Python web technologies with advanced technical analysis and includes a **comprehensive 90%+ test coverage suite** with 219 tests across 11 test files.

### Core Components

**Strategy Engine (Primary Business Logic):**
- `app/analysis/strategy_engine/` - **Phase 5 Complete**: Integrated strategy engine combining all analysis methods
  - `unified_analyzer.py` - Combines Dow Theory + Elliott Wave with confidence scoring
  - `multi_timeframe_analyzer.py` - Analyzes across multiple timeframes (MN1 to M1)
  - `signal_generator.py` - Generates actionable trading signals with risk management
  - `confidence_calculator.py` - 6-factor confidence scoring with A-D grading
  - `entry_exit_engine.py` - Entry/exit condition evaluation with 8 entry + 6 exit conditions
  - `stop_loss_calculator.py` - 6-method stop loss calculation (ATR, swing points, Fibonacci, etc.)
  - `take_profit_calculator.py` - 7-method take profit calculation with partial exit strategies
  - `risk_reward_calculator.py` - Comprehensive risk/reward analysis with position sizing
  - `signal_tester.py` - Testing framework with 6 test types across 7 market conditions
  - `backtest_engine.py` - Full backtesting with 3 execution modes and realistic simulation
  - `strategy_validator.py` - Statistical validation with 4 levels and Monte Carlo analysis

**Analysis Engines:**
- `app/analysis/dow_theory/` - Complete 6-principle Dow Theory implementation with trend detection
- `app/analysis/elliott_wave/` - Full Elliott Wave pattern recognition with Fibonacci analysis, wave prediction, and multi-degree labeling
- `app/analysis/indicators/` - Technical indicators (ATR, ZigZag, momentum, swing detection)

**Trading System (Phase 6 Complete):**
- `app/trading/` - Complete trading execution system with order management, position tracking, and risk control
  - `trading_engine.py` - Main orchestration engine with session management and signal processing
  - `order_manager.py` - Order placement, modification, cancellation with MT5 integration
  - `execution_engine.py` - 5 execution modes (immediate, aggressive, passive, iceberg, TWAP)
  - `position_manager.py` - Position tracking, modification, partial/full closing with real-time PnL
  - `risk_manager.py` - Comprehensive risk monitoring, position sizing, emergency stops

**REST API + WebSocket + UI System (Phase 7 Complete):**
- `app/api/` - Comprehensive REST API with 25+ endpoints for complete system control
  - `dashboard.py` - 7 dashboard endpoints (status, performance, positions, trades, orders, risk, market data)
  - `trading.py` - 8 trading control endpoints (session management, signal processing, order/position management, emergency controls)
  - `analysis.py` - 5 analysis endpoints (Dow Theory, Elliott Wave, unified analysis, signal generation, multi-timeframe)
  - `settings.py` - 1 settings management endpoint (configuration, export, dynamic updates)
  - `mt5_control.py` - 4 MT5 control endpoints (connect, disconnect, status, health check)
  - `dashboard_ui.py` - UI routing for real-time dashboard
  - `websockets.py` - WebSocket server with 7 subscription types and real-time data streaming

**Real-time Web Interface:**
- `frontend/templates/dashboard.html` - Modern glassmorphism trading dashboard with gradient header and integrated status indicators
- `frontend/static/js/dashboard.js` - Advanced WebSocket client with auto-reconnection, real-time updates, and connection status management
- `frontend/static/css/dashboard.css` - Modern responsive styling with glassmorphism effects, animations, and micro-interactions

**WebSocket Real-time System:**
- Real-time price streaming, signal distribution, trading event notifications
- 7 subscription types: market_data, signals, trading_events, system_status, positions, orders, risk_alerts
- Connection management with automatic reconnection and client tracking

**Data & Infrastructure:**
- `app/mt5/` - MetaTrader 5 integration with demo account connection (XMTrading-MT5 3)
- `app/db/` - SQLAlchemy models optimized for time-series data with proper indexing
- `app/services/` - Business logic layer handling data processing and signal generation
- `app/integrations/websocket_integration.py` - WebSocket integration layer for trading system notifications

### Comprehensive Test Suite (90%+ Coverage)

**Testing Architecture:**
- **Total Tests**: 322 test functions across 17 test files
- **API Tests**: Multiple test files covering all REST endpoints and WebSocket functionality
- **Analysis Tests**: Tests for Dow Theory, Elliott Wave, and Fibonacci calculations
- **Frontend Tests**: Tests for UI components, responsive design, and browser compatibility
- **Integration Tests**: Tests for complete system workflows and error handling

**Test Categories:**
- `tests/test_api/` - API endpoint testing (5 files, 101 tests)
  - Dashboard API, Trading API, Analysis API, WebSocket integration, MT5 integration
- `tests/test_analysis/` - Analysis engine testing (3 files, 57 tests)
  - Dow Theory, Elliott Wave, Fibonacci accuracy, multi-currency validation
- `tests/test_frontend/` - UI functionality testing (1 file, 31 tests)
  - Component structure, responsive design, glassmorphism styling
- `tests/test_integration/` - System integration testing (1 file, 17 tests)
  - Complete workflows, concurrent operations, performance testing
- `tests/conftest.py` - Shared fixtures and test configuration

### Key Architectural Patterns

**Layered Architecture:** API → Services → Strategy Engine → Analysis → Data with clear separation of concerns

**Strategy Pattern:** Modular analysis engines that can be combined. Each analysis module (Dow Theory, Elliott Wave) implements common interfaces for extensibility.

**Factory Pattern:** Multiple algorithm implementations (e.g., swing detection supports ZigZag, Fractals, Pivot Points, Adaptive algorithms)

**Configuration-Driven:** Extensive use of Pydantic models for type-safe configuration throughout the system

### Complete Trading System Integration

**Strategy Engine to Trading Execution to API Flow:**
1. `UnifiedAnalyzer` combines Dow Theory + Elliott Wave analysis with configurable weighting
2. `MultiTimeframeAnalyzer` provides hierarchy-based analysis across timeframes
3. `SignalGenerator` creates actionable signals with Market/Limit order determination
4. `ConfidenceCalculator` provides 6-factor confidence scoring with statistical grading
5. `EntryExitEngine` evaluates 8 entry conditions and 6 exit conditions with weighted scoring
6. Risk management calculators provide comprehensive stop loss, take profit, and position sizing
7. **`TradingEngine` orchestrates complete signal-to-execution workflow**
8. **`OrderManager` handles order placement and execution through MT5**
9. **`PositionManager` tracks and manages open positions with real-time PnL**
10. **`RiskManager` monitors and controls trading risk in real-time**
11. **REST API provides external access to all system components and real-time data**

**Testing & Validation Framework:**
- `SignalTester` runs 6 test types (accuracy, performance, stress, robustness, consistency, integration)
- `BacktestEngine` provides realistic simulation with slippage, commission, and risk limits
- `StrategyValidator` performs statistical validation with walk-forward and Monte Carlo analysis
- **Comprehensive test suite with 219 tests ensuring 90%+ code coverage**

### Database Schema

Time-series optimized with composite indexes:
- `PriceData` - OHLCV data with symbol/timeframe/timestamp indexing
- `Signal` - Trading signals with strategy attribution and confidence scoring
- `Trade` - Execution records with strategy tracking
- `SystemSettings` - Dynamic configuration with type conversion

### Analysis Engine Integration

**Dow Theory Analyzer:** Implements all 6 principles with configurable sensitivity. Uses ATR-based noise filtering and pseudo-volume confirmation for FX markets.

**Elliott Wave Analyzer:** Complete 5-wave impulse and 3-wave corrective pattern detection with strict validation rules, Fibonacci level integration, wave prediction capabilities, and multi-degree labeling system. Includes incomplete pattern detection with price target projections.

**Strategy Engine:** Combines both engines with multi-timeframe analysis, sophisticated signal generation, comprehensive risk management, and extensive testing/validation frameworks.

### Real-Time Processing

The system runs continuous data collection (1-minute intervals) with analysis execution (5-minute intervals). APScheduler handles automated tasks including health monitoring and data cleanup.

### Testing Strategy

Comprehensive test coverage with specialized test suites for each component:

**API Testing (Multiple test files):**
- Dashboard API: System status, performance metrics, position/order management
- Trading API: Session control, signal processing, order execution, risk management
- Analysis API: Dow Theory, Elliott Wave, unified analysis, multi-timeframe
- WebSocket: Real-time communication, subscription management, error handling
- MT5 Integration: Connection management, account info, health monitoring

**Analysis Engine Testing (57 tests):**
- Elliott Wave: Pattern detection, Fibonacci accuracy (10 decimal places), multi-currency validation
- Dow Theory: Trend detection, swing point analysis, volume confirmation
- Strategy Engine: Built-in testing framework with 6 test types across 7 market conditions

**Frontend Testing (31 tests):**
- UI components, responsive design, glassmorphism styling
- JavaScript functionality, WebSocket integration, cross-browser compatibility

**Integration Testing (17 tests):**
- Complete system workflows, concurrent operations, performance benchmarks
- Error handling, graceful degradation, resource usage monitoring

### Development Status

**Phase 7 Complete (2025-06-10)** - Complete real-time trading system with UI, MT5 integration, and comprehensive test coverage:

**Completed Phases:**
- **Phase 1-5:** Complete strategy engine with Dow Theory and Elliott Wave analysis
- **Phase 6:** Complete trading execution system implementation and testing
- **Phase 7:** Complete API + WebSocket + UI + MT5 integration + 90%+ test coverage

**Phase 7 Full Implementation:**
- **Phase 7.1:** 25+ REST Endpoints with complete API coverage
- **Phase 7.2:** WebSocket real-time streaming with 7 subscription types
- **Phase 7.3:** Complete real-time dashboard with Bootstrap 5 + Chart.js
- **MT5 Integration:** Demo account connection established (XMTrading-MT5 3, $10,000 balance)
- **Comprehensive Testing:** 322 tests across 17 files
- **Technical Stack:** FastAPI + WebSocket + Bootstrap5 + Chart.js + MT5 integration

**Current Status:** Production-ready real-time trading system with modern UI operational at `http://localhost:8000`

## Key Dependencies & Tech Stack

**Core Framework:**
- FastAPI 0.104.1 - Modern async web framework with auto-generated docs
- Uvicorn - ASGI server with hot reload support
- Pydantic 2.5.0 - Data validation and settings management with type safety

**Data & Analysis:**
- Pandas 2.1.4 - Time series data manipulation optimized for financial data
- NumPy 1.24.4 - Numerical computations for technical indicators
- SQLAlchemy 2.0.23 - Database ORM with async support
- Alembic 1.12.1 - Database schema migrations

**Trading & Real-time:**
- WebSockets 12.0 - Real-time bidirectional communication
- APScheduler 3.10.4 - Background task scheduling for data collection
- Numba 0.58.1 - JIT compilation for performance-critical calculations

**Testing & Quality:**
- Pytest 7.4.3 & pytest-asyncio 0.21.1 - Comprehensive async testing framework
- Black, isort, flake8 - Code formatting and linting tools

**Security & Auth:**
- python-jose[cryptography] - JWT token handling and authentication
- passlib[bcrypt] - Secure password hashing

## Common Issues & Solutions

### Testing Framework Compatibility
The project uses httpx<0.28.0 to maintain compatibility with FastAPI TestClient. Never upgrade httpx beyond 0.27.2 without testing all API endpoints.

**Test Suite Status:** The test suite has achieved significant coverage improvements with Dashboard API, Analysis API, and Trading API tests all fully functional. Key test categories working include:
- Dashboard API (100% passing)
- Trading API (100% passing) 
- Analysis API (100% passing)
- Basic endpoint tests (100% passing)

### Trading API Test Patterns
Trading API tests require specific mock object structures that match the actual implementation:

```python
# Order execution mocks must have proper attributes
mock_execution = MagicMock()
mock_execution.success = True
mock_execution.order_id = "ord_001"
mock_execution.mt5_ticket = 12345
mock_execution.executed_volume = 0.1
mock_execution.execution_price = 1.0851
mock_execution.commission = 2.5
mock_execution.error_message = None
mock_execution.execution_time = datetime.now()

# Position close result mocks
mock_close_result = MagicMock()
mock_close_result.success = True
mock_close_result.closed_volume = 0.1
mock_close_result.close_price = 1.0875
mock_close_result.realized_pnl = 25.0
mock_close_result.commission = 2.5
mock_close_result.error_message = None
mock_close_result.close_time = datetime.now()
```

### Analysis API Test Failures
When analysis API tests fail with HTTP 500 errors, the issue is typically insufficient mocking of the MT5 data fetcher. Tests require both data fetcher and analyzer mocking:

```python
@patch('app.api.analysis.get_data_fetcher')
@patch('app.api.analysis.dow_theory_analyzer')
def test_analysis_endpoint(self, mock_analyzer, mock_get_data_fetcher):
    # Mock data fetcher
    mock_data_fetcher = AsyncMock()
    mock_get_data_fetcher.return_value = mock_data_fetcher
    
    # Create realistic mock price data
    import pandas as pd
    mock_price_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='4H'),
        'open': [1.0800 + i*0.0001 for i in range(100)],
        'high': [1.0810 + i*0.0001 for i in range(100)],
        'low': [1.0790 + i*0.0001 for i in range(100)],
        'close': [1.0805 + i*0.0001 for i in range(100)],
        'volume': [1000 + i*10 for i in range(100)]
    })
    mock_data_fetcher.get_historical_data.return_value = mock_price_data
    
    # Mock analyzer result with proper object structure
    mock_analysis_result = MagicMock()
    mock_analysis_result.primary_trend.value = "bullish"
    mock_analyzer.analyze.return_value = mock_analysis_result
```

### DataFrame Validation Patterns
The system uses proper DataFrame validation to avoid ambiguous truth value errors:

```python
# Correct pattern
if price_data is None or price_data.empty:
    raise HTTPException(status_code=404, detail="No data available")

# Avoid this (causes "ambiguous truth value" errors)
if not price_data or len(price_data) == 0:
```

### API Response Structure Patterns
Trading API endpoints return specific response structures that tests must match:

```python
# Signal processing returns boolean success, not nested objects
mock_trading_engine.process_signal = AsyncMock(return_value=True)  # Not a dict

# Order cancellation uses DELETE method, not POST
response = client.delete("/api/trading/orders/ord_001")  # Correct
# response = client.post("/api/trading/orders/ord_001/cancel")  # Wrong

# Emergency stop methods return None, not dict  
mock_risk_manager.emergency_stop = AsyncMock(return_value=None)

# Order types must match exact enum values
"SELL_LIMIT"  # Correct
"LIMIT_SELL"  # Wrong - will cause 400 validation error
```

### Method Signature Issues
Data fetcher methods expect `start_date` and `end_date` parameters, not `count`:

```python
# Correct usage
end_date = datetime.now()
start_date = end_date - timedelta(days=days_back)
price_data = await data_fetcher.get_historical_data(
    symbol=symbol,
    timeframe=timeframe,
    start_date=start_date,
    end_date=end_date
)

# Incorrect (will cause TypeError)
price_data = await data_fetcher.get_historical_data(
    symbol=symbol,
    timeframe=timeframe,
    count=periods  # This parameter doesn't exist
)
```

## Important Configuration

System settings are managed through `app/config.py` with environment variable support. Trading parameters, risk management settings, and analysis parameters are all configurable without code changes.

The complete system runs on `http://localhost:8000` with comprehensive access:
- **Real-time Dashboard:** `/ui/dashboard` - Complete trading interface with live charts and controls
- **Interactive API Documentation:** `/docs` - Auto-generated Swagger UI
- **Root Landing Page:** `/` - System overview with auto-redirect to dashboard
- **Health Check:** `/health` - System health status
- **WebSocket Connection:** `ws://localhost:8000/ws` - Real-time data streaming

**Key URLs:**
- **Primary Dashboard**: `http://localhost:8000/ui/dashboard` - Modern real-time trading interface
- **API Documentation**: `http://localhost:8000/docs` - Interactive Swagger UI
- **System Status**: `http://localhost:8000/api/dashboard/status` - Real-time system health
- **MT5 Status**: `http://localhost:8000/api/mt5/status` - MT5 connection and account info
- **WebSocket Endpoint**: `ws://localhost:8000/ws` - Real-time data streaming

## Complete Trading System Usage

The system provides end-to-end trading from analysis to execution with full API access:

### Programmatic Usage (Python)
```python
# Strategy Analysis
from app.analysis.strategy_engine import (
    unified_analyzer, signal_generator, confidence_calculator,
    entry_exit_engine, stop_loss_calculator, take_profit_calculator,
    risk_reward_calculator
)

# Trading Execution  
from app.trading import (
    TradingEngine, OrderManager, ExecutionEngine, 
    PositionManager, RiskManager
)

# Complete workflow
unified_result = unified_analyzer.analyze(price_data, symbol, timeframe)
signal = signal_generator.generate_signal_from_unified(unified_result, current_price)
confidence = confidence_calculator.calculate_unified_confidence(unified_result, signal)

# Risk management
stop_loss_rec = stop_loss_calculator.calculate_stop_loss(signal, unified_result, price_data)
take_profit_rec = take_profit_calculator.calculate_take_profit(signal, unified_result, price_data)
risk_analysis = risk_reward_calculator.calculate_risk_reward(signal, stop_loss_rec, take_profit_rec)

# Trading execution
trading_engine = TradingEngine()
await trading_engine.start_trading_session(TradingMode.DEMO)
success = await trading_engine.process_signal(signal)
```

### REST API Usage
```bash
# Get system status
curl http://localhost:8000/api/dashboard/status

# Connect to MT5
curl -X POST http://localhost:8000/api/mt5/connect

# Start trading session
curl -X POST http://localhost:8000/api/trading/session/start \
  -H "Content-Type: application/json" \
  -d '{"mode": "demo"}'

# Get analysis for USDJPY
curl http://localhost:8000/api/analysis/unified/USDJPY?timeframe=H1

# Get current positions
curl http://localhost:8000/api/dashboard/positions

# Update trading settings
curl -X PUT http://localhost:8000/api/settings/trading \
  -H "Content-Type: application/json" \
  -d '{"max_positions": 10, "risk_per_trade": 0.02}'
```

### WebSocket Usage (JavaScript)
```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws');

// Subscribe to market data
ws.send(JSON.stringify({
    action: 'subscribe',
    subscription_type: 'market_data',
    symbols: ['EURUSD', 'USDJPY']
}));

// Handle real-time updates
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Real-time update:', data);
};
```

### Real-time Dashboard Access
Navigate to `http://localhost:8000/ui/dashboard` for the production-ready trading interface featuring:
- **Modern Glassmorphism Design**: Gradient header with integrated connection status using pulse animations
- **Live Price Charts**: Real-time Chart.js implementation with 50-point history and multi-symbol support
- **Integrated Controls**: MT5 connection, trading session management, and emergency controls in unified header
- **Real-time Monitoring**: WebSocket-powered live position tracking, market data, and trading events
- **Responsive Design**: Mobile-optimized interface with backdrop blur effects and micro-interactions

### Testing and Quality Assurance

**Comprehensive Test Execution:**
```bash
# Run full test suite (322 tests)
python -m pytest tests/ -v

# Test specific components
python -m pytest tests/test_api/test_dashboard.py::TestDashboardAPI::test_dashboard_status_success -v
python -m pytest tests/test_analysis/test_elliott_wave.py::TestElliottWaveAnalyzer -v
python -m pytest tests/test_frontend/test_ui_functionality.py::TestFrontendUI::test_dashboard_page_access -v

# Coverage analysis
python -m pytest tests/ --cov=app --cov-report=html
```

**Quality Metrics Achieved:**
- **API Coverage**: Multiple test files across all 25+ REST endpoints and WebSocket functionality
- **Analysis Coverage**: Tests with mathematical precision validation (10 decimal places)
- **UI Coverage**: Tests for responsive design and modern glassmorphism interface
- **Integration Coverage**: Tests for complete system workflows and performance
- **Overall Coverage**: Comprehensive coverage across all major system components

### Frontend Development Notes
- **UI Framework**: Bootstrap 5 with custom glassmorphism styling
- **Real-time Updates**: WebSocket client with automatic reconnection and status indicators
- **Design System**: Modern gradient-based design with pulse animations and hover effects
- **Connection Status**: Seamlessly integrated in header to avoid UI overlay issues
- **Performance**: Optimized for real-time data streaming with efficient DOM updates