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

### MT5 Connection Setup
```bash
# Test MT5 demo connection setup
python scripts/setup_mt5_demo.py
# Test MT5 connection functionality
python scripts/test_mt5_connection.py
```

### Testing
```bash
# Run all tests
pytest tests/
# Run specific analysis engine tests
pytest tests/test_analysis/test_dow_theory.py -v
pytest tests/test_analysis/test_elliott_wave.py -v
pytest tests/test_analysis/test_fibonacci_accuracy.py -v
pytest tests/test_analysis/test_multi_currency_validation.py -v
# Run all analysis tests
pytest tests/test_analysis/ -v
# Run with coverage
pytest tests/ --cov=app --cov-report=html
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

### Trading System Testing
```bash
# Run complete trading system integration test
python scripts/test_trading_system.py
# Test individual trading components
pytest tests/test_trading/ -v
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

This is a **sophisticated FX automated trading system** implementing **Dow Theory** and **Elliott Wave Theory** for algorithmic trading. The system combines modern Python web technologies with advanced technical analysis.

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

**Data & Infrastructure:**
- `app/mt5/` - MetaTrader 5 integration for live data and trade execution
- `app/db/` - SQLAlchemy models optimized for time-series data with proper indexing
- `app/services/` - Business logic layer handling data processing and signal generation

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

Comprehensive test coverage with specialized test suites for each analysis engine:

**Elliott Wave Test Suite:**
- `test_elliott_wave.py` - Core functionality, pattern detection, wave labeling, and prediction workflows
- `test_fibonacci_accuracy.py` - Mathematical precision testing (10 decimal places), performance benchmarks, and real market data validation
- `test_multi_currency_validation.py` - Cross-currency testing (EUR/USD, USD/JPY, GBP/JPY, XAU/USD), volatility adaptation, and correlation analysis

**Strategy Engine Test Suite:**
- Built-in testing framework with `SignalTester` for automated validation across 6 test types and 7 market conditions
- `BacktestEngine` for historical performance validation with 3 execution modes and realistic simulation
- `StrategyValidator` for statistical significance testing with 4 validation levels
- **Operational Testing Scripts:** `scripts/run_strategy_tests.py` (comprehensive), `scripts/demo_strategy_test.py` (demo), `scripts/simple_strategy_test.py` (basic check)

**Key Testing Features:**
- Parametrized testing for noise tolerance and currency-specific behavior
- Performance requirements: 1000 calculations under 100ms
- Mathematical precision: Fibonacci calculations accurate to 1e-10
- Confidence scoring validation: All patterns include reliability metrics (0.0-1.0 scale)
- Real market simulation with realistic price movements and volatility patterns

### Development Status

**Phase 7 Complete (2025-06-10)** - Complete real-time trading system with UI and MT5 integration:

**Completed Phases:**
- **Phase 1-5:** Complete strategy engine with Dow Theory and Elliott Wave analysis
- **Phase 6:** Complete trading execution system implementation and testing
- **Phase 7:** Complete API + WebSocket + UI + MT5 integration

**Phase 7 Full Implementation:**
- **Phase 7.1:** 25+ REST Endpoints with complete API coverage
- **Phase 7.2:** WebSocket real-time streaming with 7 subscription types
- **Phase 7.3:** Complete real-time dashboard with Bootstrap 5 + Chart.js
- **MT5 Integration:** Demo account connection established (XMTrading-MT5 3, $10,000 balance)
- **Technical Stack:** FastAPI + WebSocket + Bootstrap5 + Chart.js + MT5 integration
- **Real-time Features:** Live price charts, position tracking, trading controls, system monitoring

**Current Status:** Production-ready real-time trading system with modern UI operational at `http://localhost:8000`

**Latest Updates (Phase 7 Complete):**
- **Modern Glassmorphism UI**: Redesigned dashboard with gradient header, backdrop blur effects, and micro-interactions
- **Integrated Connection Status**: WebSocket status seamlessly integrated into header with pulse animations
- **MT5 Demo Integration**: Live connection to XMTrading-MT5 3 server with $10,000 demo account
- **Real-time WebSocket Streaming**: 7 subscription types with automatic reconnection
- **Complete API Coverage**: 25+ REST endpoints for full system control

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

### Frontend Development Notes
- **UI Framework**: Bootstrap 5 with custom glassmorphism styling
- **Real-time Updates**: WebSocket client with automatic reconnection and status indicators
- **Design System**: Modern gradient-based design with pulse animations and hover effects
- **Connection Status**: Seamlessly integrated in header to avoid UI overlay issues
- **Performance**: Optimized for real-time data streaming with efficient DOM updates