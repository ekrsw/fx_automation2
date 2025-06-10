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

### Database Operations
```bash
# Apply migrations
alembic upgrade head
# Create new migration
alembic revision --autogenerate -m "description"
# Check migration status
alembic current
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

**Analysis Engines (Primary Business Logic):**
- `app/analysis/dow_theory/` - Complete 6-principle Dow Theory implementation with trend detection
- `app/analysis/elliott_wave/` - Full Elliott Wave pattern recognition with Fibonacci analysis, wave prediction, and multi-degree labeling
- `app/analysis/indicators/` - Technical indicators (ATR, ZigZag, momentum, swing detection)

**Data & Trading Infrastructure:**
- `app/mt5/` - MetaTrader 5 integration for live data and trade execution
- `app/db/` - SQLAlchemy models optimized for time-series data with proper indexing
- `app/services/` - Business logic layer handling data processing and signal generation

**API & Interface:**
- `app/api/` - FastAPI endpoints for dashboard and system monitoring
- `frontend/` - Web dashboard for visualization and system control

### Key Architectural Patterns

**Layered Architecture:** API → Services → Analysis → Data with clear separation of concerns

**Strategy Pattern:** Modular analysis engines that can be combined. Each analysis module (Dow Theory, Elliott Wave) implements common interfaces for extensibility.

**Factory Pattern:** Multiple algorithm implementations (e.g., swing detection supports ZigZag, Fractals, Pivot Points, Adaptive algorithms)

**Configuration-Driven:** Extensive use of Pydantic models for type-safe configuration throughout the system

### Database Schema

Time-series optimized with composite indexes:
- `PriceData` - OHLCV data with symbol/timeframe/timestamp indexing
- `Signal` - Trading signals with strategy attribution and confidence scoring
- `Trade` - Execution records with strategy tracking
- `SystemSettings` - Dynamic configuration with type conversion

### Analysis Engine Integration

**Dow Theory Analyzer:** Implements all 6 principles with configurable sensitivity. Uses ATR-based noise filtering and pseudo-volume confirmation for FX markets.

**Elliott Wave Analyzer:** Complete 5-wave impulse and 3-wave corrective pattern detection with strict validation rules, Fibonacci level integration, wave prediction capabilities, and multi-degree labeling system. Includes incomplete pattern detection with price target projections.

Both engines generate confidence-scored signals that feed into a unified decision system.

### Real-Time Processing

The system runs continuous data collection (1-minute intervals) with analysis execution (5-minute intervals). APScheduler handles automated tasks including health monitoring and data cleanup.

### Testing Strategy

Comprehensive test coverage with specialized test suites for each analysis engine:

**Elliott Wave Test Suite:**
- `test_elliott_wave.py` - Core functionality, pattern detection, wave labeling, and prediction workflows
- `test_fibonacci_accuracy.py` - Mathematical precision testing (10 decimal places), performance benchmarks, and real market data validation
- `test_multi_currency_validation.py` - Cross-currency testing (EUR/USD, USD/JPY, GBP/JPY, XAU/USD), volatility adaptation, and correlation analysis

**Key Testing Features:**
- Parametrized testing for noise tolerance and currency-specific behavior
- Performance requirements: 1000 calculations under 100ms
- Mathematical precision: Fibonacci calculations accurate to 1e-10
- Confidence scoring validation: All patterns include reliability metrics (0.0-1.0 scale)
- Real market simulation with realistic price movements and volatility patterns

### Development Status

**Phase 4 Complete** - Elliott Wave analysis engine fully implemented with comprehensive testing:

**Completed Components:**
- Complete Elliott Wave pattern detection (5-wave impulse, 3-wave corrective)
- Advanced Fibonacci analysis with 10-decimal precision calculations
- Multi-degree wave labeling system (Primary → Intermediate → Minor → Minute → Minuette)
- Wave prediction engine with price targets and risk assessments
- Comprehensive test suite covering accuracy, performance, and multi-currency validation

**Ready for Phase 5:** Strategy integration combining Dow Theory and Elliott Wave analysis for unified signal generation.

## Important Configuration

System settings are managed through `app/config.py` with environment variable support. Trading parameters, risk management settings, and analysis parameters are all configurable without code changes.

The development server runs on `http://localhost:8000` with a live dashboard showing system health and analysis results.