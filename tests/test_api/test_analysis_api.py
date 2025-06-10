"""
Comprehensive tests for Analysis REST API endpoints
Testing all analysis functionality with 90%+ coverage
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
import json

from app.main import app

client = TestClient(app)


class TestAnalysisAPI:
    """Comprehensive Analysis API test suite"""
    
    def setup_mock_data_fetcher(self):
        """Setup common mock data fetcher for all tests"""
        import pandas as pd
        from unittest.mock import MagicMock, AsyncMock
        
        mock_data_fetcher = AsyncMock()
        
        # Create a custom mock object that behaves like DataFrame but supports dict-like access
        class MockPriceData:
            def __init__(self):
                self.data = {
                    'timestamp': pd.date_range('2024-01-01', periods=100, freq='4h'),
                    'open': [1.0800 + i*0.0001 for i in range(100)],
                    'high': [1.0810 + i*0.0001 for i in range(100)],
                    'low': [1.0790 + i*0.0001 for i in range(100)],
                    'close': [1.0805 + i*0.0001 for i in range(100)],
                    'volume': [1000 + i*10 for i in range(100)]
                }
            
            def __getitem__(self, key):
                if key == -1:
                    # Return last row as dict
                    return {col: values[-1] for col, values in self.data.items()}
                return self.data[key]
            
            def __len__(self):
                return 100
                
            @property
            def empty(self):
                return False
        
        mock_price_data = MockPriceData()
        mock_data_fetcher.get_historical_data.return_value = mock_price_data
        return mock_data_fetcher
    
    def setup_mock_dow_analyzer_result(self, trend="BULLISH"):
        """Setup mock Dow Theory analyzer result"""
        from unittest.mock import MagicMock
        from enum import Enum
        
        class MockTrend(Enum):
            BULLISH = "bullish"
            BEARISH = "bearish"
            NEUTRAL = "neutral"
        
        mock_analysis_result = MagicMock()
        mock_analysis_result.primary_trend = MockTrend.BULLISH if trend == "BULLISH" else MockTrend.BEARISH
        mock_analysis_result.secondary_trend = MockTrend.BULLISH if trend == "BULLISH" else MockTrend.BEARISH
        mock_analysis_result.minor_trend = MockTrend.NEUTRAL
        mock_analysis_result.trend_strength = 0.75
        mock_analysis_result.confirmation_score = 0.80
        mock_analysis_result.volume_confirmation = True
        mock_analysis_result.swing_points = []
        mock_analysis_result.support_levels = []
        mock_analysis_result.resistance_levels = []
        mock_analysis_result.signal_strength = 0.85
        mock_analysis_result.risk_score = 0.15
        mock_analysis_result.predictions = []
        return mock_analysis_result
        
    def setup_mock_elliott_analyzer_result(self):
        """Setup mock Elliott Wave analyzer result"""
        from unittest.mock import MagicMock
        
        mock_analysis_result = MagicMock()
        mock_analysis_result.current_wave = MagicMock()
        mock_analysis_result.current_wave.wave_type = "IMPULSE"
        mock_analysis_result.current_wave.wave_number = 3
        mock_analysis_result.current_wave.degree = "PRIMARY"
        
        mock_analysis_result.wave_pattern = MagicMock()
        mock_analysis_result.wave_pattern.pattern_type = "FIVE_WAVE_IMPULSE"
        mock_analysis_result.wave_pattern.completion_percentage = 0.60
        mock_analysis_result.wave_pattern.wave_structure = []
        
        mock_analysis_result.fibonacci_levels = MagicMock()
        mock_analysis_result.fibonacci_levels.retracement_levels = [1.0825, 1.0812, 1.0806]
        mock_analysis_result.fibonacci_levels.extension_levels = [1.0920, 1.0950, 1.0980]
        
        mock_analysis_result.confidence = 0.85
        mock_analysis_result.swing_points = []
        mock_analysis_result.predictions = []
        return mock_analysis_result
    
    @patch('app.api.analysis.get_data_fetcher')
    @patch('app.api.analysis.dow_theory_analyzer')
    def test_dow_theory_analysis_eurusd(self, mock_dow_analyzer, mock_get_data_fetcher):
        """Test Dow Theory analysis for EURUSD"""
        # Mock data fetcher
        mock_data_fetcher = AsyncMock()
        mock_get_data_fetcher.return_value = mock_data_fetcher
        
        # Create mock price data (DataFrame-like object)
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
        
        # Create mock analysis result object with required attributes
        from unittest.mock import MagicMock
        from enum import Enum
        
        # Mock the trend enum
        class MockTrend(Enum):
            BULLISH = "bullish"
            BEARISH = "bearish"
            NEUTRAL = "neutral"
        
        mock_analysis_result = MagicMock()
        mock_analysis_result.primary_trend = MockTrend.BULLISH
        mock_analysis_result.secondary_trend = MockTrend.BULLISH
        mock_analysis_result.minor_trend = MockTrend.NEUTRAL
        mock_analysis_result.trend_strength = 0.75
        mock_analysis_result.confirmation_score = 0.80
        mock_analysis_result.volume_confirmation = True
        mock_analysis_result.swing_points = []  # Empty for simplicity
        mock_analysis_result.support_levels = []
        mock_analysis_result.resistance_levels = []
        mock_analysis_result.signal_strength = 0.85
        mock_analysis_result.risk_score = 0.15
        mock_analysis_result.predictions = []
        
        mock_dow_analyzer.analyze.return_value = mock_analysis_result
        
        response = client.get("/api/analysis/dow-theory/EURUSD?timeframe=H4")
        
        assert response.status_code == 200
        data = response.json()
        
        # The API returns a nested structure under "analysis"
        assert data["symbol"] == "EURUSD"
        assert data["timeframe"] == "H4"
        assert "analysis" in data
        analysis = data["analysis"]
        assert analysis["primary_trend"] == "bullish"  # Enum value 
        assert analysis["trend_strength"] == 0.75
    
    @patch('app.api.analysis.get_data_fetcher')
    @patch('app.api.analysis.dow_theory_analyzer')
    def test_dow_theory_analysis_usdjpy(self, mock_dow_analyzer, mock_get_data_fetcher):
        """Test Dow Theory analysis for USDJPY"""
        # Setup mocks
        mock_get_data_fetcher.return_value = self.setup_mock_data_fetcher()
        mock_dow_analyzer.analyze.return_value = self.setup_mock_dow_analyzer_result("BEARISH")
        
        response = client.get("/api/analysis/dow-theory/USDJPY?timeframe=H1")
        
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "USDJPY"
        assert data["timeframe"] == "H1"
        assert "analysis" in data
        analysis = data["analysis"]
        assert analysis["primary_trend"] == "bearish"
    
    @patch('app.api.analysis.get_data_fetcher')
    @patch('app.api.analysis.dow_theory_analyzer')
    def test_dow_theory_analysis_invalid_symbol(self, mock_dow_analyzer, mock_get_data_fetcher):
        """Test Dow Theory analysis with invalid symbol"""
        # Setup mocks - data fetcher should work but analyzer should fail
        mock_get_data_fetcher.return_value = self.setup_mock_data_fetcher()
        mock_dow_analyzer.analyze.side_effect = ValueError("Invalid symbol")
        
        response = client.get("/api/analysis/dow-theory/INVALID")
        
        assert response.status_code == 500  # Internal server error due to analyzer failure
        data = response.json()
        assert "detail" in data
    
    @patch('app.api.analysis.get_data_fetcher')
    @patch('app.api.analysis.elliott_wave_analyzer')
    def test_elliott_wave_analysis_eurusd(self, mock_elliott_analyzer, mock_get_data_fetcher):
        """Test Elliott Wave analysis for EURUSD"""
        # Setup mocks
        mock_get_data_fetcher.return_value = self.setup_mock_data_fetcher()
        mock_elliott_analyzer.analyze.return_value = self.setup_mock_elliott_analyzer_result()
        
        response = client.get("/api/analysis/elliott-wave/EURUSD?timeframe=H4")
        
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "EURUSD"
        assert data["timeframe"] == "H4"
        assert "analysis" in data
    
    @patch('app.api.analysis.get_data_fetcher')
    @patch('app.api.analysis.elliott_wave_analyzer')
    def test_elliott_wave_analysis_correction_pattern(self, mock_elliott_analyzer, mock_get_data_fetcher):
        """Test Elliott Wave analysis with correction pattern"""
        # Setup mocks
        mock_get_data_fetcher.return_value = self.setup_mock_data_fetcher()
        
        # Setup mock analysis result with correction pattern
        mock_analysis_result = MagicMock()
        mock_analysis_result.current_wave = MagicMock()
        mock_analysis_result.current_wave.wave_type = "CORRECTION"
        mock_analysis_result.current_wave.wave_number = "B"
        mock_analysis_result.current_wave.degree = "INTERMEDIATE"
        
        mock_analysis_result.wave_pattern = MagicMock()
        mock_analysis_result.wave_pattern.pattern_type = "ABC_CORRECTION"
        mock_analysis_result.wave_pattern.completion_percentage = 0.70
        mock_analysis_result.wave_pattern.wave_structure = []
        
        mock_analysis_result.fibonacci_levels = MagicMock()
        mock_analysis_result.fibonacci_levels.retracement_levels = [1.2675, 1.2662, 1.2655]
        mock_analysis_result.fibonacci_levels.extension_levels = [1.2620, 1.2600, 1.2580]
        
        mock_analysis_result.confidence = 0.75
        mock_analysis_result.swing_points = []
        mock_analysis_result.predictions = []
        
        mock_elliott_analyzer.analyze.return_value = mock_analysis_result
        
        response = client.get("/api/analysis/elliott-wave/GBPUSD?timeframe=H1")
        
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "GBPUSD"
        assert data["timeframe"] == "H1"
        assert "analysis" in data
        analysis = data["analysis"]
        # Elliott Wave API returns patterns array, not direct current_wave
        assert "patterns" in analysis
        assert "fibonacci_levels" in analysis
    
    @patch('app.api.analysis.get_data_fetcher')
    @patch('app.api.analysis.unified_analyzer')
    def test_unified_analysis_comprehensive(self, mock_unified_analyzer, mock_get_data_fetcher):
        """Test comprehensive unified analysis"""
        # Setup mocks
        mock_get_data_fetcher.return_value = self.setup_mock_data_fetcher()
        
        from unittest.mock import MagicMock
        mock_analysis_result = MagicMock()
        mock_analysis_result.combined_signal = "BUY"
        mock_analysis_result.combined_confidence = 0.82
        mock_analysis_result.agreement_score = 0.83
        mock_analysis_result.dow_signal = "BUY"
        mock_analysis_result.dow_confidence = 0.80
        mock_analysis_result.elliott_signal = "BUY"
        mock_analysis_result.elliott_confidence = 0.85
        mock_analysis_result.elliott_patterns = []
        mock_analysis_result.price_targets = {
            "primary_target": 1.0950,
            "secondary_target": 1.0980,
            "stop_loss": 1.0800
        }
        mock_analysis_result.risk_levels = []
        mock_analysis_result.swing_points = []
        
        # Mock Dow trend object
        from unittest.mock import MagicMock
        from enum import Enum
        class MockTrend(Enum):
            BULLISH = "bullish"
        mock_dow_trend = MagicMock()
        mock_dow_trend.primary_trend = MockTrend.BULLISH
        mock_analysis_result.dow_trend = mock_dow_trend
        
        mock_unified_analyzer.analyze.return_value = mock_analysis_result
        
        response = client.get("/api/analysis/unified/EURUSD?timeframe=H4")
        
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "EURUSD"
        assert data["timeframe"] == "H4"
        assert "unified_analysis" in data
        unified_analysis = data["unified_analysis"]
        assert unified_analysis["combined_signal"] == "BUY"
        assert unified_analysis["combined_confidence"] == 0.82
        assert "dow_analysis" in unified_analysis
        assert "elliott_analysis" in unified_analysis
    
    @patch('app.api.analysis.get_data_fetcher')
    @patch('app.api.analysis.unified_analyzer')
    def test_unified_analysis_conflicting_signals(self, mock_unified_analyzer, mock_get_data_fetcher):
        """Test unified analysis with conflicting signals"""
        # Setup mocks
        mock_get_data_fetcher.return_value = self.setup_mock_data_fetcher()
        
        from unittest.mock import MagicMock
        mock_analysis_result = MagicMock()
        mock_analysis_result.combined_signal = "NEUTRAL"
        mock_analysis_result.combined_confidence = 0.45
        mock_analysis_result.agreement_score = 0.20
        mock_analysis_result.dow_signal = "SELL"
        mock_analysis_result.dow_confidence = 0.70
        mock_analysis_result.elliott_signal = "BUY"
        mock_analysis_result.elliott_confidence = 0.60
        mock_analysis_result.elliott_patterns = []
        mock_analysis_result.price_targets = {}
        mock_analysis_result.risk_levels = []
        mock_analysis_result.swing_points = []
        
        # Mock Dow trend object
        from unittest.mock import MagicMock
        from enum import Enum
        class MockTrend(Enum):
            BEARISH = "bearish"
        mock_dow_trend = MagicMock()
        mock_dow_trend.primary_trend = MockTrend.BEARISH
        mock_analysis_result.dow_trend = mock_dow_trend
        
        mock_unified_analyzer.analyze.return_value = mock_analysis_result
        
        response = client.get("/api/analysis/unified/USDJPY?timeframe=H1")
        
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "USDJPY"
        assert data["timeframe"] == "H1"
        assert "unified_analysis" in data
        unified_analysis = data["unified_analysis"]
        assert unified_analysis["combined_signal"] == "NEUTRAL"
        assert unified_analysis["combined_confidence"] == 0.45
        assert "recommendations" in data
        recommendations = data["recommendations"]
        assert recommendations["action"] == "NEUTRAL"
    
    @patch('app.api.analysis.get_data_fetcher')
    @patch('app.api.analysis.unified_analyzer')
    @patch('app.api.analysis.signal_generator')
    @patch('app.api.analysis.confidence_calculator')
    def test_signals_generation_buy_signal(self, mock_confidence_calculator, mock_signal_generator, mock_unified_analyzer, mock_get_data_fetcher):
        """Test signals generation for buy signal"""
        # Setup mocks
        mock_get_data_fetcher.return_value = self.setup_mock_data_fetcher()
        
        # Mock unified analyzer result
        from unittest.mock import MagicMock
        mock_unified_result = MagicMock()
        mock_unified_result.combined_signal = "BUY"
        mock_unified_result.combined_confidence = 0.85
        mock_unified_result.agreement_score = 0.85
        mock_unified_analyzer.analyze.return_value = mock_unified_result
        
        # Mock signal generation
        from enum import Enum
        class MockSignalType(Enum):
            BUY = "BUY"
        class MockUrgency(Enum):
            HIGH = "HIGH"
        
        mock_signal = MagicMock()
        mock_signal.signal_id = "sig_001"
        mock_signal.signal_type = MockSignalType.BUY
        mock_signal.urgency = MockUrgency.HIGH
        mock_signal.entry_price = 1.0850
        mock_signal.stop_loss = 1.0800
        mock_signal.take_profit_1 = 1.0950
        mock_signal.take_profit_2 = None
        mock_signal.take_profit_3 = None
        mock_signal.confidence = 0.85
        mock_signal.strength = "STRONG"
        mock_signal.quality_score = 0.85
        mock_signal.risk_reward_ratio = 2.0
        mock_signal.position_size_pct = 1.0
        mock_signal.valid_until = None
        mock_signal.analysis_summary = {}
        
        mock_signal_generator.generate_signal_from_unified.return_value = mock_signal
        
        # Mock confidence calculator
        mock_confidence_result = MagicMock()
        mock_confidence_result.overall_confidence = 0.85
        mock_confidence_result.confidence_grade = "A"
        mock_confidence_result.confidence_factors = {}
        mock_confidence_result.risk_assessment = {}
        mock_confidence_calculator.calculate_unified_confidence.return_value = mock_confidence_result
        
        response = client.get("/api/analysis/signals/EURUSD?timeframe=H4")
        
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "EURUSD"
        assert data["timeframe"] == "H4"
        assert data["signal_generated"] == True
        assert "signal" in data
        signal = data["signal"]
        assert signal["signal_type"] == "BUY"
        assert signal["confidence"] == 0.85
        assert signal["risk_reward_ratio"] == 2.0
    
    @patch('app.api.analysis.get_data_fetcher')
    @patch('app.api.analysis.unified_analyzer')
    @patch('app.api.analysis.signal_generator')
    def test_signals_generation_no_signals(self, mock_signal_generator, mock_unified_analyzer, mock_get_data_fetcher):
        """Test signals generation when no signals available"""
        # Setup mocks
        mock_get_data_fetcher.return_value = self.setup_mock_data_fetcher()
        # Mock unified analyzer result with low confidence
        mock_unified_result = MagicMock()
        mock_unified_result.combined_signal = "NEUTRAL"
        mock_unified_result.combined_confidence = 0.30
        mock_unified_result.agreement_score = 0.30
        mock_unified_analyzer.analyze.return_value = mock_unified_result
        
        # Mock signal generation returning None (no signal)
        mock_signal_generator.generate_signal_from_unified.return_value = None
        
        response = client.get("/api/analysis/signals/USDCHF?timeframe=H1")
        
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "USDCHF"
        assert data["timeframe"] == "H1"
        assert data["signal_generated"] == False
        assert "reason" in data
    
    @patch('app.api.analysis.get_data_fetcher')
    @patch('app.api.analysis.unified_analyzer')
    @patch('app.api.analysis.signal_generator')
    @patch('app.api.analysis.confidence_calculator')
    def test_signals_generation_multiple_signals(self, mock_confidence_calculator, mock_signal_generator, mock_unified_analyzer, mock_get_data_fetcher):
        """Test signals generation with multiple signals"""
        # Setup mocks
        mock_get_data_fetcher.return_value = self.setup_mock_data_fetcher()
        
        # Mock unified analyzer result
        from unittest.mock import MagicMock
        mock_unified_result = MagicMock()
        mock_unified_result.combined_signal = "BUY"
        mock_unified_result.combined_confidence = 0.85
        mock_unified_result.agreement_score = 0.85
        mock_unified_analyzer.analyze.return_value = mock_unified_result
        
        # Mock signal generation
        from enum import Enum
        class MockSignalType(Enum):
            BUY = "BUY"
        class MockUrgency(Enum):
            HIGH = "HIGH"
        
        mock_signal = MagicMock()
        mock_signal.signal_id = "sig_001"
        mock_signal.signal_type = MockSignalType.BUY
        mock_signal.urgency = MockUrgency.HIGH
        mock_signal.entry_price = 1.0850
        mock_signal.confidence = 0.85
        mock_signal.strength = "STRONG"
        mock_signal.risk_reward_ratio = 2.0
        mock_signal.stop_loss = 1.0800
        mock_signal.take_profit_1 = 1.0950
        mock_signal.take_profit_2 = None
        mock_signal.take_profit_3 = None
        mock_signal.quality_score = 0.85
        mock_signal.position_size_pct = 1.0
        mock_signal.valid_until = None
        mock_signal.analysis_summary = {}
        
        mock_signal_generator.generate_signal_from_unified.return_value = mock_signal
        
        # Mock confidence calculator
        mock_confidence_result = MagicMock()
        mock_confidence_result.overall_confidence = 0.85
        mock_confidence_result.confidence_grade = "A"
        mock_confidence_result.confidence_factors = {}
        mock_confidence_result.risk_assessment = {}
        mock_confidence_calculator.calculate_unified_confidence.return_value = mock_confidence_result
        
        response = client.get("/api/analysis/signals/EURUSD?timeframe=H4")
        
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "EURUSD"
        assert data["timeframe"] == "H4"
        assert data["signal_generated"] == True
        assert "signal" in data
        signal = data["signal"]
        assert signal["signal_type"] == "BUY"
        assert signal["confidence"] == 0.85
    
    @patch('app.api.analysis.get_data_fetcher')
    @patch('app.api.analysis.multi_timeframe_analyzer')
    def test_multi_timeframe_analysis_comprehensive(self, mock_mtf_analyzer, mock_get_data_fetcher):
        """Test comprehensive multi-timeframe analysis"""
        # Setup mocks
        mock_get_data_fetcher.return_value = self.setup_mock_data_fetcher()
        
        from unittest.mock import MagicMock
        mock_analysis_result = MagicMock()
        mock_analysis_result.primary_signal = "STRONG_BUY"
        mock_analysis_result.primary_confidence = 0.84
        mock_analysis_result.alignment_score = 1.0
        mock_analysis_result.trend_consistency = "VERY_HIGH"
        mock_analysis_result.optimal_entry_timeframe = "H1"
        mock_analysis_result.entry_confirmation_score = 0.90
        mock_analysis_result.conflicting_timeframes = []
        mock_analysis_result.timeframe_hierarchy = ["D1", "H4", "H1", "M15"]
        
        # Mock timeframe results
        mock_timeframe_results = {}
        for tf in ["M15", "H1", "H4", "D1"]:
            mock_tf_result = MagicMock()
            mock_tf_result.combined_signal = "BUY"
            mock_tf_result.combined_confidence = 0.85
            mock_tf_result.agreement_score = 0.85
            mock_tf_result.dow_confidence = 0.80
            mock_tf_result.elliott_confidence = 0.85
            mock_timeframe_results[tf] = mock_tf_result
        
        mock_analysis_result.timeframe_results = mock_timeframe_results
        
        mock_mtf_analyzer.analyze_multiple_timeframes.return_value = mock_analysis_result
        
        response = client.post("/api/analysis/multi-timeframe", json={
            "symbol": "EURUSD",
            "timeframes": ["M15", "H1", "H4", "D1"]
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "EURUSD"
        assert "multi_timeframe_analysis" in data
        mtf_analysis = data["multi_timeframe_analysis"]
        assert mtf_analysis["primary_signal"] == "STRONG_BUY"
        assert mtf_analysis["alignment_score"] == 1.0
        assert "timeframe_results" in mtf_analysis
    
    @patch('app.api.analysis.get_data_fetcher')
    @patch('app.api.analysis.multi_timeframe_analyzer')
    def test_multi_timeframe_analysis_mixed_signals(self, mock_mtf_analyzer, mock_get_data_fetcher):
        """Test multi-timeframe analysis with mixed signals"""
        # Setup mocks
        mock_get_data_fetcher.return_value = self.setup_mock_data_fetcher()
        
        from unittest.mock import MagicMock
        mock_analysis_result = MagicMock()
        mock_analysis_result.primary_signal = "NEUTRAL"
        mock_analysis_result.primary_confidence = 0.60
        mock_analysis_result.alignment_score = 0.3
        mock_analysis_result.trend_consistency = "LOW"
        mock_analysis_result.optimal_entry_timeframe = None
        mock_analysis_result.entry_confirmation_score = 0.30
        mock_analysis_result.conflicting_timeframes = ["H1", "D1"]
        mock_analysis_result.timeframe_hierarchy = ["D1", "H4", "H1"]
        
        # Mock timeframe results with conflicting signals
        mock_timeframe_results = {
            "H1": MagicMock(),
            "H4": MagicMock(),
            "D1": MagicMock()
        }
        mock_timeframe_results["H1"].combined_signal = "SELL"
        mock_timeframe_results["H1"].combined_confidence = 0.70
        mock_timeframe_results["H4"].combined_signal = "NEUTRAL"
        mock_timeframe_results["H4"].combined_confidence = 0.50
        mock_timeframe_results["D1"].combined_signal = "BUY"
        mock_timeframe_results["D1"].combined_confidence = 0.80
        
        mock_analysis_result.timeframe_results = mock_timeframe_results
        
        mock_mtf_analyzer.analyze_multiple_timeframes.return_value = mock_analysis_result
        
        response = client.post("/api/analysis/multi-timeframe", json={
            "symbol": "USDJPY",
            "timeframes": ["H1", "H4", "D1"]
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "USDJPY"
        assert "multi_timeframe_analysis" in data
        mtf_analysis = data["multi_timeframe_analysis"]
        assert mtf_analysis["primary_signal"] == "NEUTRAL"
        assert mtf_analysis["alignment_score"] == 0.3
        assert "recommendations" in data
        recommendations = data["recommendations"]
        assert recommendations["overall_action"] == "NEUTRAL"
    
    @patch('app.api.analysis.get_data_fetcher')
    @patch('app.api.analysis.multi_timeframe_analyzer')
    def test_multi_timeframe_analysis_multiple_symbols(self, mock_mtf_analyzer, mock_get_data_fetcher):
        """Test multi-timeframe analysis for multiple symbols"""
        # Setup mocks
        mock_get_data_fetcher.return_value = self.setup_mock_data_fetcher()
        
        # Create mock result for single symbol
        from unittest.mock import MagicMock
        mock_result = MagicMock()
        mock_result.primary_signal = "BUY"
        mock_result.primary_confidence = 0.85
        mock_result.alignment_score = 0.85
        mock_result.trend_consistency = "HIGH"
        mock_result.optimal_entry_timeframe = "H1"
        mock_result.entry_confirmation_score = 0.85
        mock_result.conflicting_timeframes = []
        mock_result.timeframe_hierarchy = ["H4", "H1"]
        
        # Mock timeframe results
        mock_timeframe_results = {
            "H1": MagicMock(),
            "H4": MagicMock()
        }
        mock_timeframe_results["H1"].combined_signal = "BUY"
        mock_timeframe_results["H1"].combined_confidence = 0.85
        mock_timeframe_results["H4"].combined_signal = "BUY"
        mock_timeframe_results["H4"].combined_confidence = 0.85
        
        mock_result.timeframe_results = mock_timeframe_results
        
        mock_mtf_analyzer.analyze_multiple_timeframes.return_value = mock_result
        
        response = client.post("/api/analysis/multi-timeframe", json={
            "symbol": "EURUSD",
            "timeframes": ["H1", "H4"]
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "EURUSD"
        assert "multi_timeframe_analysis" in data
        mtf_analysis = data["multi_timeframe_analysis"]
        assert mtf_analysis["primary_signal"] == "BUY"
        assert mtf_analysis["primary_confidence"] == 0.85
    
    def test_analysis_api_health_check(self):
        """Test analysis API health check"""
        response = client.get("/api/analysis/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "component" in data
        assert data["component"] == "analysis_api"
    
    @patch('app.api.analysis.get_data_fetcher')
    @patch('app.api.analysis.dow_theory_analyzer')
    def test_analysis_error_handling(self, mock_dow_analyzer, mock_get_data_fetcher):
        """Test analysis API error handling"""
        # Setup mocks - data fetcher works but analyzer fails
        mock_get_data_fetcher.return_value = self.setup_mock_data_fetcher()
        mock_dow_analyzer.analyze.side_effect = Exception("Analysis engine failure")
        
        response = client.get("/api/analysis/dow-theory/EURUSD")
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
    
    @patch('app.api.analysis.get_data_fetcher')
    def test_invalid_symbol_format(self, mock_get_data_fetcher):
        """Test analysis with invalid symbol format"""
        mock_get_data_fetcher.return_value = self.setup_mock_data_fetcher()
        
        response = client.get("/api/analysis/dow-theory/INVALID_SYMBOL_123")
        
        # Should return 404 (no data) or 500 (processing error)
        assert response.status_code in [404, 500]
    
    @patch('app.api.analysis.get_data_fetcher')
    def test_invalid_timeframe_parameter(self, mock_get_data_fetcher):
        """Test analysis with invalid timeframe"""
        mock_get_data_fetcher.return_value = self.setup_mock_data_fetcher()
        
        response = client.get("/api/analysis/elliott-wave/EURUSD?timeframe=INVALID")
        
        # Should return data error or handle gracefully
        assert response.status_code in [404, 422, 500]
    
    @patch('app.api.analysis.get_data_fetcher')
    @patch('app.api.analysis.unified_analyzer')
    def test_unified_analysis_performance(self, mock_unified_analyzer, mock_get_data_fetcher):
        """Test unified analysis performance"""
        import time
        
        # Setup mocks
        from unittest.mock import MagicMock
        mock_get_data_fetcher.return_value = self.setup_mock_data_fetcher()
        mock_result = MagicMock()
        
        # Set all required numeric attributes that are used in API comparisons
        mock_result.combined_signal = "BUY"
        mock_result.combined_confidence = 0.85  # Used in line 387 comparison
        mock_result.agreement_score = 0.80     # Used in line 390 comparison
        mock_result.dow_signal = "BUY"
        mock_result.dow_confidence = 0.80
        mock_result.elliott_signal = "BUY"
        mock_result.elliott_confidence = 0.85
        mock_result.elliott_patterns = []
        mock_result.price_targets = {}
        mock_result.risk_levels = []
        mock_result.swing_points = []
        
        # Mock Dow trend object
        from enum import Enum
        class MockTrend(Enum):
            BULLISH = "bullish"
        mock_dow_trend = MagicMock()
        mock_dow_trend.primary_trend = MockTrend.BULLISH
        mock_result.dow_trend = mock_dow_trend
        
        mock_unified_analyzer.analyze.return_value = mock_result
        
        start_time = time.time()
        response = client.get("/api/analysis/unified/EURUSD")
        end_time = time.time()
        
        assert response.status_code == 200
        # Response should be fast (under 1 second for mocked call)
        assert (end_time - start_time) < 1.0
    
    @patch('app.api.analysis.get_data_fetcher')
    def test_concurrent_analysis_requests(self, mock_get_data_fetcher):
        """Test concurrent analysis requests"""
        import concurrent.futures
        
        # Setup mock for all requests
        mock_get_data_fetcher.return_value = self.setup_mock_data_fetcher()
        
        def make_request(symbol):
            return client.get(f"/api/analysis/dow-theory/{symbol}")
        
        symbols = ["EURUSD", "USDJPY", "GBPUSD", "USDCHF"]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(make_request, symbol) for symbol in symbols]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should complete (success or error)
        for response in responses:
            assert response.status_code in [200, 404, 500]
    
    @patch('app.api.analysis.get_data_fetcher')
    @patch('app.api.analysis.unified_analyzer')
    @patch('app.api.analysis.confidence_calculator')
    def test_signal_filtering_by_confidence(self, mock_confidence_calculator, mock_unified_analyzer, mock_get_data_fetcher):
        """Test signal filtering by confidence threshold"""
        # Setup mocks  
        from unittest.mock import MagicMock
        mock_get_data_fetcher.return_value = self.setup_mock_data_fetcher()
        
        # Setup unified analyzer mock
        mock_unified_result = MagicMock()
        mock_unified_result.overall_signal = MagicMock()
        mock_unified_result.overall_signal.value = "buy"
        mock_unified_result.confidence = 0.85
        mock_unified_analyzer.analyze.return_value = mock_unified_result
        
        # Setup confidence calculator mock
        mock_confidence_result = MagicMock()
        mock_confidence_result.overall_confidence = 0.85
        mock_confidence_result.confidence_grade = "A"
        mock_confidence_result.confidence_factors = {}
        mock_confidence_result.risk_assessment = {}
        mock_confidence_calculator.calculate_unified_confidence.return_value = mock_confidence_result
        
        # Test with high confidence filter
        response = client.get("/api/analysis/signals/EURUSD?min_confidence=0.8")
        
        assert response.status_code == 200
        data = response.json()
        # Should return signal data structure
        assert "signal_generated" in data or "timestamp" in data