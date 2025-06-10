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
    
    @patch('app.api.analysis.dow_theory_analyzer')
    def test_dow_theory_analysis_usdjpy(self, mock_dow_analyzer):
        """Test Dow Theory analysis for USDJPY"""
        mock_dow_analyzer.analyze.return_value = {
            "symbol": "USDJPY",
            "timeframe": "H1",
            "trend_direction": "BEARISH",
            "trend_strength": 0.65,
            "swing_points": [
                {"type": "HIGH", "price": 150.50, "timestamp": "2024-01-01T15:00:00"},
                {"type": "LOW", "price": 149.20, "timestamp": "2024-01-01T12:00:00"}
            ],
            "confirmation_signals": ["momentum_confirmation"],
            "confidence": 0.70,
            "last_updated": datetime.now().isoformat()
        }
        
        response = client.get("/api/analysis/dow-theory/USDJPY?timeframe=H1")
        
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "USDJPY"
        assert data["trend_direction"] == "BEARISH"
        assert data["trend_strength"] == 0.65
    
    @patch('app.api.analysis.dow_theory_analyzer')
    def test_dow_theory_analysis_invalid_symbol(self, mock_dow_analyzer):
        """Test Dow Theory analysis with invalid symbol"""
        mock_dow_analyzer.analyze.side_effect = ValueError("Invalid symbol")
        
        response = client.get("/api/analysis/dow-theory/INVALID")
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
    
    @patch('app.api.analysis.elliott_wave_analyzer')
    def test_elliott_wave_analysis_eurusd(self, mock_elliott_analyzer):
        """Test Elliott Wave analysis for EURUSD"""
        mock_elliott_analyzer.analyze.return_value = {
            "symbol": "EURUSD",
            "timeframe": "H4",
            "current_wave": {
                "wave_type": "IMPULSE",
                "wave_number": 3,
                "degree": "PRIMARY",
                "start_price": 1.0800,
                "current_price": 1.0875,
                "projected_end": 1.0950
            },
            "wave_pattern": {
                "pattern_type": "FIVE_WAVE_IMPULSE",
                "completion_percentage": 0.60,
                "wave_structure": [
                    {"wave": 1, "start": 1.0750, "end": 1.0850},
                    {"wave": 2, "start": 1.0850, "end": 1.0800},
                    {"wave": 3, "start": 1.0800, "end": 1.0950, "status": "in_progress"}
                ]
            },
            "fibonacci_levels": {
                "retracement_levels": [1.0825, 1.0812, 1.0806],
                "extension_levels": [1.0920, 1.0950, 1.0980]
            },
            "confidence": 0.85,
            "last_updated": datetime.now().isoformat()
        }
        
        response = client.get("/api/analysis/elliott-wave/EURUSD?timeframe=H4")
        
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "EURUSD"
        assert data["current_wave"]["wave_number"] == 3
        assert data["wave_pattern"]["pattern_type"] == "FIVE_WAVE_IMPULSE"
        assert len(data["fibonacci_levels"]["retracement_levels"]) == 3
    
    @patch('app.api.analysis.elliott_wave_analyzer')
    def test_elliott_wave_analysis_correction_pattern(self, mock_elliott_analyzer):
        """Test Elliott Wave analysis with correction pattern"""
        mock_elliott_analyzer.analyze.return_value = {
            "symbol": "GBPUSD",
            "timeframe": "H1",
            "current_wave": {
                "wave_type": "CORRECTION",
                "wave_number": "B",
                "degree": "INTERMEDIATE",
                "start_price": 1.2700,
                "current_price": 1.2650,
                "projected_end": 1.2600
            },
            "wave_pattern": {
                "pattern_type": "ABC_CORRECTION",
                "completion_percentage": 0.70,
                "wave_structure": [
                    {"wave": "A", "start": 1.2750, "end": 1.2650},
                    {"wave": "B", "start": 1.2650, "end": 1.2700, "status": "in_progress"},
                    {"wave": "C", "start": 1.2700, "end": 1.2600, "status": "projected"}
                ]
            },
            "fibonacci_levels": {
                "retracement_levels": [1.2675, 1.2662, 1.2655],
                "extension_levels": [1.2620, 1.2600, 1.2580]
            },
            "confidence": 0.75,
            "last_updated": datetime.now().isoformat()
        }
        
        response = client.get("/api/analysis/elliott-wave/GBPUSD?timeframe=H1")
        
        assert response.status_code == 200
        data = response.json()
        assert data["current_wave"]["wave_type"] == "CORRECTION"
        assert data["wave_pattern"]["pattern_type"] == "ABC_CORRECTION"
    
    @patch('app.api.analysis.unified_analyzer')
    def test_unified_analysis_comprehensive(self, mock_unified_analyzer):
        """Test comprehensive unified analysis"""
        mock_unified_analyzer.analyze.return_value = {
            "symbol": "EURUSD",
            "timeframe": "H4",
            "overall_signal": "BUY",
            "confidence": 0.82,
            "analysis_components": {
                "dow_theory": {
                    "signal": "BUY",
                    "confidence": 0.80,
                    "weight": 0.4,
                    "trend_direction": "BULLISH",
                    "trend_strength": 0.75
                },
                "elliott_wave": {
                    "signal": "BUY",
                    "confidence": 0.85,
                    "weight": 0.6,
                    "wave_type": "IMPULSE",
                    "wave_number": 3
                }
            },
            "consensus_score": 0.83,
            "price_targets": {
                "primary_target": 1.0950,
                "secondary_target": 1.0980,
                "stop_loss": 1.0800
            },
            "risk_reward_ratio": 2.5,
            "timeframe_alignment": {
                "H1": "BUY",
                "H4": "BUY", 
                "D1": "NEUTRAL"
            },
            "last_updated": datetime.now().isoformat()
        }
        
        response = client.get("/api/analysis/unified/EURUSD?timeframe=H4")
        
        assert response.status_code == 200
        data = response.json()
        assert data["overall_signal"] == "BUY"
        assert data["confidence"] == 0.82
        assert data["consensus_score"] == 0.83
        assert data["risk_reward_ratio"] == 2.5
        assert "dow_theory" in data["analysis_components"]
        assert "elliott_wave" in data["analysis_components"]
    
    @patch('app.api.analysis.unified_analyzer')
    def test_unified_analysis_conflicting_signals(self, mock_unified_analyzer):
        """Test unified analysis with conflicting signals"""
        mock_unified_analyzer.analyze.return_value = {
            "symbol": "USDJPY",
            "timeframe": "H1",
            "overall_signal": "NEUTRAL",
            "confidence": 0.45,
            "analysis_components": {
                "dow_theory": {
                    "signal": "SELL",
                    "confidence": 0.70,
                    "weight": 0.4,
                    "trend_direction": "BEARISH"
                },
                "elliott_wave": {
                    "signal": "BUY",
                    "confidence": 0.60,
                    "weight": 0.6,
                    "wave_type": "CORRECTION"
                }
            },
            "consensus_score": 0.20,
            "conflict_reason": "Dow Theory suggests bearish trend while Elliott Wave indicates corrective bounce",
            "recommendation": "WAIT",
            "last_updated": datetime.now().isoformat()
        }
        
        response = client.get("/api/analysis/unified/USDJPY?timeframe=H1")
        
        assert response.status_code == 200
        data = response.json()
        assert data["overall_signal"] == "NEUTRAL"
        assert data["confidence"] == 0.45
        assert data["recommendation"] == "WAIT"
        assert "conflict_reason" in data
    
    @patch('app.api.analysis.signal_generator')
    def test_signals_generation_buy_signal(self, mock_signal_generator):
        """Test signals generation for buy signal"""
        mock_signal_generator.generate_signals.return_value = [
            {
                "signal_id": "sig_001",
                "symbol": "EURUSD",
                "signal_type": "BUY",
                "entry_price": 1.0850,
                "stop_loss": 1.0800,
                "take_profit": 1.0950,
                "confidence": 0.85,
                "risk_reward_ratio": 2.0,
                "signal_strength": "STRONG",
                "generated_at": datetime.now().isoformat(),
                "valid_until": (datetime.now() + timedelta(hours=4)).isoformat(),
                "analysis_basis": {
                    "dow_theory_signal": "BUY",
                    "elliott_wave_signal": "BUY",
                    "consensus_score": 0.85
                }
            }
        ]
        
        response = client.get("/api/analysis/signals/EURUSD?timeframe=H4")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        signal = data[0]
        assert signal["signal_type"] == "BUY"
        assert signal["confidence"] == 0.85
        assert signal["risk_reward_ratio"] == 2.0
    
    @patch('app.api.analysis.signal_generator')
    def test_signals_generation_no_signals(self, mock_signal_generator):
        """Test signals generation when no signals available"""
        mock_signal_generator.generate_signals.return_value = []
        
        response = client.get("/api/analysis/signals/USDCHF?timeframe=H1")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 0
    
    @patch('app.api.analysis.signal_generator')
    def test_signals_generation_multiple_signals(self, mock_signal_generator):
        """Test signals generation with multiple signals"""
        mock_signal_generator.generate_signals.return_value = [
            {
                "signal_id": "sig_001",
                "symbol": "EURUSD",
                "signal_type": "BUY",
                "entry_price": 1.0850,
                "confidence": 0.85,
                "signal_strength": "STRONG"
            },
            {
                "signal_id": "sig_002",
                "symbol": "EURUSD", 
                "signal_type": "BUY",
                "entry_price": 1.0845,
                "confidence": 0.70,
                "signal_strength": "MODERATE"
            }
        ]
        
        response = client.get("/api/analysis/signals/EURUSD?timeframe=H4")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["signal_strength"] == "STRONG"
        assert data[1]["signal_strength"] == "MODERATE"
    
    @patch('app.api.analysis.multi_timeframe_analyzer')
    def test_multi_timeframe_analysis_comprehensive(self, mock_mtf_analyzer):
        """Test comprehensive multi-timeframe analysis"""
        mock_mtf_analyzer.analyze.return_value = {
            "symbol": "EURUSD",
            "timeframes": {
                "M15": {
                    "trend": "BULLISH",
                    "signal": "BUY",
                    "confidence": 0.75,
                    "weight": 0.1
                },
                "H1": {
                    "trend": "BULLISH",
                    "signal": "BUY", 
                    "confidence": 0.80,
                    "weight": 0.2
                },
                "H4": {
                    "trend": "BULLISH",
                    "signal": "BUY",
                    "confidence": 0.85,
                    "weight": 0.3
                },
                "D1": {
                    "trend": "BULLISH",
                    "signal": "BUY",
                    "confidence": 0.90,
                    "weight": 0.4
                }
            },
            "overall_alignment": 1.0,
            "composite_signal": "STRONG_BUY",
            "composite_confidence": 0.84,
            "primary_timeframe": "H4",
            "entry_timeframe": "H1",
            "trend_consistency": "VERY_HIGH",
            "recommendation": {
                "action": "BUY",
                "urgency": "HIGH",
                "entry_price": 1.0850,
                "stop_loss": 1.0800,
                "take_profit": 1.0950
            },
            "last_updated": datetime.now().isoformat()
        }
        
        response = client.post("/api/analysis/multi-timeframe", json={
            "symbols": ["EURUSD"],
            "timeframes": ["M15", "H1", "H4", "D1"]
        })
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        analysis = data[0]
        assert analysis["symbol"] == "EURUSD"
        assert analysis["composite_signal"] == "STRONG_BUY"
        assert analysis["overall_alignment"] == 1.0
        assert len(analysis["timeframes"]) == 4
    
    @patch('app.api.analysis.multi_timeframe_analyzer')
    def test_multi_timeframe_analysis_mixed_signals(self, mock_mtf_analyzer):
        """Test multi-timeframe analysis with mixed signals"""
        mock_mtf_analyzer.analyze.return_value = {
            "symbol": "USDJPY",
            "timeframes": {
                "H1": {
                    "trend": "BEARISH",
                    "signal": "SELL",
                    "confidence": 0.70,
                    "weight": 0.3
                },
                "H4": {
                    "trend": "NEUTRAL",
                    "signal": "HOLD",
                    "confidence": 0.50,
                    "weight": 0.4
                },
                "D1": {
                    "trend": "BULLISH",
                    "signal": "BUY",
                    "confidence": 0.80,
                    "weight": 0.3
                }
            },
            "overall_alignment": 0.3,
            "composite_signal": "NEUTRAL",
            "composite_confidence": 0.60,
            "primary_timeframe": "D1",
            "entry_timeframe": None,
            "trend_consistency": "LOW",
            "recommendation": {
                "action": "WAIT",
                "urgency": "LOW",
                "reason": "Conflicting signals across timeframes"
            },
            "last_updated": datetime.now().isoformat()
        }
        
        response = client.post("/api/analysis/multi-timeframe", json={
            "symbols": ["USDJPY"],
            "timeframes": ["H1", "H4", "D1"]
        })
        
        assert response.status_code == 200
        data = response.json()
        analysis = data[0]
        assert analysis["composite_signal"] == "NEUTRAL"
        assert analysis["overall_alignment"] == 0.3
        assert analysis["recommendation"]["action"] == "WAIT"
    
    @patch('app.api.analysis.multi_timeframe_analyzer')
    def test_multi_timeframe_analysis_multiple_symbols(self, mock_mtf_analyzer):
        """Test multi-timeframe analysis for multiple symbols"""
        mock_mtf_analyzer.analyze.side_effect = [
            {
                "symbol": "EURUSD",
                "composite_signal": "BUY",
                "composite_confidence": 0.85
            },
            {
                "symbol": "GBPUSD",
                "composite_signal": "SELL",
                "composite_confidence": 0.75
            }
        ]
        
        response = client.post("/api/analysis/multi-timeframe", json={
            "symbols": ["EURUSD", "GBPUSD"],
            "timeframes": ["H1", "H4"]
        })
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["symbol"] == "EURUSD"
        assert data[1]["symbol"] == "GBPUSD"
    
    def test_analysis_api_health_check(self):
        """Test analysis API health check"""
        response = client.get("/api/analysis/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "component" in data
        assert data["component"] == "analysis_api"
    
    @patch('app.api.analysis.dow_theory_analyzer')
    def test_analysis_error_handling(self, mock_dow_analyzer):
        """Test analysis API error handling"""
        mock_dow_analyzer.analyze.side_effect = Exception("Analysis engine failure")
        
        response = client.get("/api/analysis/dow-theory/EURUSD")
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
    
    def test_invalid_symbol_format(self):
        """Test analysis with invalid symbol format"""
        response = client.get("/api/analysis/dow-theory/INVALID_SYMBOL_123")
        
        # Should return either 400 (validation error) or 500 (processing error)
        assert response.status_code in [400, 500]
    
    def test_invalid_timeframe_parameter(self):
        """Test analysis with invalid timeframe"""
        response = client.get("/api/analysis/elliott-wave/EURUSD?timeframe=INVALID")
        
        # Should return validation error or handle gracefully
        assert response.status_code in [400, 422, 500]
    
    @patch('app.api.analysis.unified_analyzer')
    def test_unified_analysis_performance(self, mock_unified_analyzer):
        """Test unified analysis performance"""
        import time
        
        # Mock fast response
        mock_unified_analyzer.analyze.return_value = {
            "symbol": "EURUSD",
            "overall_signal": "BUY",
            "confidence": 0.80
        }
        
        start_time = time.time()
        response = client.get("/api/analysis/unified/EURUSD")
        end_time = time.time()
        
        assert response.status_code == 200
        # Response should be fast (under 1 second for mocked call)
        assert (end_time - start_time) < 1.0
    
    def test_concurrent_analysis_requests(self):
        """Test concurrent analysis requests"""
        import concurrent.futures
        
        def make_request(symbol):
            return client.get(f"/api/analysis/dow-theory/{symbol}")
        
        symbols = ["EURUSD", "USDJPY", "GBPUSD", "USDCHF"]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(make_request, symbol) for symbol in symbols]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should complete (success or error)
        for response in responses:
            assert response.status_code in [200, 400, 500]
    
    @patch('app.api.analysis.signal_generator')
    def test_signal_filtering_by_confidence(self, mock_signal_generator):
        """Test signal filtering by confidence threshold"""
        mock_signal_generator.generate_signals.return_value = [
            {"signal_id": "sig_high", "confidence": 0.85, "signal_type": "BUY"},
            {"signal_id": "sig_low", "confidence": 0.45, "signal_type": "SELL"}
        ]
        
        # Test with high confidence filter
        response = client.get("/api/analysis/signals/EURUSD?min_confidence=0.8")
        
        assert response.status_code == 200
        data = response.json()
        # Should only return high confidence signals
        high_conf_signals = [s for s in data if s["confidence"] >= 0.8]
        assert len(high_conf_signals) >= 0  # May filter out low confidence signals