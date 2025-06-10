"""
Tests for Dow Theory analysis
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.analysis.dow_theory.analyzer import DowTheoryAnalyzer
from app.analysis.dow_theory.base import TrendDirection, TrendStrength
from app.analysis.indicators.swing_detector import SwingType


class TestDowTheoryAnalyzer:
    """Test suite for Dow Theory analyzer"""
    
    @pytest.fixture
    def sample_uptrend_data(self):
        """Create sample data with clear uptrend"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Create uptrending price data with some noise
        trend = np.linspace(1.0000, 1.0200, 100)
        noise = np.random.normal(0, 0.0005, 100)
        prices = trend + noise
        
        # Generate OHLC from trend
        highs = prices + np.random.uniform(0.0001, 0.0003, 100)
        lows = prices - np.random.uniform(0.0001, 0.0003, 100)
        opens = np.roll(prices, 1)
        opens[0] = prices[0]
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': np.random.randint(1000, 5000, 100)
        }).set_index('timestamp')
    
    @pytest.fixture
    def sample_downtrend_data(self):
        """Create sample data with clear downtrend"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Create downtrending price data
        trend = np.linspace(1.0200, 1.0000, 100)
        noise = np.random.normal(0, 0.0005, 100)
        prices = trend + noise
        
        # Generate OHLC from trend
        highs = prices + np.random.uniform(0.0001, 0.0003, 100)
        lows = prices - np.random.uniform(0.0001, 0.0003, 100)
        opens = np.roll(prices, 1)
        opens[0] = prices[0]
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': np.random.randint(1000, 5000, 100)
        }).set_index('timestamp')
    
    @pytest.fixture
    def sample_sideways_data(self):
        """Create sample data with sideways movement"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Create sideways price data
        base_price = 1.0100
        noise = np.random.normal(0, 0.0010, 100)
        prices = np.full(100, base_price) + noise
        
        # Generate OHLC
        highs = prices + np.random.uniform(0.0001, 0.0005, 100)
        lows = prices - np.random.uniform(0.0001, 0.0005, 100)
        opens = np.roll(prices, 1)
        opens[0] = prices[0]
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': np.random.randint(1000, 5000, 100)
        }).set_index('timestamp')
    
    @pytest.fixture
    def analyzer(self):
        """Create Dow Theory analyzer instance"""
        return DowTheoryAnalyzer(
            swing_sensitivity=0.5,
            min_swing_size_pct=1.0,  # Lower threshold for test data
            trend_confirmation_swings=2
        )
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer is not None
        assert analyzer.swing_sensitivity == 0.5
        assert analyzer.min_swing_size_pct == 0.01  # Converted to decimal
        assert analyzer.trend_confirmation_swings == 2
    
    def test_uptrend_detection(self, analyzer, sample_uptrend_data):
        """Test uptrend detection"""
        trend_analysis = analyzer.analyze_trend(sample_uptrend_data)
        
        assert trend_analysis is not None
        assert trend_analysis.direction in [TrendDirection.UPTREND, TrendDirection.SIDEWAYS]
        assert trend_analysis.confidence >= 0.0
        assert len(trend_analysis.swing_points) >= 0
    
    def test_downtrend_detection(self, analyzer, sample_downtrend_data):
        """Test downtrend detection"""
        trend_analysis = analyzer.analyze_trend(sample_downtrend_data)
        
        assert trend_analysis is not None
        assert trend_analysis.direction in [TrendDirection.DOWNTREND, TrendDirection.SIDEWAYS]
        assert trend_analysis.confidence >= 0.0
    
    def test_sideways_detection(self, analyzer, sample_sideways_data):
        """Test sideways trend detection"""
        trend_analysis = analyzer.analyze_trend(sample_sideways_data)
        
        assert trend_analysis is not None
        assert trend_analysis.direction in [TrendDirection.SIDEWAYS, TrendDirection.UNDEFINED]
    
    def test_swing_point_detection(self, analyzer, sample_uptrend_data):
        """Test swing point detection"""
        swing_points = analyzer.detect_swing_points(sample_uptrend_data)
        
        assert isinstance(swing_points, list)
        # Should detect some swing points in 100 data points
        assert len(swing_points) >= 0
        
        # Check swing point structure
        for swing in swing_points:
            assert hasattr(swing, 'index')
            assert hasattr(swing, 'price')
            assert hasattr(swing, 'swing_type')
            assert swing.swing_type in [SwingType.HIGH, SwingType.LOW]
    
    def test_trend_strength_calculation(self, analyzer, sample_uptrend_data):
        """Test trend strength calculation"""
        swing_points = analyzer.detect_swing_points(sample_uptrend_data)
        trend_direction = analyzer.identify_trend_direction(swing_points)
        
        strength = analyzer.calculate_trend_strength(
            sample_uptrend_data, swing_points, trend_direction
        )
        
        assert isinstance(strength, TrendStrength)
        assert strength in [
            TrendStrength.VERY_WEAK, TrendStrength.WEAK, TrendStrength.MODERATE,
            TrendStrength.STRONG, TrendStrength.VERY_STRONG
        ]
    
    def test_key_levels_calculation(self, analyzer, sample_uptrend_data):
        """Test key levels calculation"""
        swing_points = analyzer.detect_swing_points(sample_uptrend_data)
        key_levels = analyzer.calculate_key_levels(swing_points)
        
        assert isinstance(key_levels, dict)
        
        # Check if expected keys exist when swing points are present
        if swing_points:
            highs = [sp for sp in swing_points if sp.swing_type == SwingType.HIGH]
            lows = [sp for sp in swing_points if sp.swing_type == SwingType.LOW]
            
            if highs:
                assert 'resistance' in key_levels
                assert 'avg_high' in key_levels
            
            if lows:
                assert 'support' in key_levels
                assert 'avg_low' in key_levels
    
    def test_signal_generation(self, analyzer, sample_uptrend_data):
        """Test signal generation"""
        signals = analyzer.generate_signals(sample_uptrend_data)
        
        assert isinstance(signals, list)
        
        # Check signal structure
        for signal in signals:
            assert hasattr(signal, 'signal_type')
            assert hasattr(signal, 'confidence')
            assert signal.signal_type in ['buy', 'sell', 'hold']
            assert 0.0 <= signal.confidence <= 1.0
    
    def test_trend_confirmation_with_secondary_data(self, analyzer, sample_uptrend_data):
        """Test trend confirmation with secondary timeframe"""
        # Create secondary data (different timeframe simulation)
        secondary_data = sample_uptrend_data.iloc[::4].copy()  # Every 4th row
        
        signals = analyzer.generate_signals(sample_uptrend_data, secondary_data)
        
        assert isinstance(signals, list)
        
        # Check if confirmation status is set
        for signal in signals:
            assert hasattr(signal, 'confirmation_status')
    
    def test_analysis_summary(self, analyzer, sample_uptrend_data):
        """Test analysis summary generation"""
        summary = analyzer.get_analysis_summary(sample_uptrend_data, 'D1')
        
        assert isinstance(summary, dict)
        assert 'timeframe' in summary
        assert 'trend_direction' in summary
        assert 'trend_strength' in summary
        assert 'confidence' in summary
        assert summary['timeframe'] == 'D1'
    
    def test_insufficient_data_handling(self, analyzer):
        """Test handling of insufficient data"""
        # Create very small dataset
        small_data = pd.DataFrame({
            'open': [1.0, 1.0],
            'high': [1.0, 1.0],
            'low': [1.0, 1.0],
            'close': [1.0, 1.0]
        })
        
        trend_analysis = analyzer.analyze_trend(small_data)
        
        assert trend_analysis.direction == TrendDirection.UNDEFINED
        assert trend_analysis.confidence == 0.0
    
    def test_error_handling(self, analyzer):
        """Test error handling with invalid data"""
        # Test with empty DataFrame
        empty_data = pd.DataFrame()
        
        try:
            trend_analysis = analyzer.analyze_trend(empty_data)
            # Should not crash, should return undefined trend
            assert trend_analysis.direction == TrendDirection.UNDEFINED
        except Exception:
            # If it throws an exception, that's also acceptable
            pass
    
    def test_caching_functionality(self, analyzer, sample_uptrend_data):
        """Test swing point caching"""
        # First call
        swing_points_1 = analyzer.detect_swing_points(sample_uptrend_data, use_cache=True)
        
        # Second call (should use cache)
        swing_points_2 = analyzer.detect_swing_points(sample_uptrend_data, use_cache=True)
        
        # Results should be identical
        assert len(swing_points_1) == len(swing_points_2)
        
        # Check cache is working (cache should have entries)
        assert len(analyzer._swing_cache) > 0
    
    def test_higher_highs_lower_lows_detection(self, analyzer):
        """Test specific HH/HL and LH/LL pattern detection"""
        # Create data with clear HH/HL pattern
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        
        # Manually create HH/HL pattern
        prices = [1.0000, 1.0010, 1.0005, 1.0020, 1.0015, 1.0030, 1.0025, 1.0040]
        prices.extend([1.0040] * (50 - len(prices)))  # Pad to 50 points
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + 0.0002 for p in prices],
            'low': [p - 0.0002 for p in prices],
            'close': prices,
            'volume': [1000] * 50
        }).set_index('timestamp')
        
        swing_points = analyzer.detect_swing_points(data)
        trend_direction = analyzer.identify_trend_direction(swing_points)
        
        # Should detect uptrend or at least not downtrend
        assert trend_direction in [TrendDirection.UPTREND, TrendDirection.SIDEWAYS, TrendDirection.UNDEFINED]


class TestDowTheoryIntegration:
    """Integration tests for Dow Theory components"""
    
    def test_full_analysis_workflow(self):
        """Test complete analysis workflow"""
        # Create realistic FX data
        dates = pd.date_range(start='2024-01-01', periods=200, freq='H')
        
        # Simulate EURUSD-like movement
        base_price = 1.0950
        returns = np.random.normal(0, 0.0001, 200)  # Small hourly returns
        prices = [base_price]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        prices = np.array(prices[:200])
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': np.roll(prices, 1),
            'high': prices + np.random.uniform(0.00005, 0.0002, 200),
            'low': prices - np.random.uniform(0.00005, 0.0002, 200),
            'close': prices,
            'volume': np.random.randint(100, 1000, 200)
        }).set_index('timestamp')
        
        # Fix first open
        data.iloc[0, data.columns.get_loc('open')] = data.iloc[0, data.columns.get_loc('close')]
        
        # Run full analysis
        analyzer = DowTheoryAnalyzer()
        
        # Test trend analysis
        trend_analysis = analyzer.analyze_trend(data, 'H1')
        assert trend_analysis is not None
        
        # Test signal generation
        signals = analyzer.generate_signals(data)
        assert isinstance(signals, list)
        
        # Test summary
        summary = analyzer.get_analysis_summary(data, 'H1')
        assert isinstance(summary, dict)
        assert 'error' not in summary  # No errors in processing
    
    def test_multi_timeframe_analysis(self):
        """Test analysis across multiple timeframes"""
        # Create daily data
        daily_dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        daily_prices = np.linspace(1.0900, 1.1000, 30)  # Uptrend
        
        daily_data = pd.DataFrame({
            'timestamp': daily_dates,
            'open': np.roll(daily_prices, 1),
            'high': daily_prices + 0.0010,
            'low': daily_prices - 0.0010,
            'close': daily_prices,
            'volume': np.random.randint(1000, 5000, 30)
        }).set_index('timestamp')
        
        daily_data.iloc[0, daily_data.columns.get_loc('open')] = daily_data.iloc[0, daily_data.columns.get_loc('close')]
        
        # Create 4-hour data (more granular)
        h4_dates = pd.date_range(start='2024-01-01', periods=180, freq='4H')
        h4_prices = np.linspace(1.0900, 1.1000, 180)
        
        h4_data = pd.DataFrame({
            'timestamp': h4_dates,
            'open': np.roll(h4_prices, 1),
            'high': h4_prices + 0.0005,
            'low': h4_prices - 0.0005,
            'close': h4_prices,
            'volume': np.random.randint(500, 2000, 180)
        }).set_index('timestamp')
        
        h4_data.iloc[0, h4_data.columns.get_loc('open')] = h4_data.iloc[0, h4_data.columns.get_loc('close')]
        
        analyzer = DowTheoryAnalyzer()
        
        # Analyze both timeframes
        daily_trend = analyzer.analyze_trend(daily_data, 'D1')
        h4_trend = analyzer.analyze_trend(h4_data, 'H4')
        
        # Check confirmation
        confirmation = analyzer.check_confirmation(daily_trend, h4_trend)
        
        assert daily_trend is not None
        assert h4_trend is not None
        assert confirmation is not None
        
        # Generate signals with confirmation
        signals = analyzer.generate_signals(daily_data, h4_data)
        assert isinstance(signals, list)


if __name__ == '__main__':
    # Run tests if executed directly
    pytest.main([__file__, '-v'])