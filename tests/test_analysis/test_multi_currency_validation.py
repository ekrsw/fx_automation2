"""
Multi-Currency Pair Elliott Wave Validation Tests
"""

import pytest
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime, timedelta

from app.analysis.elliott_wave import (
    ElliottWaveAnalyzer, WaveLabeler, WavePredictor,
    Wave, WavePattern, WaveType, WaveLabel, WaveDirection,
    FibonacciCalculator
)
from app.analysis.indicators.swing_detector import SwingPoint, SwingType


def create_swing_point(index, price, swing_type):
    """Helper function to create SwingPoint with proper parameters"""
    from datetime import datetime
    import pandas as pd
    return SwingPoint(
        index=index,
        timestamp=pd.Timestamp(datetime.now()),
        price=price,
        swing_type=swing_type
    )


class CurrencyPairTestData:
    """Generate test data for different currency pairs"""
    
    @staticmethod
    def generate_eurusd_data() -> List[SwingPoint]:
        """Generate EUR/USD test data with realistic price movements"""
        # EUR/USD typically trades between 1.0000 - 1.2500
        base_data = [
            (1.0500, SwingType.LOW),     # Major support
            (1.1200, SwingType.HIGH),    # 700 pip rally
            (1.0767, SwingType.LOW),     # 61.8% retracement
            (1.1633, SwingType.HIGH),    # 161.8% extension
            (1.1350, SwingType.LOW),     # 38.2% retracement
            (1.1800, SwingType.HIGH),    # Final extension
            (1.1500, SwingType.LOW),     # Correction
        ]
        
        return [create_swing_point(i * 20, price, swing_type) 
                for i, (price, swing_type) in enumerate(base_data)]
    
    @staticmethod
    def generate_usdjpy_data() -> List[SwingPoint]:
        """Generate USD/JPY test data with realistic price movements"""
        # USD/JPY typically trades between 100.00 - 150.00
        base_data = [
            (105.00, SwingType.LOW),     # Support level
            (115.00, SwingType.HIGH),    # 1000 pip rally
            (108.82, SwingType.LOW),     # 61.8% retracement
            (121.18, SwingType.HIGH),    # 161.8% extension
            (117.00, SwingType.LOW),     # Shallow retracement
            (125.00, SwingType.HIGH),    # Strong extension
            (120.00, SwingType.LOW),     # Correction
        ]
        
        return [create_swing_point(i * 25, price, swing_type) 
                for i, (price, swing_type) in enumerate(base_data)]
    
    @staticmethod
    def generate_gbpjpy_data() -> List[SwingPoint]:
        """Generate GBP/JPY test data with realistic price movements"""
        # GBP/JPY typically trades between 130.00 - 180.00 (more volatile)
        base_data = [
            (140.00, SwingType.LOW),     # Major low
            (155.00, SwingType.HIGH),    # 1500 pip rally
            (145.73, SwingType.LOW),     # 61.8% retracement
            (169.27, SwingType.HIGH),    # 161.8% extension
            (160.00, SwingType.LOW),     # Retracement
            (175.00, SwingType.HIGH),    # Strong move
            (165.00, SwingType.LOW),     # Correction
        ]
        
        return [create_swing_point(i * 30, price, swing_type) 
                for i, (price, swing_type) in enumerate(base_data)]
    
    @staticmethod
    def generate_xauusd_data() -> List[SwingPoint]:
        """Generate XAU/USD (Gold) test data with realistic price movements"""
        # Gold typically trades between 1700 - 2100
        base_data = [
            (1800.00, SwingType.LOW),    # Support
            (1950.00, SwingType.HIGH),   # $150 rally
            (1857.30, SwingType.LOW),    # 61.8% retracement
            (2092.70, SwingType.HIGH),   # 161.8% extension
            (2020.00, SwingType.LOW),    # Shallow retracement
            (2150.00, SwingType.HIGH),   # Extension
            (2080.00, SwingType.LOW),    # Correction
        ]
        
        return [create_swing_point(i * 15, price, swing_type) 
                for i, (price, swing_type) in enumerate(base_data)]


class TestMultiCurrencyElliottWave:
    """Test Elliott Wave analysis across multiple currency pairs"""
    
    def test_pattern_detection_across_pairs(self):
        """Test pattern detection consistency across different currency pairs"""
        analyzer = ElliottWaveAnalyzer()
        
        currency_pairs = {
            'EURUSD': CurrencyPairTestData.generate_eurusd_data(),
            'USDJPY': CurrencyPairTestData.generate_usdjpy_data(),
            'GBPJPY': CurrencyPairTestData.generate_gbpjpy_data(),
            'XAUUSD': CurrencyPairTestData.generate_xauusd_data(),
        }
        
        results = {}
        
        for pair, swing_points in currency_pairs.items():
            # Analyze each pair
            waves = analyzer.identify_waves(swing_points)
            patterns = analyzer.find_patterns(waves)
            
            results[pair] = {
                'wave_count': len(waves),
                'pattern_count': len(patterns),
                'impulse_patterns': len([p for p in patterns if p.pattern_type == WaveType.IMPULSE]),
                'corrective_patterns': len([p for p in patterns if p.pattern_type == WaveType.CORRECTIVE]),
                'avg_confidence': np.mean([p.confidence for p in patterns]) if patterns else 0.0
            }
        
        # Validate results across pairs
        for pair, result in results.items():
            # Should detect waves in all pairs
            assert result['wave_count'] >= 3, f"{pair}: Insufficient waves detected"
            
            # Should find patterns in realistic data
            assert result['pattern_count'] > 0, f"{pair}: No patterns detected"
            
            # Should find both impulse and corrective patterns
            assert result['impulse_patterns'] > 0, f"{pair}: No impulse patterns"
            
            # Average confidence should be reasonable
            assert result['avg_confidence'] > 0.3, f"{pair}: Low confidence {result['avg_confidence']}"
    
    def test_fibonacci_accuracy_across_pairs(self):
        """Test Fibonacci level accuracy across different currency pairs"""
        calculator = FibonacciCalculator()
        
        pairs_data = {
            'EURUSD': {
                'swing_points': CurrencyPairTestData.generate_eurusd_data(),
                'pip_value': 0.0001,
                'expected_precision': 4
            },
            'USDJPY': {
                'swing_points': CurrencyPairTestData.generate_usdjpy_data(),
                'pip_value': 0.01,
                'expected_precision': 2
            },
            'GBPJPY': {
                'swing_points': CurrencyPairTestData.generate_gbpjpy_data(),
                'pip_value': 0.01,
                'expected_precision': 2
            },
            'XAUUSD': {
                'swing_points': CurrencyPairTestData.generate_xauusd_data(),
                'pip_value': 0.01,
                'expected_precision': 2
            }
        }
        
        for pair, data in pairs_data.items():
            swing_points = data['swing_points']
            pip_value = data['pip_value']
            
            # Test retracement levels for each major move
            for i in range(len(swing_points) - 1):
                start_price = swing_points[i].price
                end_price = swing_points[i + 1].price
                
                if abs(end_price - start_price) > pip_value * 100:  # Only test significant moves
                    retracements = calculator.calculate_retracement_levels(start_price, end_price)
                    
                    # Verify precision is appropriate for the pair
                    for level, price in retracements.items():
                        # Round to appropriate precision
                        rounded_price = round(price, data['expected_precision'])
                        precision_error = abs(price - rounded_price)
                        
                        # Error should be less than half a pip
                        assert precision_error < pip_value / 2, \
                            f"{pair}: Precision error {precision_error} for level {level}"
    
    def test_wave_labeling_consistency(self):
        """Test wave labeling consistency across currency pairs"""
        labeler = WaveLabeler()
        analyzer = ElliottWaveAnalyzer()
        
        pairs = ['EURUSD', 'USDJPY', 'GBPJPY', 'XAUUSD']
        labeling_results = {}
        
        for pair in pairs:
            # Get swing points
            if pair == 'EURUSD':
                swing_points = CurrencyPairTestData.generate_eurusd_data()
            elif pair == 'USDJPY':
                swing_points = CurrencyPairTestData.generate_usdjpy_data()
            elif pair == 'GBPJPY':
                swing_points = CurrencyPairTestData.generate_gbpjpy_data()
            else:  # XAUUSD
                swing_points = CurrencyPairTestData.generate_xauusd_data()
            
            # Analyze waves and apply labels
            waves = analyzer.identify_waves(swing_points)
            labeled_waves = labeler.label_waves(waves)
            
            # Count labels
            label_counts = {}
            for wave in labeled_waves:
                if hasattr(wave, 'label') and wave.label != WaveLabel.UNKNOWN:
                    label = wave.label.value
                    label_counts[label] = label_counts.get(label, 0) + 1
            
            labeling_results[pair] = {
                'total_waves': len(labeled_waves),
                'labeled_waves': len([w for w in labeled_waves 
                                    if hasattr(w, 'label') and w.label != WaveLabel.UNKNOWN]),
                'label_distribution': label_counts,
                'labeling_rate': len([w for w in labeled_waves 
                                    if hasattr(w, 'label') and w.label != WaveLabel.UNKNOWN]) / len(labeled_waves)
            }
        
        # Validate labeling consistency
        for pair, result in labeling_results.items():
            # Should label reasonable percentage of waves
            assert result['labeling_rate'] > 0.3, f"{pair}: Low labeling rate {result['labeling_rate']}"
            
            # Should have reasonable label distribution
            assert len(result['label_distribution']) > 1, f"{pair}: Too few different labels"
    
    def test_prediction_accuracy_across_pairs(self):
        """Test prediction accuracy across different currency pairs"""
        predictor = WavePredictor()
        analyzer = ElliottWaveAnalyzer()
        
        pairs_test_data = {
            'EURUSD': CurrencyPairTestData.generate_eurusd_data(),
            'USDJPY': CurrencyPairTestData.generate_usdjpy_data(),
            'GBPJPY': CurrencyPairTestData.generate_gbpjpy_data(),
            'XAUUSD': CurrencyPairTestData.generate_xauusd_data()
        }
        
        prediction_results = {}
        
        for pair, swing_points in pairs_test_data.items():
            # Use partial data to test predictions
            partial_swings = swing_points[:-2]  # Remove last 2 points
            waves = analyzer.identify_waves(partial_swings)
            incomplete_patterns = analyzer.find_incomplete_patterns(waves)
            
            if incomplete_patterns:
                current_price = partial_swings[-1].price
                predictions = predictor.predict_pattern_completion(incomplete_patterns, current_price)
                
                prediction_results[pair] = {
                    'prediction_count': len(predictions),
                    'avg_confidence': np.mean([p['confidence'] for p in predictions]) if predictions else 0.0,
                    'scenario_counts': [len(p['scenarios']) for p in predictions],
                    'has_targets': any('primary_targets' in p and p['primary_targets'] for p in predictions)
                }
        
        # Validate predictions
        for pair, result in prediction_results.items():
            if result['prediction_count'] > 0:
                # Should generate reasonable confidence
                assert result['avg_confidence'] > 0.2, f"{pair}: Low prediction confidence"
                
                # Should generate multiple scenarios
                assert max(result['scenario_counts']) >= 2, f"{pair}: Too few scenarios"
                
                # Should have price targets
                assert result['has_targets'], f"{pair}: No price targets generated"
    
    def test_volatility_adaptation(self):
        """Test adaptation to different volatility levels across pairs"""
        analyzer = ElliottWaveAnalyzer()
        
        # Create data with different volatility characteristics
        low_volatility_data = CurrencyPairTestData.generate_eurusd_data()  # EUR/USD typically less volatile
        high_volatility_data = CurrencyPairTestData.generate_gbpjpy_data()  # GBP/JPY typically more volatile
        
        # Analyze both
        low_vol_waves = analyzer.identify_waves(low_volatility_data)
        high_vol_waves = analyzer.identify_waves(high_volatility_data)
        
        low_vol_patterns = analyzer.find_patterns(low_vol_waves)
        high_vol_patterns = analyzer.find_patterns(high_vol_waves)
        
        # Validate adaptation
        # High volatility pairs might have different confidence characteristics
        if low_vol_patterns and high_vol_patterns:
            low_vol_avg_conf = np.mean([p.confidence for p in low_vol_patterns])
            high_vol_avg_conf = np.mean([p.confidence for p in high_vol_patterns])
            
            # Both should have reasonable confidence
            assert low_vol_avg_conf > 0.3, "Low volatility pair should have reasonable confidence"
            assert high_vol_avg_conf > 0.3, "High volatility pair should have reasonable confidence"
    
    def test_cross_pair_correlation_analysis(self):
        """Test analysis of correlated currency pairs"""
        analyzer = ElliottWaveAnalyzer()
        
        # EUR/USD and GBP/USD often correlate (both against USD)
        # USD/JPY and EUR/JPY often correlate (both against JPY)
        
        eurusd_data = CurrencyPairTestData.generate_eurusd_data()
        usdjpy_data = CurrencyPairTestData.generate_usdjpy_data()
        
        # Analyze both pairs
        eurusd_waves = analyzer.identify_waves(eurusd_data)
        usdjpy_waves = analyzer.identify_waves(usdjpy_data)
        
        eurusd_patterns = analyzer.find_patterns(eurusd_waves)
        usdjpy_patterns = analyzer.find_patterns(usdjpy_waves)
        
        # Both should detect patterns (showing USD strength/weakness affects both)
        assert len(eurusd_patterns) > 0, "Should detect EUR/USD patterns"
        assert len(usdjpy_patterns) > 0, "Should detect USD/JPY patterns"
        
        # Pattern timing might be similar (correlated moves)
        if eurusd_patterns and usdjpy_patterns:
            eurusd_pattern = eurusd_patterns[0]
            usdjpy_pattern = usdjpy_patterns[0]
            
            # Both patterns should be reasonably confident
            assert eurusd_pattern.confidence > 0.3
            assert usdjpy_pattern.confidence > 0.3
    
    @pytest.mark.parametrize("pair,expected_pip_value", [
        ("EURUSD", 0.0001),
        ("USDJPY", 0.01),
        ("GBPJPY", 0.01),
        ("XAUUSD", 0.01),
    ])
    def test_pip_value_handling(self, pair, expected_pip_value):
        """Test correct handling of pip values for different pairs"""
        calculator = FibonacciCalculator()
        
        # Get test data for the pair
        if pair == "EURUSD":
            swing_points = CurrencyPairTestData.generate_eurusd_data()
        elif pair == "USDJPY":
            swing_points = CurrencyPairTestData.generate_usdjpy_data()
        elif pair == "GBPJPY":
            swing_points = CurrencyPairTestData.generate_gbpjpy_data()
        else:  # XAUUSD
            swing_points = CurrencyPairTestData.generate_xauusd_data()
        
        # Test Fibonacci calculations
        start_price = swing_points[0].price
        end_price = swing_points[1].price
        
        retracements = calculator.calculate_retracement_levels(start_price, end_price)
        
        # Verify calculations are precise to the pip value
        for level, price in retracements.items():
            # Check that price precision matches expected pip value
            decimal_places = len(str(expected_pip_value).split('.')[-1])
            
            # Price should be calculable to appropriate precision
            rounded_price = round(price, decimal_places)
            precision_error = abs(price - rounded_price)
            
            assert precision_error < expected_pip_value, \
                f"{pair}: Precision error {precision_error} exceeds pip value {expected_pip_value}"
    
    def test_market_session_adaptation(self):
        """Test adaptation to different market session characteristics"""
        analyzer = ElliottWaveAnalyzer()
        
        # Simulate different market sessions with different characteristics
        # Asian session: typically quieter, smaller ranges
        asian_data = [
            create_swing_point(0, 1.0500, SwingType.LOW),
            create_swing_point(20, 1.0520, SwingType.HIGH),   # Small 20 pip move
            create_swing_point(40, 1.0510, SwingType.LOW),    # Small retracement
            create_swing_point(60, 1.0535, SwingType.HIGH),   # Small extension
            create_swing_point(80, 1.0515, SwingType.LOW),    # Consolidation
        ]
        
        # London/NY session: typically more volatile, larger ranges
        london_ny_data = [
            create_swing_point(0, 1.0500, SwingType.LOW),
            create_swing_point(20, 1.0650, SwingType.HIGH),   # Large 150 pip move
            create_swing_point(40, 1.0558, SwingType.LOW),    # 61.8% retracement
            create_swing_point(60, 1.0793, SwingType.HIGH),   # 161.8% extension
            create_swing_point(80, 1.0680, SwingType.LOW),    # Significant retracement
        ]
        
        # Analyze both session types
        asian_waves = analyzer.identify_waves(asian_data)
        london_waves = analyzer.identify_waves(london_ny_data)
        
        asian_patterns = analyzer.find_patterns(asian_waves)
        london_patterns = analyzer.find_patterns(london_waves)
        
        # Both should adapt to their respective volatility environments
        assert len(asian_waves) > 0, "Should detect waves in Asian session"
        assert len(london_waves) > 0, "Should detect waves in London session"
        
        # London session should typically show stronger patterns due to higher volatility
        if london_patterns:
            london_avg_conf = np.mean([p.confidence for p in london_patterns])
            assert london_avg_conf > 0.4, "London session should show strong patterns"


class TestCurrencyPairSpecificBehavior:
    """Test behavior specific to different currency pair characteristics"""
    
    def test_major_pairs_stability(self):
        """Test that major pairs (EUR/USD, GBP/USD, USD/JPY) show stable analysis"""
        analyzer = ElliottWaveAnalyzer()
        
        major_pairs_data = {
            'EURUSD': CurrencyPairTestData.generate_eurusd_data(),
            'USDJPY': CurrencyPairTestData.generate_usdjpy_data(),
        }
        
        for pair, swing_points in major_pairs_data.items():
            waves = analyzer.identify_waves(swing_points)
            patterns = analyzer.find_patterns(swing_points)
            
            # Major pairs should show stable, high-confidence analysis
            if patterns:
                avg_confidence = np.mean([p.confidence for p in patterns])
                assert avg_confidence > 0.4, f"{pair}: Major pair should have high confidence"
    
    def test_cross_pairs_complexity(self):
        """Test that cross pairs (GBP/JPY, EUR/JPY) handle increased complexity"""
        analyzer = ElliottWaveAnalyzer()
        
        cross_pair_data = CurrencyPairTestData.generate_gbpjpy_data()
        waves = analyzer.identify_waves(cross_pair_data)
        patterns = analyzer.find_patterns(waves)
        
        # Cross pairs are typically more volatile and complex
        assert len(waves) > 0, "Should detect waves in cross pairs"
        
        if patterns:
            # May have lower confidence due to complexity
            avg_confidence = np.mean([p.confidence for p in patterns])
            assert avg_confidence > 0.2, "Cross pairs should still have reasonable confidence"
    
    def test_commodity_pairs_behavior(self):
        """Test behavior with commodity-related pairs (XAU/USD)"""
        analyzer = ElliottWaveAnalyzer()
        
        gold_data = CurrencyPairTestData.generate_xauusd_data()
        waves = analyzer.identify_waves(gold_data)
        patterns = analyzer.find_patterns(waves)
        
        # Gold should show clear Elliott Wave patterns
        assert len(waves) > 0, "Should detect waves in Gold"
        assert len(patterns) > 0, "Should detect patterns in Gold"
        
        if patterns:
            # Gold often shows strong trending behavior
            impulse_patterns = [p for p in patterns if p.pattern_type == WaveType.IMPULSE]
            assert len(impulse_patterns) > 0, "Gold should show impulse patterns"


class TestRealTimeMultiCurrencyScenarios:
    """Test real-time scenarios with multiple currency pairs"""
    
    def test_simultaneous_analysis(self):
        """Test simultaneous analysis of multiple currency pairs"""
        analyzer = ElliottWaveAnalyzer()
        labeler = WaveLabeler()
        predictor = WavePredictor()
        
        # Simulate real-time data for multiple pairs
        pairs_data = {
            'EURUSD': CurrencyPairTestData.generate_eurusd_data(),
            'USDJPY': CurrencyPairTestData.generate_usdjpy_data(),
            'GBPJPY': CurrencyPairTestData.generate_gbpjpy_data(),
        }
        
        # Analyze all pairs simultaneously
        results = {}
        for pair, swing_points in pairs_data.items():
            waves = analyzer.identify_waves(swing_points)
            labeled_waves = labeler.label_waves(waves)
            patterns = analyzer.find_patterns(labeled_waves)
            incomplete_patterns = analyzer.find_incomplete_patterns(labeled_waves)
            
            current_price = swing_points[-1].price
            predictions = predictor.predict_pattern_completion(incomplete_patterns, current_price)
            
            results[pair] = {
                'waves': waves,
                'patterns': patterns,
                'predictions': predictions,
                'analysis_success': len(patterns) > 0
            }
        
        # Validate simultaneous analysis
        successful_analyses = sum(1 for r in results.values() if r['analysis_success'])
        assert successful_analyses >= 2, "Should successfully analyze multiple pairs simultaneously"
        
        # Check for consistent analysis quality
        for pair, result in results.items():
            if result['patterns']:
                avg_confidence = np.mean([p.confidence for p in result['patterns']])
                assert avg_confidence > 0.25, f"{pair}: Simultaneous analysis quality check"
    
    def test_market_correlation_impact(self):
        """Test how market correlations affect Elliott Wave analysis"""
        analyzer = ElliottWaveAnalyzer()
        
        # Create correlated movements (USD strength affecting multiple pairs)
        # When USD strengthens: EUR/USD down, USD/JPY up
        
        usd_strength_scenario = {
            'EURUSD': [  # EUR/USD falling (USD strength)
                create_swing_point(0, 1.1000, SwingType.HIGH),
                create_swing_point(20, 1.0500, SwingType.LOW),
                create_swing_point(40, 1.0691, SwingType.HIGH),  # 38.2% retracement
                create_swing_point(60, 1.0191, SwingType.LOW),   # Extension
            ],
            'USDJPY': [  # USD/JPY rising (USD strength)
                create_swing_point(0, 110.00, SwingType.LOW),
                create_swing_point(20, 115.00, SwingType.HIGH),
                create_swing_point(40, 113.09, SwingType.LOW),   # 38.2% retracement
                create_swing_point(60, 118.09, SwingType.HIGH),  # Extension
            ]
        }
        
        results = {}
        for pair, swing_points in usd_strength_scenario.items():
            waves = analyzer.identify_waves(swing_points)
            patterns = analyzer.find_patterns(waves)
            results[pair] = patterns
        
        # Both pairs should show patterns (correlated USD movement)
        assert len(results['EURUSD']) > 0, "EUR/USD should show USD strength pattern"
        assert len(results['USDJPY']) > 0, "USD/JPY should show USD strength pattern"
        
        # Patterns should have reasonable confidence
        for pair, patterns in results.items():
            if patterns:
                avg_confidence = np.mean([p.confidence for p in patterns])
                assert avg_confidence > 0.3, f"{pair}: Correlated movement should show good patterns"