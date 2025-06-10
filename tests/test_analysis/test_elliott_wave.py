"""
Elliott Wave Analysis Tests
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from app.analysis.elliott_wave import (
    ElliottWaveAnalyzer, WaveLabeler, WavePredictor,
    Wave, WavePattern, WaveType, WaveLabel, WaveDirection,
    FibonacciCalculator, WaveValidator
)
from app.analysis.indicators.swing_detector import SwingPoint, SwingType


class TestFibonacciCalculator:
    """Test Fibonacci calculation functionality"""
    
    def test_retracement_levels_bullish(self):
        """Test Fibonacci retracement calculation for bullish move"""
        calculator = FibonacciCalculator()
        
        # Bullish move from 100 to 150
        retracements = calculator.calculate_retracement_levels(100.0, 150.0)
        
        assert len(retracements) == 5  # Default levels
        assert abs(retracements[0.236] - 138.2) < 0.01  # 150 - (50 * 0.236)
        assert abs(retracements[0.382] - 130.9) < 0.01  # 150 - (50 * 0.382)
        assert abs(retracements[0.500] - 125.0) < 0.01  # 150 - (50 * 0.500)
        assert abs(retracements[0.618] - 119.1) < 0.01  # 150 - (50 * 0.618)
        assert abs(retracements[0.786] - 110.7) < 0.01  # 150 - (50 * 0.786)
    
    def test_retracement_levels_bearish(self):
        """Test Fibonacci retracement calculation for bearish move"""
        calculator = FibonacciCalculator()
        
        # Bearish move from 150 to 100
        retracements = calculator.calculate_retracement_levels(150.0, 100.0)
        
        assert abs(retracements[0.236] - 111.8) < 0.01  # 100 + (50 * 0.236)
        assert abs(retracements[0.618] - 130.9) < 0.01  # 100 + (50 * 0.618)
    
    def test_extension_levels_bullish(self):
        """Test Fibonacci extension calculation for bullish move"""
        calculator = FibonacciCalculator()
        
        # Wave 1: 100 to 120, Wave 2 retracement to 110
        extensions = calculator.calculate_extension_levels(100.0, 120.0, 110.0)
        
        assert abs(extensions[1.000] - 130.0) < 0.01  # 110 + (20 * 1.0)
        assert abs(extensions[1.618] - 142.36) < 0.01  # 110 + (20 * 1.618)
        assert abs(extensions[2.618] - 162.36) < 0.01  # 110 + (20 * 2.618)
    
    def test_extension_levels_bearish(self):
        """Test Fibonacci extension calculation for bearish move"""
        calculator = FibonacciCalculator()
        
        # Wave 1: 120 to 100, Wave 2 retracement to 110
        extensions = calculator.calculate_extension_levels(120.0, 100.0, 110.0)
        
        assert abs(extensions[1.000] - 90.0) < 0.01   # 110 - (20 * 1.0)
        assert abs(extensions[1.618] - 77.64) < 0.01  # 110 - (20 * 1.618)
    
    def test_closest_fibonacci_level(self):
        """Test finding closest Fibonacci level to actual ratio"""
        calculator = FibonacciCalculator()
        
        # Test exact match
        fib_level = calculator.find_closest_fibonacci_level(0.618, tolerance=0.1)
        assert fib_level is not None
        assert abs(fib_level.value - 0.618) < 0.001
        
        # Test approximate match
        fib_level = calculator.find_closest_fibonacci_level(0.62, tolerance=0.1)
        assert fib_level is not None
        assert abs(fib_level.value - 0.618) < 0.1
        
        # Test no match
        fib_level = calculator.find_closest_fibonacci_level(0.9, tolerance=0.05)
        assert fib_level is None
    
    def test_analyze_wave_ratios(self):
        """Test wave ratio analysis"""
        calculator = FibonacciCalculator()
        
        # Create test waves
        waves = [
            Wave(WaveLabel.WAVE_1, WaveType.IMPULSE, WaveDirection.UP,
                 SwingPoint(0, 100.0, SwingType.LOW),
                 SwingPoint(10, 120.0, SwingType.HIGH)),
            Wave(WaveLabel.WAVE_2, WaveType.IMPULSE, WaveDirection.DOWN,
                 SwingPoint(10, 120.0, SwingType.HIGH),
                 SwingPoint(20, 107.64, SwingType.LOW)),  # 61.8% retracement
            Wave(WaveLabel.WAVE_3, WaveType.IMPULSE, WaveDirection.UP,
                 SwingPoint(20, 107.64, SwingType.LOW),
                 SwingPoint(30, 140.0, SwingType.HIGH))   # 1.618 extension
        ]
        
        analysis = calculator.analyze_wave_ratios(waves)
        
        assert 'wave_ratios' in analysis
        assert 'fibonacci_matches' in analysis
        assert 'ratio_quality_score' in analysis
        
        # Check specific ratios
        assert 'wave_2_vs_wave_1' in analysis['wave_ratios']
        assert 'wave_3_vs_wave_2' in analysis['wave_ratios']
        
        # Should find Fibonacci matches
        assert len(analysis['fibonacci_matches']) > 0
        assert analysis['ratio_quality_score'] > 0.5


class TestWaveValidator:
    """Test wave validation rules"""
    
    def create_test_waves_impulse_valid(self):
        """Create valid 5-wave impulse sequence"""
        return [
            # Wave 1: Up from 100 to 120
            Wave(WaveLabel.WAVE_1, WaveType.IMPULSE, WaveDirection.UP,
                 SwingPoint(0, 100.0, SwingType.LOW),
                 SwingPoint(10, 120.0, SwingType.HIGH)),
            # Wave 2: Down to 110 (50% retracement)
            Wave(WaveLabel.WAVE_2, WaveType.IMPULSE, WaveDirection.DOWN,
                 SwingPoint(10, 120.0, SwingType.HIGH),
                 SwingPoint(20, 110.0, SwingType.LOW)),
            # Wave 3: Up to 145 (1.75x Wave 1)
            Wave(WaveLabel.WAVE_3, WaveType.IMPULSE, WaveDirection.UP,
                 SwingPoint(20, 110.0, SwingType.LOW),
                 SwingPoint(30, 145.0, SwingType.HIGH)),
            # Wave 4: Down to 130 (43% retracement, no overlap with Wave 1)
            Wave(WaveLabel.WAVE_4, WaveType.IMPULSE, WaveDirection.DOWN,
                 SwingPoint(30, 145.0, SwingType.HIGH),
                 SwingPoint(40, 130.0, SwingType.LOW)),
            # Wave 5: Up to 150 (Equal to Wave 1)
            Wave(WaveLabel.WAVE_5, WaveType.IMPULSE, WaveDirection.UP,
                 SwingPoint(40, 130.0, SwingType.LOW),
                 SwingPoint(50, 150.0, SwingType.HIGH))
        ]
    
    def create_test_waves_impulse_invalid(self):
        """Create invalid 5-wave impulse sequence (Wave 2 over-retracement)"""
        return [
            # Wave 1: Up from 100 to 120
            Wave(WaveLabel.WAVE_1, WaveType.IMPULSE, WaveDirection.UP,
                 SwingPoint(0, 100.0, SwingType.LOW),
                 SwingPoint(10, 120.0, SwingType.HIGH)),
            # Wave 2: Down to 95 (125% retracement - INVALID)
            Wave(WaveLabel.WAVE_2, WaveType.IMPULSE, WaveDirection.DOWN,
                 SwingPoint(10, 120.0, SwingType.HIGH),
                 SwingPoint(20, 95.0, SwingType.LOW)),
            # Wave 3: Up to 140
            Wave(WaveLabel.WAVE_3, WaveType.IMPULSE, WaveDirection.UP,
                 SwingPoint(20, 95.0, SwingType.LOW),
                 SwingPoint(30, 140.0, SwingType.HIGH)),
            # Wave 4: Down to 125
            Wave(WaveLabel.WAVE_4, WaveType.IMPULSE, WaveDirection.DOWN,
                 SwingPoint(30, 140.0, SwingType.HIGH),
                 SwingPoint(40, 125.0, SwingType.LOW)),
            # Wave 5: Up to 150
            Wave(WaveLabel.WAVE_5, WaveType.IMPULSE, WaveDirection.UP,
                 SwingPoint(40, 125.0, SwingType.LOW),
                 SwingPoint(50, 150.0, SwingType.HIGH))
        ]
    
    def test_validate_impulse_wave_valid(self):
        """Test validation of valid impulse wave"""
        validator = WaveValidator()
        waves = self.create_test_waves_impulse_valid()
        
        validation = validator.validate_impulse_wave(waves)
        
        assert validation['is_valid'] is True
        assert validation['confidence'] > 0.8
        assert len(validation['violations']) == 0
        assert 'wave_2_retracement' in validation['rule_checks']
        assert 'wave_sizes' in validation['rule_checks']
    
    def test_validate_impulse_wave_invalid(self):
        """Test validation of invalid impulse wave"""
        validator = WaveValidator()
        waves = self.create_test_waves_impulse_invalid()
        
        validation = validator.validate_impulse_wave(waves)
        
        assert validation['is_valid'] is False
        assert validation['confidence'] < 0.5
        assert len(validation['violations']) > 0
        assert any('Wave 2 retraces more than 100%' in v for v in validation['violations'])
    
    def test_validate_corrective_wave_valid(self):
        """Test validation of valid corrective wave"""
        validator = WaveValidator()
        
        # Valid ABC correction
        waves = [
            # Wave A: Down from 150 to 130
            Wave(WaveLabel.WAVE_A, WaveType.CORRECTIVE, WaveDirection.DOWN,
                 SwingPoint(0, 150.0, SwingType.HIGH),
                 SwingPoint(10, 130.0, SwingType.LOW)),
            # Wave B: Up to 145 (75% retracement)
            Wave(WaveLabel.WAVE_B, WaveType.CORRECTIVE, WaveDirection.UP,
                 SwingPoint(10, 130.0, SwingType.LOW),
                 SwingPoint(20, 145.0, SwingType.HIGH)),
            # Wave C: Down to 115 (Equal to Wave A)
            Wave(WaveLabel.WAVE_C, WaveType.CORRECTIVE, WaveDirection.DOWN,
                 SwingPoint(20, 145.0, SwingType.HIGH),
                 SwingPoint(30, 115.0, SwingType.LOW))
        ]
        
        validation = validator.validate_corrective_wave(waves)
        
        assert validation['is_valid'] is True
        assert validation['confidence'] > 0.7
        assert len(validation['violations']) == 0
    
    def test_validate_corrective_wave_invalid(self):
        """Test validation of invalid corrective wave"""
        validator = WaveValidator()
        
        # Invalid ABC correction (Wave C too small)
        waves = [
            # Wave A: Down from 150 to 130
            Wave(WaveLabel.WAVE_A, WaveType.CORRECTIVE, WaveDirection.DOWN,
                 SwingPoint(0, 150.0, SwingType.HIGH),
                 SwingPoint(10, 130.0, SwingType.LOW)),
            # Wave B: Up to 145 (75% retracement)
            Wave(WaveLabel.WAVE_B, WaveType.CORRECTIVE, WaveDirection.UP,
                 SwingPoint(10, 130.0, SwingType.LOW),
                 SwingPoint(20, 145.0, SwingType.HIGH)),
            # Wave C: Down to 142 (Only 15% of Wave A - INVALID)
            Wave(WaveLabel.WAVE_C, WaveType.CORRECTIVE, WaveDirection.DOWN,
                 SwingPoint(20, 145.0, SwingType.HIGH),
                 SwingPoint(30, 142.0, SwingType.LOW))
        ]
        
        validation = validator.validate_corrective_wave(waves)
        
        assert validation['is_valid'] is False
        assert len(validation['violations']) > 0
        assert any('Wave C is less than 61.8%' in v for v in validation['violations'])


class TestElliottWaveAnalyzer:
    """Test Elliott Wave analyzer functionality"""
    
    def create_test_swing_points_impulse(self):
        """Create swing points representing a 5-wave impulse"""
        return [
            SwingPoint(0, 100.0, SwingType.LOW),    # Wave 1 start
            SwingPoint(10, 120.0, SwingType.HIGH),  # Wave 1 end / Wave 2 start
            SwingPoint(20, 110.0, SwingType.LOW),   # Wave 2 end / Wave 3 start
            SwingPoint(30, 145.0, SwingType.HIGH),  # Wave 3 end / Wave 4 start
            SwingPoint(40, 130.0, SwingType.LOW),   # Wave 4 end / Wave 5 start
            SwingPoint(50, 150.0, SwingType.HIGH)   # Wave 5 end
        ]
    
    def create_test_swing_points_corrective(self):
        """Create swing points representing a 3-wave correction"""
        return [
            SwingPoint(0, 150.0, SwingType.HIGH),   # Wave A start
            SwingPoint(10, 130.0, SwingType.LOW),   # Wave A end / Wave B start
            SwingPoint(20, 145.0, SwingType.HIGH),  # Wave B end / Wave C start
            SwingPoint(30, 115.0, SwingType.LOW)    # Wave C end
        ]
    
    def test_identify_waves_from_swing_points(self):
        """Test wave identification from swing points"""
        analyzer = ElliottWaveAnalyzer()
        swing_points = self.create_test_swing_points_impulse()
        
        waves = analyzer.identify_waves(swing_points)
        
        assert len(waves) == 5  # 5 waves from 6 swing points
        
        # Check first wave
        assert waves[0].direction == WaveDirection.UP
        assert waves[0].start_point.price == 100.0
        assert waves[0].end_point.price == 120.0
        assert waves[0].price_range == 20.0
        
        # Check alternating directions
        for i in range(1, len(waves)):
            assert waves[i].direction != waves[i-1].direction
    
    def test_find_impulse_patterns(self):
        """Test impulse pattern detection"""
        analyzer = ElliottWaveAnalyzer()
        swing_points = self.create_test_swing_points_impulse()
        
        waves = analyzer.identify_waves(swing_points)
        patterns = analyzer.find_patterns(waves)
        
        # Should find at least one impulse pattern
        impulse_patterns = [p for p in patterns if p.pattern_type == WaveType.IMPULSE]
        assert len(impulse_patterns) > 0
        
        # Check pattern structure
        pattern = impulse_patterns[0]
        assert len(pattern.waves) == 5
        assert pattern.is_complete is True
        assert pattern.completion_percentage == 1.0
        
        # Check wave labels
        expected_labels = [WaveLabel.WAVE_1, WaveLabel.WAVE_2, WaveLabel.WAVE_3, 
                          WaveLabel.WAVE_4, WaveLabel.WAVE_5]
        actual_labels = [w.label for w in pattern.waves]
        assert actual_labels == expected_labels
    
    def test_find_corrective_patterns(self):
        """Test corrective pattern detection"""
        analyzer = ElliottWaveAnalyzer()
        swing_points = self.create_test_swing_points_corrective()
        
        waves = analyzer.identify_waves(swing_points)
        patterns = analyzer.find_patterns(waves)
        
        # Should find at least one corrective pattern
        corrective_patterns = [p for p in patterns if p.pattern_type == WaveType.CORRECTIVE]
        assert len(corrective_patterns) > 0
        
        # Check pattern structure
        pattern = corrective_patterns[0]
        assert len(pattern.waves) == 3
        assert pattern.is_complete is True
        
        # Check wave labels
        expected_labels = [WaveLabel.WAVE_A, WaveLabel.WAVE_B, WaveLabel.WAVE_C]
        actual_labels = [w.label for w in pattern.waves]
        assert actual_labels == expected_labels
    
    def test_find_incomplete_patterns(self):
        """Test incomplete pattern detection"""
        analyzer = ElliottWaveAnalyzer()
        
        # Create incomplete 3-wave sequence (missing waves 4 and 5)
        swing_points = self.create_test_swing_points_impulse()[:4]  # Only first 4 points
        
        waves = analyzer.identify_waves(swing_points)
        incomplete_patterns = analyzer.find_incomplete_patterns(waves)
        
        assert len(incomplete_patterns) > 0
        
        # Check incomplete pattern properties
        pattern = incomplete_patterns[0]
        assert pattern.completion_percentage < 1.0
        assert pattern.next_expected_wave is not None
        assert 'price_targets' in pattern.metadata or len(pattern.price_targets) > 0
    
    def test_wave_confidence_calculation(self):
        """Test wave confidence scoring"""
        analyzer = ElliottWaveAnalyzer()
        swing_points = self.create_test_swing_points_impulse()
        
        waves = analyzer.identify_waves(swing_points)
        
        # All waves should have confidence scores
        for wave in waves:
            assert 0.0 <= wave.confidence <= 1.0
            assert hasattr(wave, 'metadata')
            assert 'size_consistency' in wave.metadata
            assert 'direction_score' in wave.metadata


class TestWaveLabeler:
    """Test wave labeling functionality"""
    
    def create_test_waves_for_labeling(self):
        """Create waves for labeling tests"""
        return [
            Wave(WaveLabel.UNKNOWN, WaveType.UNKNOWN, WaveDirection.UP,
                 SwingPoint(0, 100.0, SwingType.LOW),
                 SwingPoint(10, 120.0, SwingType.HIGH)),
            Wave(WaveLabel.UNKNOWN, WaveType.UNKNOWN, WaveDirection.DOWN,
                 SwingPoint(10, 120.0, SwingType.HIGH),
                 SwingPoint(20, 110.0, SwingType.LOW)),
            Wave(WaveLabel.UNKNOWN, WaveType.UNKNOWN, WaveDirection.UP,
                 SwingPoint(20, 110.0, SwingType.LOW),
                 SwingPoint(30, 145.0, SwingType.HIGH)),
            Wave(WaveLabel.UNKNOWN, WaveType.UNKNOWN, WaveDirection.DOWN,
                 SwingPoint(30, 145.0, SwingType.HIGH),
                 SwingPoint(40, 130.0, SwingType.LOW)),
            Wave(WaveLabel.UNKNOWN, WaveType.UNKNOWN, WaveDirection.UP,
                 SwingPoint(40, 130.0, SwingType.LOW),
                 SwingPoint(50, 150.0, SwingType.HIGH))
        ]
    
    def test_label_waves_basic(self):
        """Test basic wave labeling"""
        labeler = WaveLabeler()
        waves = self.create_test_waves_for_labeling()
        
        labeled_waves = labeler.label_waves(waves)
        
        assert len(labeled_waves) == len(waves)
        
        # Check that some waves got labeled
        labeled_count = sum(1 for w in labeled_waves 
                          if hasattr(w, 'label') and w.label != WaveLabel.UNKNOWN)
        assert labeled_count > 0
    
    def test_degree_analysis(self):
        """Test wave degree analysis"""
        labeler = WaveLabeler()
        waves = self.create_test_waves_for_labeling()
        
        degree_analysis = labeler._analyze_wave_degrees(waves)
        
        assert 'primary_patterns' in degree_analysis
        assert 'sub_patterns' in degree_analysis
        assert 'degree_assignments' in degree_analysis
        assert 'confidence_scores' in degree_analysis
        
        # Should have degree assignments for all waves
        assert len(degree_analysis['degree_assignments']) == len(waves)
    
    def test_alternative_counts(self):
        """Test alternative count generation"""
        labeler = WaveLabeler()
        waves = self.create_test_waves_for_labeling()
        
        alternatives = labeler.generate_alternative_counts(waves)
        
        assert isinstance(alternatives, list)
        # Should generate at least one alternative
        if len(alternatives) > 0:
            alt = alternatives[0]
            assert 'type' in alt
            assert 'description' in alt
            assert 'waves' in alt
            assert 'confidence' in alt


class TestWavePredictor:
    """Test wave prediction functionality"""
    
    def create_incomplete_impulse_pattern(self):
        """Create incomplete impulse pattern (3 waves)"""
        waves = [
            Wave(WaveLabel.WAVE_1, WaveType.IMPULSE, WaveDirection.UP,
                 SwingPoint(0, 100.0, SwingType.LOW),
                 SwingPoint(10, 120.0, SwingType.HIGH)),
            Wave(WaveLabel.WAVE_2, WaveType.IMPULSE, WaveDirection.DOWN,
                 SwingPoint(10, 120.0, SwingType.HIGH),
                 SwingPoint(20, 110.0, SwingType.LOW)),
            Wave(WaveLabel.WAVE_3, WaveType.IMPULSE, WaveDirection.UP,
                 SwingPoint(20, 110.0, SwingType.LOW),
                 SwingPoint(30, 145.0, SwingType.HIGH))
        ]
        
        pattern = WavePattern(
            pattern_type=WaveType.IMPULSE,
            waves=waves,
            completion_percentage=0.6,  # 3/5 waves complete
            next_expected_wave=WaveLabel.WAVE_4
        )
        
        return pattern
    
    def create_incomplete_corrective_pattern(self):
        """Create incomplete corrective pattern (1 wave)"""
        waves = [
            Wave(WaveLabel.WAVE_A, WaveType.CORRECTIVE, WaveDirection.DOWN,
                 SwingPoint(0, 150.0, SwingType.HIGH),
                 SwingPoint(10, 130.0, SwingType.LOW))
        ]
        
        pattern = WavePattern(
            pattern_type=WaveType.CORRECTIVE,
            waves=waves,
            completion_percentage=0.33,  # 1/3 waves complete
            next_expected_wave=WaveLabel.WAVE_B
        )
        
        return pattern
    
    def test_predict_wave_4_scenarios(self):
        """Test Wave 4 prediction scenarios"""
        predictor = WavePredictor()
        pattern = self.create_incomplete_impulse_pattern()
        current_price = 145.0  # At Wave 3 high
        
        prediction = predictor.predict_next_wave(pattern, current_price)
        
        assert prediction['next_expected_wave'] == WaveLabel.WAVE_4.value
        assert 'scenarios' in prediction
        assert len(prediction['scenarios']) > 0
        
        # Check scenario structure
        scenario = prediction['scenarios'][0]
        assert 'wave' in scenario
        assert 'type' in scenario
        assert 'target_price' in scenario
        assert 'probability' in scenario
        assert scenario['wave'] == 'Wave 4'
        assert scenario['type'] == 'retracement'
    
    def test_predict_wave_b_scenarios(self):
        """Test Wave B prediction scenarios"""
        predictor = WavePredictor()
        pattern = self.create_incomplete_corrective_pattern()
        current_price = 130.0  # At Wave A low
        
        prediction = predictor.predict_next_wave(pattern, current_price)
        
        assert prediction['next_expected_wave'] == WaveLabel.WAVE_B.value
        assert 'scenarios' in prediction
        assert len(prediction['scenarios']) > 0
        
        # Check scenario structure
        scenario = prediction['scenarios'][0]
        assert scenario['wave'] == 'Wave B'
        assert scenario['type'] == 'retracement'
    
    def test_primary_targets_extraction(self):
        """Test primary target extraction from scenarios"""
        predictor = WavePredictor()
        pattern = self.create_incomplete_impulse_pattern()
        current_price = 145.0
        
        prediction = predictor.predict_next_wave(pattern, current_price)
        
        assert 'primary_targets' in prediction
        targets = prediction['primary_targets']
        
        if targets:
            assert 'most_probable' in targets
            assert isinstance(targets['most_probable'], float)
    
    def test_risk_level_calculation(self):
        """Test risk level calculation"""
        predictor = WavePredictor()
        pattern = self.create_incomplete_impulse_pattern()
        current_price = 145.0
        
        prediction = predictor.predict_next_wave(pattern, current_price)
        
        assert 'risk_levels' in prediction
        risk_levels = prediction['risk_levels']
        
        # Should have some risk levels defined
        assert len(risk_levels) > 0
        assert all(isinstance(level, float) for level in risk_levels.values())
    
    def test_prediction_confidence(self):
        """Test prediction confidence calculation"""
        predictor = WavePredictor()
        pattern = self.create_incomplete_impulse_pattern()
        current_price = 145.0
        
        prediction = predictor.predict_next_wave(pattern, current_price)
        
        assert 'confidence' in prediction
        assert 0.0 <= prediction['confidence'] <= 1.0
    
    def test_multiple_pattern_predictions(self):
        """Test predictions for multiple incomplete patterns"""
        predictor = WavePredictor()
        
        patterns = [
            self.create_incomplete_impulse_pattern(),
            self.create_incomplete_corrective_pattern()
        ]
        current_price = 140.0
        
        predictions = predictor.predict_pattern_completion(patterns, current_price)
        
        assert isinstance(predictions, list)
        assert len(predictions) <= predictor.max_prediction_scenarios
        
        # Should be sorted by confidence
        if len(predictions) > 1:
            for i in range(1, len(predictions)):
                assert predictions[i]['confidence'] <= predictions[i-1]['confidence']


class TestElliottWaveIntegration:
    """Test integrated Elliott Wave functionality"""
    
    def test_complete_analysis_workflow(self):
        """Test complete analysis workflow from swing points to predictions"""
        # Initialize components
        analyzer = ElliottWaveAnalyzer()
        labeler = WaveLabeler()
        predictor = WavePredictor()
        
        # Create realistic swing points
        swing_points = [
            SwingPoint(0, 1.2000, SwingType.LOW),
            SwingPoint(50, 1.2100, SwingType.HIGH),
            SwingPoint(100, 1.2050, SwingType.LOW),
            SwingPoint(150, 1.2180, SwingType.HIGH),
            SwingPoint(200, 1.2080, SwingType.LOW),
            SwingPoint(250, 1.2150, SwingType.HIGH)
        ]
        
        # Step 1: Identify waves
        waves = analyzer.identify_waves(swing_points)
        assert len(waves) > 0
        
        # Step 2: Find patterns
        patterns = analyzer.find_patterns(waves)
        incomplete_patterns = analyzer.find_incomplete_patterns(waves)
        
        # Step 3: Label waves
        labeled_waves = labeler.label_waves(waves)
        
        # Step 4: Generate predictions
        current_price = 1.2150
        if incomplete_patterns:
            predictions = predictor.predict_pattern_completion(incomplete_patterns, current_price)
            
            # Verify prediction structure
            for prediction in predictions:
                assert 'pattern_type' in prediction
                assert 'scenarios' in prediction
                assert 'primary_targets' in prediction
                assert 'confidence' in prediction
    
    def test_fibonacci_integration(self):
        """Test Fibonacci level integration throughout analysis"""
        analyzer = ElliottWaveAnalyzer()
        
        # Create swing points with clear Fibonacci relationships
        swing_points = [
            SwingPoint(0, 1.0000, SwingType.LOW),
            SwingPoint(10, 1.1000, SwingType.HIGH),    # +1000 pips
            SwingPoint(20, 1.0382, SwingType.LOW),     # 61.8% retracement
            SwingPoint(30, 1.1618, SwingType.HIGH),    # 161.8% extension
            SwingPoint(40, 1.0900, SwingType.LOW),     # Shallow retracement
            SwingPoint(50, 1.2000, SwingType.HIGH)     # Final target
        ]
        
        waves = analyzer.identify_waves(swing_points)
        patterns = analyzer.find_patterns(waves)
        
        if patterns:
            pattern = patterns[0]
            # Should have Fibonacci analysis in metadata
            if 'fibonacci_analysis' in pattern.metadata:
                fib_analysis = pattern.metadata['fibonacci_analysis']
                assert 'wave_ratios' in fib_analysis
                assert 'fibonacci_matches' in fib_analysis


# Performance and accuracy tests
class TestElliottWaveAccuracy:
    """Test Elliott Wave detection accuracy with known patterns"""
    
    @pytest.mark.parametrize("noise_level", [0.0, 0.01, 0.02])
    def test_pattern_detection_with_noise(self, noise_level):
        """Test pattern detection accuracy with varying noise levels"""
        analyzer = ElliottWaveAnalyzer()
        
        # Generate perfect 5-wave pattern with added noise
        base_prices = [100, 120, 110, 145, 130, 150]
        swing_points = []
        
        for i, price in enumerate(base_prices):
            # Add random noise
            noisy_price = price + np.random.normal(0, price * noise_level)
            swing_type = SwingType.LOW if i % 2 == 0 else SwingType.HIGH
            swing_points.append(SwingPoint(i * 10, noisy_price, swing_type))
        
        waves = analyzer.identify_waves(swing_points)
        patterns = analyzer.find_patterns(waves)
        
        # Should still detect impulse pattern even with noise
        impulse_patterns = [p for p in patterns if p.pattern_type == WaveType.IMPULSE]
        
        if noise_level <= 0.01:  # Low noise should definitely detect pattern
            assert len(impulse_patterns) > 0
        
        # Pattern confidence should decrease with noise
        if impulse_patterns:
            pattern = impulse_patterns[0]
            expected_min_confidence = max(0.3, 0.8 - noise_level * 20)
            assert pattern.confidence >= expected_min_confidence
    
    def test_fibonacci_accuracy(self):
        """Test accuracy of Fibonacci level detection"""
        calculator = FibonacciCalculator()
        
        # Test with known Fibonacci relationships
        test_cases = [
            (100, 161.8, 161.8),  # 161.8% extension
            (100, 138.2, 138.2),  # 38.2% retracement from 200
            (100, 150, 123.6),    # 50% retracement
        ]
        
        for start, end, expected_level in test_cases:
            if expected_level > end:  # Extension case
                extensions = calculator.calculate_extension_levels(start, end, start)
                found_match = any(abs(price - expected_level) < 1.0 
                                for price in extensions.values())
                assert found_match, f"Failed to find {expected_level} in extensions"
            else:  # Retracement case
                retracements = calculator.calculate_retracement_levels(start, end)
                found_match = any(abs(price - expected_level) < 1.0 
                                for price in retracements.values())
                assert found_match, f"Failed to find {expected_level} in retracements"