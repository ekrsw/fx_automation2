"""
Elliott Wave Prediction Engine
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.analysis.elliott_wave.base import (
    Wave, WavePattern, WaveType, WaveLabel, WaveDirection,
    FibonacciCalculator
)
from app.analysis.indicators.swing_detector import SwingPoint, SwingType
from app.utils.logger import analysis_logger


class WavePredictor:
    """
    Advanced Elliott Wave prediction engine
    
    Provides:
    - Next wave price targets based on current incomplete patterns
    - Time projections for wave completion
    - Probability analysis for different scenarios
    - Risk/reward assessments
    """
    
    def __init__(self,
                 fibonacci_tolerance: float = 0.1,
                 confidence_threshold: float = 0.4,
                 max_prediction_scenarios: int = 5):
        """
        Initialize wave predictor
        
        Args:
            fibonacci_tolerance: Tolerance for Fibonacci level matching
            confidence_threshold: Minimum confidence for predictions
            max_prediction_scenarios: Maximum scenarios to generate
        """
        self.fibonacci_tolerance = fibonacci_tolerance
        self.confidence_threshold = confidence_threshold
        self.max_prediction_scenarios = max_prediction_scenarios
        
        self.fibonacci_calculator = FibonacciCalculator()
        
        # Common Fibonacci ratios for different wave types
        self.impulse_ratios = {
            'wave_2_retracement': [0.382, 0.500, 0.618],
            'wave_3_extension': [1.618, 2.618, 1.000],
            'wave_4_retracement': [0.236, 0.382, 0.500],
            'wave_5_projection': [0.618, 1.000, 1.618]
        }
        
        self.corrective_ratios = {
            'wave_b_retracement': [0.382, 0.500, 0.618, 0.786],
            'wave_c_projection': [0.618, 1.000, 1.618, 2.618]
        }
    
    def predict_next_wave(self, 
                         current_pattern: WavePattern,
                         current_price: float,
                         current_time: datetime = None) -> Dict[str, Any]:
        """
        Predict the next wave in an incomplete pattern
        
        Args:
            current_pattern: Incomplete wave pattern
            current_price: Current market price
            current_time: Current time (defaults to now)
            
        Returns:
            Prediction results with targets and scenarios
        """
        if current_time is None:
            current_time = datetime.utcnow()
        
        try:
            prediction = {
                'pattern_type': current_pattern.pattern_type.value,
                'current_wave_count': len(current_pattern.waves),
                'completion_percentage': current_pattern.completion_percentage,
                'next_expected_wave': current_pattern.next_expected_wave.value if current_pattern.next_expected_wave else None,
                'current_price': current_price,
                'scenarios': [],
                'primary_targets': {},
                'risk_levels': {},
                'time_projections': {},
                'confidence': 0.0
            }
            
            if current_pattern.pattern_type == WaveType.IMPULSE:
                prediction = self._predict_impulse_continuation(
                    current_pattern, current_price, current_time, prediction
                )
            elif current_pattern.pattern_type == WaveType.CORRECTIVE:
                prediction = self._predict_corrective_continuation(
                    current_pattern, current_price, current_time, prediction
                )
            
            # Calculate overall prediction confidence
            prediction['confidence'] = self._calculate_prediction_confidence(
                current_pattern, prediction['scenarios']
            )
            
            # Sort scenarios by probability
            prediction['scenarios'].sort(
                key=lambda x: x.get('probability', 0), reverse=True
            )
            
            analysis_logger.debug(f"Generated {len(prediction['scenarios'])} prediction scenarios")
            return prediction
            
        except Exception as e:
            analysis_logger.error(f"Error predicting next wave: {e}")
            return {'error': str(e)}
    
    def _predict_impulse_continuation(self,
                                    pattern: WavePattern,
                                    current_price: float,
                                    current_time: datetime,
                                    prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Predict continuation of impulse pattern"""
        wave_count = len(pattern.waves)
        
        if wave_count == 1:
            # After wave 1, predict wave 2 retracement
            prediction['scenarios'] = self._predict_wave_2_scenarios(
                pattern.waves[0], current_price
            )
            
        elif wave_count == 2:
            # After wave 2, predict wave 3 extension
            prediction['scenarios'] = self._predict_wave_3_scenarios(
                pattern.waves[0], pattern.waves[1], current_price
            )
            
        elif wave_count == 3:
            # After wave 3, predict wave 4 retracement
            prediction['scenarios'] = self._predict_wave_4_scenarios(
                pattern.waves, current_price
            )
            
        elif wave_count == 4:
            # After wave 4, predict wave 5 final target
            prediction['scenarios'] = self._predict_wave_5_scenarios(
                pattern.waves, current_price
            )
        
        # Generate primary targets from scenarios
        prediction['primary_targets'] = self._extract_primary_targets(
            prediction['scenarios']
        )
        
        # Calculate risk levels
        prediction['risk_levels'] = self._calculate_impulse_risk_levels(
            pattern.waves, current_price
        )
        
        return prediction
    
    def _predict_corrective_continuation(self,
                                       pattern: WavePattern,
                                       current_price: float,
                                       current_time: datetime,
                                       prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Predict continuation of corrective pattern"""
        wave_count = len(pattern.waves)
        
        if wave_count == 1:
            # After wave A, predict wave B retracement
            prediction['scenarios'] = self._predict_wave_b_scenarios(
                pattern.waves[0], current_price
            )
            
        elif wave_count == 2:
            # After wave B, predict wave C target
            prediction['scenarios'] = self._predict_wave_c_scenarios(
                pattern.waves[0], pattern.waves[1], current_price
            )
        
        # Generate primary targets from scenarios
        prediction['primary_targets'] = self._extract_primary_targets(
            prediction['scenarios']
        )
        
        # Calculate risk levels
        prediction['risk_levels'] = self._calculate_corrective_risk_levels(
            pattern.waves, current_price
        )
        
        return prediction
    
    def _predict_wave_2_scenarios(self, 
                                 wave_1: Wave, 
                                 current_price: float) -> List[Dict[str, Any]]:
        """Predict Wave 2 retracement scenarios"""
        scenarios = []
        
        wave_1_start = wave_1.start_point.price
        wave_1_end = wave_1.end_point.price
        
        # Calculate retracement levels
        retracements = self.fibonacci_calculator.calculate_retracement_levels(
            wave_1_start, wave_1_end, self.impulse_ratios['wave_2_retracement']
        )
        
        for i, (level, target_price) in enumerate(retracements.items()):
            # Calculate probability based on common retracement levels
            if level == 0.618:
                probability = 0.4  # Most common
            elif level == 0.500:
                probability = 0.35
            elif level == 0.382:
                probability = 0.25
            else:
                probability = 0.1
            
            scenario = {
                'wave': 'Wave 2',
                'type': 'retracement',
                'fibonacci_level': level,
                'target_price': target_price,
                'distance_from_current': abs(target_price - current_price),
                'probability': probability,
                'description': f"Wave 2 retraces {level:.1%} of Wave 1"
            }
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _predict_wave_3_scenarios(self,
                                 wave_1: Wave,
                                 wave_2: Wave,
                                 current_price: float) -> List[Dict[str, Any]]:
        """Predict Wave 3 extension scenarios"""
        scenarios = []
        
        wave_1_size = wave_1.price_range
        wave_2_end = wave_2.end_point.price
        
        # Calculate extension targets from wave 2 end
        for ratio in self.impulse_ratios['wave_3_extension']:
            if wave_1.is_bullish:
                target_price = wave_2_end + (wave_1_size * ratio)
            else:
                target_price = wave_2_end - (wave_1_size * ratio)
            
            # Probability based on common Wave 3 relationships
            if ratio == 1.618:
                probability = 0.5  # Most common extension
            elif ratio == 1.000:
                probability = 0.3  # Equal to Wave 1
            elif ratio == 2.618:
                probability = 0.2  # Strong extension
            else:
                probability = 0.1
            
            scenario = {
                'wave': 'Wave 3',
                'type': 'extension',
                'fibonacci_level': ratio,
                'target_price': target_price,
                'distance_from_current': abs(target_price - current_price),
                'probability': probability,
                'description': f"Wave 3 extends {ratio:.2f} times Wave 1"
            }
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _predict_wave_4_scenarios(self,
                                 waves: List[Wave],
                                 current_price: float) -> List[Dict[str, Any]]:
        """Predict Wave 4 retracement scenarios"""
        scenarios = []
        
        wave_3_start = waves[2].start_point.price
        wave_3_end = waves[2].end_point.price
        wave_1_end = waves[0].end_point.price
        
        # Calculate retracement levels for Wave 4
        retracements = self.fibonacci_calculator.calculate_retracement_levels(
            wave_3_start, wave_3_end, self.impulse_ratios['wave_4_retracement']
        )
        
        for level, target_price in retracements.items():
            # Check overlap with Wave 1 (invalid for impulse)
            valid = True
            if waves[0].is_bullish and target_price <= wave_1_end:
                valid = False
            elif not waves[0].is_bullish and target_price >= wave_1_end:
                valid = False
            
            if not valid:
                continue
            
            # Probability based on common Wave 4 patterns
            if level == 0.382:
                probability = 0.4  # Most common
            elif level == 0.236:
                probability = 0.35
            elif level == 0.500:
                probability = 0.25
            else:
                probability = 0.1
            
            scenario = {
                'wave': 'Wave 4',
                'type': 'retracement',
                'fibonacci_level': level,
                'target_price': target_price,
                'distance_from_current': abs(target_price - current_price),
                'probability': probability,
                'description': f"Wave 4 retraces {level:.1%} of Wave 3",
                'validation': 'valid' if valid else 'overlaps_wave_1'
            }
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _predict_wave_5_scenarios(self,
                                 waves: List[Wave],
                                 current_price: float) -> List[Dict[str, Any]]:
        """Predict Wave 5 target scenarios"""
        scenarios = []
        
        wave_1 = waves[0]
        wave_4_end = waves[3].end_point.price
        
        # Calculate Wave 5 projections
        for ratio in self.impulse_ratios['wave_5_projection']:
            if wave_1.is_bullish:
                target_price = wave_4_end + (wave_1.price_range * ratio)
            else:
                target_price = wave_4_end - (wave_1.price_range * ratio)
            
            # Probability based on common Wave 5 relationships
            if ratio == 1.000:
                probability = 0.4  # Equal to Wave 1
            elif ratio == 0.618:
                probability = 0.35  # Truncated 5th
            elif ratio == 1.618:
                probability = 0.25  # Extended 5th
            else:
                probability = 0.1
            
            scenario = {
                'wave': 'Wave 5',
                'type': 'projection',
                'fibonacci_level': ratio,
                'target_price': target_price,
                'distance_from_current': abs(target_price - current_price),
                'probability': probability,
                'description': f"Wave 5 projects {ratio:.2f} times Wave 1"
            }
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _predict_wave_b_scenarios(self,
                                 wave_a: Wave,
                                 current_price: float) -> List[Dict[str, Any]]:
        """Predict Wave B retracement scenarios"""
        scenarios = []
        
        wave_a_start = wave_a.start_point.price
        wave_a_end = wave_a.end_point.price
        
        # Calculate retracement levels
        retracements = self.fibonacci_calculator.calculate_retracement_levels(
            wave_a_start, wave_a_end, self.corrective_ratios['wave_b_retracement']
        )
        
        for level, target_price in retracements.items():
            # Probability based on common Wave B patterns
            if level == 0.618:
                probability = 0.3
            elif level == 0.500:
                probability = 0.25
            elif level == 0.382:
                probability = 0.25
            elif level == 0.786:
                probability = 0.2  # Deep retracement
            else:
                probability = 0.1
            
            scenario = {
                'wave': 'Wave B',
                'type': 'retracement',
                'fibonacci_level': level,
                'target_price': target_price,
                'distance_from_current': abs(target_price - current_price),
                'probability': probability,
                'description': f"Wave B retraces {level:.1%} of Wave A"
            }
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _predict_wave_c_scenarios(self,
                                 wave_a: Wave,
                                 wave_b: Wave,
                                 current_price: float) -> List[Dict[str, Any]]:
        """Predict Wave C target scenarios"""
        scenarios = []
        
        wave_a_size = wave_a.price_range
        wave_b_end = wave_b.end_point.price
        
        # Calculate Wave C projections
        for ratio in self.corrective_ratios['wave_c_projection']:
            if wave_a.is_bullish:
                target_price = wave_b_end + (wave_a_size * ratio)
            else:
                target_price = wave_b_end - (wave_a_size * ratio)
            
            # Probability based on common Wave C relationships
            if ratio == 1.000:
                probability = 0.4  # Equal to Wave A
            elif ratio == 1.618:
                probability = 0.3  # Extended C
            elif ratio == 0.618:
                probability = 0.2  # Truncated C
            elif ratio == 2.618:
                probability = 0.1  # Very extended
            else:
                probability = 0.05
            
            scenario = {
                'wave': 'Wave C',
                'type': 'projection',
                'fibonacci_level': ratio,
                'target_price': target_price,
                'distance_from_current': abs(target_price - current_price),
                'probability': probability,
                'description': f"Wave C projects {ratio:.2f} times Wave A"
            }
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _extract_primary_targets(self, scenarios: List[Dict[str, Any]]) -> Dict[str, float]:
        """Extract primary price targets from scenarios"""
        if not scenarios:
            return {}
        
        # Sort by probability
        sorted_scenarios = sorted(scenarios, key=lambda x: x.get('probability', 0), reverse=True)
        
        targets = {}
        targets['most_probable'] = sorted_scenarios[0]['target_price']
        
        if len(sorted_scenarios) >= 2:
            targets['alternative'] = sorted_scenarios[1]['target_price']
        
        if len(sorted_scenarios) >= 3:
            targets['conservative'] = sorted_scenarios[-1]['target_price']
        
        # Calculate average target weighted by probability
        total_weight = sum(s.get('probability', 0) for s in scenarios)
        if total_weight > 0:
            weighted_target = sum(
                s['target_price'] * s.get('probability', 0) 
                for s in scenarios
            ) / total_weight
            targets['weighted_average'] = weighted_target
        
        return targets
    
    def _calculate_impulse_risk_levels(self, 
                                     waves: List[Wave], 
                                     current_price: float) -> Dict[str, float]:
        """Calculate risk levels for impulse patterns"""
        risk_levels = {}
        
        if len(waves) >= 2:
            # Wave 2 low as stop loss
            wave_2_extreme = waves[1].end_point.price
            risk_levels['wave_2_low'] = wave_2_extreme
        
        if len(waves) >= 4:
            # Wave 4 low as stop loss for Wave 5
            wave_4_extreme = waves[3].end_point.price
            risk_levels['wave_4_low'] = wave_4_extreme
        
        # Previous wave extreme as immediate stop
        if waves:
            last_wave = waves[-1]
            risk_levels['previous_extreme'] = last_wave.end_point.price
        
        return risk_levels
    
    def _calculate_corrective_risk_levels(self,
                                        waves: List[Wave],
                                        current_price: float) -> Dict[str, float]:
        """Calculate risk levels for corrective patterns"""
        risk_levels = {}
        
        if len(waves) >= 1:
            # Wave A start as invalidation level
            wave_a_start = waves[0].start_point.price
            risk_levels['wave_a_start'] = wave_a_start
        
        if len(waves) >= 2:
            # Wave B extreme as stop loss
            wave_b_extreme = waves[1].end_point.price
            risk_levels['wave_b_extreme'] = wave_b_extreme
        
        return risk_levels
    
    def _calculate_prediction_confidence(self,
                                       pattern: WavePattern,
                                       scenarios: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence in predictions"""
        confidence_factors = []
        
        # Factor 1: Pattern confidence
        confidence_factors.append(pattern.confidence * 0.4)
        
        # Factor 2: Number of scenarios (more = better)
        scenario_factor = min(1.0, len(scenarios) / 3.0) * 0.2
        confidence_factors.append(scenario_factor)
        
        # Factor 3: Probability distribution (higher max probability = better)
        if scenarios:
            max_probability = max(s.get('probability', 0) for s in scenarios)
            confidence_factors.append(max_probability * 0.4)
        else:
            confidence_factors.append(0.0)
        
        return sum(confidence_factors)
    
    def predict_pattern_completion(self,
                                 incomplete_patterns: List[WavePattern],
                                 current_price: float,
                                 current_time: datetime = None) -> List[Dict[str, Any]]:
        """
        Predict completion scenarios for multiple incomplete patterns
        
        Args:
            incomplete_patterns: List of incomplete patterns
            current_price: Current market price
            current_time: Current time
            
        Returns:
            List of completion predictions
        """
        if current_time is None:
            current_time = datetime.utcnow()
        
        predictions = []
        
        for pattern in incomplete_patterns:
            if pattern.completion_percentage < 1.0:  # Only incomplete patterns
                prediction = self.predict_next_wave(pattern, current_price, current_time)
                
                if prediction.get('confidence', 0) >= self.confidence_threshold:
                    predictions.append(prediction)
        
        # Sort by confidence
        predictions.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return predictions[:self.max_prediction_scenarios]


# Create default predictor instance
wave_predictor = WavePredictor()