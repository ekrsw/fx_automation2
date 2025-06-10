"""
Elliott Wave Analyzer Implementation
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import combinations

from app.analysis.elliott_wave.base import (
    ElliottWaveBase, Wave, WavePattern, WaveType, WaveLabel, WaveDirection,
    FibonacciCalculator, WaveValidator
)
from app.analysis.indicators.swing_detector import SwingPoint, SwingType
from app.utils.logger import analysis_logger


class ElliottWaveAnalyzer(ElliottWaveBase):
    """
    Complete Elliott Wave analyzer implementation
    
    Implements advanced Elliott Wave pattern recognition with:
    - 5-wave impulse pattern detection
    - 3-wave corrective pattern detection
    - Fibonacci ratio analysis
    - Wave degree classification
    - Price and time projections
    """
    
    def __init__(self,
                 fibonacci_tolerance: float = 0.1,
                 min_wave_size_pct: float = 0.1,
                 max_waves_to_analyze: int = 20,
                 wave_degree_levels: int = 3):
        """
        Initialize Elliott Wave analyzer
        
        Args:
            fibonacci_tolerance: Tolerance for Fibonacci ratio matching
            min_wave_size_pct: Minimum wave size as percentage
            max_waves_to_analyze: Maximum number of waves to analyze
            wave_degree_levels: Number of wave degree levels
        """
        super().__init__(
            fibonacci_tolerance=fibonacci_tolerance,
            min_wave_size_pct=min_wave_size_pct,
            max_waves_to_analyze=max_waves_to_analyze,
            wave_degree_levels=wave_degree_levels
        )
        
        # Wave pattern templates for matching
        self.impulse_template = [WaveLabel.WAVE_1, WaveLabel.WAVE_2, WaveLabel.WAVE_3, 
                               WaveLabel.WAVE_4, WaveLabel.WAVE_5]
        self.corrective_template = [WaveLabel.WAVE_A, WaveLabel.WAVE_B, WaveLabel.WAVE_C]
    
    def identify_waves(self, swing_points: List[SwingPoint]) -> List[Wave]:
        """
        Identify Elliott Waves from swing points
        
        Args:
            swing_points: List of swing points
            
        Returns:
            List of identified waves
        """
        if len(swing_points) < 2:
            return []
        
        try:
            waves = []
            
            # Convert swing points to waves
            for i in range(len(swing_points) - 1):
                start_point = swing_points[i]
                end_point = swing_points[i + 1]
                
                # Determine wave direction
                if end_point.price > start_point.price:
                    direction = WaveDirection.UP
                elif end_point.price < start_point.price:
                    direction = WaveDirection.DOWN
                else:
                    direction = WaveDirection.SIDEWAYS
                
                # Create wave
                wave = Wave(
                    label=WaveLabel.UNKNOWN,  # Will be determined later
                    wave_type=WaveType.UNKNOWN,
                    direction=direction,
                    start_point=start_point,
                    end_point=end_point,
                    degree=0,
                    confidence=0.5  # Initial confidence
                )
                
                # Calculate basic wave metrics
                wave.metadata = {
                    'price_move': wave.price_move,
                    'price_range': wave.price_range,
                    'duration_bars': wave.duration_bars,
                    'slope': wave.price_move / max(1, wave.duration_bars)
                }
                
                waves.append(wave)
            
            # Filter waves by minimum size
            filtered_waves = self._filter_waves_by_size(waves)
            
            # Analyze wave relationships and assign confidence
            analyzed_waves = self._analyze_wave_relationships(filtered_waves)
            
            analysis_logger.debug(f"Identified {len(analyzed_waves)} Elliott waves from {len(swing_points)} swing points")
            return analyzed_waves
            
        except Exception as e:
            analysis_logger.error(f"Error identifying Elliott waves: {e}")
            return []
    
    def find_patterns(self, waves: List[Wave]) -> List[WavePattern]:
        """
        Find Elliott Wave patterns in identified waves
        
        Args:
            waves: List of waves
            
        Returns:
            List of wave patterns
        """
        patterns = []
        
        try:
            # Find impulse patterns (5 waves)
            impulse_patterns = self._find_impulse_patterns(waves)
            patterns.extend(impulse_patterns)
            
            # Find corrective patterns (3 waves)
            corrective_patterns = self._find_corrective_patterns(waves)
            patterns.extend(corrective_patterns)
            
            # Sort patterns by confidence
            patterns.sort(key=lambda p: p.confidence, reverse=True)
            
            # Limit to max patterns
            max_patterns = min(10, len(patterns))  # Reasonable limit
            patterns = patterns[:max_patterns]
            
            analysis_logger.debug(f"Found {len(patterns)} Elliott wave patterns")
            return patterns
            
        except Exception as e:
            analysis_logger.error(f"Error finding Elliott wave patterns: {e}")
            return []
    
    def _filter_waves_by_size(self, waves: List[Wave]) -> List[Wave]:
        """Filter waves by minimum size requirement"""
        if not waves:
            return []
        
        # Calculate average wave size for reference
        avg_wave_size = np.mean([w.price_range for w in waves])
        min_size = avg_wave_size * self.min_wave_size_pct
        
        filtered_waves = []
        for wave in waves:
            if wave.price_range >= min_size:
                filtered_waves.append(wave)
            else:
                analysis_logger.debug(f"Filtered out small wave: {wave.price_range:.5f} < {min_size:.5f}")
        
        return filtered_waves
    
    def _analyze_wave_relationships(self, waves: List[Wave]) -> List[Wave]:
        """Analyze relationships between waves and update confidence"""
        if len(waves) < 2:
            return waves
        
        for i, wave in enumerate(waves):
            confidence_factors = []
            
            # Factor 1: Wave size relative to neighbors
            size_consistency = self._calculate_size_consistency(waves, i)
            confidence_factors.append(size_consistency * 0.3)
            
            # Factor 2: Direction alternation
            direction_score = self._calculate_direction_score(waves, i)
            confidence_factors.append(direction_score * 0.3)
            
            # Factor 3: Time proportion
            time_score = self._calculate_time_proportion_score(waves, i)
            confidence_factors.append(time_score * 0.2)
            
            # Factor 4: Fibonacci ratios (if applicable)
            fib_score = self._calculate_fibonacci_score(waves, i)
            confidence_factors.append(fib_score * 0.2)
            
            # Update wave confidence
            wave.confidence = min(1.0, sum(confidence_factors))
            
            # Update metadata
            wave.metadata.update({
                'size_consistency': size_consistency,
                'direction_score': direction_score,
                'time_score': time_score,
                'fibonacci_score': fib_score
            })
        
        return waves
    
    def _calculate_size_consistency(self, waves: List[Wave], index: int) -> float:
        """Calculate how consistent wave size is with neighbors"""
        if len(waves) < 3 or index == 0 or index == len(waves) - 1:
            return 0.5
        
        current_size = waves[index].price_range
        prev_size = waves[index - 1].price_range
        next_size = waves[index + 1].price_range
        
        # Calculate size ratios
        ratios = []
        if prev_size > 0:
            ratios.append(current_size / prev_size)
        if next_size > 0:
            ratios.append(current_size / next_size)
        
        if not ratios:
            return 0.5
        
        # Good ratios are between 0.2 and 5.0 (not too different)
        avg_ratio = np.mean(ratios)
        if 0.2 <= avg_ratio <= 5.0:
            return 1.0 - abs(np.log(avg_ratio)) / np.log(5)  # Scale to 0-1
        else:
            return 0.1
    
    def _calculate_direction_score(self, waves: List[Wave], index: int) -> float:
        """Calculate direction alternation score"""
        if index == 0:
            return 0.5
        
        current_direction = waves[index].direction
        prev_direction = waves[index - 1].direction
        
        # Elliott waves should generally alternate direction
        if current_direction != prev_direction and current_direction != WaveDirection.SIDEWAYS:
            return 1.0
        elif current_direction == WaveDirection.SIDEWAYS:
            return 0.3
        else:
            return 0.1
    
    def _calculate_time_proportion_score(self, waves: List[Wave], index: int) -> float:
        """Calculate time proportion score"""
        if len(waves) < 2:
            return 0.5
        
        current_duration = waves[index].duration_bars
        if current_duration == 0:
            return 0.1
        
        # Calculate average duration
        avg_duration = np.mean([w.duration_bars for w in waves if w.duration_bars > 0])
        
        if avg_duration > 0:
            ratio = current_duration / avg_duration
            # Good time proportions are between 0.3 and 3.0
            if 0.3 <= ratio <= 3.0:
                return 1.0 - abs(np.log(ratio)) / np.log(3)
            else:
                return 0.2
        
        return 0.5
    
    def _calculate_fibonacci_score(self, waves: List[Wave], index: int) -> float:
        """Calculate Fibonacci relationship score"""
        if index < 2:
            return 0.5
        
        current_wave = waves[index]
        
        # Look for Fibonacci relationships with previous waves
        fib_scores = []
        
        for i in range(max(0, index - 3), index):
            prev_wave = waves[i]
            if prev_wave.price_range > 0:
                ratio = current_wave.price_range / prev_wave.price_range
                fib_match = self.fibonacci_calculator.find_closest_fibonacci_level(
                    ratio, self.fibonacci_tolerance
                )
                if fib_match:
                    fib_scores.append(1.0)
                else:
                    fib_scores.append(0.1)
        
        return np.mean(fib_scores) if fib_scores else 0.5
    
    def _find_impulse_patterns(self, waves: List[Wave]) -> List[WavePattern]:
        """Find 5-wave impulse patterns"""
        patterns = []
        
        if len(waves) < 5:
            return patterns
        
        # Look for 5-wave sequences
        for i in range(len(waves) - 4):
            sequence = waves[i:i+5]
            
            # Create potential impulse pattern
            pattern = WavePattern(
                pattern_type=WaveType.IMPULSE,
                waves=sequence.copy()
            )
            
            # Label waves
            for j, wave in enumerate(pattern.waves):
                wave.label = self.impulse_template[j]
                wave.wave_type = WaveType.IMPULSE
            
            # Validate pattern
            validation = self.validate_pattern(pattern)
            pattern.confidence = validation.get('confidence', 0.0)
            
            # Calculate completion percentage
            pattern.completion_percentage = 1.0  # 5 waves = complete
            
            # Calculate price targets
            pattern.price_targets = self.calculate_price_targets(pattern)
            
            # Add Fibonacci analysis
            fib_analysis = self.fibonacci_calculator.analyze_wave_ratios(sequence)
            pattern.metadata = {
                'validation': validation,
                'fibonacci_analysis': fib_analysis,
                'pattern_strength': self._calculate_pattern_strength(sequence)
            }
            
            # Only add patterns above confidence threshold
            if pattern.confidence >= self.analysis_config.get('confidence_threshold', 0.6):
                patterns.append(pattern)
        
        return patterns
    
    def _find_corrective_patterns(self, waves: List[Wave]) -> List[WavePattern]:
        """Find 3-wave corrective patterns"""
        patterns = []
        
        if len(waves) < 3:
            return patterns
        
        # Look for 3-wave sequences
        for i in range(len(waves) - 2):
            sequence = waves[i:i+3]
            
            # Create potential corrective pattern
            pattern = WavePattern(
                pattern_type=WaveType.CORRECTIVE,
                waves=sequence.copy()
            )
            
            # Label waves
            for j, wave in enumerate(pattern.waves):
                wave.label = self.corrective_template[j]
                wave.wave_type = WaveType.CORRECTIVE
            
            # Validate pattern
            validation = self.validate_pattern(pattern)
            pattern.confidence = validation.get('confidence', 0.0)
            
            # Calculate completion percentage
            pattern.completion_percentage = 1.0  # 3 waves = complete
            
            # Calculate price targets
            pattern.price_targets = self.calculate_price_targets(pattern)
            
            # Add Fibonacci analysis
            fib_analysis = self.fibonacci_calculator.analyze_wave_ratios(sequence)
            pattern.metadata = {
                'validation': validation,
                'fibonacci_analysis': fib_analysis,
                'pattern_strength': self._calculate_pattern_strength(sequence)
            }
            
            # Only add patterns above confidence threshold
            if pattern.confidence >= self.analysis_config.get('confidence_threshold', 0.6):
                patterns.append(pattern)
        
        return patterns
    
    def _calculate_pattern_strength(self, waves: List[Wave]) -> float:
        """Calculate overall pattern strength"""
        if not waves:
            return 0.0
        
        strength_factors = []
        
        # Factor 1: Average wave confidence
        avg_confidence = np.mean([w.confidence for w in waves])
        strength_factors.append(avg_confidence * 0.4)
        
        # Factor 2: Direction consistency
        direction_changes = 0
        for i in range(1, len(waves)):
            if waves[i].direction != waves[i-1].direction:
                direction_changes += 1
        
        expected_changes = len(waves) - 1
        direction_consistency = direction_changes / expected_changes if expected_changes > 0 else 0
        strength_factors.append(direction_consistency * 0.3)
        
        # Factor 3: Size relationships
        sizes = [w.price_range for w in waves]
        size_std = np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 1
        size_consistency = max(0, 1 - size_std)  # Lower std = higher consistency
        strength_factors.append(size_consistency * 0.3)
        
        return sum(strength_factors)
    
    def find_incomplete_patterns(self, waves: List[Wave]) -> List[WavePattern]:
        """
        Find incomplete Elliott Wave patterns for projection
        
        Args:
            waves: List of waves
            
        Returns:
            List of incomplete patterns with projections
        """
        incomplete_patterns = []
        
        try:
            # Look for incomplete impulse patterns (3-4 waves)
            for wave_count in [3, 4]:
                for i in range(len(waves) - wave_count + 1):
                    sequence = waves[i:i+wave_count]
                    
                    # Create partial impulse pattern
                    pattern = WavePattern(
                        pattern_type=WaveType.IMPULSE,
                        waves=sequence.copy()
                    )
                    
                    # Label known waves
                    for j, wave in enumerate(pattern.waves):
                        wave.label = self.impulse_template[j]
                        wave.wave_type = WaveType.IMPULSE
                    
                    # Calculate completion percentage
                    pattern.completion_percentage = wave_count / 5.0
                    
                    # Determine next expected wave
                    if wave_count == 3:
                        pattern.next_expected_wave = WaveLabel.WAVE_4
                    elif wave_count == 4:
                        pattern.next_expected_wave = WaveLabel.WAVE_5
                    
                    # Validate partial pattern (relaxed rules)
                    validation = self._validate_partial_impulse(pattern.waves)
                    pattern.confidence = validation.get('confidence', 0.0)
                    
                    if pattern.confidence >= 0.4:  # Lower threshold for incomplete
                        # Calculate projections
                        pattern.price_targets = self._calculate_wave_projections(pattern)
                        pattern.metadata = {
                            'validation': validation,
                            'is_incomplete': True,
                            'waves_remaining': 5 - wave_count
                        }
                        
                        incomplete_patterns.append(pattern)
            
            # Look for incomplete corrective patterns (1-2 waves)
            for wave_count in [1, 2]:
                for i in range(len(waves) - wave_count + 1):
                    sequence = waves[i:i+wave_count]
                    
                    # Create partial corrective pattern
                    pattern = WavePattern(
                        pattern_type=WaveType.CORRECTIVE,
                        waves=sequence.copy()
                    )
                    
                    # Label known waves
                    for j, wave in enumerate(pattern.waves):
                        wave.label = self.corrective_template[j]
                        wave.wave_type = WaveType.CORRECTIVE
                    
                    # Calculate completion percentage
                    pattern.completion_percentage = wave_count / 3.0
                    
                    # Determine next expected wave
                    if wave_count == 1:
                        pattern.next_expected_wave = WaveLabel.WAVE_B
                    elif wave_count == 2:
                        pattern.next_expected_wave = WaveLabel.WAVE_C
                    
                    # Basic validation for partial corrective
                    pattern.confidence = 0.5  # Neutral confidence for incomplete
                    
                    # Calculate projections
                    pattern.price_targets = self._calculate_wave_projections(pattern)
                    pattern.metadata = {
                        'is_incomplete': True,
                        'waves_remaining': 3 - wave_count
                    }
                    
                    incomplete_patterns.append(pattern)
            
            return incomplete_patterns
            
        except Exception as e:
            analysis_logger.error(f"Error finding incomplete patterns: {e}")
            return []
    
    def _validate_partial_impulse(self, waves: List[Wave]) -> Dict[str, Any]:
        """Validate partial impulse pattern with relaxed rules"""
        validation = {
            'is_valid': False,
            'violations': [],
            'confidence': 0.0,
            'rule_checks': {}
        }
        
        if len(waves) < 3:
            validation['violations'].append("Need at least 3 waves for partial impulse validation")
            return validation
        
        # Relaxed rules for partial patterns
        violations = 0
        total_checks = 3
        
        # Check 1: Wave 2 retracement
        if len(waves) >= 2:
            wave_1_size = waves[0].price_range
            wave_2_size = waves[1].price_range
            
            if wave_1_size > 0:
                retracement = wave_2_size / wave_1_size
                if retracement > 1.0:
                    violations += 1
                    validation['violations'].append("Wave 2 retraces more than 100% of Wave 1")
        
        # Check 2: Direction alternation
        direction_ok = True
        for i in range(1, len(waves)):
            if waves[i].direction == waves[i-1].direction:
                direction_ok = False
                break
        
        if not direction_ok:
            violations += 1
            validation['violations'].append("Waves do not alternate direction properly")
        
        # Check 3: Wave 3 size (if available)
        if len(waves) >= 3:
            wave_1_size = waves[0].price_range
            wave_3_size = waves[2].price_range
            
            # Wave 3 should be at least as large as Wave 1
            if wave_3_size < wave_1_size * 0.618:  # Relaxed rule
                violations += 1
                validation['violations'].append("Wave 3 appears too small relative to Wave 1")
        
        validation['confidence'] = max(0.0, (total_checks - violations) / total_checks)
        validation['is_valid'] = violations == 0
        
        return validation
    
    def _calculate_wave_projections(self, pattern: WavePattern) -> Dict[str, float]:
        """Calculate projections for incomplete patterns"""
        projections = {}
        
        if not pattern.waves:
            return projections
        
        if pattern.pattern_type == WaveType.IMPULSE:
            if len(pattern.waves) == 3:
                # Project Wave 4 retracement levels
                wave_3_start = pattern.waves[2].start_point.price
                wave_3_end = pattern.waves[2].end_point.price
                
                retracements = self.fibonacci_calculator.calculate_retracement_levels(
                    wave_3_start, wave_3_end, [0.236, 0.382, 0.500, 0.618]
                )
                
                for level, price in retracements.items():
                    projections[f'wave_4_retracement_{level}'] = price
            
            elif len(pattern.waves) == 4:
                # Project Wave 5 targets
                wave_1 = pattern.waves[0]
                wave_4_end = pattern.waves[3].end_point.price
                
                if wave_1.is_bullish:
                    projections['wave_5_equal_wave_1'] = wave_4_end + wave_1.price_range
                    projections['wave_5_618_wave_1'] = wave_4_end + (wave_1.price_range * 0.618)
                else:
                    projections['wave_5_equal_wave_1'] = wave_4_end - wave_1.price_range
                    projections['wave_5_618_wave_1'] = wave_4_end - (wave_1.price_range * 0.618)
        
        elif pattern.pattern_type == WaveType.CORRECTIVE:
            if len(pattern.waves) == 1:
                # Project Wave B retracement
                wave_a_start = pattern.waves[0].start_point.price
                wave_a_end = pattern.waves[0].end_point.price
                
                retracements = self.fibonacci_calculator.calculate_retracement_levels(
                    wave_a_start, wave_a_end, [0.382, 0.500, 0.618]
                )
                
                for level, price in retracements.items():
                    projections[f'wave_b_retracement_{level}'] = price
            
            elif len(pattern.waves) == 2:
                # Project Wave C targets
                wave_a = pattern.waves[0]
                wave_b_end = pattern.waves[1].end_point.price
                
                if wave_a.is_bullish:
                    projections['wave_c_equal_wave_a'] = wave_b_end + wave_a.price_range
                    projections['wave_c_618_wave_a'] = wave_b_end + (wave_a.price_range * 0.618)
                else:
                    projections['wave_c_equal_wave_a'] = wave_b_end - wave_a.price_range
                    projections['wave_c_618_wave_a'] = wave_b_end - (wave_a.price_range * 0.618)
        
        return projections


# Create default analyzer instance
elliott_wave_analyzer = ElliottWaveAnalyzer()