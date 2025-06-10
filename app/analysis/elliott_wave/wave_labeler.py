"""
Elliott Wave Labeling System
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from app.analysis.elliott_wave.base import (
    Wave, WavePattern, WaveType, WaveLabel, WaveDirection,
    FibonacciCalculator
)
from app.analysis.indicators.swing_detector import SwingPoint, SwingType
from app.utils.logger import analysis_logger


class WaveLabeler:
    """
    Advanced Elliott Wave labeling system with multiple degree support
    
    Provides automatic wave labeling with:
    - Multiple wave degrees (Primary, Intermediate, Minor, etc.)
    - Context-aware labeling
    - Confidence scoring
    - Alternative count suggestions
    """
    
    def __init__(self,
                 max_degree_levels: int = 4,
                 labeling_confidence_threshold: float = 0.6):
        """
        Initialize wave labeler
        
        Args:
            max_degree_levels: Maximum number of wave degrees to analyze
            labeling_confidence_threshold: Minimum confidence for labeling
        """
        self.max_degree_levels = max_degree_levels
        self.labeling_confidence_threshold = labeling_confidence_threshold
        
        # Wave degree hierarchy (highest to lowest)
        self.degree_hierarchy = [
            "Primary",      # Degree 0 - Largest waves
            "Intermediate", # Degree 1
            "Minor",        # Degree 2
            "Minute",       # Degree 3
            "Minuette"      # Degree 4 - Smallest waves
        ]
        
        # Labeling symbols for different degrees
        self.degree_symbols = {
            0: {"impulse": ["I", "II", "III", "IV", "V"], 
                "corrective": ["A", "B", "C"]},
            1: {"impulse": ["1", "2", "3", "4", "5"], 
                "corrective": ["a", "b", "c"]},
            2: {"impulse": ["i", "ii", "iii", "iv", "v"], 
                "corrective": ["w", "x", "y"]},
            3: {"impulse": ["(1)", "(2)", "(3)", "(4)", "(5)"], 
                "corrective": ["(a)", "(b)", "(c)"]},
            4: {"impulse": ["[1]", "[2]", "[3]", "[4]", "[5]"], 
                "corrective": ["[a]", "[b]", "[c]"]},
        }
        
        self.fibonacci_calculator = FibonacciCalculator()
    
    def label_waves(self, waves: List[Wave]) -> List[Wave]:
        """
        Apply Elliott Wave labels to waves
        
        Args:
            waves: List of waves to label
            
        Returns:
            List of labeled waves
        """
        if len(waves) < 3:
            return waves
        
        try:
            # Analyze wave structure at multiple degrees
            degree_analysis = self._analyze_wave_degrees(waves)
            
            # Apply labels based on analysis
            labeled_waves = self._apply_wave_labels(waves, degree_analysis)
            
            # Validate and adjust labels
            validated_waves = self._validate_wave_labels(labeled_waves)
            
            analysis_logger.debug(f"Labeled {len(validated_waves)} waves with Elliott Wave notation")
            return validated_waves
            
        except Exception as e:
            analysis_logger.error(f"Error labeling waves: {e}")
            return waves
    
    def _analyze_wave_degrees(self, waves: List[Wave]) -> Dict[str, Any]:
        """
        Analyze wave structure to determine degrees
        
        Args:
            waves: List of waves
            
        Returns:
            Degree analysis results
        """
        analysis = {
            'primary_patterns': [],
            'sub_patterns': [],
            'degree_assignments': {},
            'confidence_scores': {}
        }
        
        # Find major patterns (degree 0 - primary)
        primary_patterns = self._identify_primary_patterns(waves)
        analysis['primary_patterns'] = primary_patterns
        
        # For each primary pattern, look for sub-patterns
        for pattern in primary_patterns:
            sub_patterns = self._identify_sub_patterns(pattern.waves)
            analysis['sub_patterns'].extend(sub_patterns)
        
        # Assign degrees based on pattern hierarchy
        for i, wave in enumerate(waves):
            degree = self._determine_wave_degree(wave, primary_patterns, analysis['sub_patterns'])
            analysis['degree_assignments'][i] = degree
            
            # Calculate confidence for degree assignment
            confidence = self._calculate_degree_confidence(wave, degree, waves)
            analysis['confidence_scores'][i] = confidence
        
        return analysis
    
    def _identify_primary_patterns(self, waves: List[Wave]) -> List[WavePattern]:
        """Identify primary degree patterns"""
        patterns = []
        
        # Look for 5-wave and 3-wave patterns at the largest scale
        wave_sizes = [w.price_range for w in waves]
        avg_size = np.mean(wave_sizes)
        
        # Focus on larger waves for primary patterns
        significant_waves = [w for w in waves if w.price_range >= avg_size * 0.5]
        
        if len(significant_waves) >= 5:
            # Try to identify 5-wave impulse
            for i in range(len(significant_waves) - 4):
                sequence = significant_waves[i:i+5]
                
                if self._is_valid_impulse_structure(sequence):
                    pattern = WavePattern(
                        pattern_type=WaveType.IMPULSE,
                        waves=sequence,
                        confidence=self._calculate_pattern_confidence(sequence)
                    )
                    
                    # Assign primary degree labels
                    for j, wave in enumerate(pattern.waves):
                        wave.degree = 0
                        wave.label = self._get_wave_label(j, WaveType.IMPULSE, 0)
                    
                    patterns.append(pattern)
        
        if len(significant_waves) >= 3:
            # Try to identify 3-wave correction
            for i in range(len(significant_waves) - 2):
                sequence = significant_waves[i:i+3]
                
                if self._is_valid_corrective_structure(sequence):
                    pattern = WavePattern(
                        pattern_type=WaveType.CORRECTIVE,
                        waves=sequence,
                        confidence=self._calculate_pattern_confidence(sequence)
                    )
                    
                    # Assign primary degree labels
                    for j, wave in enumerate(pattern.waves):
                        wave.degree = 0
                        wave.label = self._get_wave_label(j, WaveType.CORRECTIVE, 0)
                    
                    patterns.append(pattern)
        
        return patterns
    
    def _identify_sub_patterns(self, primary_waves: List[Wave]) -> List[WavePattern]:
        """Identify sub-patterns within primary waves"""
        sub_patterns = []
        
        for primary_wave in primary_waves:
            # For each primary wave, look for internal structure
            if hasattr(primary_wave, 'sub_waves') and primary_wave.sub_waves:
                sub_waves = primary_wave.sub_waves
                
                # Look for 5-wave and 3-wave sub-patterns
                if len(sub_waves) >= 5:
                    for i in range(len(sub_waves) - 4):
                        sequence = sub_waves[i:i+5]
                        
                        if self._is_valid_impulse_structure(sequence):
                            pattern = WavePattern(
                                pattern_type=WaveType.IMPULSE,
                                waves=sequence,
                                confidence=self._calculate_pattern_confidence(sequence)
                            )
                            
                            # Assign intermediate degree labels
                            for j, wave in enumerate(pattern.waves):
                                wave.degree = 1
                                wave.label = self._get_wave_label(j, WaveType.IMPULSE, 1)
                            
                            sub_patterns.append(pattern)
                
                if len(sub_waves) >= 3:
                    for i in range(len(sub_waves) - 2):
                        sequence = sub_waves[i:i+3]
                        
                        if self._is_valid_corrective_structure(sequence):
                            pattern = WavePattern(
                                pattern_type=WaveType.CORRECTIVE,
                                waves=sequence,
                                confidence=self._calculate_pattern_confidence(sequence)
                            )
                            
                            # Assign intermediate degree labels
                            for j, wave in enumerate(pattern.waves):
                                wave.degree = 1
                                wave.label = self._get_wave_label(j, WaveType.CORRECTIVE, 1)
                            
                            sub_patterns.append(pattern)
        
        return sub_patterns
    
    def _determine_wave_degree(self, wave: Wave, primary_patterns: List[WavePattern], 
                             sub_patterns: List[WavePattern]) -> int:
        """Determine the appropriate degree for a wave"""
        # Check if wave is part of primary pattern
        for pattern in primary_patterns:
            if wave in pattern.waves:
                return 0  # Primary degree
        
        # Check if wave is part of sub-pattern
        for pattern in sub_patterns:
            if wave in pattern.waves:
                return 1  # Intermediate degree
        
        # Default to minor degree
        return 2
    
    def _calculate_degree_confidence(self, wave: Wave, degree: int, all_waves: List[Wave]) -> float:
        """Calculate confidence in degree assignment"""
        confidence_factors = []
        
        # Factor 1: Wave size relative to others at same degree
        same_degree_waves = [w for w in all_waves if getattr(w, 'degree', 2) == degree]
        if same_degree_waves:
            sizes = [w.price_range for w in same_degree_waves]
            avg_size = np.mean(sizes)
            
            if avg_size > 0:
                size_ratio = wave.price_range / avg_size
                # Good size ratio is between 0.5 and 2.0
                if 0.5 <= size_ratio <= 2.0:
                    confidence_factors.append(1.0)
                else:
                    confidence_factors.append(0.5)
        
        # Factor 2: Duration consistency
        same_degree_durations = [w.duration_bars for w in same_degree_waves if w.duration_bars > 0]
        if same_degree_durations:
            avg_duration = np.mean(same_degree_durations)
            
            if avg_duration > 0 and wave.duration_bars > 0:
                duration_ratio = wave.duration_bars / avg_duration
                if 0.5 <= duration_ratio <= 2.0:
                    confidence_factors.append(1.0)
                else:
                    confidence_factors.append(0.5)
        
        # Factor 3: Position in pattern
        if hasattr(wave, 'label') and wave.label != WaveLabel.UNKNOWN:
            confidence_factors.append(0.8)  # Bonus for being in recognized pattern
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _apply_wave_labels(self, waves: List[Wave], degree_analysis: Dict[str, Any]) -> List[Wave]:
        """Apply labels to waves based on degree analysis"""
        labeled_waves = waves.copy()
        
        for i, wave in enumerate(labeled_waves):
            degree = degree_analysis['degree_assignments'].get(i, 2)
            confidence = degree_analysis['confidence_scores'].get(i, 0.5)
            
            # Only apply labels if confidence is above threshold
            if confidence >= self.labeling_confidence_threshold:
                wave.degree = degree
                
                # Determine label based on context
                if not hasattr(wave, 'label') or wave.label == WaveLabel.UNKNOWN:
                    # Try to determine label from pattern context
                    label = self._determine_contextual_label(wave, labeled_waves, i)
                    if label:
                        wave.label = label
                
                # Update metadata
                wave.metadata.update({
                    'degree_name': self.degree_hierarchy[min(degree, len(self.degree_hierarchy)-1)],
                    'labeling_confidence': confidence,
                    'label_symbol': self._get_label_symbol(wave)
                })
        
        return labeled_waves
    
    def _determine_contextual_label(self, wave: Wave, waves: List[Wave], index: int) -> Optional[WaveLabel]:
        """Determine wave label based on context"""
        # Look at surrounding waves to determine position in pattern
        
        # Simple heuristic: alternating impulse/corrective based on direction
        if index == 0:
            return WaveLabel.WAVE_1 if wave.direction in [WaveDirection.UP, WaveDirection.DOWN] else WaveLabel.WAVE_A
        
        prev_wave = waves[index - 1]
        
        # If previous wave was labeled, try to continue the sequence
        if hasattr(prev_wave, 'label') and prev_wave.label != WaveLabel.UNKNOWN:
            return self._get_next_label_in_sequence(prev_wave.label)
        
        # Default labeling based on position and direction
        position_in_pattern = index % 5
        if wave.direction != prev_wave.direction:  # Alternating direction suggests impulse
            impulse_labels = [WaveLabel.WAVE_1, WaveLabel.WAVE_2, WaveLabel.WAVE_3, 
                            WaveLabel.WAVE_4, WaveLabel.WAVE_5]
            return impulse_labels[position_in_pattern]
        
        return None
    
    def _get_next_label_in_sequence(self, current_label: WaveLabel) -> Optional[WaveLabel]:
        """Get the next label in an Elliott Wave sequence"""
        impulse_sequence = [WaveLabel.WAVE_1, WaveLabel.WAVE_2, WaveLabel.WAVE_3, 
                          WaveLabel.WAVE_4, WaveLabel.WAVE_5]
        corrective_sequence = [WaveLabel.WAVE_A, WaveLabel.WAVE_B, WaveLabel.WAVE_C]
        
        if current_label in impulse_sequence:
            current_index = impulse_sequence.index(current_label)
            if current_index < len(impulse_sequence) - 1:
                return impulse_sequence[current_index + 1]
        
        elif current_label in corrective_sequence:
            current_index = corrective_sequence.index(current_label)
            if current_index < len(corrective_sequence) - 1:
                return corrective_sequence[current_index + 1]
        
        return None
    
    def _validate_wave_labels(self, waves: List[Wave]) -> List[Wave]:
        """Validate and adjust wave labels for consistency"""
        validated_waves = waves.copy()
        
        # Check for label sequence consistency
        for i in range(1, len(validated_waves)):
            current_wave = validated_waves[i]
            prev_wave = validated_waves[i - 1]
            
            # Ensure logical label progression
            if (hasattr(current_wave, 'label') and hasattr(prev_wave, 'label') and
                current_wave.label != WaveLabel.UNKNOWN and prev_wave.label != WaveLabel.UNKNOWN):
                
                if not self._is_valid_label_sequence(prev_wave.label, current_wave.label):
                    # Reset invalid labels
                    current_wave.label = WaveLabel.UNKNOWN
                    analysis_logger.debug(f"Reset invalid label sequence at wave {i}")
        
        return validated_waves
    
    def _is_valid_label_sequence(self, prev_label: WaveLabel, current_label: WaveLabel) -> bool:
        """Check if label sequence is valid"""
        impulse_sequence = [WaveLabel.WAVE_1, WaveLabel.WAVE_2, WaveLabel.WAVE_3, 
                          WaveLabel.WAVE_4, WaveLabel.WAVE_5]
        corrective_sequence = [WaveLabel.WAVE_A, WaveLabel.WAVE_B, WaveLabel.WAVE_C]
        
        # Check impulse sequence
        if prev_label in impulse_sequence and current_label in impulse_sequence:
            prev_index = impulse_sequence.index(prev_label)
            current_index = impulse_sequence.index(current_label)
            return current_index == prev_index + 1
        
        # Check corrective sequence
        if prev_label in corrective_sequence and current_label in corrective_sequence:
            prev_index = corrective_sequence.index(prev_label)
            current_index = corrective_sequence.index(current_label)
            return current_index == prev_index + 1
        
        # Mixed sequences are allowed (end of one pattern, start of another)
        return True
    
    def _get_wave_label(self, position: int, wave_type: WaveType, degree: int) -> WaveLabel:
        """Get appropriate wave label for position, type, and degree"""
        if wave_type == WaveType.IMPULSE:
            labels = [WaveLabel.WAVE_1, WaveLabel.WAVE_2, WaveLabel.WAVE_3, 
                     WaveLabel.WAVE_4, WaveLabel.WAVE_5]
        else:
            labels = [WaveLabel.WAVE_A, WaveLabel.WAVE_B, WaveLabel.WAVE_C]
        
        if 0 <= position < len(labels):
            return labels[position]
        
        return WaveLabel.UNKNOWN
    
    def _get_label_symbol(self, wave: Wave) -> str:
        """Get display symbol for wave label based on degree"""
        if not hasattr(wave, 'label') or not hasattr(wave, 'degree'):
            return "?"
        
        degree = min(wave.degree, len(self.degree_symbols) - 1)
        symbols = self.degree_symbols.get(degree, {})
        
        if wave.wave_type == WaveType.IMPULSE:
            impulse_symbols = symbols.get('impulse', ['1', '2', '3', '4', '5'])
            label_map = {
                WaveLabel.WAVE_1: impulse_symbols[0] if len(impulse_symbols) > 0 else '1',
                WaveLabel.WAVE_2: impulse_symbols[1] if len(impulse_symbols) > 1 else '2',
                WaveLabel.WAVE_3: impulse_symbols[2] if len(impulse_symbols) > 2 else '3',
                WaveLabel.WAVE_4: impulse_symbols[3] if len(impulse_symbols) > 3 else '4',
                WaveLabel.WAVE_5: impulse_symbols[4] if len(impulse_symbols) > 4 else '5',
            }
        else:
            corrective_symbols = symbols.get('corrective', ['A', 'B', 'C'])
            label_map = {
                WaveLabel.WAVE_A: corrective_symbols[0] if len(corrective_symbols) > 0 else 'A',
                WaveLabel.WAVE_B: corrective_symbols[1] if len(corrective_symbols) > 1 else 'B',
                WaveLabel.WAVE_C: corrective_symbols[2] if len(corrective_symbols) > 2 else 'C',
            }
        
        return label_map.get(wave.label, "?")
    
    def _is_valid_impulse_structure(self, waves: List[Wave]) -> bool:
        """Quick check for valid impulse wave structure"""
        if len(waves) != 5:
            return False
        
        # Check alternating directions
        for i in range(1, len(waves)):
            if waves[i].direction == waves[i-1].direction:
                return False
        
        # Motive waves (1, 3, 5) should be in same general direction
        motive_directions = [waves[0].direction, waves[2].direction, waves[4].direction]
        return len(set(motive_directions)) == 1
    
    def _is_valid_corrective_structure(self, waves: List[Wave]) -> bool:
        """Quick check for valid corrective wave structure"""
        if len(waves) != 3:
            return False
        
        # A and C should be in same direction, B opposite
        return (waves[0].direction == waves[2].direction and 
                waves[1].direction != waves[0].direction)
    
    def _calculate_pattern_confidence(self, waves: List[Wave]) -> float:
        """Calculate confidence score for a wave pattern"""
        if not waves:
            return 0.0
        
        confidence_factors = []
        
        # Factor 1: Size consistency
        sizes = [w.price_range for w in waves]
        if len(sizes) > 1:
            size_consistency = 1.0 - (np.std(sizes) / np.mean(sizes))
            confidence_factors.append(max(0.0, size_consistency) * 0.3)
        
        # Factor 2: Direction alternation
        direction_changes = 0
        for i in range(1, len(waves)):
            if waves[i].direction != waves[i-1].direction:
                direction_changes += 1
        
        expected_changes = len(waves) - 1
        direction_score = direction_changes / expected_changes if expected_changes > 0 else 0
        confidence_factors.append(direction_score * 0.4)
        
        # Factor 3: Average wave confidence
        wave_confidences = [getattr(w, 'confidence', 0.5) for w in waves]
        confidence_factors.append(np.mean(wave_confidences) * 0.3)
        
        return sum(confidence_factors)
    
    def generate_alternative_counts(self, waves: List[Wave]) -> List[Dict[str, Any]]:
        """
        Generate alternative Elliott Wave counts
        
        Args:
            waves: List of waves
            
        Returns:
            List of alternative count scenarios
        """
        alternatives = []
        
        try:
            # Alternative 1: Different degree interpretation
            alt_1 = self._create_degree_alternative(waves)
            if alt_1:
                alternatives.append(alt_1)
            
            # Alternative 2: Different pattern type (impulse vs corrective)
            alt_2 = self._create_pattern_type_alternative(waves)
            if alt_2:
                alternatives.append(alt_2)
            
            # Alternative 3: Extended waves (if applicable)
            alt_3 = self._create_extended_wave_alternative(waves)
            if alt_3:
                alternatives.append(alt_3)
            
            # Sort by confidence
            alternatives.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            return alternatives[:3]  # Return top 3 alternatives
            
        except Exception as e:
            analysis_logger.error(f"Error generating alternative counts: {e}")
            return []
    
    def _create_degree_alternative(self, waves: List[Wave]) -> Optional[Dict[str, Any]]:
        """Create alternative count with different degree interpretation"""
        # This is a simplified implementation
        # In practice, this would involve complex reanalysis
        
        alternative_waves = []
        for wave in waves:
            alt_wave = Wave(
                label=wave.label,
                wave_type=wave.wave_type,
                direction=wave.direction,
                start_point=wave.start_point,
                end_point=wave.end_point,
                degree=min(wave.degree + 1, self.max_degree_levels - 1),  # One degree lower
                confidence=wave.confidence * 0.8  # Lower confidence for alternative
            )
            alternative_waves.append(alt_wave)
        
        return {
            'type': 'degree_alternative',
            'description': 'Alternative interpretation with different wave degree',
            'waves': alternative_waves,
            'confidence': np.mean([w.confidence for w in alternative_waves])
        }
    
    def _create_pattern_type_alternative(self, waves: List[Wave]) -> Optional[Dict[str, Any]]:
        """Create alternative count with different pattern type"""
        # This would involve re-labeling waves as different pattern type
        return None  # Placeholder for complex implementation
    
    def _create_extended_wave_alternative(self, waves: List[Wave]) -> Optional[Dict[str, Any]]:
        """Create alternative count considering wave extensions"""
        # This would consider possibilities like extended 3rd or 5th waves
        return None  # Placeholder for complex implementation


# Create default labeler instance
wave_labeler = WaveLabeler()