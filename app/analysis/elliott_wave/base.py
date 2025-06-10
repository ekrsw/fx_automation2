"""
Elliott Wave Analysis Base Classes and Framework
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from app.analysis.indicators.swing_detector import SwingPoint, SwingType
from app.utils.logger import analysis_logger


class WaveType(Enum):
    """Elliott Wave types"""
    IMPULSE = "impulse"  # 5-wave impulse
    CORRECTIVE = "corrective"  # 3-wave correction
    UNKNOWN = "unknown"


class WaveLabel(Enum):
    """Elliott Wave labels"""
    # Impulse waves
    WAVE_1 = "1"
    WAVE_2 = "2"
    WAVE_3 = "3"
    WAVE_4 = "4"
    WAVE_5 = "5"
    
    # Corrective waves
    WAVE_A = "A"
    WAVE_B = "B"
    WAVE_C = "C"
    
    # Sub-waves (for detailed analysis)
    WAVE_I = "I"
    WAVE_II = "II"
    WAVE_III = "III"
    WAVE_IV = "IV"
    WAVE_V = "V"
    
    UNKNOWN = "?"


class WaveDirection(Enum):
    """Wave direction"""
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"


class FibonacciLevel(Enum):
    """Standard Fibonacci retracement/extension levels"""
    # Retracement levels
    RETRACEMENT_23_6 = 0.236
    RETRACEMENT_38_2 = 0.382
    RETRACEMENT_50_0 = 0.500
    RETRACEMENT_61_8 = 0.618
    RETRACEMENT_78_6 = 0.786
    
    # Extension levels
    EXTENSION_100_0 = 1.000
    EXTENSION_127_2 = 1.272
    EXTENSION_161_8 = 1.618
    EXTENSION_200_0 = 2.000
    EXTENSION_261_8 = 2.618


@dataclass
class Wave:
    """Individual Elliott Wave"""
    label: WaveLabel
    wave_type: WaveType
    direction: WaveDirection
    start_point: SwingPoint
    end_point: SwingPoint
    degree: int = 0  # Wave degree (0=primary, 1=intermediate, etc.)
    confidence: float = 0.0  # Confidence in wave identification (0-1)
    fibonacci_ratios: Dict[str, float] = field(default_factory=dict)
    sub_waves: List['Wave'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def price_move(self) -> float:
        """Calculate price movement of the wave"""
        return self.end_point.price - self.start_point.price
    
    @property
    def price_range(self) -> float:
        """Calculate absolute price range of the wave"""
        return abs(self.price_move)
    
    @property
    def duration_bars(self) -> int:
        """Calculate duration in bars"""
        return self.end_point.index - self.start_point.index
    
    @property
    def is_bullish(self) -> bool:
        """Check if wave is bullish (upward)"""
        return self.price_move > 0
    
    @property
    def is_bearish(self) -> bool:
        """Check if wave is bearish (downward)"""
        return self.price_move < 0


@dataclass
class WavePattern:
    """Elliott Wave pattern (collection of waves)"""
    pattern_type: WaveType
    waves: List[Wave]
    confidence: float = 0.0
    completion_percentage: float = 0.0  # How complete the pattern is (0-1)
    next_expected_wave: Optional[WaveLabel] = None
    price_targets: Dict[str, float] = field(default_factory=dict)
    time_targets: Dict[str, datetime] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_complete(self) -> bool:
        """Check if pattern is complete"""
        if self.pattern_type == WaveType.IMPULSE:
            return len(self.waves) == 5
        elif self.pattern_type == WaveType.CORRECTIVE:
            return len(self.waves) == 3
        return False
    
    @property
    def total_price_move(self) -> float:
        """Total price movement from start to end"""
        if not self.waves:
            return 0.0
        return self.waves[-1].end_point.price - self.waves[0].start_point.price
    
    @property
    def total_duration(self) -> int:
        """Total duration in bars"""
        if not self.waves:
            return 0
        return self.waves[-1].end_point.index - self.waves[0].start_point.index


class FibonacciCalculator:
    """
    Calculator for Fibonacci retracements and extensions
    """
    
    @staticmethod
    def calculate_retracement_levels(
        start_price: float, 
        end_price: float,
        levels: List[float] = None
    ) -> Dict[float, float]:
        """
        Calculate Fibonacci retracement levels
        
        Args:
            start_price: Starting price of the move
            end_price: Ending price of the move
            levels: Custom Fibonacci levels (default: standard retracements)
            
        Returns:
            Dictionary mapping Fibonacci level to price
        """
        if levels is None:
            levels = [0.236, 0.382, 0.500, 0.618, 0.786]
        
        price_diff = end_price - start_price
        retracements = {}
        
        for level in levels:
            retracement_price = end_price - (price_diff * level)
            retracements[level] = retracement_price
        
        return retracements
    
    @staticmethod
    def calculate_extension_levels(
        wave_start: float,
        wave_end: float,
        retracement_end: float,
        levels: List[float] = None
    ) -> Dict[float, float]:
        """
        Calculate Fibonacci extension levels
        
        Args:
            wave_start: Start of the initial wave
            wave_end: End of the initial wave
            retracement_end: End of the retracement
            levels: Custom Fibonacci levels (default: standard extensions)
            
        Returns:
            Dictionary mapping Fibonacci level to price
        """
        if levels is None:
            levels = [1.000, 1.272, 1.618, 2.000, 2.618]
        
        wave_size = abs(wave_end - wave_start)
        extensions = {}
        
        # Determine direction
        if wave_end > wave_start:  # Upward wave
            for level in levels:
                extension_price = retracement_end + (wave_size * level)
                extensions[level] = extension_price
        else:  # Downward wave
            for level in levels:
                extension_price = retracement_end - (wave_size * level)
                extensions[level] = extension_price
        
        return extensions
    
    @staticmethod
    def find_closest_fibonacci_level(
        actual_ratio: float,
        tolerance: float = 0.1
    ) -> Optional[FibonacciLevel]:
        """
        Find the closest standard Fibonacci level to an actual ratio
        
        Args:
            actual_ratio: Actual measured ratio
            tolerance: Maximum allowed deviation
            
        Returns:
            Closest FibonacciLevel or None if no match within tolerance
        """
        best_match = None
        min_difference = float('inf')
        
        for fib_level in FibonacciLevel:
            difference = abs(actual_ratio - fib_level.value)
            if difference < min_difference and difference <= tolerance:
                min_difference = difference
                best_match = fib_level
        
        return best_match
    
    @staticmethod
    def analyze_wave_ratios(waves: List[Wave]) -> Dict[str, Any]:
        """
        Analyze Fibonacci ratios between waves
        
        Args:
            waves: List of waves to analyze
            
        Returns:
            Dictionary with ratio analysis results
        """
        analysis = {
            'wave_ratios': {},
            'fibonacci_matches': {},
            'ratio_quality_score': 0.0
        }
        
        if len(waves) < 2:
            return analysis
        
        # Calculate ratios between consecutive waves
        for i in range(1, len(waves)):
            wave_curr = waves[i]
            wave_prev = waves[i-1]
            
            if wave_prev.price_range > 0:
                ratio = wave_curr.price_range / wave_prev.price_range
                ratio_key = f"wave_{i+1}_vs_wave_{i}"
                
                analysis['wave_ratios'][ratio_key] = ratio
                
                # Find closest Fibonacci level
                fib_match = FibonacciCalculator.find_closest_fibonacci_level(ratio)
                if fib_match:
                    analysis['fibonacci_matches'][ratio_key] = {
                        'fibonacci_level': fib_match.value,
                        'actual_ratio': ratio,
                        'deviation': abs(ratio - fib_match.value)
                    }
        
        # Calculate quality score based on Fibonacci matches
        if analysis['fibonacci_matches']:
            total_deviation = sum(
                match['deviation'] for match in analysis['fibonacci_matches'].values()
            )
            avg_deviation = total_deviation / len(analysis['fibonacci_matches'])
            analysis['ratio_quality_score'] = max(0.0, 1.0 - (avg_deviation * 5))  # Scale to 0-1
        
        return analysis


class WaveValidator:
    """
    Validator for Elliott Wave rules and guidelines
    """
    
    @staticmethod
    def validate_impulse_wave(waves: List[Wave]) -> Dict[str, Any]:
        """
        Validate 5-wave impulse pattern according to Elliott Wave rules
        
        Rules:
        1. Wave 2 cannot retrace more than 100% of Wave 1
        2. Wave 3 cannot be the shortest of waves 1, 3, and 5
        3. Wave 4 cannot overlap with Wave 1 (except in diagonal triangles)
        4. Waves 1, 3, 5 should be in the same direction as the overall trend
        5. Waves 2, 4 should be counter-trend corrections
        
        Args:
            waves: List of 5 waves forming potential impulse
            
        Returns:
            Validation results dictionary
        """
        validation = {
            'is_valid': False,
            'violations': [],
            'confidence': 0.0,
            'rule_checks': {}
        }
        
        if len(waves) != 5:
            validation['violations'].append(f"Impulse wave must have 5 waves, got {len(waves)}")
            return validation
        
        # Rule 1: Wave 2 retracement
        wave_1_size = waves[0].price_range
        wave_2_size = waves[1].price_range
        
        if wave_1_size > 0:
            wave_2_retracement = wave_2_size / wave_1_size
            validation['rule_checks']['wave_2_retracement'] = wave_2_retracement
            
            if wave_2_retracement > 1.0:
                validation['violations'].append("Wave 2 retraces more than 100% of Wave 1")
        
        # Rule 2: Wave 3 cannot be shortest
        wave_1_size = waves[0].price_range
        wave_3_size = waves[2].price_range
        wave_5_size = waves[4].price_range
        
        motive_waves = [wave_1_size, wave_3_size, wave_5_size]
        validation['rule_checks']['wave_sizes'] = {
            'wave_1': wave_1_size,
            'wave_3': wave_3_size,
            'wave_5': wave_5_size
        }
        
        if wave_3_size == min(motive_waves):
            validation['violations'].append("Wave 3 is the shortest of waves 1, 3, and 5")
        
        # Rule 3: Wave 4 overlap with Wave 1
        wave_1_end = waves[0].end_point.price
        wave_4_extreme = waves[3].end_point.price
        
        # Check overlap based on trend direction
        if waves[0].is_bullish:  # Uptrend
            if wave_4_extreme <= wave_1_end:
                validation['violations'].append("Wave 4 overlaps with Wave 1 (bearish overlap)")
        else:  # Downtrend
            if wave_4_extreme >= wave_1_end:
                validation['violations'].append("Wave 4 overlaps with Wave 1 (bullish overlap)")
        
        # Rule 4: Direction consistency for motive waves (1, 3, 5)
        motive_directions = [waves[0].is_bullish, waves[2].is_bullish, waves[4].is_bullish]
        if not all(motive_directions) and not all(not d for d in motive_directions):
            validation['violations'].append("Motive waves (1, 3, 5) have inconsistent directions")
        
        # Rule 5: Corrective waves (2, 4) should be counter-trend
        corrective_directions = [waves[1].is_bullish, waves[3].is_bullish]
        expected_corrective_direction = not waves[0].is_bullish
        
        for i, is_bullish in enumerate(corrective_directions):
            if is_bullish != expected_corrective_direction:
                validation['violations'].append(f"Wave {i*2 + 2} direction inconsistent with correction")
        
        # Calculate confidence
        total_rules = 5
        violations_count = len(validation['violations'])
        validation['confidence'] = max(0.0, (total_rules - violations_count) / total_rules)
        validation['is_valid'] = violations_count == 0
        
        return validation
    
    @staticmethod
    def validate_corrective_wave(waves: List[Wave]) -> Dict[str, Any]:
        """
        Validate 3-wave corrective pattern (A-B-C)
        
        Rules:
        1. Wave A and C should be in the same direction
        2. Wave B should be counter to A and C
        3. Wave C should be at least 61.8% of Wave A
        4. Wave B should not exceed 138.2% of Wave A
        
        Args:
            waves: List of 3 waves forming potential correction
            
        Returns:
            Validation results dictionary
        """
        validation = {
            'is_valid': False,
            'violations': [],
            'confidence': 0.0,
            'rule_checks': {}
        }
        
        if len(waves) != 3:
            validation['violations'].append(f"Corrective wave must have 3 waves, got {len(waves)}")
            return validation
        
        wave_a, wave_b, wave_c = waves
        
        # Rule 1: Wave A and C same direction
        if wave_a.is_bullish != wave_c.is_bullish:
            validation['violations'].append("Wave A and C have different directions")
        
        # Rule 2: Wave B counter direction
        if wave_b.is_bullish == wave_a.is_bullish:
            validation['violations'].append("Wave B has same direction as Wave A")
        
        # Rule 3: Wave C size relative to Wave A
        if wave_a.price_range > 0:
            c_to_a_ratio = wave_c.price_range / wave_a.price_range
            validation['rule_checks']['c_to_a_ratio'] = c_to_a_ratio
            
            if c_to_a_ratio < 0.618:
                validation['violations'].append("Wave C is less than 61.8% of Wave A")
        
        # Rule 4: Wave B size relative to Wave A
        if wave_a.price_range > 0:
            b_to_a_ratio = wave_b.price_range / wave_a.price_range
            validation['rule_checks']['b_to_a_ratio'] = b_to_a_ratio
            
            if b_to_a_ratio > 1.382:
                validation['violations'].append("Wave B exceeds 138.2% of Wave A")
        
        # Calculate confidence
        total_rules = 4
        violations_count = len(validation['violations'])
        validation['confidence'] = max(0.0, (total_rules - violations_count) / total_rules)
        validation['is_valid'] = violations_count == 0
        
        return validation


class ElliottWaveBase(ABC):
    """
    Base class for Elliott Wave analysis implementations
    
    Provides common functionality and interface for different
    Elliott Wave analysis approaches with maximum flexibility.
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
            wave_degree_levels: Number of wave degree levels to analyze
        """
        self.fibonacci_tolerance = fibonacci_tolerance
        self.min_wave_size_pct = min_wave_size_pct / 100.0
        self.max_waves_to_analyze = max_waves_to_analyze
        self.wave_degree_levels = wave_degree_levels
        
        # Initialize sub-components
        self.fibonacci_calculator = FibonacciCalculator()
        self.wave_validator = WaveValidator()
        
        # Configuration for flexible analysis
        self.analysis_config = {
            'strict_rules': True,  # Whether to enforce strict Elliott Wave rules
            'allow_diagonal_triangles': True,  # Allow diagonal triangle patterns
            'enable_fibonacci_extensions': True,  # Calculate Fibonacci extensions
            'enable_time_analysis': False,  # Enable time-based analysis
            'confidence_threshold': 0.6,  # Minimum confidence for pattern acceptance
        }
        
        # Cache for performance
        self._wave_cache = {}
        self._pattern_cache = {}
    
    @abstractmethod
    def identify_waves(self, swing_points: List[SwingPoint]) -> List[Wave]:
        """
        Identify Elliott Waves from swing points
        
        Args:
            swing_points: List of swing points
            
        Returns:
            List of identified waves
        """
        pass
    
    @abstractmethod
    def find_patterns(self, waves: List[Wave]) -> List[WavePattern]:
        """
        Find Elliott Wave patterns in identified waves
        
        Args:
            waves: List of waves
            
        Returns:
            List of wave patterns
        """
        pass
    
    def set_analysis_config(self, config: Dict[str, Any]):
        """Update analysis configuration"""
        self.analysis_config.update(config)
        analysis_logger.info(f"Updated Elliott Wave analysis config: {config}")
    
    def calculate_fibonacci_ratios(self, waves: List[Wave]) -> Dict[str, Any]:
        """Calculate Fibonacci ratios for waves"""
        return self.fibonacci_calculator.analyze_wave_ratios(waves)
    
    def validate_pattern(self, pattern: WavePattern) -> Dict[str, Any]:
        """
        Validate wave pattern according to Elliott Wave rules
        
        Args:
            pattern: Wave pattern to validate
            
        Returns:
            Validation results
        """
        if pattern.pattern_type == WaveType.IMPULSE:
            return self.wave_validator.validate_impulse_wave(pattern.waves)
        elif pattern.pattern_type == WaveType.CORRECTIVE:
            return self.wave_validator.validate_corrective_wave(pattern.waves)
        else:
            return {'is_valid': False, 'violations': ['Unknown pattern type']}
    
    def calculate_price_targets(self, pattern: WavePattern) -> Dict[str, float]:
        """
        Calculate price targets based on Elliott Wave pattern
        
        Args:
            pattern: Wave pattern
            
        Returns:
            Dictionary with price targets
        """
        targets = {}
        
        if not pattern.waves:
            return targets
        
        if pattern.pattern_type == WaveType.IMPULSE and len(pattern.waves) >= 3:
            # Calculate Wave 5 targets based on Wave 1 and 3
            wave_1 = pattern.waves[0]
            wave_3 = pattern.waves[2]
            
            if len(pattern.waves) >= 4:
                wave_4_end = pattern.waves[3].end_point.price
                
                # Common Wave 5 relationships
                wave_1_size = wave_1.price_range
                wave_3_size = wave_3.price_range
                
                if wave_1.is_bullish:
                    targets['wave_5_equal_wave_1'] = wave_4_end + wave_1_size
                    targets['wave_5_618_wave_1'] = wave_4_end + (wave_1_size * 0.618)
                    targets['wave_5_equal_wave_3'] = wave_4_end + wave_3_size
                else:
                    targets['wave_5_equal_wave_1'] = wave_4_end - wave_1_size
                    targets['wave_5_618_wave_1'] = wave_4_end - (wave_1_size * 0.618)
                    targets['wave_5_equal_wave_3'] = wave_4_end - wave_3_size
        
        elif pattern.pattern_type == WaveType.CORRECTIVE and len(pattern.waves) >= 2:
            # Calculate Wave C targets based on Wave A
            wave_a = pattern.waves[0]
            
            if len(pattern.waves) >= 3:
                wave_b_end = pattern.waves[1].end_point.price
                wave_a_size = wave_a.price_range
                
                if wave_a.is_bullish:
                    targets['wave_c_equal_wave_a'] = wave_b_end + wave_a_size
                    targets['wave_c_618_wave_a'] = wave_b_end + (wave_a_size * 0.618)
                    targets['wave_c_162_wave_a'] = wave_b_end + (wave_a_size * 1.618)
                else:
                    targets['wave_c_equal_wave_a'] = wave_b_end - wave_a_size
                    targets['wave_c_618_wave_a'] = wave_b_end - (wave_a_size * 0.618)
                    targets['wave_c_162_wave_a'] = wave_b_end - (wave_a_size * 1.618)
        
        return targets
    
    def get_analysis_summary(self, swing_points: List[SwingPoint]) -> Dict[str, Any]:
        """
        Get comprehensive Elliott Wave analysis summary
        
        Args:
            swing_points: List of swing points
            
        Returns:
            Analysis summary dictionary
        """
        try:
            # Identify waves
            waves = self.identify_waves(swing_points)
            
            # Find patterns
            patterns = self.find_patterns(waves)
            
            # Analyze patterns
            pattern_analysis = []
            for pattern in patterns:
                validation = self.validate_pattern(pattern)
                price_targets = self.calculate_price_targets(pattern)
                
                pattern_analysis.append({
                    'pattern_type': pattern.pattern_type.value,
                    'wave_count': len(pattern.waves),
                    'confidence': pattern.confidence,
                    'is_valid': validation.get('is_valid', False),
                    'violations': validation.get('violations', []),
                    'price_targets': price_targets,
                    'completion_percentage': pattern.completion_percentage
                })
            
            return {
                'total_waves': len(waves),
                'total_patterns': len(patterns),
                'valid_patterns': sum(1 for p in pattern_analysis if p['is_valid']),
                'patterns': pattern_analysis,
                'config': self.analysis_config,
                'last_updated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            analysis_logger.error(f"Error in Elliott Wave analysis summary: {e}")
            return {
                'error': str(e),
                'last_updated': datetime.utcnow().isoformat()
            }