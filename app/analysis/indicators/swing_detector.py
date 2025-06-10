"""
Advanced swing point detection algorithms
"""

from typing import List, Tuple, Optional, Dict, Any, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum

from app.analysis.indicators.zigzag import SwingPoint, SwingType, ZigZagIndicator
from app.analysis.indicators.atr import ATRIndicator
from app.utils.logger import analysis_logger


class SwingDetectionMethod(Enum):
    """Methods for swing point detection"""
    ZIGZAG = "zigzag"
    FRACTALS = "fractals"
    PIVOT_POINTS = "pivot_points"
    ADAPTIVE = "adaptive"


@dataclass
class SwingPattern:
    """Represents a pattern of swing points"""
    swing_points: List[SwingPoint]
    pattern_type: str
    confidence: float = 0.0
    timeframe_hours: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


class SwingDetector:
    """
    Advanced swing point detector with multiple algorithms
    
    Provides flexible swing detection using various methods including
    ZigZag, Fractals, Pivot Points, and adaptive algorithms.
    """
    
    def __init__(self,
                 method: SwingDetectionMethod = SwingDetectionMethod.ADAPTIVE,
                 sensitivity: float = 0.5,
                 min_swing_size_pct: float = 2.0,
                 use_atr_normalization: bool = True,
                 atr_period: int = 14):
        """
        Initialize swing detector
        
        Args:
            method: Detection method to use
            sensitivity: Detection sensitivity (0.0 = less sensitive, 1.0 = more sensitive)
            min_swing_size_pct: Minimum swing size as percentage
            use_atr_normalization: Whether to normalize using ATR
            atr_period: Period for ATR calculation
        """
        self.method = method
        self.sensitivity = max(0.1, min(1.0, sensitivity))
        self.min_swing_size_pct = min_swing_size_pct / 100.0
        self.use_atr_normalization = use_atr_normalization
        self.atr_period = atr_period
        
        # Initialize sub-indicators
        if self.use_atr_normalization:
            self.atr_indicator = ATRIndicator(period=atr_period)
        
        # Configure detection parameters based on sensitivity
        self._configure_parameters()
    
    def _configure_parameters(self):
        """Configure detection parameters based on sensitivity"""
        # Adjust thresholds based on sensitivity
        # Higher sensitivity = lower thresholds = more swing points
        sensitivity_factor = 2.0 - self.sensitivity
        
        self.zigzag_threshold = self.min_swing_size_pct * sensitivity_factor
        self.fractal_period = max(3, int(7 * sensitivity_factor))
        self.pivot_period = max(5, int(10 * sensitivity_factor))
        
        # ATR multiplier for noise filtering
        self.atr_multiplier = 0.3 + (0.4 * sensitivity_factor)
    
    def detect_swings(self, data: pd.DataFrame) -> List[SwingPoint]:
        """
        Detect swing points using the configured method
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            List of detected SwingPoint objects
        """
        if len(data) < 10:
            return []
        
        try:
            if self.method == SwingDetectionMethod.ZIGZAG:
                return self._detect_zigzag_swings(data)
            elif self.method == SwingDetectionMethod.FRACTALS:
                return self._detect_fractal_swings(data)
            elif self.method == SwingDetectionMethod.PIVOT_POINTS:
                return self._detect_pivot_swings(data)
            elif self.method == SwingDetectionMethod.ADAPTIVE:
                return self._detect_adaptive_swings(data)
            else:
                raise ValueError(f"Unknown detection method: {self.method}")
                
        except Exception as e:
            analysis_logger.error(f"Error detecting swings: {e}")
            return []
    
    def _detect_zigzag_swings(self, data: pd.DataFrame) -> List[SwingPoint]:
        """Detect swings using ZigZag method"""
        zigzag = ZigZagIndicator(
            threshold_pct=self.zigzag_threshold * 100,
            use_atr_filter=self.use_atr_normalization,
            atr_multiplier=self.atr_multiplier
        )
        return zigzag.find_swing_points(data)
    
    def _detect_fractal_swings(self, data: pd.DataFrame) -> List[SwingPoint]:
        """Detect swings using Fractal method"""
        swing_points = []
        
        for i in range(self.fractal_period, len(data) - self.fractal_period):
            # Check for fractal high
            current_high = data['high'].iloc[i]
            left_highs = data['high'].iloc[i-self.fractal_period:i]
            right_highs = data['high'].iloc[i+1:i+self.fractal_period+1]
            
            if (current_high > left_highs.max() and 
                current_high > right_highs.max()):
                
                if self._is_significant_swing(data, i, current_high, SwingType.HIGH):
                    swing_point = self._create_swing_point(data, i, current_high, SwingType.HIGH)
                    swing_points.append(swing_point)
            
            # Check for fractal low
            current_low = data['low'].iloc[i]
            left_lows = data['low'].iloc[i-self.fractal_period:i]
            right_lows = data['low'].iloc[i+1:i+self.fractal_period+1]
            
            if (current_low < left_lows.min() and 
                current_low < right_lows.min()):
                
                if self._is_significant_swing(data, i, current_low, SwingType.LOW):
                    swing_point = self._create_swing_point(data, i, current_low, SwingType.LOW)
                    swing_points.append(swing_point)
        
        return self._filter_consecutive_swings(swing_points)
    
    def _detect_pivot_swings(self, data: pd.DataFrame) -> List[SwingPoint]:
        """Detect swings using Pivot Point method"""
        swing_points = []
        
        # Calculate rolling highs and lows
        rolling_high = data['high'].rolling(window=self.pivot_period, center=True).max()
        rolling_low = data['low'].rolling(window=self.pivot_period, center=True).min()
        
        for i in range(self.pivot_period, len(data) - self.pivot_period):
            # Pivot high: current high equals rolling max
            if data['high'].iloc[i] == rolling_high.iloc[i]:
                if self._is_significant_swing(data, i, data['high'].iloc[i], SwingType.HIGH):
                    swing_point = self._create_swing_point(data, i, data['high'].iloc[i], SwingType.HIGH)
                    swing_points.append(swing_point)
            
            # Pivot low: current low equals rolling min
            if data['low'].iloc[i] == rolling_low.iloc[i]:
                if self._is_significant_swing(data, i, data['low'].iloc[i], SwingType.LOW):
                    swing_point = self._create_swing_point(data, i, data['low'].iloc[i], SwingType.LOW)
                    swing_points.append(swing_point)
        
        return self._filter_consecutive_swings(swing_points)
    
    def _detect_adaptive_swings(self, data: pd.DataFrame) -> List[SwingPoint]:
        """Detect swings using adaptive method combining multiple approaches"""
        # Get swings from multiple methods
        zigzag_swings = self._detect_zigzag_swings(data)
        fractal_swings = self._detect_fractal_swings(data)
        
        # Combine and score swing points
        all_swings = {}  # index -> SwingPoint
        
        # Add ZigZag swings with higher weight
        for swing in zigzag_swings:
            all_swings[swing.index] = swing
            swing.strength += 0.3  # Bonus for ZigZag detection
        
        # Add Fractal swings, boost existing ones
        for swing in fractal_swings:
            if swing.index in all_swings:
                # Boost existing swing
                all_swings[swing.index].strength += 0.2
            else:
                # Add new swing with lower initial strength
                swing.strength *= 0.7
                all_swings[swing.index] = swing
        
        # Convert back to list and sort
        combined_swings = list(all_swings.values())
        combined_swings.sort(key=lambda x: x.index)
        
        # Filter based on combined strength
        min_strength = 0.1 + (0.2 * (1.0 - self.sensitivity))
        filtered_swings = [s for s in combined_swings if s.strength >= min_strength]
        
        return self._filter_consecutive_swings(filtered_swings)
    
    def _is_significant_swing(self, data: pd.DataFrame, index: int, 
                            price: float, swing_type: SwingType) -> bool:
        """Check if a potential swing is significant enough"""
        if index < 1 or index >= len(data) - 1:
            return False
        
        # Check percentage threshold
        reference_price = data['close'].iloc[max(0, index - 5):index + 1].mean()
        price_change_pct = abs(price - reference_price) / reference_price
        
        if price_change_pct < self.min_swing_size_pct:
            return False
        
        # Check ATR threshold if enabled
        if self.use_atr_normalization:
            atr_values = self.atr_indicator.calculate(data)
            if index < len(atr_values):
                current_atr = atr_values.iloc[index]
                if not pd.isna(current_atr) and current_atr > 0:
                    atr_threshold = current_atr * self.atr_multiplier
                    price_change = abs(price - reference_price)
                    if price_change < atr_threshold:
                        return False
        
        return True
    
    def _create_swing_point(self, data: pd.DataFrame, index: int, 
                          price: float, swing_type: SwingType) -> SwingPoint:
        """Create a SwingPoint object"""
        timestamp = data.index[index] if hasattr(data.index, 'to_pydatetime') else pd.Timestamp('1970-01-01')
        
        # Calculate basic strength
        strength = self._calculate_swing_strength(data, index, swing_type)
        
        return SwingPoint(
            index=index,
            timestamp=timestamp,
            price=price,
            swing_type=swing_type,
            strength=strength,
            atr_ratio=0.0
        )
    
    def _calculate_swing_strength(self, data: pd.DataFrame, index: int, 
                                swing_type: SwingType) -> float:
        """Calculate swing strength based on price action"""
        lookback = min(10, index)
        lookahead = min(10, len(data) - index - 1)
        
        if lookback < 2 or lookahead < 2:
            return 0.1
        
        current_price = data['high'].iloc[index] if swing_type == SwingType.HIGH else data['low'].iloc[index]
        
        # Compare with surrounding prices
        if swing_type == SwingType.HIGH:
            left_max = data['high'].iloc[index-lookback:index].max()
            right_max = data['high'].iloc[index+1:index+lookahead+1].max()
            max_surrounding = max(left_max, right_max)
            
            if max_surrounding > 0:
                return min(1.0, (current_price - max_surrounding) / max_surrounding + 0.1)
        else:
            left_min = data['low'].iloc[index-lookback:index].min()
            right_min = data['low'].iloc[index+1:index+lookahead+1].min()
            min_surrounding = min(left_min, right_min)
            
            if min_surrounding > 0:
                return min(1.0, (min_surrounding - current_price) / min_surrounding + 0.1)
        
        return 0.1
    
    def _filter_consecutive_swings(self, swing_points: List[SwingPoint]) -> List[SwingPoint]:
        """Filter out consecutive swings of the same type, keeping the strongest"""
        if len(swing_points) < 2:
            return swing_points
        
        filtered = []
        current_group = [swing_points[0]]
        
        for swing in swing_points[1:]:
            if swing.swing_type == current_group[-1].swing_type:
                # Same type, add to current group
                current_group.append(swing)
            else:
                # Different type, finalize current group
                best_swing = max(current_group, key=lambda x: x.strength)
                filtered.append(best_swing)
                current_group = [swing]
        
        # Add the last group
        if current_group:
            best_swing = max(current_group, key=lambda x: x.strength)
            filtered.append(best_swing)
        
        return filtered
    
    def find_swing_patterns(self, data: pd.DataFrame, 
                          pattern_types: List[str] = None) -> List[SwingPattern]:
        """
        Find specific swing patterns in the data
        
        Args:
            data: OHLC data
            pattern_types: List of pattern types to find ['double_top', 'double_bottom', 'head_shoulders', etc.]
            
        Returns:
            List of detected SwingPattern objects
        """
        if pattern_types is None:
            pattern_types = ['double_top', 'double_bottom', 'triple_top', 'triple_bottom']
        
        swing_points = self.detect_swings(data)
        patterns = []
        
        if 'double_top' in pattern_types:
            patterns.extend(self._find_double_tops(swing_points))
        
        if 'double_bottom' in pattern_types:
            patterns.extend(self._find_double_bottoms(swing_points))
        
        return patterns
    
    def _find_double_tops(self, swing_points: List[SwingPoint]) -> List[SwingPattern]:
        """Find double top patterns"""
        patterns = []
        highs = [s for s in swing_points if s.swing_type == SwingType.HIGH]
        
        for i in range(len(highs) - 1):
            for j in range(i + 1, len(highs)):
                high1, high2 = highs[i], highs[j]
                
                # Check if prices are similar (within 2%)
                price_diff_pct = abs(high1.price - high2.price) / max(high1.price, high2.price)
                
                if price_diff_pct <= 0.02:  # Within 2%
                    # Find intervening low
                    intervening_lows = [s for s in swing_points 
                                      if (s.swing_type == SwingType.LOW and 
                                          high1.index < s.index < high2.index)]
                    
                    if intervening_lows:
                        lowest_point = min(intervening_lows, key=lambda x: x.price)
                        
                        # Check if low is significantly below highs
                        low_diff = min(high1.price, high2.price) - lowest_point.price
                        if low_diff / lowest_point.price > 0.01:  # At least 1% pullback
                            
                            pattern = SwingPattern(
                                swing_points=[high1, lowest_point, high2],
                                pattern_type='double_top',
                                confidence=1.0 - price_diff_pct,
                                metadata={
                                    'height': low_diff,
                                    'width_bars': high2.index - high1.index,
                                    'price_similarity': 1.0 - price_diff_pct
                                }
                            )
                            patterns.append(pattern)
        
        return patterns
    
    def _find_double_bottoms(self, swing_points: List[SwingPoint]) -> List[SwingPattern]:
        """Find double bottom patterns"""
        patterns = []
        lows = [s for s in swing_points if s.swing_type == SwingType.LOW]
        
        for i in range(len(lows) - 1):
            for j in range(i + 1, len(lows)):
                low1, low2 = lows[i], lows[j]
                
                # Check if prices are similar (within 2%)
                price_diff_pct = abs(low1.price - low2.price) / min(low1.price, low2.price)
                
                if price_diff_pct <= 0.02:  # Within 2%
                    # Find intervening high
                    intervening_highs = [s for s in swing_points 
                                       if (s.swing_type == SwingType.HIGH and 
                                           low1.index < s.index < low2.index)]
                    
                    if intervening_highs:
                        highest_point = max(intervening_highs, key=lambda x: x.price)
                        
                        # Check if high is significantly above lows
                        high_diff = highest_point.price - max(low1.price, low2.price)
                        if high_diff / highest_point.price > 0.01:  # At least 1% bounce
                            
                            pattern = SwingPattern(
                                swing_points=[low1, highest_point, low2],
                                pattern_type='double_bottom',
                                confidence=1.0 - price_diff_pct,
                                metadata={
                                    'height': high_diff,
                                    'width_bars': low2.index - low1.index,
                                    'price_similarity': 1.0 - price_diff_pct
                                }
                            )
                            patterns.append(pattern)
        
        return patterns
    
    def get_detection_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of swing detection results"""
        swing_points = self.detect_swings(data)
        patterns = self.find_swing_patterns(data)
        
        return {
            'total_swings': len(swing_points),
            'swing_highs': len([s for s in swing_points if s.swing_type == SwingType.HIGH]),
            'swing_lows': len([s for s in swing_points if s.swing_type == SwingType.LOW]),
            'avg_strength': np.mean([s.strength for s in swing_points]) if swing_points else 0.0,
            'detected_patterns': len(patterns),
            'pattern_types': list(set(p.pattern_type for p in patterns)),
            'method': self.method.value,
            'sensitivity': self.sensitivity
        }


# Convenience functions
def detect_swings(data: pd.DataFrame, 
                 method: str = 'adaptive',
                 sensitivity: float = 0.5,
                 min_swing_size_pct: float = 2.0) -> List[SwingPoint]:
    """
    Convenience function to detect swing points
    
    Args:
        data: OHLC data
        method: Detection method ('zigzag', 'fractals', 'pivot_points', 'adaptive')
        sensitivity: Detection sensitivity (0.0 to 1.0)
        min_swing_size_pct: Minimum swing size percentage
        
    Returns:
        List of SwingPoint objects
    """
    detector = SwingDetector(
        method=SwingDetectionMethod(method),
        sensitivity=sensitivity,
        min_swing_size_pct=min_swing_size_pct
    )
    return detector.detect_swings(data)


# Create default detectors
adaptive_detector = SwingDetector(method=SwingDetectionMethod.ADAPTIVE, sensitivity=0.5)
sensitive_detector = SwingDetector(method=SwingDetectionMethod.ADAPTIVE, sensitivity=0.8)
conservative_detector = SwingDetector(method=SwingDetectionMethod.ADAPTIVE, sensitivity=0.2)