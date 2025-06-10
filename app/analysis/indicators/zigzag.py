"""
ZigZag indicator implementation for swing point detection
"""

from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

from app.analysis.indicators.atr import ATRIndicator, IndicatorBase
from app.utils.logger import analysis_logger


class SwingType(Enum):
    """Types of swing points"""
    HIGH = "high"
    LOW = "low"


@dataclass
class SwingPoint:
    """Represents a swing point in price data"""
    index: int
    timestamp: pd.Timestamp
    price: float
    swing_type: SwingType
    strength: float = 0.0  # Relative strength/significance
    atr_ratio: float = 0.0  # Price move relative to ATR


class ZigZagIndicator(IndicatorBase):
    """
    ZigZag Indicator for identifying significant price swings
    
    Filters out minor price movements and highlights significant
    trend changes based on percentage or ATR-based thresholds.
    """
    
    def __init__(self, 
                 threshold_pct: float = 5.0,
                 use_atr_filter: bool = True,
                 atr_period: int = 14,
                 atr_multiplier: float = 0.5,
                 min_bars_between_swings: int = 5):
        """
        Initialize ZigZag indicator
        
        Args:
            threshold_pct: Minimum percentage change to register a swing
            use_atr_filter: Whether to use ATR-based filtering
            atr_period: Period for ATR calculation
            atr_multiplier: ATR multiplier for significance threshold
            min_bars_between_swings: Minimum bars between swing points
        """
        super().__init__(period=atr_period)
        self.threshold_pct = threshold_pct / 100.0  # Convert to decimal
        self.use_atr_filter = use_atr_filter
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.min_bars_between_swings = min_bars_between_swings
        
        # Initialize ATR indicator if needed
        if self.use_atr_filter:
            self.atr_indicator = ATRIndicator(period=atr_period)
    
    def get_required_columns(self) -> List[str]:
        """Required OHLC columns"""
        return ['high', 'low', 'close']
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate ZigZag indicator
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            Series with ZigZag values (NaN where no swing point)
        """
        try:
            swing_points = self.find_swing_points(data)
            
            # Create series with ZigZag values
            zigzag_series = pd.Series(index=data.index, dtype=float)
            
            for point in swing_points:
                if point.index < len(zigzag_series):
                    zigzag_series.iloc[point.index] = point.price
            
            zigzag_series.name = f'ZigZag_{self.threshold_pct*100:.1f}%'
            
            analysis_logger.debug(f"Calculated ZigZag with {len(swing_points)} swing points")
            return zigzag_series
            
        except Exception as e:
            analysis_logger.error(f"Error calculating ZigZag: {e}")
            raise
    
    def find_swing_points(self, data: pd.DataFrame) -> List[SwingPoint]:
        """
        Find swing points in price data
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            List of SwingPoint objects
        """
        if not self.validate_data(data):
            raise ValueError(f"Data missing required columns: {self.get_required_columns()}")
        
        if len(data) < self.min_bars_between_swings * 2:
            return []
        
        swing_points = []
        
        # Calculate ATR if using ATR filter
        atr_values = None
        if self.use_atr_filter:
            atr_values = self.atr_indicator.calculate(data)
        
        # Find potential swing highs and lows
        potential_highs = self._find_potential_highs(data)
        potential_lows = self._find_potential_lows(data)
        
        # Combine and sort by index
        all_potentials = []
        
        for idx in potential_highs:
            all_potentials.append((idx, data['high'].iloc[idx], SwingType.HIGH))
        
        for idx in potential_lows:
            all_potentials.append((idx, data['low'].iloc[idx], SwingType.LOW))
        
        all_potentials.sort(key=lambda x: x[0])
        
        # Filter based on significance
        filtered_points = self._filter_significant_swings(
            data, all_potentials, atr_values
        )
        
        # Create SwingPoint objects
        for idx, price, swing_type in filtered_points:
            timestamp = data.index[idx] if hasattr(data.index, 'to_pydatetime') else pd.Timestamp('1970-01-01')
            
            # Calculate strength and ATR ratio
            strength = self._calculate_swing_strength(data, idx, swing_type)
            atr_ratio = 0.0
            
            if atr_values is not None and idx < len(atr_values):
                current_atr = atr_values.iloc[idx]
                if not pd.isna(current_atr) and current_atr > 0:
                    # Calculate price move from previous swing
                    if swing_points:
                        prev_price = swing_points[-1].price
                        price_move = abs(price - prev_price)
                        atr_ratio = price_move / current_atr
            
            swing_point = SwingPoint(
                index=idx,
                timestamp=timestamp,
                price=price,
                swing_type=swing_type,
                strength=strength,
                atr_ratio=atr_ratio
            )
            
            swing_points.append(swing_point)
        
        return swing_points
    
    def _find_potential_highs(self, data: pd.DataFrame) -> List[int]:
        """Find potential swing high indices"""
        potential_highs = []
        
        for i in range(self.min_bars_between_swings, 
                      len(data) - self.min_bars_between_swings):
            
            current_high = data['high'].iloc[i]
            
            # Check if current high is higher than surrounding bars
            left_range = data['high'].iloc[i-self.min_bars_between_swings:i]
            right_range = data['high'].iloc[i+1:i+self.min_bars_between_swings+1]
            
            if (current_high > left_range.max() and 
                current_high > right_range.max()):
                potential_highs.append(i)
        
        return potential_highs
    
    def _find_potential_lows(self, data: pd.DataFrame) -> List[int]:
        """Find potential swing low indices"""
        potential_lows = []
        
        for i in range(self.min_bars_between_swings, 
                      len(data) - self.min_bars_between_swings):
            
            current_low = data['low'].iloc[i]
            
            # Check if current low is lower than surrounding bars
            left_range = data['low'].iloc[i-self.min_bars_between_swings:i]
            right_range = data['low'].iloc[i+1:i+self.min_bars_between_swings+1]
            
            if (current_low < left_range.min() and 
                current_low < right_range.min()):
                potential_lows.append(i)
        
        return potential_lows
    
    def _filter_significant_swings(self, 
                                  data: pd.DataFrame,
                                  potentials: List[Tuple[int, float, SwingType]],
                                  atr_values: Optional[pd.Series] = None) -> List[Tuple[int, float, SwingType]]:
        """Filter swing points based on significance thresholds"""
        if not potentials:
            return []
        
        filtered = []
        last_swing = potentials[0]
        filtered.append(last_swing)
        
        for current in potentials[1:]:
            idx, price, swing_type = current
            last_idx, last_price, last_swing_type = last_swing
            
            # Skip if same type and not significant enough
            if swing_type == last_swing_type:
                if swing_type == SwingType.HIGH and price > last_price:
                    # New higher high
                    filtered[-1] = current
                    last_swing = current
                elif swing_type == SwingType.LOW and price < last_price:
                    # New lower low
                    filtered[-1] = current
                    last_swing = current
                continue
            
            # Different swing type, check significance
            price_change = abs(price - last_price)
            price_change_pct = price_change / last_price
            
            # Check percentage threshold
            if price_change_pct < self.threshold_pct:
                continue
            
            # Check ATR threshold if enabled
            if self.use_atr_filter and atr_values is not None:
                if idx < len(atr_values):
                    current_atr = atr_values.iloc[idx]
                    if not pd.isna(current_atr) and current_atr > 0:
                        atr_threshold = current_atr * self.atr_multiplier
                        if price_change < atr_threshold:
                            continue
            
            # Check minimum bars between swings
            if idx - last_idx < self.min_bars_between_swings:
                continue
            
            filtered.append(current)
            last_swing = current
        
        return filtered
    
    def _calculate_swing_strength(self, data: pd.DataFrame, idx: int, 
                                swing_type: SwingType) -> float:
        """Calculate the strength/significance of a swing point"""
        if idx < self.min_bars_between_swings or idx >= len(data) - self.min_bars_between_swings:
            return 0.0
        
        lookback = min(self.min_bars_between_swings * 2, idx)
        lookahead = min(self.min_bars_between_swings * 2, len(data) - idx - 1)
        
        if swing_type == SwingType.HIGH:
            current_price = data['high'].iloc[idx]
            surrounding_highs = pd.concat([
                data['high'].iloc[idx-lookback:idx],
                data['high'].iloc[idx+1:idx+lookahead+1]
            ])
            
            if len(surrounding_highs) > 0:
                max_surrounding = surrounding_highs.max()
                if max_surrounding > 0:
                    return (current_price - max_surrounding) / max_surrounding
        
        else:  # SwingType.LOW
            current_price = data['low'].iloc[idx]
            surrounding_lows = pd.concat([
                data['low'].iloc[idx-lookback:idx],
                data['low'].iloc[idx+1:idx+lookahead+1]
            ])
            
            if len(surrounding_lows) > 0:
                min_surrounding = surrounding_lows.min()
                if min_surrounding > 0:
                    return (min_surrounding - current_price) / min_surrounding
        
        return 0.0
    
    def get_swing_point_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics of swing points"""
        swing_points = self.find_swing_points(data)
        
        if not swing_points:
            return {
                'total_swings': 0,
                'swing_highs': 0,
                'swing_lows': 0,
                'avg_strength': 0.0,
                'avg_atr_ratio': 0.0
            }
        
        highs = [p for p in swing_points if p.swing_type == SwingType.HIGH]
        lows = [p for p in swing_points if p.swing_type == SwingType.LOW]
        
        strengths = [p.strength for p in swing_points if p.strength > 0]
        atr_ratios = [p.atr_ratio for p in swing_points if p.atr_ratio > 0]
        
        return {
            'total_swings': len(swing_points),
            'swing_highs': len(highs),
            'swing_lows': len(lows),
            'avg_strength': np.mean(strengths) if strengths else 0.0,
            'avg_atr_ratio': np.mean(atr_ratios) if atr_ratios else 0.0,
            'max_strength': max(strengths) if strengths else 0.0,
            'max_atr_ratio': max(atr_ratios) if atr_ratios else 0.0
        }


def find_zigzag_swings(data: pd.DataFrame, 
                      threshold_pct: float = 5.0,
                      use_atr_filter: bool = True,
                      atr_multiplier: float = 0.5) -> List[SwingPoint]:
    """
    Convenience function to find swing points using ZigZag
    
    Args:
        data: DataFrame with OHLC data
        threshold_pct: Minimum percentage change threshold
        use_atr_filter: Whether to use ATR filtering
        atr_multiplier: ATR multiplier for filtering
        
    Returns:
        List of SwingPoint objects
    """
    zigzag = ZigZagIndicator(
        threshold_pct=threshold_pct,
        use_atr_filter=use_atr_filter,
        atr_multiplier=atr_multiplier
    )
    return zigzag.find_swing_points(data)


# Create default ZigZag instances
zigzag_5pct = ZigZagIndicator(threshold_pct=5.0, use_atr_filter=True)
zigzag_3pct = ZigZagIndicator(threshold_pct=3.0, use_atr_filter=True)