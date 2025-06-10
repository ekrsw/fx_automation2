"""
Dow Theory Analysis Base Classes
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

from app.analysis.indicators.swing_detector import SwingDetector, SwingPoint, SwingType
from app.analysis.indicators.atr import ATRIndicator
from app.analysis.indicators.momentum import MomentumAnalyzer
from app.utils.logger import analysis_logger


class TrendDirection(Enum):
    """Trend direction according to Dow Theory"""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"
    UNDEFINED = "undefined"


class TrendStrength(Enum):
    """Trend strength classification"""
    VERY_STRONG = "very_strong"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    VERY_WEAK = "very_weak"


class ConfirmationStatus(Enum):
    """Confirmation status between indices/timeframes"""
    CONFIRMED = "confirmed"
    UNCONFIRMED = "unconfirmed"
    DIVERGENT = "divergent"
    PENDING = "pending"


@dataclass
class TrendAnalysis:
    """Results of trend analysis"""
    direction: TrendDirection
    strength: TrendStrength
    confidence: float  # 0.0 to 1.0
    start_date: Optional[datetime] = None
    duration_days: int = 0
    swing_points: List[SwingPoint] = field(default_factory=list)
    key_levels: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DowTheorySignal:
    """Dow Theory trading signal"""
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0.0 to 1.0
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    confirmation_status: ConfirmationStatus = ConfirmationStatus.PENDING
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class DowTheoryBase(ABC):
    """
    Base class for Dow Theory analysis implementations
    
    Provides common functionality and interface for different
    Dow Theory analysis approaches.
    """
    
    def __init__(self,
                 swing_sensitivity: float = 0.5,
                 min_swing_size_pct: float = 2.0,
                 trend_confirmation_swings: int = 3,
                 use_volume_confirmation: bool = True,
                 atr_period: int = 14):
        """
        Initialize Dow Theory analyzer
        
        Args:
            swing_sensitivity: Swing detection sensitivity (0.0 to 1.0)
            min_swing_size_pct: Minimum swing size percentage
            trend_confirmation_swings: Number of swings needed for trend confirmation
            use_volume_confirmation: Whether to use volume for confirmation
            atr_period: ATR period for volatility analysis
        """
        self.swing_sensitivity = max(0.1, min(1.0, swing_sensitivity))
        self.min_swing_size_pct = min_swing_size_pct
        self.trend_confirmation_swings = max(2, trend_confirmation_swings)
        self.use_volume_confirmation = use_volume_confirmation
        self.atr_period = atr_period
        
        # Initialize supporting indicators
        self.swing_detector = SwingDetector(
            sensitivity=swing_sensitivity,
            min_swing_size_pct=min_swing_size_pct
        )
        self.atr_indicator = ATRIndicator(period=atr_period)
        self.momentum_analyzer = MomentumAnalyzer()
        
        # Cache for performance
        self._swing_cache = {}
        self._trend_cache = {}
    
    @abstractmethod
    def analyze_trend(self, data: pd.DataFrame, timeframe: str = 'D1') -> TrendAnalysis:
        """
        Analyze trend according to Dow Theory
        
        Args:
            data: OHLC data
            timeframe: Timeframe string ('D1', 'H4', etc.)
            
        Returns:
            TrendAnalysis object
        """
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, 
                        secondary_data: Optional[pd.DataFrame] = None) -> List[DowTheorySignal]:
        """
        Generate trading signals based on Dow Theory
        
        Args:
            data: Primary timeframe OHLC data
            secondary_data: Secondary timeframe data for confirmation
            
        Returns:
            List of DowTheorySignal objects
        """
        pass
    
    def detect_swing_points(self, data: pd.DataFrame, 
                          use_cache: bool = True) -> List[SwingPoint]:
        """
        Detect swing points in price data
        
        Args:
            data: OHLC data
            use_cache: Whether to use cached results
            
        Returns:
            List of SwingPoint objects
        """
        # Create cache key based on data hash
        cache_key = f"{len(data)}_{data['close'].iloc[-1]}_{self.swing_sensitivity}"
        
        if use_cache and cache_key in self._swing_cache:
            return self._swing_cache[cache_key]
        
        swing_points = self.swing_detector.detect_swings(data)
        
        if use_cache:
            self._swing_cache[cache_key] = swing_points
        
        return swing_points
    
    def identify_trend_direction(self, swing_points: List[SwingPoint]) -> TrendDirection:
        """
        Identify trend direction based on swing points
        
        Dow Theory rules:
        - Uptrend: Series of higher highs and higher lows
        - Downtrend: Series of lower highs and lower lows
        - Sideways: Mixed or inconclusive pattern
        
        Args:
            swing_points: List of swing points
            
        Returns:
            TrendDirection enum value
        """
        if len(swing_points) < self.trend_confirmation_swings * 2:
            return TrendDirection.UNDEFINED
        
        # Separate highs and lows
        highs = [sp for sp in swing_points if sp.swing_type == SwingType.HIGH]
        lows = [sp for sp in swing_points if sp.swing_type == SwingType.LOW]
        
        if len(highs) < 2 or len(lows) < 2:
            return TrendDirection.UNDEFINED
        
        # Sort by index to get chronological order
        highs.sort(key=lambda x: x.index)
        lows.sort(key=lambda x: x.index)
        
        # Check for higher highs and higher lows (uptrend)
        higher_highs = self._check_higher_highs(highs[-3:])  # Check last 3 highs
        higher_lows = self._check_higher_lows(lows[-3:])    # Check last 3 lows
        
        # Check for lower highs and lower lows (downtrend)
        lower_highs = self._check_lower_highs(highs[-3:])
        lower_lows = self._check_lower_lows(lows[-3:])
        
        # Determine trend based on patterns
        if higher_highs and higher_lows:
            return TrendDirection.UPTREND
        elif lower_highs and lower_lows:
            return TrendDirection.DOWNTREND
        else:
            return TrendDirection.SIDEWAYS
    
    def _check_higher_highs(self, highs: List[SwingPoint]) -> bool:
        """Check if we have a pattern of higher highs"""
        if len(highs) < 2:
            return False
        
        ascending_count = 0
        for i in range(1, len(highs)):
            if highs[i].price > highs[i-1].price:
                ascending_count += 1
        
        # At least 75% of highs should be ascending
        return ascending_count / (len(highs) - 1) >= 0.75
    
    def _check_higher_lows(self, lows: List[SwingPoint]) -> bool:
        """Check if we have a pattern of higher lows"""
        if len(lows) < 2:
            return False
        
        ascending_count = 0
        for i in range(1, len(lows)):
            if lows[i].price > lows[i-1].price:
                ascending_count += 1
        
        # At least 75% of lows should be ascending
        return ascending_count / (len(lows) - 1) >= 0.75
    
    def _check_lower_highs(self, highs: List[SwingPoint]) -> bool:
        """Check if we have a pattern of lower highs"""
        if len(highs) < 2:
            return False
        
        descending_count = 0
        for i in range(1, len(highs)):
            if highs[i].price < highs[i-1].price:
                descending_count += 1
        
        # At least 75% of highs should be descending
        return descending_count / (len(highs) - 1) >= 0.75
    
    def _check_lower_lows(self, lows: List[SwingPoint]) -> bool:
        """Check if we have a pattern of lower lows"""
        if len(lows) < 2:
            return False
        
        descending_count = 0
        for i in range(1, len(lows)):
            if lows[i].price < lows[i-1].price:
                descending_count += 1
        
        # At least 75% of lows should be descending
        return descending_count / (len(lows) - 1) >= 0.75
    
    def calculate_trend_strength(self, data: pd.DataFrame, 
                               swing_points: List[SwingPoint],
                               trend_direction: TrendDirection) -> TrendStrength:
        """
        Calculate trend strength based on multiple factors
        
        Args:
            data: OHLC data
            swing_points: Swing points
            trend_direction: Current trend direction
            
        Returns:
            TrendStrength enum value
        """
        if trend_direction == TrendDirection.UNDEFINED:
            return TrendStrength.VERY_WEAK
        
        strength_score = 0.0
        
        # Factor 1: Swing point consistency
        consistency_score = self._calculate_swing_consistency(swing_points, trend_direction)
        strength_score += consistency_score * 0.3
        
        # Factor 2: Momentum analysis
        momentum_analysis = self.momentum_analyzer.analyze(data)
        momentum_score = momentum_analysis.get('composite_score', 0.0)
        
        if trend_direction == TrendDirection.UPTREND:
            strength_score += max(0, momentum_score) * 0.3
        else:
            strength_score += max(0, -momentum_score) * 0.3
        
        # Factor 3: ATR-based volatility
        atr_values = self.atr_indicator.calculate(data)
        if not atr_values.empty:
            volatility_score = self._calculate_volatility_score(data, atr_values)
            strength_score += volatility_score * 0.2
        
        # Factor 4: Volume confirmation (if available and enabled)
        if self.use_volume_confirmation and 'volume' in data.columns:
            volume_score = self._calculate_volume_score(data, trend_direction)
            strength_score += volume_score * 0.2
        
        # Convert score to strength enum
        return self._score_to_strength(strength_score)
    
    def _calculate_swing_consistency(self, swing_points: List[SwingPoint], 
                                   trend_direction: TrendDirection) -> float:
        """Calculate consistency of swing points with trend direction"""
        if len(swing_points) < 4:
            return 0.0
        
        highs = [sp for sp in swing_points if sp.swing_type == SwingType.HIGH]
        lows = [sp for sp in swing_points if sp.swing_type == SwingType.LOW]
        
        if len(highs) < 2 or len(lows) < 2:
            return 0.0
        
        highs.sort(key=lambda x: x.index)
        lows.sort(key=lambda x: x.index)
        
        if trend_direction == TrendDirection.UPTREND:
            # Check for consistent higher highs and higher lows
            hh_score = sum(1 for i in range(1, len(highs)) if highs[i].price > highs[i-1].price) / (len(highs) - 1)
            hl_score = sum(1 for i in range(1, len(lows)) if lows[i].price > lows[i-1].price) / (len(lows) - 1)
            return (hh_score + hl_score) / 2
        
        elif trend_direction == TrendDirection.DOWNTREND:
            # Check for consistent lower highs and lower lows
            lh_score = sum(1 for i in range(1, len(highs)) if highs[i].price < highs[i-1].price) / (len(highs) - 1)
            ll_score = sum(1 for i in range(1, len(lows)) if lows[i].price < lows[i-1].price) / (len(lows) - 1)
            return (lh_score + ll_score) / 2
        
        return 0.0
    
    def _calculate_volatility_score(self, data: pd.DataFrame, atr_values: pd.Series) -> float:
        """Calculate volatility-based strength score"""
        if len(atr_values) < 20:
            return 0.0
        
        current_atr = atr_values.iloc[-1]
        avg_atr = atr_values.tail(20).mean()
        
        # Lower volatility relative to average indicates stronger trend
        if avg_atr > 0:
            volatility_ratio = current_atr / avg_atr
            # Score is higher when current volatility is moderate (not too high, not too low)
            if 0.7 <= volatility_ratio <= 1.3:
                return 1.0
            elif 0.5 <= volatility_ratio <= 1.5:
                return 0.7
            else:
                return 0.3
        
        return 0.0
    
    def _calculate_volume_score(self, data: pd.DataFrame, 
                              trend_direction: TrendDirection) -> float:
        """Calculate volume-based confirmation score"""
        if 'volume' not in data.columns or len(data) < 20:
            return 0.0
        
        # Use ATR * tick count as pseudo-volume for forex
        atr_values = self.atr_indicator.calculate(data)
        if atr_values.empty:
            return 0.0
        
        # Calculate pseudo-volume trend
        recent_atr = atr_values.tail(10).mean()
        older_atr = atr_values.tail(20).head(10).mean()
        
        if older_atr > 0:
            volume_trend = (recent_atr - older_atr) / older_atr
            
            # For uptrends, increasing volume is bullish
            # For downtrends, increasing volume is bearish
            if trend_direction == TrendDirection.UPTREND:
                return max(0, min(1, volume_trend * 2 + 0.5))
            elif trend_direction == TrendDirection.DOWNTREND:
                return max(0, min(1, -volume_trend * 2 + 0.5))
        
        return 0.5  # Neutral if no clear volume trend
    
    def _score_to_strength(self, score: float) -> TrendStrength:
        """Convert numerical score to TrendStrength enum"""
        if score >= 0.8:
            return TrendStrength.VERY_STRONG
        elif score >= 0.6:
            return TrendStrength.STRONG
        elif score >= 0.4:
            return TrendStrength.MODERATE
        elif score >= 0.2:
            return TrendStrength.WEAK
        else:
            return TrendStrength.VERY_WEAK
    
    def calculate_key_levels(self, swing_points: List[SwingPoint]) -> Dict[str, float]:
        """
        Calculate key support and resistance levels
        
        Args:
            swing_points: List of swing points
            
        Returns:
            Dictionary with key levels
        """
        if not swing_points:
            return {}
        
        highs = [sp.price for sp in swing_points if sp.swing_type == SwingType.HIGH]
        lows = [sp.price for sp in swing_points if sp.swing_type == SwingType.LOW]
        
        key_levels = {}
        
        if highs:
            key_levels['resistance'] = max(highs)
            key_levels['avg_high'] = np.mean(highs)
        
        if lows:
            key_levels['support'] = min(lows)
            key_levels['avg_low'] = np.mean(lows)
        
        if highs and lows:
            key_levels['midpoint'] = (max(highs) + min(lows)) / 2
            key_levels['range'] = max(highs) - min(lows)
        
        return key_levels
    
    def check_confirmation(self, primary_trend: TrendAnalysis, 
                         secondary_trend: Optional[TrendAnalysis]) -> ConfirmationStatus:
        """
        Check trend confirmation between timeframes
        
        Args:
            primary_trend: Primary timeframe trend analysis
            secondary_trend: Secondary timeframe trend analysis
            
        Returns:
            ConfirmationStatus enum value
        """
        if secondary_trend is None:
            return ConfirmationStatus.PENDING
        
        if primary_trend.direction == secondary_trend.direction:
            # Same direction - check strength alignment
            primary_strength_value = self._strength_to_value(primary_trend.strength)
            secondary_strength_value = self._strength_to_value(secondary_trend.strength)
            
            # If both are reasonably strong, it's confirmed
            if primary_strength_value >= 2 and secondary_strength_value >= 2:
                return ConfirmationStatus.CONFIRMED
            else:
                return ConfirmationStatus.UNCONFIRMED
        
        elif (primary_trend.direction in [TrendDirection.UPTREND, TrendDirection.DOWNTREND] and
              secondary_trend.direction in [TrendDirection.UPTREND, TrendDirection.DOWNTREND]):
            # Opposite directions
            return ConfirmationStatus.DIVERGENT
        
        else:
            # One or both are sideways/undefined
            return ConfirmationStatus.UNCONFIRMED
    
    def _strength_to_value(self, strength: TrendStrength) -> int:
        """Convert TrendStrength to numerical value"""
        strength_map = {
            TrendStrength.VERY_WEAK: 0,
            TrendStrength.WEAK: 1,
            TrendStrength.MODERATE: 2,
            TrendStrength.STRONG: 3,
            TrendStrength.VERY_STRONG: 4
        }
        return strength_map.get(strength, 0)
    
    def get_analysis_summary(self, data: pd.DataFrame, 
                           timeframe: str = 'D1') -> Dict[str, Any]:
        """
        Get comprehensive analysis summary
        
        Args:
            data: OHLC data
            timeframe: Timeframe string
            
        Returns:
            Dictionary with analysis summary
        """
        try:
            trend_analysis = self.analyze_trend(data, timeframe)
            signals = self.generate_signals(data)
            
            return {
                'timeframe': timeframe,
                'trend_direction': trend_analysis.direction.value,
                'trend_strength': trend_analysis.strength.value,
                'confidence': trend_analysis.confidence,
                'duration_days': trend_analysis.duration_days,
                'swing_points_count': len(trend_analysis.swing_points),
                'key_levels': trend_analysis.key_levels,
                'active_signals': len([s for s in signals if s.signal_type in ['buy', 'sell']]),
                'last_updated': datetime.utcnow().isoformat(),
                'metadata': trend_analysis.metadata
            }
            
        except Exception as e:
            analysis_logger.error(f"Error generating analysis summary: {e}")
            return {
                'error': str(e),
                'timeframe': timeframe,
                'last_updated': datetime.utcnow().isoformat()
            }