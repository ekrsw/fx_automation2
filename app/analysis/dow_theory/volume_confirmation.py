"""
Volume confirmation for Dow Theory analysis
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

from app.analysis.indicators.atr import ATRIndicator
from app.analysis.dow_theory.base import TrendDirection
from app.utils.logger import analysis_logger


class VolumePattern(Enum):
    """Volume pattern types"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


class VolumeConfirmation(Enum):
    """Volume confirmation status"""
    CONFIRMED = "confirmed"
    NEUTRAL = "neutral"
    DIVERGENT = "divergent"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class VolumeAnalysis:
    """Results of volume analysis"""
    pattern: VolumePattern
    confirmation: VolumeConfirmation
    strength: float  # 0.0 to 1.0
    pseudo_volume_trend: float  # Trend in pseudo-volume
    metadata: Dict[str, Any]


class VolumeConfirmationAnalyzer:
    """
    Volume confirmation analyzer for Dow Theory
    
    Since FX markets don't have traditional volume, this analyzer uses:
    1. ATR (Average True Range) as proxy for activity
    2. Tick count estimation
    3. Price movement characteristics
    4. Spread analysis (if available)
    """
    
    def __init__(self,
                 atr_period: int = 14,
                 volume_trend_period: int = 20,
                 confirmation_threshold: float = 0.6):
        """
        Initialize volume confirmation analyzer
        
        Args:
            atr_period: Period for ATR calculation
            volume_trend_period: Period for volume trend analysis
            confirmation_threshold: Threshold for confirmation (0.0 to 1.0)
        """
        self.atr_period = atr_period
        self.volume_trend_period = volume_trend_period
        self.confirmation_threshold = confirmation_threshold
        
        self.atr_indicator = ATRIndicator(period=atr_period)
    
    def analyze_volume_confirmation(self, 
                                  data: pd.DataFrame,
                                  trend_direction: TrendDirection) -> VolumeAnalysis:
        """
        Analyze volume confirmation for given trend direction
        
        Args:
            data: OHLC data (may include volume column)
            trend_direction: Current trend direction
            
        Returns:
            VolumeAnalysis object
        """
        try:
            if len(data) < self.volume_trend_period:
                return VolumeAnalysis(
                    pattern=VolumePattern.STABLE,
                    confirmation=VolumeConfirmation.INSUFFICIENT_DATA,
                    strength=0.0,
                    pseudo_volume_trend=0.0,
                    metadata={'reason': 'insufficient_data'}
                )
            
            # Calculate pseudo-volume metrics
            pseudo_volume_series = self._calculate_pseudo_volume(data)
            
            # Analyze volume pattern
            volume_pattern = self._identify_volume_pattern(pseudo_volume_series)
            
            # Calculate volume trend
            volume_trend = self._calculate_volume_trend(pseudo_volume_series)
            
            # Determine confirmation status
            confirmation = self._determine_confirmation(
                volume_pattern, volume_trend, trend_direction
            )
            
            # Calculate confirmation strength
            strength = self._calculate_confirmation_strength(
                pseudo_volume_series, trend_direction, confirmation
            )
            
            # Additional analysis metadata
            metadata = {
                'atr_period': self.atr_period,
                'trend_period': self.volume_trend_period,
                'avg_pseudo_volume': pseudo_volume_series.tail(10).mean(),
                'volume_volatility': pseudo_volume_series.tail(20).std(),
                'price_volume_correlation': self._calculate_price_volume_correlation(data, pseudo_volume_series)
            }
            
            return VolumeAnalysis(
                pattern=volume_pattern,
                confirmation=confirmation,
                strength=strength,
                pseudo_volume_trend=volume_trend,
                metadata=metadata
            )
            
        except Exception as e:
            analysis_logger.error(f"Error in volume confirmation analysis: {e}")
            return VolumeAnalysis(
                pattern=VolumePattern.STABLE,
                confirmation=VolumeConfirmation.INSUFFICIENT_DATA,
                strength=0.0,
                pseudo_volume_trend=0.0,
                metadata={'error': str(e)}
            )
    
    def _calculate_pseudo_volume(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate pseudo-volume for FX using multiple methods
        
        Methods:
        1. ATR * normalized tick count
        2. Range * estimated activity
        3. Price movement velocity
        """
        # Method 1: ATR-based pseudo volume
        atr_values = self.atr_indicator.calculate(data)
        
        # Method 2: Range-based activity estimation
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - data['close'].shift(1)).abs(),
            (data['low'] - data['close'].shift(1)).abs()
        ], axis=1).max(axis=1)
        
        # Method 3: Price velocity (rate of change)
        price_velocity = data['close'].diff().abs()
        
        # Method 4: Intrabar movement estimation
        intrabar_movement = (data['high'] - data['low']) / data['close']
        
        # Combine methods with weights
        pseudo_volume = (
            atr_values * 0.4 +
            true_range * 0.3 +
            price_velocity * 0.2 +
            intrabar_movement * 0.1
        )
        
        # Normalize and smooth
        pseudo_volume = pseudo_volume.rolling(window=3).mean()
        
        # If actual volume is available, incorporate it
        if 'volume' in data.columns:
            actual_volume = data['volume']
            # Weight: 70% actual volume, 30% pseudo volume
            combined_volume = actual_volume * 0.7 + pseudo_volume * 0.3
            return combined_volume.fillna(pseudo_volume)
        
        return pseudo_volume.fillna(0)
    
    def _identify_volume_pattern(self, volume_series: pd.Series) -> VolumePattern:
        """Identify the pattern in volume data"""
        if len(volume_series) < 10:
            return VolumePattern.STABLE
        
        recent_volume = volume_series.tail(self.volume_trend_period)
        
        # Calculate trend
        x = np.arange(len(recent_volume))
        coeffs = np.polyfit(x, recent_volume.values, 1)
        trend_slope = coeffs[0]
        
        # Calculate volatility
        volume_volatility = recent_volume.std() / recent_volume.mean()
        
        # Classify pattern
        if volume_volatility > 0.5:
            return VolumePattern.VOLATILE
        elif trend_slope > recent_volume.mean() * 0.01:
            return VolumePattern.INCREASING
        elif trend_slope < -recent_volume.mean() * 0.01:
            return VolumePattern.DECREASING
        else:
            return VolumePattern.STABLE
    
    def _calculate_volume_trend(self, volume_series: pd.Series) -> float:
        """Calculate volume trend as percentage change"""
        if len(volume_series) < self.volume_trend_period:
            return 0.0
        
        recent_volume = volume_series.tail(self.volume_trend_period)
        early_avg = recent_volume.head(self.volume_trend_period // 2).mean()
        late_avg = recent_volume.tail(self.volume_trend_period // 2).mean()
        
        if early_avg > 0:
            return (late_avg - early_avg) / early_avg
        
        return 0.0
    
    def _determine_confirmation(self, 
                              volume_pattern: VolumePattern,
                              volume_trend: float,
                              trend_direction: TrendDirection) -> VolumeConfirmation:
        """
        Determine volume confirmation status based on Dow Theory principles
        
        Dow Theory states:
        - Volume should expand in the direction of the main trend
        - Volume should contract during counter-trend moves
        """
        if trend_direction == TrendDirection.UNDEFINED:
            return VolumeConfirmation.NEUTRAL
        
        # For uptrends, increasing volume is confirming
        if trend_direction == TrendDirection.UPTREND:
            if volume_pattern == VolumePattern.INCREASING or volume_trend > 0.05:
                return VolumeConfirmation.CONFIRMED
            elif volume_pattern == VolumePattern.DECREASING or volume_trend < -0.05:
                return VolumeConfirmation.DIVERGENT
            else:
                return VolumeConfirmation.NEUTRAL
        
        # For downtrends, increasing volume is also confirming (selling pressure)
        elif trend_direction == TrendDirection.DOWNTREND:
            if volume_pattern == VolumePattern.INCREASING or volume_trend > 0.05:
                return VolumeConfirmation.CONFIRMED
            elif volume_pattern == VolumePattern.DECREASING or volume_trend < -0.05:
                return VolumeConfirmation.DIVERGENT
            else:
                return VolumeConfirmation.NEUTRAL
        
        # For sideways trends
        elif trend_direction == TrendDirection.SIDEWAYS:
            if volume_pattern == VolumePattern.DECREASING:
                return VolumeConfirmation.CONFIRMED  # Low volume in consolidation
            elif volume_pattern == VolumePattern.INCREASING:
                return VolumeConfirmation.DIVERGENT  # High volume may indicate breakout
            else:
                return VolumeConfirmation.NEUTRAL
        
        return VolumeConfirmation.NEUTRAL
    
    def _calculate_confirmation_strength(self, 
                                       volume_series: pd.Series,
                                       trend_direction: TrendDirection,
                                       confirmation: VolumeConfirmation) -> float:
        """Calculate the strength of volume confirmation"""
        if confirmation == VolumeConfirmation.INSUFFICIENT_DATA:
            return 0.0
        
        base_strength = {
            VolumeConfirmation.CONFIRMED: 0.8,
            VolumeConfirmation.NEUTRAL: 0.5,
            VolumeConfirmation.DIVERGENT: 0.2
        }.get(confirmation, 0.5)
        
        # Adjust based on volume consistency
        consistency_factor = self._calculate_volume_consistency(volume_series)
        
        # Adjust based on volume magnitude
        magnitude_factor = self._calculate_volume_magnitude_factor(volume_series)
        
        # Final strength calculation
        strength = base_strength * consistency_factor * magnitude_factor
        
        return max(0.0, min(1.0, strength))
    
    def _calculate_volume_consistency(self, volume_series: pd.Series) -> float:
        """Calculate how consistent the volume pattern is"""
        if len(volume_series) < 10:
            return 0.5
        
        recent_volume = volume_series.tail(10)
        
        # Calculate coefficient of variation (lower is more consistent)
        cv = recent_volume.std() / recent_volume.mean() if recent_volume.mean() > 0 else 1.0
        
        # Convert to consistency factor (0 to 1, higher is more consistent)
        consistency = max(0.0, min(1.0, 1.0 - cv))
        
        return 0.5 + (consistency * 0.5)  # Range: 0.5 to 1.0
    
    def _calculate_volume_magnitude_factor(self, volume_series: pd.Series) -> float:
        """Calculate factor based on volume magnitude relative to historical levels"""
        if len(volume_series) < 20:
            return 1.0
        
        current_avg = volume_series.tail(5).mean()
        historical_avg = volume_series.tail(20).mean()
        
        if historical_avg > 0:
            magnitude_ratio = current_avg / historical_avg
            
            # Optimal range is 0.8 to 1.5 (not too low, not too extreme)
            if 0.8 <= magnitude_ratio <= 1.5:
                return 1.0
            elif 0.5 <= magnitude_ratio <= 2.0:
                return 0.8
            else:
                return 0.6
        
        return 1.0
    
    def _calculate_price_volume_correlation(self, 
                                          data: pd.DataFrame,
                                          volume_series: pd.Series) -> float:
        """Calculate correlation between price changes and volume"""
        if len(data) < 20 or len(volume_series) < 20:
            return 0.0
        
        # Calculate price changes
        price_changes = data['close'].pct_change().tail(20)
        recent_volume = volume_series.tail(20)
        
        # Align series
        min_length = min(len(price_changes), len(recent_volume))
        price_changes = price_changes.tail(min_length)
        recent_volume = recent_volume.tail(min_length)
        
        # Calculate correlation
        try:
            correlation = price_changes.corr(recent_volume)
            return correlation if not pd.isna(correlation) else 0.0
        except:
            return 0.0
    
    def get_volume_signals(self, 
                          data: pd.DataFrame,
                          trend_direction: TrendDirection) -> List[Dict[str, Any]]:
        """
        Generate volume-based signals
        
        Args:
            data: OHLC data
            trend_direction: Current trend direction
            
        Returns:
            List of volume signal dictionaries
        """
        signals = []
        
        try:
            volume_analysis = self.analyze_volume_confirmation(data, trend_direction)
            
            # Volume breakout signal
            if volume_analysis.pattern == VolumePattern.INCREASING:
                pseudo_volume = self._calculate_pseudo_volume(data)
                current_volume = pseudo_volume.iloc[-1]
                avg_volume = pseudo_volume.tail(20).mean()
                
                if current_volume > avg_volume * 1.5:  # 50% above average
                    signals.append({
                        'type': 'volume_breakout',
                        'strength': min(1.0, current_volume / avg_volume - 1.0),
                        'description': f'Volume spike: {current_volume/avg_volume:.1f}x average',
                        'trend_supporting': volume_analysis.confirmation == VolumeConfirmation.CONFIRMED
                    })
            
            # Volume divergence signal
            if volume_analysis.confirmation == VolumeConfirmation.DIVERGENT:
                signals.append({
                    'type': 'volume_divergence',
                    'strength': 1.0 - volume_analysis.strength,
                    'description': f'Volume diverging from {trend_direction.value} trend',
                    'warning': True
                })
            
            # Volume confirmation signal
            if (volume_analysis.confirmation == VolumeConfirmation.CONFIRMED and
                volume_analysis.strength >= self.confirmation_threshold):
                signals.append({
                    'type': 'volume_confirmation',
                    'strength': volume_analysis.strength,
                    'description': f'Volume confirming {trend_direction.value} trend',
                    'bullish': trend_direction in [TrendDirection.UPTREND]
                })
        
        except Exception as e:
            analysis_logger.error(f"Error generating volume signals: {e}")
        
        return signals
    
    def get_volume_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of volume analysis"""
        try:
            pseudo_volume = self._calculate_pseudo_volume(data)
            
            return {
                'current_volume': pseudo_volume.iloc[-1] if not pseudo_volume.empty else 0,
                'avg_volume_10': pseudo_volume.tail(10).mean(),
                'avg_volume_20': pseudo_volume.tail(20).mean(),
                'volume_trend': self._calculate_volume_trend(pseudo_volume),
                'volume_pattern': self._identify_volume_pattern(pseudo_volume).value,
                'volume_volatility': pseudo_volume.tail(20).std() / pseudo_volume.tail(20).mean()
                                   if pseudo_volume.tail(20).mean() > 0 else 0
            }
            
        except Exception as e:
            analysis_logger.error(f"Error in volume summary: {e}")
            return {'error': str(e)}


# Create default analyzer instance
volume_confirmation_analyzer = VolumeConfirmationAnalyzer()