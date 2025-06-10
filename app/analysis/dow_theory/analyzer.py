"""
Dow Theory Analyzer Implementation
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.analysis.dow_theory.base import (
    DowTheoryBase, TrendAnalysis, DowTheorySignal, TrendDirection, 
    TrendStrength, ConfirmationStatus
)
from app.analysis.indicators.swing_detector import SwingPoint, SwingType
from app.utils.logger import analysis_logger


class DowTheoryAnalyzer(DowTheoryBase):
    """
    Complete Dow Theory analyzer implementation
    
    Implements the core principles of Dow Theory:
    1. The averages (indices) discount everything
    2. The market has three trends: primary, secondary, minor
    3. Primary trends have three phases
    4. The averages must confirm each other
    5. Volume must confirm the trend
    6. A trend is assumed to continue until it gives definite signals of reversal
    """
    
    def __init__(self,
                 swing_sensitivity: float = 0.5,
                 min_swing_size_pct: float = 2.0,
                 trend_confirmation_swings: int = 3,
                 use_volume_confirmation: bool = True,
                 atr_period: int = 14,
                 primary_trend_min_days: int = 21,
                 secondary_trend_min_days: int = 7):
        """
        Initialize Dow Theory analyzer
        
        Args:
            swing_sensitivity: Swing detection sensitivity
            min_swing_size_pct: Minimum swing size percentage
            trend_confirmation_swings: Number of swings for trend confirmation
            use_volume_confirmation: Whether to use volume confirmation
            atr_period: ATR period for volatility analysis
            primary_trend_min_days: Minimum days for primary trend classification
            secondary_trend_min_days: Minimum days for secondary trend classification
        """
        super().__init__(
            swing_sensitivity=swing_sensitivity,
            min_swing_size_pct=min_swing_size_pct,
            trend_confirmation_swings=trend_confirmation_swings,
            use_volume_confirmation=use_volume_confirmation,
            atr_period=atr_period
        )
        
        self.primary_trend_min_days = primary_trend_min_days
        self.secondary_trend_min_days = secondary_trend_min_days
    
    def analyze_trend(self, data: pd.DataFrame, timeframe: str = 'D1') -> TrendAnalysis:
        """
        Analyze trend according to Dow Theory principles
        
        Args:
            data: OHLC data
            timeframe: Timeframe string
            
        Returns:
            TrendAnalysis object with comprehensive trend information
        """
        try:
            # Detect swing points
            swing_points = self.detect_swing_points(data)
            
            if len(swing_points) < self.trend_confirmation_swings:
                return TrendAnalysis(
                    direction=TrendDirection.UNDEFINED,
                    strength=TrendStrength.VERY_WEAK,
                    confidence=0.0,
                    swing_points=swing_points,
                    metadata={'reason': 'insufficient_swing_points'}
                )
            
            # Identify trend direction using HH/HL and LH/LL patterns
            trend_direction = self.identify_trend_direction(swing_points)
            
            # Calculate trend strength
            trend_strength = self.calculate_trend_strength(data, swing_points, trend_direction)
            
            # Calculate confidence based on multiple factors
            confidence = self._calculate_trend_confidence(data, swing_points, trend_direction, trend_strength)
            
            # Find trend start date and duration
            start_date, duration_days = self._calculate_trend_duration(swing_points, trend_direction)
            
            # Calculate key support/resistance levels
            key_levels = self.calculate_key_levels(swing_points)
            
            # Additional metadata
            metadata = {
                'timeframe': timeframe,
                'swing_count': len(swing_points),
                'trend_phases': self._identify_trend_phases(swing_points, trend_direction),
                'recent_swing_strength': self._calculate_recent_swing_strength(swing_points),
                'volatility_regime': self._classify_volatility_regime(data)
            }
            
            return TrendAnalysis(
                direction=trend_direction,
                strength=trend_strength,
                confidence=confidence,
                start_date=start_date,
                duration_days=duration_days,
                swing_points=swing_points,
                key_levels=key_levels,
                metadata=metadata
            )
            
        except Exception as e:
            analysis_logger.error(f"Error in trend analysis: {e}")
            return TrendAnalysis(
                direction=TrendDirection.UNDEFINED,
                strength=TrendStrength.VERY_WEAK,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
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
        signals = []
        
        try:
            # Analyze primary trend
            primary_trend = self.analyze_trend(data, 'primary')
            
            # Analyze secondary trend if data provided
            secondary_trend = None
            if secondary_data is not None:
                secondary_trend = self.analyze_trend(secondary_data, 'secondary')
            
            # Check for trend confirmation
            confirmation_status = self.check_confirmation(primary_trend, secondary_trend)
            
            # Generate signals based on trend analysis
            signals.extend(self._generate_trend_following_signals(primary_trend, confirmation_status))
            
            # Generate reversal signals
            signals.extend(self._generate_reversal_signals(data, primary_trend))
            
            # Generate breakout signals
            signals.extend(self._generate_breakout_signals(data, primary_trend))
            
            analysis_logger.debug(f"Generated {len(signals)} Dow Theory signals")
            
        except Exception as e:
            analysis_logger.error(f"Error generating signals: {e}")
        
        return signals
    
    def _calculate_trend_confidence(self, data: pd.DataFrame, swing_points: List[SwingPoint],
                                  trend_direction: TrendDirection, trend_strength: TrendStrength) -> float:
        """Calculate overall confidence in trend analysis"""
        if trend_direction == TrendDirection.UNDEFINED:
            return 0.0
        
        confidence_factors = []
        
        # Factor 1: Swing point consistency
        consistency = self._calculate_swing_consistency(swing_points, trend_direction)
        confidence_factors.append(consistency * 0.3)
        
        # Factor 2: Trend strength
        strength_value = self._strength_to_value(trend_strength)
        confidence_factors.append((strength_value / 4) * 0.25)
        
        # Factor 3: Number of confirming swings
        confirming_swings = self._count_confirming_swings(swing_points, trend_direction)
        swing_confidence = min(1.0, confirming_swings / self.trend_confirmation_swings)
        confidence_factors.append(swing_confidence * 0.25)
        
        # Factor 4: Recent price action alignment
        recent_alignment = self._check_recent_price_alignment(data, trend_direction)
        confidence_factors.append(recent_alignment * 0.2)
        
        return sum(confidence_factors)
    
    def _calculate_trend_duration(self, swing_points: List[SwingPoint], 
                                trend_direction: TrendDirection) -> Tuple[Optional[datetime], int]:
        """Calculate when the current trend started and its duration"""
        if not swing_points or trend_direction == TrendDirection.UNDEFINED:
            return None, 0
        
        # Find the swing point that started the current trend
        trend_start_point = self._find_trend_start_point(swing_points, trend_direction)
        
        if trend_start_point:
            start_date = trend_start_point.timestamp
            if isinstance(start_date, datetime):
                duration = (datetime.utcnow() - start_date).days
                return start_date, duration
        
        return None, 0
    
    def _find_trend_start_point(self, swing_points: List[SwingPoint], 
                              trend_direction: TrendDirection) -> Optional[SwingPoint]:
        """Find the swing point where the current trend began"""
        if len(swing_points) < 3:
            return None
        
        # Sort swing points by index
        sorted_swings = sorted(swing_points, key=lambda x: x.index)
        
        if trend_direction == TrendDirection.UPTREND:
            # Look for the low that started the uptrend
            for i in range(len(sorted_swings) - 2):
                current = sorted_swings[i]
                if current.swing_type == SwingType.LOW:
                    # Check if subsequent swings confirm uptrend
                    subsequent_highs = [s for s in sorted_swings[i+1:] if s.swing_type == SwingType.HIGH]
                    subsequent_lows = [s for s in sorted_swings[i+1:] if s.swing_type == SwingType.LOW]
                    
                    if (len(subsequent_highs) >= 1 and len(subsequent_lows) >= 1 and
                        subsequent_highs[0].price > current.price and
                        subsequent_lows[0].price > current.price):
                        return current
        
        elif trend_direction == TrendDirection.DOWNTREND:
            # Look for the high that started the downtrend
            for i in range(len(sorted_swings) - 2):
                current = sorted_swings[i]
                if current.swing_type == SwingType.HIGH:
                    # Check if subsequent swings confirm downtrend
                    subsequent_highs = [s for s in sorted_swings[i+1:] if s.swing_type == SwingType.HIGH]
                    subsequent_lows = [s for s in sorted_swings[i+1:] if s.swing_type == SwingType.LOW]
                    
                    if (len(subsequent_highs) >= 1 and len(subsequent_lows) >= 1 and
                        subsequent_highs[0].price < current.price and
                        subsequent_lows[0].price < current.price):
                        return current
        
        return sorted_swings[0] if sorted_swings else None
    
    def _identify_trend_phases(self, swing_points: List[SwingPoint], 
                             trend_direction: TrendDirection) -> str:
        """
        Identify which phase of the trend we're in
        
        Dow Theory identifies three phases:
        1. Accumulation/Distribution phase
        2. Public participation phase  
        3. Excess/Panic phase
        """
        if len(swing_points) < 4:
            return 'undefined'
        
        # Analyze momentum and volatility patterns
        recent_swings = swing_points[-4:]
        
        # Calculate swing magnitudes
        swing_magnitudes = []
        for i in range(1, len(recent_swings)):
            magnitude = abs(recent_swings[i].price - recent_swings[i-1].price)
            swing_magnitudes.append(magnitude)
        
        if len(swing_magnitudes) < 2:
            return 'undefined'
        
        # Analyze pattern of swing magnitudes
        avg_early = np.mean(swing_magnitudes[:len(swing_magnitudes)//2])
        avg_recent = np.mean(swing_magnitudes[len(swing_magnitudes)//2:])
        
        magnitude_ratio = avg_recent / avg_early if avg_early > 0 else 1.0
        
        # Classify phase based on magnitude patterns
        if magnitude_ratio < 0.8:
            return 'accumulation'  # Decreasing volatility, consolidation
        elif magnitude_ratio > 1.5:
            return 'excess'        # Increasing volatility, potential exhaustion
        else:
            return 'public_participation'  # Steady momentum
    
    def _calculate_recent_swing_strength(self, swing_points: List[SwingPoint]) -> float:
        """Calculate the strength of recent swing points"""
        if len(swing_points) < 2:
            return 0.0
        
        recent_swings = swing_points[-3:] if len(swing_points) >= 3 else swing_points
        return np.mean([sp.strength for sp in recent_swings])
    
    def _classify_volatility_regime(self, data: pd.DataFrame) -> str:
        """Classify current volatility regime"""
        atr_values = self.atr_indicator.calculate(data)
        
        if len(atr_values) < 20:
            return 'unknown'
        
        current_atr = atr_values.iloc[-1]
        historical_atr = atr_values.tail(50).mean()
        
        if current_atr / historical_atr > 1.5:
            return 'high_volatility'
        elif current_atr / historical_atr < 0.7:
            return 'low_volatility'
        else:
            return 'normal_volatility'
    
    def _count_confirming_swings(self, swing_points: List[SwingPoint], 
                               trend_direction: TrendDirection) -> int:
        """Count the number of swings that confirm the trend"""
        if len(swing_points) < 2:
            return 0
        
        highs = [sp for sp in swing_points if sp.swing_type == SwingType.HIGH]
        lows = [sp for sp in swing_points if sp.swing_type == SwingType.LOW]
        
        highs.sort(key=lambda x: x.index)
        lows.sort(key=lambda x: x.index)
        
        confirming_swings = 0
        
        if trend_direction == TrendDirection.UPTREND:
            # Count higher highs
            for i in range(1, len(highs)):
                if highs[i].price > highs[i-1].price:
                    confirming_swings += 1
            
            # Count higher lows
            for i in range(1, len(lows)):
                if lows[i].price > lows[i-1].price:
                    confirming_swings += 1
        
        elif trend_direction == TrendDirection.DOWNTREND:
            # Count lower highs
            for i in range(1, len(highs)):
                if highs[i].price < highs[i-1].price:
                    confirming_swings += 1
            
            # Count lower lows
            for i in range(1, len(lows)):
                if lows[i].price < lows[i-1].price:
                    confirming_swings += 1
        
        return confirming_swings
    
    def _check_recent_price_alignment(self, data: pd.DataFrame, 
                                    trend_direction: TrendDirection) -> float:
        """Check if recent price action aligns with trend direction"""
        if len(data) < 10:
            return 0.0
        
        recent_data = data.tail(10)
        price_changes = recent_data['close'].pct_change().dropna()
        
        if trend_direction == TrendDirection.UPTREND:
            # Count positive price changes
            positive_changes = (price_changes > 0).sum()
            return positive_changes / len(price_changes)
        
        elif trend_direction == TrendDirection.DOWNTREND:
            # Count negative price changes
            negative_changes = (price_changes < 0).sum()
            return negative_changes / len(price_changes)
        
        return 0.5  # Neutral for sideways trend
    
    def _generate_trend_following_signals(self, trend_analysis: TrendAnalysis, 
                                        confirmation_status: ConfirmationStatus) -> List[DowTheorySignal]:
        """Generate trend-following signals"""
        signals = []
        
        if (trend_analysis.direction in [TrendDirection.UPTREND, TrendDirection.DOWNTREND] and
            trend_analysis.strength in [TrendStrength.STRONG, TrendStrength.VERY_STRONG] and
            trend_analysis.confidence >= 0.6):
            
            signal_type = 'buy' if trend_analysis.direction == TrendDirection.UPTREND else 'sell'
            
            # Calculate entry based on recent swing points
            entry_price = self._calculate_trend_entry_price(trend_analysis)
            stop_loss = self._calculate_trend_stop_loss(trend_analysis)
            take_profit = self._calculate_trend_take_profit(trend_analysis, entry_price, stop_loss)
            
            risk_reward = None
            if entry_price and stop_loss and take_profit:
                risk = abs(entry_price - stop_loss)
                reward = abs(take_profit - entry_price)
                risk_reward = reward / risk if risk > 0 else None
            
            signal = DowTheorySignal(
                signal_type=signal_type,
                confidence=trend_analysis.confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward,
                confirmation_status=confirmation_status,
                reasoning=f"Strong {trend_analysis.direction.value} trend with {trend_analysis.strength.value} strength",
                metadata={
                    'signal_source': 'trend_following',
                    'trend_duration_days': trend_analysis.duration_days,
                    'swing_points_count': len(trend_analysis.swing_points)
                }
            )
            signals.append(signal)
        
        return signals
    
    def _generate_reversal_signals(self, data: pd.DataFrame, 
                                 trend_analysis: TrendAnalysis) -> List[DowTheorySignal]:
        """Generate trend reversal signals"""
        signals = []
        
        # Look for potential reversal patterns
        if len(trend_analysis.swing_points) >= 4:
            recent_swings = trend_analysis.swing_points[-4:]
            
            # Check for failure to make new highs/lows
            reversal_detected = self._detect_potential_reversal(recent_swings, trend_analysis.direction)
            
            if reversal_detected:
                signal_type = 'sell' if trend_analysis.direction == TrendDirection.UPTREND else 'buy'
                
                signal = DowTheorySignal(
                    signal_type=signal_type,
                    confidence=0.4,  # Lower confidence for reversal signals
                    confirmation_status=ConfirmationStatus.PENDING,
                    reasoning=f"Potential {trend_analysis.direction.value} reversal detected",
                    metadata={
                        'signal_source': 'reversal',
                        'reversal_pattern': 'failure_to_confirm'
                    }
                )
                signals.append(signal)
        
        return signals
    
    def _generate_breakout_signals(self, data: pd.DataFrame, 
                                 trend_analysis: TrendAnalysis) -> List[DowTheorySignal]:
        """Generate breakout signals"""
        signals = []
        
        if not trend_analysis.key_levels:
            return signals
        
        current_price = data['close'].iloc[-1]
        
        # Check for resistance breakout
        if 'resistance' in trend_analysis.key_levels:
            resistance = trend_analysis.key_levels['resistance']
            if current_price > resistance * 1.001:  # 0.1% above resistance
                signal = DowTheorySignal(
                    signal_type='buy',
                    confidence=0.5,
                    entry_price=current_price,
                    stop_loss=resistance * 0.995,  # Just below resistance
                    reasoning=f"Breakout above resistance at {resistance:.5f}",
                    metadata={
                        'signal_source': 'breakout',
                        'breakout_level': resistance,
                        'breakout_type': 'resistance'
                    }
                )
                signals.append(signal)
        
        # Check for support breakdown
        if 'support' in trend_analysis.key_levels:
            support = trend_analysis.key_levels['support']
            if current_price < support * 0.999:  # 0.1% below support
                signal = DowTheorySignal(
                    signal_type='sell',
                    confidence=0.5,
                    entry_price=current_price,
                    stop_loss=support * 1.005,  # Just above support
                    reasoning=f"Breakdown below support at {support:.5f}",
                    metadata={
                        'signal_source': 'breakout',
                        'breakout_level': support,
                        'breakout_type': 'support'
                    }
                )
                signals.append(signal)
        
        return signals
    
    def _calculate_trend_entry_price(self, trend_analysis: TrendAnalysis) -> Optional[float]:
        """Calculate optimal entry price for trend-following signal"""
        if not trend_analysis.swing_points:
            return None
        
        recent_swings = trend_analysis.swing_points[-2:]
        
        if trend_analysis.direction == TrendDirection.UPTREND:
            # Enter on pullback to recent swing low
            lows = [sp.price for sp in recent_swings if sp.swing_type == SwingType.LOW]
            return max(lows) if lows else None
        
        elif trend_analysis.direction == TrendDirection.DOWNTREND:
            # Enter on bounce to recent swing high
            highs = [sp.price for sp in recent_swings if sp.swing_type == SwingType.HIGH]
            return min(highs) if highs else None
        
        return None
    
    def _calculate_trend_stop_loss(self, trend_analysis: TrendAnalysis) -> Optional[float]:
        """Calculate stop loss for trend-following signal"""
        if not trend_analysis.swing_points:
            return None
        
        if trend_analysis.direction == TrendDirection.UPTREND:
            # Stop below recent swing low
            lows = [sp.price for sp in trend_analysis.swing_points if sp.swing_type == SwingType.LOW]
            return min(lows[-2:]) * 0.995 if len(lows) >= 2 else None
        
        elif trend_analysis.direction == TrendDirection.DOWNTREND:
            # Stop above recent swing high
            highs = [sp.price for sp in trend_analysis.swing_points if sp.swing_type == SwingType.HIGH]
            return max(highs[-2:]) * 1.005 if len(highs) >= 2 else None
        
        return None
    
    def _calculate_trend_take_profit(self, trend_analysis: TrendAnalysis, 
                                   entry_price: Optional[float], 
                                   stop_loss: Optional[float]) -> Optional[float]:
        """Calculate take profit target"""
        if not entry_price or not stop_loss:
            return None
        
        risk = abs(entry_price - stop_loss)
        risk_reward_ratio = 2.0  # Target 2:1 risk reward
        
        if trend_analysis.direction == TrendDirection.UPTREND:
            return entry_price + (risk * risk_reward_ratio)
        elif trend_analysis.direction == TrendDirection.DOWNTREND:
            return entry_price - (risk * risk_reward_ratio)
        
        return None
    
    def _detect_potential_reversal(self, recent_swings: List[SwingPoint], 
                                 trend_direction: TrendDirection) -> bool:
        """Detect potential trend reversal patterns"""
        if len(recent_swings) < 4:
            return False
        
        if trend_direction == TrendDirection.UPTREND:
            # Look for failure to make new highs and breakdown of recent low
            highs = [sp for sp in recent_swings if sp.swing_type == SwingType.HIGH]
            if len(highs) >= 2:
                # Check if recent high is lower than previous high
                return highs[-1].price < highs[-2].price
        
        elif trend_direction == TrendDirection.DOWNTREND:
            # Look for failure to make new lows and breakout of recent high
            lows = [sp for sp in recent_swings if sp.swing_type == SwingType.LOW]
            if len(lows) >= 2:
                # Check if recent low is higher than previous low
                return lows[-1].price > lows[-2].price
        
        return False


# Create default analyzer instance
dow_theory_analyzer = DowTheoryAnalyzer()