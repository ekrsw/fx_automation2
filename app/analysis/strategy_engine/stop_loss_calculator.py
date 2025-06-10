"""
Stop Loss Calculation Engine

Advanced stop loss calculation system using multiple methods including
ATR, swing points, Fibonacci levels, and pattern-based stops.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from app.analysis.strategy_engine.unified_analyzer import UnifiedAnalysisResult
from app.analysis.strategy_engine.signal_generator import TradingSignal, SignalType
from app.analysis.indicators.atr import ATRIndicator
from app.utils.logger import analysis_logger


class StopLossType(Enum):
    """Types of stop loss calculation methods"""
    ATR_BASED = "atr_based"
    SWING_POINT = "swing_point"
    FIBONACCI = "fibonacci"
    PATTERN_BASED = "pattern_based"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    PERCENTAGE = "percentage"
    SUPPORT_RESISTANCE = "support_resistance"


@dataclass
class StopLossLevel:
    """Individual stop loss level calculation"""
    method: StopLossType
    level: float
    confidence: float  # 0.0 to 1.0
    risk_percent: float  # Risk as percentage of account
    distance_pips: float  # Distance in pips from entry
    rationale: str
    supporting_data: Dict[str, Any]


@dataclass
class StopLossRecommendation:
    """Complete stop loss recommendation"""
    primary_stop: float
    secondary_stop: Optional[float]
    initial_stop: float
    trailing_stop: Optional[float]
    
    # Risk metrics
    risk_percent: float
    risk_reward_ratio: float
    max_loss_amount: float
    
    # Analysis breakdown
    calculations: List[StopLossLevel]
    method_used: StopLossType
    confidence: float
    
    # Metadata
    calculation_timestamp: datetime
    valid_for_hours: int


class StopLossCalculator:
    """
    Advanced stop loss calculation engine
    
    Calculates optimal stop loss levels using multiple methods and
    combines them to provide the best risk management approach.
    """
    
    def __init__(self,
                 default_atr_multiplier: float = 2.0,
                 max_risk_percent: float = 2.0,
                 min_risk_reward_ratio: float = 1.5):
        """
        Initialize stop loss calculator
        
        Args:
            default_atr_multiplier: Default ATR multiplier for stops
            max_risk_percent: Maximum risk per trade
            min_risk_reward_ratio: Minimum acceptable risk/reward ratio
        """
        self.default_atr_multiplier = default_atr_multiplier
        self.max_risk_percent = max_risk_percent
        self.min_risk_reward_ratio = min_risk_reward_ratio
        
        # Calculation configuration
        self.config = {
            'atr_period': 14,
            'atr_multipliers': [1.5, 2.0, 2.5, 3.0],  # Different ATR options
            'swing_lookback': 20,  # Periods to look back for swing points
            'fibonacci_levels': [0.236, 0.382, 0.500, 0.618, 0.786],
            'volatility_window': 20,  # Period for volatility calculation
            'min_stop_distance_pips': 5,  # Minimum stop distance
            'max_stop_distance_pips': 100,  # Maximum stop distance
            'confidence_weights': {
                StopLossType.ATR_BASED: 0.25,
                StopLossType.SWING_POINT: 0.20,
                StopLossType.FIBONACCI: 0.15,
                StopLossType.PATTERN_BASED: 0.20,
                StopLossType.VOLATILITY_ADJUSTED: 0.10,
                StopLossType.SUPPORT_RESISTANCE: 0.10
            }
        }
        
        # Initialize ATR indicator
        self.atr_indicator = ATRIndicator(period=self.config['atr_period'])
    
    def calculate_stop_loss(self,
                          signal: TradingSignal,
                          unified_result: UnifiedAnalysisResult,
                          price_data: pd.DataFrame,
                          account_balance: float = 10000.0) -> StopLossRecommendation:
        """
        Calculate comprehensive stop loss recommendation
        
        Args:
            signal: Trading signal
            unified_result: Unified analysis result
            price_data: Historical price data
            account_balance: Account balance for risk calculation
            
        Returns:
            Complete stop loss recommendation
        """
        try:
            calculations = []
            current_price = signal.entry_price or price_data['close'].iloc[-1]
            
            # Method 1: ATR-based stop loss
            atr_stops = self._calculate_atr_stops(price_data, current_price, signal)
            calculations.extend(atr_stops)
            
            # Method 2: Swing point stops
            swing_stops = self._calculate_swing_point_stops(
                unified_result, current_price, signal
            )
            calculations.extend(swing_stops)
            
            # Method 3: Fibonacci-based stops
            fib_stops = self._calculate_fibonacci_stops(
                unified_result, current_price, signal
            )
            calculations.extend(fib_stops)
            
            # Method 4: Pattern-based stops
            pattern_stops = self._calculate_pattern_stops(
                unified_result, current_price, signal
            )
            calculations.extend(pattern_stops)
            
            # Method 5: Volatility-adjusted stops
            volatility_stops = self._calculate_volatility_stops(
                price_data, current_price, signal
            )
            calculations.extend(volatility_stops)
            
            # Method 6: Support/Resistance stops
            sr_stops = self._calculate_support_resistance_stops(
                unified_result, current_price, signal
            )
            calculations.extend(sr_stops)
            
            # Select optimal stop loss
            primary_stop, method_used, confidence = self._select_optimal_stop(calculations)
            
            # Calculate secondary and trailing stops
            secondary_stop = self._calculate_secondary_stop(calculations, primary_stop)
            initial_stop = primary_stop  # Could be different for staged entries
            trailing_stop = self._calculate_trailing_stop(primary_stop, current_price, signal)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(
                primary_stop, current_price, signal, account_balance
            )
            
            recommendation = StopLossRecommendation(
                primary_stop=primary_stop,
                secondary_stop=secondary_stop,
                initial_stop=initial_stop,
                trailing_stop=trailing_stop,
                risk_percent=risk_metrics['risk_percent'],
                risk_reward_ratio=risk_metrics['risk_reward_ratio'],
                max_loss_amount=risk_metrics['max_loss_amount'],
                calculations=calculations,
                method_used=method_used,
                confidence=confidence,
                calculation_timestamp=datetime.utcnow(),
                valid_for_hours=4
            )
            
            analysis_logger.info(
                f"Calculated stop loss: {signal.symbol} - "
                f"Primary: {primary_stop:.5f}, Method: {method_used.value}, "
                f"Risk: {risk_metrics['risk_percent']:.1f}%"
            )
            
            return recommendation
            
        except Exception as e:
            analysis_logger.error(f"Error calculating stop loss: {e}")
            return self._create_default_stop_loss(signal, current_price, account_balance)
    
    def _calculate_atr_stops(self,
                           price_data: pd.DataFrame,
                           current_price: float,
                           signal: TradingSignal) -> List[StopLossLevel]:
        """Calculate ATR-based stop loss levels"""
        stops = []
        
        try:
            # Calculate ATR
            atr_values = self.atr_indicator.calculate(price_data)
            if atr_values is None or len(atr_values) == 0:
                return stops
            
            current_atr = atr_values.iloc[-1]
            
            # Calculate stops for different ATR multipliers
            for multiplier in self.config['atr_multipliers']:
                if signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
                    stop_level = current_price - (current_atr * multiplier)
                else:
                    stop_level = current_price + (current_atr * multiplier)
                
                # Calculate risk metrics
                risk_distance = abs(current_price - stop_level)
                risk_percent = (risk_distance / current_price) * 100
                distance_pips = self._calculate_pips(current_price, stop_level, signal.symbol)
                
                # Confidence based on ATR multiplier (2.0 is optimal)
                confidence = 1.0 - abs(multiplier - 2.0) * 0.2
                confidence = max(0.3, min(1.0, confidence))
                
                stop = StopLossLevel(
                    method=StopLossType.ATR_BASED,
                    level=stop_level,
                    confidence=confidence,
                    risk_percent=risk_percent,
                    distance_pips=distance_pips,
                    rationale=f"ATR({self.config['atr_period']}) * {multiplier}",
                    supporting_data={
                        'atr_value': current_atr,
                        'multiplier': multiplier,
                        'atr_period': self.config['atr_period']
                    }
                )
                stops.append(stop)
                
        except Exception as e:
            analysis_logger.error(f"Error calculating ATR stops: {e}")
        
        return stops
    
    def _calculate_swing_point_stops(self,
                                   unified_result: UnifiedAnalysisResult,
                                   current_price: float,
                                   signal: TradingSignal) -> List[StopLossLevel]:
        """Calculate swing point based stops"""
        stops = []
        
        try:
            swing_points = unified_result.swing_points
            if not swing_points:
                return stops
            
            # Find relevant swing points
            if signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
                # For buy signals, look for swing lows below current price
                relevant_swings = [
                    sp for sp in swing_points 
                    if sp.swing_type.value.lower() == 'low' and sp.price < current_price
                ]
                relevant_swings.sort(key=lambda x: x.price, reverse=True)  # Highest first
            else:
                # For sell signals, look for swing highs above current price
                relevant_swings = [
                    sp for sp in swing_points 
                    if sp.swing_type.value.lower() == 'high' and sp.price > current_price
                ]
                relevant_swings.sort(key=lambda x: x.price)  # Lowest first
            
            # Take top 3 most relevant swing points
            for i, swing in enumerate(relevant_swings[:3]):
                stop_level = swing.price
                
                # Adjust slightly beyond the swing point
                adjustment = current_price * 0.0005  # 0.05% adjustment
                if signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
                    stop_level -= adjustment
                else:
                    stop_level += adjustment
                
                risk_distance = abs(current_price - stop_level)
                risk_percent = (risk_distance / current_price) * 100
                distance_pips = self._calculate_pips(current_price, stop_level, signal.symbol)
                
                # Confidence decreases with distance and swing age
                distance_factor = max(0.3, 1.0 - (risk_distance / current_price) * 10)
                recency_factor = max(0.5, 1.0 - i * 0.2)  # Newer swings get higher confidence
                confidence = distance_factor * recency_factor * swing.strength
                
                stop = StopLossLevel(
                    method=StopLossType.SWING_POINT,
                    level=stop_level,
                    confidence=confidence,
                    risk_percent=risk_percent,
                    distance_pips=distance_pips,
                    rationale=f"Swing {swing.swing_type.value} at {swing.price:.5f}",
                    supporting_data={
                        'swing_point': {
                            'price': swing.price,
                            'type': swing.swing_type.value,
                            'strength': swing.strength,
                            'timestamp': swing.timestamp.isoformat() if swing.timestamp else None
                        },
                        'adjustment': adjustment
                    }
                )
                stops.append(stop)
                
        except Exception as e:
            analysis_logger.error(f"Error calculating swing point stops: {e}")
        
        return stops
    
    def _calculate_fibonacci_stops(self,
                                 unified_result: UnifiedAnalysisResult,
                                 current_price: float,
                                 signal: TradingSignal) -> List[StopLossLevel]:
        """Calculate Fibonacci-based stops"""
        stops = []
        
        try:
            # Use Elliott Wave Fibonacci levels if available
            if hasattr(unified_result, 'price_targets') and unified_result.price_targets:
                fib_levels = []
                
                # Extract Fibonacci levels from price targets
                for key, level in unified_result.price_targets.items():
                    if level and 'fib' in key.lower():
                        fib_levels.append(level)
                
                # Also check risk levels for Fibonacci-based stops
                if hasattr(unified_result, 'risk_levels') and unified_result.risk_levels:
                    for key, level in unified_result.risk_levels.items():
                        if level and 'fib' in key.lower():
                            fib_levels.append(level)
                
                # Filter relevant levels for stop loss
                relevant_levels = []
                if signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
                    relevant_levels = [level for level in fib_levels if level < current_price]
                else:
                    relevant_levels = [level for level in fib_levels if level > current_price]
                
                # Create stops for each relevant Fibonacci level
                for level in relevant_levels[:3]:  # Top 3 levels
                    risk_distance = abs(current_price - level)
                    risk_percent = (risk_distance / current_price) * 100
                    distance_pips = self._calculate_pips(current_price, level, signal.symbol)
                    
                    # Confidence based on how close to standard Fibonacci ratios
                    confidence = 0.7  # Base confidence for Fibonacci levels
                    
                    stop = StopLossLevel(
                        method=StopLossType.FIBONACCI,
                        level=level,
                        confidence=confidence,
                        risk_percent=risk_percent,
                        distance_pips=distance_pips,
                        rationale=f"Fibonacci support/resistance at {level:.5f}",
                        supporting_data={
                            'fibonacci_level': level,
                            'level_type': 'retracement' if level != current_price else 'extension'
                        }
                    )
                    stops.append(stop)
                    
        except Exception as e:
            analysis_logger.error(f"Error calculating Fibonacci stops: {e}")
        
        return stops
    
    def _calculate_pattern_stops(self,
                               unified_result: UnifiedAnalysisResult,
                               current_price: float,
                               signal: TradingSignal) -> List[StopLossLevel]:
        """Calculate pattern-based stops"""
        stops = []
        
        try:
            # Elliott Wave pattern invalidation levels
            if unified_result.elliott_patterns:
                for pattern in unified_result.elliott_patterns:
                    if hasattr(pattern, 'invalidation_level') and pattern.invalidation_level:
                        stop_level = pattern.invalidation_level
                        
                        # Check if this is a relevant stop level
                        if signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
                            if stop_level < current_price:
                                valid_stop = True
                            else:
                                continue
                        else:
                            if stop_level > current_price:
                                valid_stop = True
                            else:
                                continue
                        
                        risk_distance = abs(current_price - stop_level)
                        risk_percent = (risk_distance / current_price) * 100
                        distance_pips = self._calculate_pips(current_price, stop_level, signal.symbol)
                        
                        # Confidence based on pattern confidence
                        confidence = pattern.confidence * 0.8  # Slightly reduced for stops
                        
                        stop = StopLossLevel(
                            method=StopLossType.PATTERN_BASED,
                            level=stop_level,
                            confidence=confidence,
                            risk_percent=risk_percent,
                            distance_pips=distance_pips,
                            rationale=f"Elliott Wave pattern invalidation level",
                            supporting_data={
                                'pattern': {
                                    'wave_type': pattern.wave_type.value if hasattr(pattern, 'wave_type') else 'unknown',
                                    'confidence': pattern.confidence,
                                    'invalidation_level': pattern.invalidation_level
                                }
                            }
                        )
                        stops.append(stop)
            
            # Dow Theory invalidation levels
            if unified_result.dow_trend and hasattr(unified_result.dow_trend, 'invalidation_level'):
                invalidation_level = unified_result.dow_trend.invalidation_level
                if invalidation_level:
                    # Similar logic as Elliott Wave
                    if signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
                        if invalidation_level < current_price:
                            risk_distance = abs(current_price - invalidation_level)
                            risk_percent = (risk_distance / current_price) * 100
                            distance_pips = self._calculate_pips(current_price, invalidation_level, signal.symbol)
                            
                            confidence = 0.7  # Standard confidence for trend invalidation
                            
                            stop = StopLossLevel(
                                method=StopLossType.PATTERN_BASED,
                                level=invalidation_level,
                                confidence=confidence,
                                risk_percent=risk_percent,
                                distance_pips=distance_pips,
                                rationale="Dow Theory trend invalidation level",
                                supporting_data={
                                    'trend_invalidation': invalidation_level,
                                    'current_trend': unified_result.dow_trend.primary_trend.value if hasattr(unified_result.dow_trend, 'primary_trend') else 'unknown'
                                }
                            )
                            stops.append(stop)
                            
        except Exception as e:
            analysis_logger.error(f"Error calculating pattern stops: {e}")
        
        return stops
    
    def _calculate_volatility_stops(self,
                                  price_data: pd.DataFrame,
                                  current_price: float,
                                  signal: TradingSignal) -> List[StopLossLevel]:
        """Calculate volatility-adjusted stops"""
        stops = []
        
        try:
            # Calculate recent volatility
            returns = price_data['close'].pct_change().dropna()
            recent_vol = returns.rolling(self.config['volatility_window']).std().iloc[-1]
            
            if pd.isna(recent_vol):
                return stops
            
            # Volatility-based stop distances
            vol_multipliers = [1.0, 1.5, 2.0]
            
            for multiplier in vol_multipliers:
                vol_distance = recent_vol * multiplier * current_price
                
                if signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
                    stop_level = current_price - vol_distance
                else:
                    stop_level = current_price + vol_distance
                
                risk_distance = abs(current_price - stop_level)
                risk_percent = (risk_distance / current_price) * 100
                distance_pips = self._calculate_pips(current_price, stop_level, signal.symbol)
                
                # Confidence based on volatility appropriateness
                confidence = 0.6 + (0.2 * (2.0 - multiplier))  # 1.5x gets highest confidence
                confidence = max(0.4, min(0.8, confidence))
                
                stop = StopLossLevel(
                    method=StopLossType.VOLATILITY_ADJUSTED,
                    level=stop_level,
                    confidence=confidence,
                    risk_percent=risk_percent,
                    distance_pips=distance_pips,
                    rationale=f"Volatility-based stop ({multiplier}x recent volatility)",
                    supporting_data={
                        'recent_volatility': recent_vol,
                        'multiplier': multiplier,
                        'volatility_window': self.config['volatility_window']
                    }
                )
                stops.append(stop)
                
        except Exception as e:
            analysis_logger.error(f"Error calculating volatility stops: {e}")
        
        return stops
    
    def _calculate_support_resistance_stops(self,
                                          unified_result: UnifiedAnalysisResult,
                                          current_price: float,
                                          signal: TradingSignal) -> List[StopLossLevel]:
        """Calculate support/resistance based stops"""
        stops = []
        
        try:
            # Use swing points as support/resistance levels
            swing_points = unified_result.swing_points
            if not swing_points:
                return stops
            
            # Identify support/resistance levels
            if signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
                # Look for support levels (swing lows) below current price
                support_levels = [
                    sp.price for sp in swing_points 
                    if sp.swing_type.value.lower() == 'low' and sp.price < current_price
                ]
                support_levels = sorted(set(support_levels), reverse=True)  # Highest first
                
                for i, level in enumerate(support_levels[:2]):  # Top 2 support levels
                    # Place stop slightly below support
                    stop_level = level - (current_price * 0.001)  # 0.1% below
                    
                    risk_distance = abs(current_price - stop_level)
                    risk_percent = (risk_distance / current_price) * 100
                    distance_pips = self._calculate_pips(current_price, stop_level, signal.symbol)
                    
                    confidence = 0.6 - (i * 0.1)  # Decrease confidence for lower supports
                    
                    stop = StopLossLevel(
                        method=StopLossType.SUPPORT_RESISTANCE,
                        level=stop_level,
                        confidence=confidence,
                        risk_percent=risk_percent,
                        distance_pips=distance_pips,
                        rationale=f"Below support level at {level:.5f}",
                        supporting_data={
                            'support_level': level,
                            'adjustment': current_price * 0.001
                        }
                    )
                    stops.append(stop)
            else:
                # Look for resistance levels (swing highs) above current price
                resistance_levels = [
                    sp.price for sp in swing_points 
                    if sp.swing_type.value.lower() == 'high' and sp.price > current_price
                ]
                resistance_levels = sorted(set(resistance_levels))  # Lowest first
                
                for i, level in enumerate(resistance_levels[:2]):  # Top 2 resistance levels
                    # Place stop slightly above resistance
                    stop_level = level + (current_price * 0.001)  # 0.1% above
                    
                    risk_distance = abs(current_price - stop_level)
                    risk_percent = (risk_distance / current_price) * 100
                    distance_pips = self._calculate_pips(current_price, stop_level, signal.symbol)
                    
                    confidence = 0.6 - (i * 0.1)  # Decrease confidence for higher resistances
                    
                    stop = StopLossLevel(
                        method=StopLossType.SUPPORT_RESISTANCE,
                        level=stop_level,
                        confidence=confidence,
                        risk_percent=risk_percent,
                        distance_pips=distance_pips,
                        rationale=f"Above resistance level at {level:.5f}",
                        supporting_data={
                            'resistance_level': level,
                            'adjustment': current_price * 0.001
                        }
                    )
                    stops.append(stop)
                    
        except Exception as e:
            analysis_logger.error(f"Error calculating support/resistance stops: {e}")
        
        return stops
    
    def _select_optimal_stop(self, calculations: List[StopLossLevel]) -> Tuple[float, StopLossType, float]:
        """Select the optimal stop loss from all calculations"""
        if not calculations:
            # Fallback to simple percentage stop
            return 0.0, StopLossType.PERCENTAGE, 0.1
        
        # Weight calculations by method confidence and individual confidence
        weighted_scores = []
        
        for calc in calculations:
            method_weight = self.config['confidence_weights'].get(calc.method, 0.1)
            weighted_score = calc.confidence * method_weight
            
            # Bonus for reasonable risk levels (1-3%)
            if 1.0 <= calc.risk_percent <= 3.0:
                weighted_score *= 1.2
            elif calc.risk_percent > 5.0:
                weighted_score *= 0.7  # Penalty for high risk
            
            # Bonus for appropriate distance
            if self.config['min_stop_distance_pips'] <= calc.distance_pips <= self.config['max_stop_distance_pips']:
                weighted_score *= 1.1
            
            weighted_scores.append((weighted_score, calc))
        
        # Sort by weighted score and select best
        weighted_scores.sort(key=lambda x: x[0], reverse=True)
        best_score, best_calc = weighted_scores[0]
        
        return best_calc.level, best_calc.method, best_calc.confidence
    
    def _calculate_secondary_stop(self, calculations: List[StopLossLevel], primary_stop: float) -> Optional[float]:
        """Calculate secondary (wider) stop loss"""
        # Find a calculation that's further away than primary stop
        secondary_candidates = []
        
        for calc in calculations:
            distance_diff = abs(calc.level - primary_stop)
            if distance_diff > 0:  # Different from primary
                secondary_candidates.append((distance_diff, calc.level))
        
        if secondary_candidates:
            # Sort by distance and pick a reasonable secondary
            secondary_candidates.sort(key=lambda x: x[0])
            return secondary_candidates[0][1] if len(secondary_candidates) > 1 else None
        
        return None
    
    def _calculate_trailing_stop(self, primary_stop: float, current_price: float, signal: TradingSignal) -> Optional[float]:
        """Calculate trailing stop activation level"""
        # Set trailing stop to activate when in 50% profit toward first target
        if signal.take_profit_1:
            profit_distance = abs(signal.take_profit_1 - current_price)
            half_profit = profit_distance * 0.5
            
            if signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
                return current_price + half_profit
            else:
                return current_price - half_profit
        
        return None
    
    def _calculate_risk_metrics(self,
                              stop_loss: float,
                              current_price: float,
                              signal: TradingSignal,
                              account_balance: float) -> Dict[str, float]:
        """Calculate risk metrics for the stop loss"""
        risk_distance = abs(current_price - stop_loss)
        risk_percent = (risk_distance / current_price) * 100
        
        # Calculate maximum loss amount
        position_size_pct = signal.position_size_pct or 0.02
        position_value = account_balance * position_size_pct
        max_loss_amount = position_value * (risk_distance / current_price)
        
        # Calculate risk/reward ratio
        risk_reward_ratio = 0.0
        if signal.take_profit_1:
            reward_distance = abs(signal.take_profit_1 - current_price)
            risk_reward_ratio = reward_distance / risk_distance if risk_distance > 0 else 0.0
        
        return {
            'risk_percent': risk_percent,
            'risk_reward_ratio': risk_reward_ratio,
            'max_loss_amount': max_loss_amount,
            'risk_distance': risk_distance
        }
    
    def _calculate_pips(self, price1: float, price2: float, symbol: str) -> float:
        """Calculate distance in pips between two prices"""
        # Simplified pip calculation (would need more sophisticated logic for different pairs)
        if 'JPY' in symbol:
            pip_value = 0.01
        else:
            pip_value = 0.0001
        
        return abs(price1 - price2) / pip_value
    
    def _create_default_stop_loss(self, signal: TradingSignal, current_price: float, account_balance: float) -> StopLossRecommendation:
        """Create default stop loss for error cases"""
        # Simple 2% stop loss
        risk_distance = current_price * 0.02
        
        if signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
            primary_stop = current_price - risk_distance
        else:
            primary_stop = current_price + risk_distance
        
        risk_metrics = self._calculate_risk_metrics(primary_stop, current_price, signal, account_balance)
        
        return StopLossRecommendation(
            primary_stop=primary_stop,
            secondary_stop=None,
            initial_stop=primary_stop,
            trailing_stop=None,
            risk_percent=2.0,
            risk_reward_ratio=risk_metrics['risk_reward_ratio'],
            max_loss_amount=risk_metrics['max_loss_amount'],
            calculations=[],
            method_used=StopLossType.PERCENTAGE,
            confidence=0.5,
            calculation_timestamp=datetime.utcnow(),
            valid_for_hours=4
        )
    
    def get_stop_loss_summary(self, recommendation: StopLossRecommendation) -> Dict[str, Any]:
        """Get detailed summary of stop loss calculation"""
        return {
            'primary_stop': recommendation.primary_stop,
            'method_used': recommendation.method_used.value,
            'confidence': recommendation.confidence,
            'risk_metrics': {
                'risk_percent': recommendation.risk_percent,
                'risk_reward_ratio': recommendation.risk_reward_ratio,
                'max_loss_amount': recommendation.max_loss_amount
            },
            'alternative_stops': {
                'secondary_stop': recommendation.secondary_stop,
                'trailing_stop': recommendation.trailing_stop
            },
            'calculation_breakdown': [
                {
                    'method': calc.method.value,
                    'level': calc.level,
                    'confidence': calc.confidence,
                    'risk_percent': calc.risk_percent,
                    'rationale': calc.rationale
                }
                for calc in recommendation.calculations
            ],
            'metadata': {
                'calculation_timestamp': recommendation.calculation_timestamp.isoformat(),
                'valid_for_hours': recommendation.valid_for_hours
            }
        }


# Create default stop loss calculator instance
stop_loss_calculator = StopLossCalculator()