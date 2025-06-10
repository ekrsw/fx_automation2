"""
Take Profit Calculation Engine

Advanced take profit calculation system using multiple methods including
Fibonacci targets, Elliott Wave projections, support/resistance levels,
and pattern-based targets.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from app.analysis.strategy_engine.unified_analyzer import UnifiedAnalysisResult
from app.analysis.strategy_engine.signal_generator import TradingSignal, SignalType
from app.utils.logger import analysis_logger


class TakeProfitType(Enum):
    """Types of take profit calculation methods"""
    FIBONACCI_EXTENSION = "fibonacci_extension"
    FIBONACCI_RETRACEMENT = "fibonacci_retracement"
    ELLIOTT_WAVE_TARGET = "elliott_wave_target"
    PATTERN_TARGET = "pattern_target"
    SUPPORT_RESISTANCE = "support_resistance"
    ATR_MULTIPLE = "atr_multiple"
    PERCENTAGE_TARGET = "percentage_target"
    SWING_PROJECTION = "swing_projection"


@dataclass
class TakeProfitLevel:
    """Individual take profit level calculation"""
    method: TakeProfitType
    level: float
    confidence: float  # 0.0 to 1.0
    probability: float  # Probability of reaching this level
    distance_pips: float  # Distance in pips from entry
    reward_risk_ratio: float  # Reward to risk ratio
    rationale: str
    supporting_data: Dict[str, Any]


@dataclass
class TakeProfitRecommendation:
    """Complete take profit recommendation"""
    primary_targets: List[float]  # Main take profit levels (1-3)
    secondary_targets: List[float]  # Additional targets for scaling out
    final_target: Optional[float]  # Ultimate target if trend continues
    
    # Profit taking strategy
    partial_exit_levels: Dict[float, float]  # Level -> percentage to close
    trailing_profit_activation: Optional[float]  # When to start trailing profits
    
    # Risk metrics
    average_reward_risk: float
    total_profit_potential: float
    success_probability: float
    
    # Analysis breakdown
    calculations: List[TakeProfitLevel]
    methods_used: List[TakeProfitType]
    confidence: float
    
    # Metadata
    calculation_timestamp: datetime
    valid_for_hours: int


class TakeProfitCalculator:
    """
    Advanced take profit calculation engine
    
    Calculates optimal take profit levels using multiple methods and
    combines them to provide comprehensive profit-taking strategies.
    """
    
    def __init__(self,
                 default_profit_ratios: List[float] = None,
                 enable_multiple_targets: bool = True,
                 max_targets: int = 3):
        """
        Initialize take profit calculator
        
        Args:
            default_profit_ratios: Default profit taking ratios
            enable_multiple_targets: Whether to calculate multiple targets
            max_targets: Maximum number of take profit targets
        """
        self.default_profit_ratios = default_profit_ratios or [1.5, 2.5, 4.0]
        self.enable_multiple_targets = enable_multiple_targets
        self.max_targets = max_targets
        
        # Calculation configuration
        self.config = {
            'fibonacci_levels': [1.272, 1.414, 1.618, 2.000, 2.618],  # Extension levels
            'retracement_levels': [0.236, 0.382, 0.500, 0.618, 0.786],  # Retracement levels
            'atr_multiples': [2.0, 3.0, 4.0, 5.0],  # ATR-based targets
            'percentage_targets': [1.0, 2.0, 3.0, 5.0],  # Percentage targets
            'min_reward_risk_ratio': 1.5,  # Minimum acceptable R:R
            'max_distance_multiplier': 10.0,  # Maximum distance as multiple of stop
            'probability_weights': {
                TakeProfitType.FIBONACCI_EXTENSION: 0.25,
                TakeProfitType.ELLIOTT_WAVE_TARGET: 0.25,
                TakeProfitType.SUPPORT_RESISTANCE: 0.20,
                TakeProfitType.PATTERN_TARGET: 0.15,
                TakeProfitType.ATR_MULTIPLE: 0.10,
                TakeProfitType.PERCENTAGE_TARGET: 0.05
            },
            'partial_exit_strategy': {
                'first_target': 0.33,   # Close 33% at first target
                'second_target': 0.50,  # Close 50% of remaining at second target
                'third_target': 0.75    # Close 75% of remaining at third target
            }
        }
    
    def calculate_take_profit(self,
                            signal: TradingSignal,
                            unified_result: UnifiedAnalysisResult,
                            price_data: pd.DataFrame,
                            stop_loss: Optional[float] = None) -> TakeProfitRecommendation:
        """
        Calculate comprehensive take profit recommendation
        
        Args:
            signal: Trading signal
            unified_result: Unified analysis result
            price_data: Historical price data
            stop_loss: Stop loss level for R:R calculation
            
        Returns:
            Complete take profit recommendation
        """
        try:
            calculations = []
            current_price = signal.entry_price or price_data['close'].iloc[-1]
            
            # Method 1: Fibonacci extension targets
            fib_ext_targets = self._calculate_fibonacci_extensions(
                unified_result, current_price, signal
            )
            calculations.extend(fib_ext_targets)
            
            # Method 2: Elliott Wave targets
            elliott_targets = self._calculate_elliott_wave_targets(
                unified_result, current_price, signal
            )
            calculations.extend(elliott_targets)
            
            # Method 3: Pattern-based targets
            pattern_targets = self._calculate_pattern_targets(
                unified_result, current_price, signal
            )
            calculations.extend(pattern_targets)
            
            # Method 4: Support/Resistance targets
            sr_targets = self._calculate_support_resistance_targets(
                unified_result, current_price, signal
            )
            calculations.extend(sr_targets)
            
            # Method 5: ATR-based targets
            atr_targets = self._calculate_atr_targets(
                price_data, current_price, signal, stop_loss
            )
            calculations.extend(atr_targets)
            
            # Method 6: Swing projection targets
            swing_targets = self._calculate_swing_projection_targets(
                unified_result, current_price, signal
            )
            calculations.extend(swing_targets)
            
            # Select optimal targets
            primary_targets = self._select_optimal_targets(calculations, stop_loss, current_price)
            
            # Calculate secondary targets and strategy
            secondary_targets = self._calculate_secondary_targets(calculations, primary_targets)
            final_target = self._calculate_final_target(calculations, primary_targets)
            
            # Create profit-taking strategy
            partial_exit_levels = self._create_partial_exit_strategy(primary_targets)
            trailing_activation = self._calculate_trailing_activation(primary_targets, current_price)
            
            # Calculate overall metrics
            metrics = self._calculate_overall_metrics(
                primary_targets, calculations, stop_loss, current_price
            )
            
            recommendation = TakeProfitRecommendation(
                primary_targets=primary_targets,
                secondary_targets=secondary_targets,
                final_target=final_target,
                partial_exit_levels=partial_exit_levels,
                trailing_profit_activation=trailing_activation,
                average_reward_risk=metrics['average_reward_risk'],
                total_profit_potential=metrics['total_profit_potential'],
                success_probability=metrics['success_probability'],
                calculations=calculations,
                methods_used=list(set(calc.method for calc in calculations)),
                confidence=metrics['confidence'],
                calculation_timestamp=datetime.utcnow(),
                valid_for_hours=8
            )
            
            analysis_logger.info(
                f"Calculated take profit: {signal.symbol} - "
                f"Targets: {len(primary_targets)}, "
                f"Avg R:R: {metrics['average_reward_risk']:.2f}, "
                f"Confidence: {metrics['confidence']:.2f}"
            )
            
            return recommendation
            
        except Exception as e:
            analysis_logger.error(f"Error calculating take profit: {e}")
            return self._create_default_take_profit(signal, current_price, stop_loss)
    
    def _calculate_fibonacci_extensions(self,
                                      unified_result: UnifiedAnalysisResult,
                                      current_price: float,
                                      signal: TradingSignal) -> List[TakeProfitLevel]:
        """Calculate Fibonacci extension targets"""
        targets = []
        
        try:
            # Look for Fibonacci levels in price targets
            if hasattr(unified_result, 'price_targets') and unified_result.price_targets:
                for key, level in unified_result.price_targets.items():
                    if level and 'fib' in key.lower():
                        # Determine if this is a valid target for our signal direction
                        if signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
                            if level > current_price:
                                valid_target = True
                            else:
                                continue
                        else:
                            if level < current_price:
                                valid_target = True
                            else:
                                continue
                        
                        distance_pips = self._calculate_pips(current_price, level, signal.symbol)
                        
                        # Determine which Fibonacci level this represents
                        fib_ratio = self._identify_fibonacci_ratio(level, current_price, unified_result)
                        
                        # Confidence based on Fibonacci ratio strength
                        if fib_ratio in [1.618, 2.618]:  # Golden ratio levels
                            confidence = 0.8
                        elif fib_ratio in [1.272, 2.000]:  # Common levels
                            confidence = 0.7
                        else:
                            confidence = 0.6
                        
                        # Probability decreases with distance
                        distance_factor = min(1.0, 100.0 / distance_pips) if distance_pips > 0 else 0.5
                        probability = confidence * distance_factor * 0.8
                        
                        target = TakeProfitLevel(
                            method=TakeProfitType.FIBONACCI_EXTENSION,
                            level=level,
                            confidence=confidence,
                            probability=probability,
                            distance_pips=distance_pips,
                            reward_risk_ratio=0.0,  # Will be calculated later
                            rationale=f"Fibonacci {fib_ratio} extension at {level:.5f}",
                            supporting_data={
                                'fibonacci_ratio': fib_ratio,
                                'level_type': 'extension',
                                'original_key': key
                            }
                        )
                        targets.append(target)
            
            # Calculate additional Fibonacci extensions based on recent swings
            if unified_result.swing_points and len(unified_result.swing_points) >= 3:
                recent_swings = unified_result.swing_points[-3:]
                swing_range = abs(recent_swings[-1].price - recent_swings[-3].price)
                
                for fib_level in self.config['fibonacci_levels']:
                    if signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
                        target_level = current_price + (swing_range * fib_level)
                    else:
                        target_level = current_price - (swing_range * fib_level)
                    
                    distance_pips = self._calculate_pips(current_price, target_level, signal.symbol)
                    
                    # Confidence based on how standard the Fibonacci level is
                    if fib_level == 1.618:
                        confidence = 0.8
                    elif fib_level in [1.272, 2.618]:
                        confidence = 0.7
                    else:
                        confidence = 0.6
                    
                    probability = confidence * 0.7  # Base probability for calculated levels
                    
                    target = TakeProfitLevel(
                        method=TakeProfitType.FIBONACCI_EXTENSION,
                        level=target_level,
                        confidence=confidence,
                        probability=probability,
                        distance_pips=distance_pips,
                        reward_risk_ratio=0.0,
                        rationale=f"Calculated Fibonacci {fib_level} extension",
                        supporting_data={
                            'fibonacci_ratio': fib_level,
                            'swing_range': swing_range,
                            'calculation_method': 'swing_projection'
                        }
                    )
                    targets.append(target)
                    
        except Exception as e:
            analysis_logger.error(f"Error calculating Fibonacci extensions: {e}")
        
        return targets
    
    def _calculate_elliott_wave_targets(self,
                                      unified_result: UnifiedAnalysisResult,
                                      current_price: float,
                                      signal: TradingSignal) -> List[TakeProfitLevel]:
        """Calculate Elliott Wave projection targets"""
        targets = []
        
        try:
            # Use Elliott Wave predictions
            if unified_result.elliott_predictions:
                for prediction in unified_result.elliott_predictions:
                    scenarios = prediction.get('scenarios', [])
                    
                    for scenario in scenarios:
                        target_price = scenario.get('target_price')
                        scenario_type = scenario.get('type', '')
                        scenario_confidence = scenario.get('confidence', 0.5)
                        
                        if target_price:
                            # Check if target is in correct direction
                            if signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
                                if target_price > current_price:
                                    valid_target = True
                                else:
                                    continue
                            else:
                                if target_price < current_price:
                                    valid_target = True
                                else:
                                    continue
                            
                            distance_pips = self._calculate_pips(current_price, target_price, signal.symbol)
                            
                            # Confidence based on scenario type and wave confidence
                            if 'impulse' in scenario_type.lower():
                                confidence = scenario_confidence * 0.9
                            elif 'corrective' in scenario_type.lower():
                                confidence = scenario_confidence * 0.7
                            else:
                                confidence = scenario_confidence * 0.8
                            
                            # Probability based on Elliott Wave theory
                            if 'wave_5' in scenario_type.lower() or 'wave_c' in scenario_type.lower():
                                probability = confidence * 0.8  # Terminal waves have good probability
                            else:
                                probability = confidence * 0.7
                            
                            target = TakeProfitLevel(
                                method=TakeProfitType.ELLIOTT_WAVE_TARGET,
                                level=target_price,
                                confidence=confidence,
                                probability=probability,
                                distance_pips=distance_pips,
                                reward_risk_ratio=0.0,
                                rationale=f"Elliott Wave {scenario_type} target",
                                supporting_data={
                                    'scenario_type': scenario_type,
                                    'wave_degree': scenario.get('degree', 'unknown'),
                                    'elliott_confidence': scenario_confidence,
                                    'prediction_id': prediction.get('id', 'unknown')
                                }
                            )
                            targets.append(target)
            
            # Use Elliott Wave patterns for additional targets
            if unified_result.elliott_patterns:
                for pattern in unified_result.elliott_patterns:
                    if hasattr(pattern, 'target_levels') and pattern.target_levels:
                        for target_level in pattern.target_levels:
                            # Similar validation as above
                            if signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
                                if target_level > current_price:
                                    distance_pips = self._calculate_pips(current_price, target_level, signal.symbol)
                                    confidence = pattern.confidence * 0.8
                                    probability = confidence * 0.75
                                    
                                    target = TakeProfitLevel(
                                        method=TakeProfitType.ELLIOTT_WAVE_TARGET,
                                        level=target_level,
                                        confidence=confidence,
                                        probability=probability,
                                        distance_pips=distance_pips,
                                        reward_risk_ratio=0.0,
                                        rationale=f"Elliott Wave pattern target",
                                        supporting_data={
                                            'pattern_type': pattern.wave_type.value if hasattr(pattern, 'wave_type') else 'unknown',
                                            'pattern_confidence': pattern.confidence
                                        }
                                    )
                                    targets.append(target)
                                    
        except Exception as e:
            analysis_logger.error(f"Error calculating Elliott Wave targets: {e}")
        
        return targets
    
    def _calculate_pattern_targets(self,
                                 unified_result: UnifiedAnalysisResult,
                                 current_price: float,
                                 signal: TradingSignal) -> List[TakeProfitLevel]:
        """Calculate pattern-based targets"""
        targets = []
        
        try:
            # Use Dow Theory measured moves
            if unified_result.dow_trend and hasattr(unified_result.dow_trend, 'measured_move_targets'):
                measured_targets = unified_result.dow_trend.measured_move_targets
                if measured_targets:
                    for target in measured_targets:
                        if signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
                            if target > current_price:
                                distance_pips = self._calculate_pips(current_price, target, signal.symbol)
                                confidence = 0.7  # Standard confidence for measured moves
                                probability = 0.6
                                
                                target_obj = TakeProfitLevel(
                                    method=TakeProfitType.PATTERN_TARGET,
                                    level=target,
                                    confidence=confidence,
                                    probability=probability,
                                    distance_pips=distance_pips,
                                    reward_risk_ratio=0.0,
                                    rationale="Dow Theory measured move target",
                                    supporting_data={
                                        'pattern_type': 'measured_move',
                                        'trend_direction': unified_result.dow_trend.primary_trend.value if hasattr(unified_result.dow_trend, 'primary_trend') else 'unknown'
                                    }
                                )
                                targets.append(target_obj)
            
            # Calculate pattern targets based on recent price action
            if unified_result.swing_points and len(unified_result.swing_points) >= 4:
                # Find recent consolidation patterns
                recent_swings = unified_result.swing_points[-4:]
                
                # Look for rectangle/channel patterns
                highs = [sp.price for sp in recent_swings if sp.swing_type.value.lower() == 'high']
                lows = [sp.price for sp in recent_swings if sp.swing_type.value.lower() == 'low']
                
                if len(highs) >= 2 and len(lows) >= 2:
                    avg_high = np.mean(highs)
                    avg_low = np.mean(lows)
                    pattern_height = avg_high - avg_low
                    
                    # Project pattern height in direction of signal
                    if signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
                        target_level = avg_high + pattern_height
                    else:
                        target_level = avg_low - pattern_height
                    
                    distance_pips = self._calculate_pips(current_price, target_level, signal.symbol)
                    confidence = 0.6  # Moderate confidence for pattern projection
                    probability = 0.5
                    
                    target = TakeProfitLevel(
                        method=TakeProfitType.PATTERN_TARGET,
                        level=target_level,
                        confidence=confidence,
                        probability=probability,
                        distance_pips=distance_pips,
                        reward_risk_ratio=0.0,
                        rationale="Rectangle pattern projection",
                        supporting_data={
                            'pattern_type': 'rectangle',
                            'pattern_height': pattern_height,
                            'avg_high': avg_high,
                            'avg_low': avg_low
                        }
                    )
                    targets.append(target)
                    
        except Exception as e:
            analysis_logger.error(f"Error calculating pattern targets: {e}")
        
        return targets
    
    def _calculate_support_resistance_targets(self,
                                            unified_result: UnifiedAnalysisResult,
                                            current_price: float,
                                            signal: TradingSignal) -> List[TakeProfitLevel]:
        """Calculate support/resistance level targets"""
        targets = []
        
        try:
            swing_points = unified_result.swing_points
            if not swing_points:
                return targets
            
            # Find resistance levels for buy signals, support levels for sell signals
            if signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
                # Look for resistance levels (swing highs) above current price
                resistance_levels = [
                    sp.price for sp in swing_points 
                    if sp.swing_type.value.lower() == 'high' and sp.price > current_price
                ]
                resistance_levels = sorted(set(resistance_levels))  # Remove duplicates and sort
                
                # Take closest 3 resistance levels
                for i, level in enumerate(resistance_levels[:3]):
                    distance_pips = self._calculate_pips(current_price, level, signal.symbol)
                    
                    # Confidence decreases with distance and order
                    base_confidence = 0.7 - (i * 0.1)
                    distance_factor = max(0.3, 1.0 - (distance_pips / 200.0))  # Penalty for very distant levels
                    confidence = base_confidence * distance_factor
                    
                    # Probability based on how strong the resistance level is
                    # (This would ideally consider how many times price has reacted to this level)
                    probability = confidence * 0.6
                    
                    target = TakeProfitLevel(
                        method=TakeProfitType.SUPPORT_RESISTANCE,
                        level=level,
                        confidence=confidence,
                        probability=probability,
                        distance_pips=distance_pips,
                        reward_risk_ratio=0.0,
                        rationale=f"Resistance level at {level:.5f}",
                        supporting_data={
                            'level_type': 'resistance',
                            'level_strength': 1.0,  # Would be calculated based on historical reactions
                            'order': i + 1
                        }
                    )
                    targets.append(target)
            else:
                # Look for support levels (swing lows) below current price
                support_levels = [
                    sp.price for sp in swing_points 
                    if sp.swing_type.value.lower() == 'low' and sp.price < current_price
                ]
                support_levels = sorted(set(support_levels), reverse=True)  # Highest first
                
                # Take closest 3 support levels
                for i, level in enumerate(support_levels[:3]):
                    distance_pips = self._calculate_pips(current_price, level, signal.symbol)
                    
                    base_confidence = 0.7 - (i * 0.1)
                    distance_factor = max(0.3, 1.0 - (distance_pips / 200.0))
                    confidence = base_confidence * distance_factor
                    probability = confidence * 0.6
                    
                    target = TakeProfitLevel(
                        method=TakeProfitType.SUPPORT_RESISTANCE,
                        level=level,
                        confidence=confidence,
                        probability=probability,
                        distance_pips=distance_pips,
                        reward_risk_ratio=0.0,
                        rationale=f"Support level at {level:.5f}",
                        supporting_data={
                            'level_type': 'support',
                            'level_strength': 1.0,
                            'order': i + 1
                        }
                    )
                    targets.append(target)
                    
        except Exception as e:
            analysis_logger.error(f"Error calculating support/resistance targets: {e}")
        
        return targets
    
    def _calculate_atr_targets(self,
                             price_data: pd.DataFrame,
                             current_price: float,
                             signal: TradingSignal,
                             stop_loss: Optional[float]) -> List[TakeProfitLevel]:
        """Calculate ATR-based targets"""
        targets = []
        
        try:
            # Calculate ATR (simplified - using true range)
            if len(price_data) < 14:
                return targets
            
            high = price_data['high']
            low = price_data['low']
            close = price_data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]
            
            if pd.isna(atr):
                return targets
            
            # Calculate targets for different ATR multiples
            for multiplier in self.config['atr_multiples']:
                if signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
                    target_level = current_price + (atr * multiplier)
                else:
                    target_level = current_price - (atr * multiplier)
                
                distance_pips = self._calculate_pips(current_price, target_level, signal.symbol)
                
                # Confidence based on ATR multiple (2-3x ATR is typically optimal)
                if 2.0 <= multiplier <= 3.0:
                    confidence = 0.7
                elif 1.5 <= multiplier <= 4.0:
                    confidence = 0.6
                else:
                    confidence = 0.5
                
                # Probability decreases with higher multiples
                probability = confidence * (1.0 - (multiplier - 2.0) * 0.1)
                probability = max(0.2, probability)
                
                target = TakeProfitLevel(
                    method=TakeProfitType.ATR_MULTIPLE,
                    level=target_level,
                    confidence=confidence,
                    probability=probability,
                    distance_pips=distance_pips,
                    reward_risk_ratio=0.0,
                    rationale=f"ATR({multiplier}x) target",
                    supporting_data={
                        'atr_value': atr,
                        'atr_multiplier': multiplier,
                        'atr_period': 14
                    }
                )
                targets.append(target)
                
        except Exception as e:
            analysis_logger.error(f"Error calculating ATR targets: {e}")
        
        return targets
    
    def _calculate_swing_projection_targets(self,
                                          unified_result: UnifiedAnalysisResult,
                                          current_price: float,
                                          signal: TradingSignal) -> List[TakeProfitLevel]:
        """Calculate swing projection targets"""
        targets = []
        
        try:
            swing_points = unified_result.swing_points
            if len(swing_points) < 3:
                return targets
            
            # Use last 3 swing points to project next move
            recent_swings = swing_points[-3:]
            
            # Calculate the last complete swing move
            if len(recent_swings) >= 2:
                last_swing_distance = abs(recent_swings[-1].price - recent_swings[-2].price)
                
                # Project equal distance (100% projection)
                if signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
                    equal_projection = current_price + last_swing_distance
                else:
                    equal_projection = current_price - last_swing_distance
                
                distance_pips = self._calculate_pips(current_price, equal_projection, signal.symbol)
                
                target = TakeProfitLevel(
                    method=TakeProfitType.SWING_PROJECTION,
                    level=equal_projection,
                    confidence=0.6,
                    probability=0.5,
                    distance_pips=distance_pips,
                    reward_risk_ratio=0.0,
                    rationale="Equal swing projection (100%)",
                    supporting_data={
                        'swing_distance': last_swing_distance,
                        'projection_ratio': 1.0,
                        'base_swing': f"{recent_swings[-2].price:.5f} to {recent_swings[-1].price:.5f}"
                    }
                )
                targets.append(target)
                
                # Also calculate 1.618 extension
                ext_projection = current_price + (last_swing_distance * 1.618) if signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY] else current_price - (last_swing_distance * 1.618)
                
                distance_pips = self._calculate_pips(current_price, ext_projection, signal.symbol)
                
                target = TakeProfitLevel(
                    method=TakeProfitType.SWING_PROJECTION,
                    level=ext_projection,
                    confidence=0.5,
                    probability=0.4,
                    distance_pips=distance_pips,
                    reward_risk_ratio=0.0,
                    rationale="Swing extension (161.8%)",
                    supporting_data={
                        'swing_distance': last_swing_distance,
                        'projection_ratio': 1.618,
                        'base_swing': f"{recent_swings[-2].price:.5f} to {recent_swings[-1].price:.5f}"
                    }
                )
                targets.append(target)
                
        except Exception as e:
            analysis_logger.error(f"Error calculating swing projection targets: {e}")
        
        return targets
    
    def _select_optimal_targets(self,
                              calculations: List[TakeProfitLevel],
                              stop_loss: Optional[float],
                              current_price: float) -> List[float]:
        """Select optimal take profit targets from all calculations"""
        if not calculations:
            return []
        
        # Filter out targets that don't meet minimum criteria
        valid_targets = []
        
        for calc in calculations:
            # Calculate reward/risk ratio if stop loss is provided
            if stop_loss:
                risk_distance = abs(current_price - stop_loss)
                reward_distance = abs(calc.level - current_price)
                calc.reward_risk_ratio = reward_distance / risk_distance if risk_distance > 0 else 0.0
                
                # Filter by minimum R:R ratio
                if calc.reward_risk_ratio < self.config['min_reward_risk_ratio']:
                    continue
            
            # Filter by maximum distance
            if calc.distance_pips > 500:  # Arbitrary max distance
                continue
            
            valid_targets.append(calc)
        
        if not valid_targets:
            return []
        
        # Score targets based on multiple factors
        scored_targets = []
        
        for target in valid_targets:
            # Base score from confidence and probability
            base_score = target.confidence * 0.6 + target.probability * 0.4
            
            # Bonus for good R:R ratios
            if target.reward_risk_ratio >= 3.0:
                rr_bonus = 0.2
            elif target.reward_risk_ratio >= 2.0:
                rr_bonus = 0.1
            else:
                rr_bonus = 0.0
            
            # Bonus for preferred methods
            method_bonus = self.config['probability_weights'].get(target.method, 0.0)
            
            # Distance penalty (prefer closer targets for first levels)
            distance_penalty = min(0.1, target.distance_pips / 1000.0)
            
            final_score = base_score + rr_bonus + method_bonus - distance_penalty
            scored_targets.append((final_score, target))
        
        # Sort by score and group by proximity
        scored_targets.sort(key=lambda x: x[0], reverse=True)
        
        # Select diverse targets (avoid clustering)
        selected_targets = []
        for score, target in scored_targets:
            # Check if this target is too close to existing selections
            too_close = False
            min_separation = current_price * 0.005  # 0.5% minimum separation
            
            for existing_target in selected_targets:
                if abs(target.level - existing_target) < min_separation:
                    too_close = True
                    break
            
            if not too_close:
                selected_targets.append(target.level)
                
                if len(selected_targets) >= self.max_targets:
                    break
        
        return selected_targets
    
    def _calculate_secondary_targets(self,
                                   calculations: List[TakeProfitLevel],
                                   primary_targets: List[float]) -> List[float]:
        """Calculate secondary targets for additional profit taking"""
        secondary = []
        
        # Find targets that weren't selected as primary but are still good
        for calc in calculations:
            if calc.level not in primary_targets and calc.confidence > 0.4:
                secondary.append(calc.level)
        
        # Limit to reasonable number
        return sorted(secondary)[:2]
    
    def _calculate_final_target(self,
                              calculations: List[TakeProfitLevel],
                              primary_targets: List[float]) -> Optional[float]:
        """Calculate ultimate target if trend continues strongly"""
        # Find the highest confidence long-term target
        long_term_targets = [
            calc for calc in calculations 
            if calc.method in [TakeProfitType.ELLIOTT_WAVE_TARGET, TakeProfitType.FIBONACCI_EXTENSION]
            and calc.distance_pips > 100  # Far targets only
        ]
        
        if long_term_targets:
            best_target = max(long_term_targets, key=lambda x: x.confidence)
            return best_target.level
        
        return None
    
    def _create_partial_exit_strategy(self, targets: List[float]) -> Dict[float, float]:
        """Create partial exit strategy for targets"""
        strategy = {}
        
        if len(targets) >= 1:
            strategy[targets[0]] = self.config['partial_exit_strategy']['first_target']
        if len(targets) >= 2:
            strategy[targets[1]] = self.config['partial_exit_strategy']['second_target']
        if len(targets) >= 3:
            strategy[targets[2]] = self.config['partial_exit_strategy']['third_target']
        
        return strategy
    
    def _calculate_trailing_activation(self, targets: List[float], current_price: float) -> Optional[float]:
        """Calculate when to activate trailing profit system"""
        if targets:
            # Activate trailing after reaching 50% of first target
            first_target_distance = abs(targets[0] - current_price)
            return current_price + (first_target_distance * 0.5)
        
        return None
    
    def _calculate_overall_metrics(self,
                                 targets: List[float],
                                 calculations: List[TakeProfitLevel],
                                 stop_loss: Optional[float],
                                 current_price: float) -> Dict[str, float]:
        """Calculate overall metrics for the recommendation"""
        if not targets:
            return {'average_reward_risk': 0.0, 'total_profit_potential': 0.0, 'success_probability': 0.0, 'confidence': 0.0}
        
        # Calculate average reward/risk ratio
        rr_ratios = []
        if stop_loss:
            risk_distance = abs(current_price - stop_loss)
            for target in targets:
                reward_distance = abs(target - current_price)
                rr_ratio = reward_distance / risk_distance if risk_distance > 0 else 0.0
                rr_ratios.append(rr_ratio)
        
        avg_rr = np.mean(rr_ratios) if rr_ratios else 0.0
        
        # Calculate total profit potential (to furthest target)
        max_profit = max([abs(t - current_price) for t in targets]) if targets else 0.0
        total_profit_potential = (max_profit / current_price) * 100  # As percentage
        
        # Calculate average success probability
        target_calcs = [calc for calc in calculations if calc.level in targets]
        avg_probability = np.mean([calc.probability for calc in target_calcs]) if target_calcs else 0.0
        
        # Calculate overall confidence
        avg_confidence = np.mean([calc.confidence for calc in target_calcs]) if target_calcs else 0.0
        
        return {
            'average_reward_risk': avg_rr,
            'total_profit_potential': total_profit_potential,
            'success_probability': avg_probability,
            'confidence': avg_confidence
        }
    
    def _identify_fibonacci_ratio(self, level: float, current_price: float, unified_result: UnifiedAnalysisResult) -> float:
        """Identify which Fibonacci ratio a level represents"""
        # This is a simplified version - would need more sophisticated logic
        # to determine exact Fibonacci relationships
        return 1.618  # Default to golden ratio
    
    def _calculate_pips(self, price1: float, price2: float, symbol: str) -> float:
        """Calculate distance in pips between two prices"""
        # Simplified pip calculation
        if 'JPY' in symbol:
            pip_value = 0.01
        else:
            pip_value = 0.0001
        
        return abs(price1 - price2) / pip_value
    
    def _create_default_take_profit(self,
                                  signal: TradingSignal,
                                  current_price: float,
                                  stop_loss: Optional[float]) -> TakeProfitRecommendation:
        """Create default take profit for error cases"""
        # Simple ratio-based targets
        targets = []
        risk_distance = abs(current_price - stop_loss) if stop_loss else current_price * 0.02
        
        for ratio in self.default_profit_ratios:
            if signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
                target = current_price + (risk_distance * ratio)
            else:
                target = current_price - (risk_distance * ratio)
            targets.append(target)
        
        return TakeProfitRecommendation(
            primary_targets=targets,
            secondary_targets=[],
            final_target=None,
            partial_exit_levels={targets[0]: 0.5} if targets else {},
            trailing_profit_activation=None,
            average_reward_risk=2.0,
            total_profit_potential=5.0,
            success_probability=0.5,
            calculations=[],
            methods_used=[TakeProfitType.PERCENTAGE_TARGET],
            confidence=0.5,
            calculation_timestamp=datetime.utcnow(),
            valid_for_hours=8
        )
    
    def get_take_profit_summary(self, recommendation: TakeProfitRecommendation) -> Dict[str, Any]:
        """Get detailed summary of take profit calculation"""
        return {
            'primary_targets': recommendation.primary_targets,
            'target_count': len(recommendation.primary_targets),
            'metrics': {
                'average_reward_risk': recommendation.average_reward_risk,
                'total_profit_potential': recommendation.total_profit_potential,
                'success_probability': recommendation.success_probability,
                'confidence': recommendation.confidence
            },
            'strategy': {
                'partial_exit_levels': recommendation.partial_exit_levels,
                'trailing_activation': recommendation.trailing_profit_activation,
                'secondary_targets': recommendation.secondary_targets,
                'final_target': recommendation.final_target
            },
            'calculation_methods': [method.value for method in recommendation.methods_used],
            'calculation_breakdown': [
                {
                    'method': calc.method.value,
                    'level': calc.level,
                    'confidence': calc.confidence,
                    'probability': calc.probability,
                    'reward_risk_ratio': calc.reward_risk_ratio,
                    'rationale': calc.rationale
                }
                for calc in recommendation.calculations
            ],
            'metadata': {
                'calculation_timestamp': recommendation.calculation_timestamp.isoformat(),
                'valid_for_hours': recommendation.valid_for_hours
            }
        }


# Create default take profit calculator instance
take_profit_calculator = TakeProfitCalculator()