"""
Entry/Exit Decision Engine

Advanced entry and exit decision system with comprehensive condition analysis.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from app.analysis.strategy_engine.unified_analyzer import UnifiedAnalysisResult
from app.analysis.strategy_engine.multi_timeframe_analyzer import MultiTimeframeResult
from app.analysis.strategy_engine.signal_generator import TradingSignal, SignalType, SignalUrgency
from app.analysis.strategy_engine.confidence_calculator import ConfidenceScore
from app.utils.logger import analysis_logger


class EntryConditionType(Enum):
    """Types of entry conditions"""
    TREND_ALIGNMENT = "trend_alignment"
    PATTERN_COMPLETION = "pattern_completion"
    FIBONACCI_CONFLUENCE = "fibonacci_confluence"
    MOMENTUM_CONFIRMATION = "momentum_confirmation"
    VOLUME_CONFIRMATION = "volume_confirmation"
    RISK_REWARD_ACCEPTABLE = "risk_reward_acceptable"
    MARKET_STRUCTURE = "market_structure"
    TIMEFRAME_CONFIRMATION = "timeframe_confirmation"


class ExitConditionType(Enum):
    """Types of exit conditions"""
    TAKE_PROFIT_HIT = "take_profit_hit"
    STOP_LOSS_HIT = "stop_loss_hit"
    PATTERN_INVALIDATION = "pattern_invalidation"
    TREND_REVERSAL = "trend_reversal"
    MOMENTUM_DIVERGENCE = "momentum_divergence"
    TIME_BASED_EXIT = "time_based_exit"
    PROFIT_TARGET_PARTIAL = "profit_target_partial"
    TRAILING_STOP = "trailing_stop"


@dataclass
class EntryCondition:
    """Individual entry condition"""
    condition_type: EntryConditionType
    name: str
    is_met: bool
    confidence: float  # 0.0 to 1.0
    weight: float  # Importance weight
    description: str
    supporting_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExitCondition:
    """Individual exit condition"""
    condition_type: ExitConditionType
    name: str
    is_triggered: bool
    urgency: float  # 0.0 to 1.0 (how urgent the exit is)
    weight: float  # Importance weight
    description: str
    supporting_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntryDecision:
    """Entry decision result"""
    should_enter: bool
    entry_confidence: float
    entry_conditions: List[EntryCondition]
    conditions_met_pct: float
    weighted_score: float
    recommended_entry_type: SignalType
    optimal_entry_price: Optional[float]
    position_size_pct: float
    risk_assessment: Dict[str, Any]
    timing_analysis: Dict[str, Any]


@dataclass
class ExitDecision:
    """Exit decision result"""
    should_exit: bool
    exit_urgency: float
    exit_conditions: List[ExitCondition]
    conditions_triggered_pct: float
    recommended_exit_type: str  # MARKET, LIMIT, STOP
    suggested_exit_price: Optional[float]
    partial_exit_recommended: bool
    partial_exit_percentage: float
    hold_duration_analysis: Dict[str, Any]


class EntryExitEngine:
    """
    Advanced entry/exit decision engine
    
    Analyzes market conditions and signals to make sophisticated
    entry and exit decisions with comprehensive condition checking.
    """
    
    def __init__(self):
        """Initialize entry/exit engine"""
        
        # Entry condition weights (must sum to 1.0)
        self.entry_condition_weights = {
            EntryConditionType.TREND_ALIGNMENT: 0.20,
            EntryConditionType.PATTERN_COMPLETION: 0.18,
            EntryConditionType.FIBONACCI_CONFLUENCE: 0.15,
            EntryConditionType.MOMENTUM_CONFIRMATION: 0.12,
            EntryConditionType.VOLUME_CONFIRMATION: 0.10,
            EntryConditionType.RISK_REWARD_ACCEPTABLE: 0.15,
            EntryConditionType.MARKET_STRUCTURE: 0.05,
            EntryConditionType.TIMEFRAME_CONFIRMATION: 0.05
        }
        
        # Exit condition weights
        self.exit_condition_weights = {
            ExitConditionType.TAKE_PROFIT_HIT: 0.25,
            ExitConditionType.STOP_LOSS_HIT: 0.25,
            ExitConditionType.PATTERN_INVALIDATION: 0.15,
            ExitConditionType.TREND_REVERSAL: 0.15,
            ExitConditionType.MOMENTUM_DIVERGENCE: 0.10,
            ExitConditionType.TIME_BASED_EXIT: 0.05,
            ExitConditionType.TRAILING_STOP: 0.05
        }
        
        # Entry/Exit configuration
        self.config = {
            'min_entry_confidence': 0.65,
            'min_conditions_met_pct': 0.70,
            'enable_partial_entries': True,
            'enable_partial_exits': True,
            'max_position_size_pct': 0.03,  # 3% max position size
            'min_risk_reward_ratio': 1.5,
            'entry_timeout_hours': 4,  # Hours before entry signal expires
            'exit_urgency_threshold': 0.7,  # Threshold for immediate exit
            'fibonacci_confluence_threshold': 0.02,  # 2% price tolerance for confluence
            'trend_alignment_threshold': 0.6  # Minimum alignment score
        }
    
    def evaluate_entry_conditions(self,
                                 signal: TradingSignal,
                                 unified_result: UnifiedAnalysisResult,
                                 mtf_result: Optional[MultiTimeframeResult] = None,
                                 confidence_score: Optional[ConfidenceScore] = None,
                                 current_price: float = None) -> EntryDecision:
        """
        Evaluate all entry conditions for a trading signal
        
        Args:
            signal: Trading signal to evaluate
            unified_result: Unified analysis result
            mtf_result: Multi-timeframe result (optional)
            confidence_score: Confidence score (optional)
            current_price: Current market price
            
        Returns:
            Entry decision with detailed analysis
        """
        try:
            current_price = current_price or signal.entry_price
            entry_conditions = []
            
            # Condition 1: Trend Alignment
            trend_condition = self._evaluate_trend_alignment(
                unified_result, mtf_result
            )
            entry_conditions.append(trend_condition)
            
            # Condition 2: Pattern Completion
            pattern_condition = self._evaluate_pattern_completion(
                unified_result, signal
            )
            entry_conditions.append(pattern_condition)
            
            # Condition 3: Fibonacci Confluence
            fib_condition = self._evaluate_fibonacci_confluence(
                unified_result, current_price
            )
            entry_conditions.append(fib_condition)
            
            # Condition 4: Momentum Confirmation
            momentum_condition = self._evaluate_momentum_confirmation(
                unified_result, signal
            )
            entry_conditions.append(momentum_condition)
            
            # Condition 5: Risk/Reward Acceptability
            rr_condition = self._evaluate_risk_reward_condition(signal)
            entry_conditions.append(rr_condition)
            
            # Condition 6: Market Structure
            structure_condition = self._evaluate_market_structure(
                unified_result, current_price
            )
            entry_conditions.append(structure_condition)
            
            # Condition 7: Timeframe Confirmation (if MTF available)
            if mtf_result:
                tf_condition = self._evaluate_timeframe_confirmation(mtf_result)
                entry_conditions.append(tf_condition)
            
            # Calculate overall metrics
            conditions_met_count = sum(1 for c in entry_conditions if c.is_met)
            conditions_met_pct = conditions_met_count / len(entry_conditions)
            
            weighted_score = self._calculate_weighted_entry_score(entry_conditions)
            
            # Make entry decision
            should_enter = (
                weighted_score >= self.config['min_entry_confidence'] and
                conditions_met_pct >= self.config['min_conditions_met_pct']
            )
            
            # Determine optimal entry type and price
            entry_type, optimal_price = self._determine_optimal_entry(
                signal, entry_conditions, current_price
            )
            
            # Calculate position size
            position_size = self._calculate_position_size(
                signal, entry_conditions, confidence_score
            )
            
            # Risk assessment
            risk_assessment = self._assess_entry_risk(
                signal, entry_conditions, unified_result
            )
            
            # Timing analysis
            timing_analysis = self._analyze_entry_timing(
                signal, entry_conditions, mtf_result
            )
            
            decision = EntryDecision(
                should_enter=should_enter,
                entry_confidence=weighted_score,
                entry_conditions=entry_conditions,
                conditions_met_pct=conditions_met_pct,
                weighted_score=weighted_score,
                recommended_entry_type=entry_type,
                optimal_entry_price=optimal_price,
                position_size_pct=position_size,
                risk_assessment=risk_assessment,
                timing_analysis=timing_analysis
            )
            
            analysis_logger.info(
                f"Entry evaluation: {signal.symbol} - "
                f"Should enter: {should_enter}, Confidence: {weighted_score:.2f}, "
                f"Conditions met: {conditions_met_pct:.1%}"
            )
            
            return decision
            
        except Exception as e:
            analysis_logger.error(f"Error evaluating entry conditions: {e}")
            return self._create_default_entry_decision()
    
    def evaluate_exit_conditions(self,
                                signal: TradingSignal,
                                current_price: float,
                                current_position: Dict[str, Any],
                                unified_result: Optional[UnifiedAnalysisResult] = None,
                                mtf_result: Optional[MultiTimeframeResult] = None) -> ExitDecision:
        """
        Evaluate exit conditions for an existing position
        
        Args:
            signal: Original entry signal
            current_price: Current market price
            current_position: Current position details
            unified_result: Updated analysis result
            mtf_result: Updated multi-timeframe result
            
        Returns:
            Exit decision with detailed analysis
        """
        try:
            exit_conditions = []
            
            # Condition 1: Take Profit Hit
            tp_condition = self._evaluate_take_profit_condition(
                signal, current_price, current_position
            )
            exit_conditions.append(tp_condition)
            
            # Condition 2: Stop Loss Hit
            sl_condition = self._evaluate_stop_loss_condition(
                signal, current_price, current_position
            )
            exit_conditions.append(sl_condition)
            
            # Condition 3: Pattern Invalidation
            if unified_result:
                pattern_condition = self._evaluate_pattern_invalidation(
                    signal, unified_result, current_price
                )
                exit_conditions.append(pattern_condition)
            
            # Condition 4: Trend Reversal
            if unified_result:
                trend_condition = self._evaluate_trend_reversal(
                    signal, unified_result, mtf_result
                )
                exit_conditions.append(trend_condition)
            
            # Condition 5: Time-based Exit
            time_condition = self._evaluate_time_based_exit(
                signal, current_position
            )
            exit_conditions.append(time_condition)
            
            # Condition 6: Trailing Stop
            trailing_condition = self._evaluate_trailing_stop(
                signal, current_price, current_position
            )
            exit_conditions.append(trailing_condition)
            
            # Calculate overall metrics
            conditions_triggered = sum(1 for c in exit_conditions if c.is_triggered)
            conditions_triggered_pct = conditions_triggered / len(exit_conditions)
            
            # Calculate exit urgency
            exit_urgency = self._calculate_exit_urgency(exit_conditions)
            
            # Make exit decision
            should_exit = (
                exit_urgency >= self.config['exit_urgency_threshold'] or
                any(c.is_triggered and c.condition_type in [
                    ExitConditionType.STOP_LOSS_HIT,
                    ExitConditionType.TAKE_PROFIT_HIT
                ] for c in exit_conditions)
            )
            
            # Determine exit type and price
            exit_type, exit_price = self._determine_optimal_exit(
                exit_conditions, current_price
            )
            
            # Partial exit analysis
            partial_exit, partial_pct = self._analyze_partial_exit(
                exit_conditions, current_position
            )
            
            # Hold duration analysis
            hold_analysis = self._analyze_hold_duration(
                signal, current_position, exit_conditions
            )
            
            decision = ExitDecision(
                should_exit=should_exit,
                exit_urgency=exit_urgency,
                exit_conditions=exit_conditions,
                conditions_triggered_pct=conditions_triggered_pct,
                recommended_exit_type=exit_type,
                suggested_exit_price=exit_price,
                partial_exit_recommended=partial_exit,
                partial_exit_percentage=partial_pct,
                hold_duration_analysis=hold_analysis
            )
            
            analysis_logger.info(
                f"Exit evaluation: {signal.symbol} - "
                f"Should exit: {should_exit}, Urgency: {exit_urgency:.2f}, "
                f"Conditions triggered: {conditions_triggered_pct:.1%}"
            )
            
            return decision
            
        except Exception as e:
            analysis_logger.error(f"Error evaluating exit conditions: {e}")
            return self._create_default_exit_decision()
    
    def _evaluate_trend_alignment(self,
                                unified_result: UnifiedAnalysisResult,
                                mtf_result: Optional[MultiTimeframeResult]) -> EntryCondition:
        """Evaluate trend alignment condition"""
        supporting_data = {}
        
        # Check Dow Theory trend
        dow_aligned = False
        if unified_result.dow_trend:
            dow_direction = unified_result.dow_trend.primary_trend.value.lower()
            signal_direction = unified_result.combined_signal.lower()
            
            if (dow_direction == 'bullish' and signal_direction == 'buy') or \
               (dow_direction == 'bearish' and signal_direction == 'sell'):
                dow_aligned = True
            
            supporting_data['dow_trend'] = dow_direction
            supporting_data['dow_aligned'] = dow_aligned
        
        # Check multi-timeframe alignment
        mtf_aligned = False
        if mtf_result:
            mtf_aligned = mtf_result.alignment_score >= self.config['trend_alignment_threshold']
            supporting_data['mtf_alignment_score'] = mtf_result.alignment_score
            supporting_data['mtf_aligned'] = mtf_aligned
        
        # Overall alignment assessment
        if mtf_result:
            is_met = dow_aligned and mtf_aligned
            confidence = (unified_result.dow_confidence + mtf_result.alignment_score) / 2
        else:
            is_met = dow_aligned
            confidence = unified_result.dow_confidence
        
        return EntryCondition(
            condition_type=EntryConditionType.TREND_ALIGNMENT,
            name="Trend Alignment",
            is_met=is_met,
            confidence=confidence,
            weight=self.entry_condition_weights[EntryConditionType.TREND_ALIGNMENT],
            description="Alignment between Dow Theory trend and signal direction",
            supporting_data=supporting_data
        )
    
    def _evaluate_pattern_completion(self,
                                   unified_result: UnifiedAnalysisResult,
                                   signal: TradingSignal) -> EntryCondition:
        """Evaluate pattern completion condition"""
        supporting_data = {}
        
        # Check Elliott Wave pattern completion
        complete_patterns = sum(1 for p in unified_result.elliott_patterns if p.is_complete)
        total_patterns = len(unified_result.elliott_patterns)
        
        if total_patterns > 0:
            completion_ratio = complete_patterns / total_patterns
            supporting_data['elliott_completion_ratio'] = completion_ratio
        else:
            completion_ratio = 0.0
        
        # Check for predictions (incomplete patterns moving toward completion)
        prediction_quality = 0.0
        if unified_result.elliott_predictions:
            avg_prediction_confidence = np.mean([
                p.get('confidence', 0) for p in unified_result.elliott_predictions
            ])
            prediction_quality = avg_prediction_confidence
            supporting_data['prediction_confidence'] = avg_prediction_confidence
        
        # Pattern strength assessment
        pattern_strength = 0.0
        if unified_result.elliott_patterns:
            avg_pattern_confidence = np.mean([p.confidence for p in unified_result.elliott_patterns])
            pattern_strength = avg_pattern_confidence
            supporting_data['pattern_strength'] = avg_pattern_confidence
        
        # Overall pattern condition
        overall_confidence = np.mean([completion_ratio, prediction_quality, pattern_strength])
        is_met = overall_confidence >= 0.6 and (complete_patterns > 0 or unified_result.elliott_predictions)
        
        supporting_data.update({
            'complete_patterns': complete_patterns,
            'total_patterns': total_patterns,
            'predictions_available': len(unified_result.elliott_predictions)
        })
        
        return EntryCondition(
            condition_type=EntryConditionType.PATTERN_COMPLETION,
            name="Pattern Completion",
            is_met=is_met,
            confidence=overall_confidence,
            weight=self.entry_condition_weights[EntryConditionType.PATTERN_COMPLETION],
            description="Quality and completion of Elliott Wave patterns",
            supporting_data=supporting_data
        )
    
    def _evaluate_fibonacci_confluence(self,
                                     unified_result: UnifiedAnalysisResult,
                                     current_price: float) -> EntryCondition:
        """Evaluate Fibonacci confluence condition"""
        supporting_data = {}
        confluence_levels = []
        
        # Check price targets near current price
        price_tolerance = current_price * self.config['fibonacci_confluence_threshold']
        
        for target_name, target_price in unified_result.price_targets.items():
            if target_price and abs(target_price - current_price) <= price_tolerance:
                confluence_levels.append({
                    'target_name': target_name,
                    'target_price': target_price,
                    'distance': abs(target_price - current_price)
                })
        
        # Check risk levels for confluence
        for risk_name, risk_price in unified_result.risk_levels.items():
            if risk_price and abs(risk_price - current_price) <= price_tolerance:
                confluence_levels.append({
                    'level_name': risk_name,
                    'level_price': risk_price,
                    'distance': abs(risk_price - current_price)
                })
        
        # Evaluate confluence quality
        confluence_count = len(confluence_levels)
        confluence_quality = min(1.0, confluence_count / 3.0)  # Normalize to max 3 levels
        
        is_met = confluence_count >= 2  # At least 2 levels in confluence
        
        supporting_data.update({
            'confluence_count': confluence_count,
            'confluence_levels': confluence_levels,
            'price_tolerance': price_tolerance
        })
        
        return EntryCondition(
            condition_type=EntryConditionType.FIBONACCI_CONFLUENCE,
            name="Fibonacci Confluence",
            is_met=is_met,
            confidence=confluence_quality,
            weight=self.entry_condition_weights[EntryConditionType.FIBONACCI_CONFLUENCE],
            description="Confluence of Fibonacci levels near current price",
            supporting_data=supporting_data
        )
    
    def _evaluate_momentum_confirmation(self,
                                      unified_result: UnifiedAnalysisResult,
                                      signal: TradingSignal) -> EntryCondition:
        """Evaluate momentum confirmation condition"""
        supporting_data = {}
        
        # Check agreement between Dow Theory and Elliott Wave
        agreement_score = unified_result.agreement_score
        
        # Check signal confidence
        signal_confidence = signal.confidence
        
        # Check Elliott Wave momentum (from predictions)
        elliott_momentum = 0.0
        if unified_result.elliott_predictions:
            # Count predictions that align with signal direction
            aligned_predictions = 0
            for prediction in unified_result.elliott_predictions:
                scenarios = prediction.get('scenarios', [])
                for scenario in scenarios:
                    target_price = scenario.get('target_price')
                    if target_price and signal.entry_price:
                        if (signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY] and 
                            target_price > signal.entry_price) or \
                           (signal.signal_type in [SignalType.MARKET_SELL, SignalType.LIMIT_SELL] and 
                            target_price < signal.entry_price):
                            aligned_predictions += 1
                            break
            
            if unified_result.elliott_predictions:
                elliott_momentum = aligned_predictions / len(unified_result.elliott_predictions)
        
        # Overall momentum assessment
        momentum_factors = [agreement_score, signal_confidence, elliott_momentum]
        overall_momentum = np.mean(momentum_factors)
        
        is_met = overall_momentum >= 0.6
        
        supporting_data.update({
            'agreement_score': agreement_score,
            'signal_confidence': signal_confidence,
            'elliott_momentum': elliott_momentum,
            'aligned_predictions': aligned_predictions if 'aligned_predictions' in locals() else 0
        })
        
        return EntryCondition(
            condition_type=EntryConditionType.MOMENTUM_CONFIRMATION,
            name="Momentum Confirmation",
            is_met=is_met,
            confidence=overall_momentum,
            weight=self.entry_condition_weights[EntryConditionType.MOMENTUM_CONFIRMATION],
            description="Momentum confirmation from multiple analysis methods",
            supporting_data=supporting_data
        )
    
    def _evaluate_risk_reward_condition(self, signal: TradingSignal) -> EntryCondition:
        """Evaluate risk/reward ratio acceptability"""
        supporting_data = {}
        
        rr_ratio = signal.risk_reward_ratio or 0.0
        is_met = rr_ratio >= self.config['min_risk_reward_ratio']
        
        # Calculate confidence based on RR ratio quality
        if rr_ratio >= 3.0:
            confidence = 1.0
        elif rr_ratio >= 2.0:
            confidence = 0.8
        elif rr_ratio >= 1.5:
            confidence = 0.6
        else:
            confidence = 0.3
        
        supporting_data.update({
            'risk_reward_ratio': rr_ratio,
            'minimum_required': self.config['min_risk_reward_ratio'],
            'stop_loss': signal.stop_loss,
            'take_profit_1': signal.take_profit_1
        })
        
        return EntryCondition(
            condition_type=EntryConditionType.RISK_REWARD_ACCEPTABLE,
            name="Risk/Reward Acceptable",
            is_met=is_met,
            confidence=confidence,
            weight=self.entry_condition_weights[EntryConditionType.RISK_REWARD_ACCEPTABLE],
            description="Risk/reward ratio meets minimum requirements",
            supporting_data=supporting_data
        )
    
    def _evaluate_market_structure(self,
                                 unified_result: UnifiedAnalysisResult,
                                 current_price: float) -> EntryCondition:
        """Evaluate market structure condition"""
        supporting_data = {}
        
        # Check swing point quality
        swing_count = len(unified_result.swing_points)
        swing_quality = min(1.0, swing_count / 5.0)  # Normalize to 5 swing points
        
        # Check if current price is at significant structure level
        structure_confluence = False
        if unified_result.swing_points:
            recent_swings = unified_result.swing_points[-5:]  # Last 5 swing points
            for swing in recent_swings:
                if abs(swing.price - current_price) <= current_price * 0.01:  # Within 1%
                    structure_confluence = True
                    break
        
        # Overall structure assessment
        structure_score = swing_quality
        if structure_confluence:
            structure_score += 0.2  # Bonus for being at structure level
        
        is_met = swing_count >= 3 and structure_score >= 0.6
        
        supporting_data.update({
            'swing_point_count': swing_count,
            'structure_confluence': structure_confluence,
            'structure_score': structure_score
        })
        
        return EntryCondition(
            condition_type=EntryConditionType.MARKET_STRUCTURE,
            name="Market Structure",
            is_met=is_met,
            confidence=min(1.0, structure_score),
            weight=self.entry_condition_weights[EntryConditionType.MARKET_STRUCTURE],
            description="Quality of market structure and swing points",
            supporting_data=supporting_data
        )
    
    def _evaluate_timeframe_confirmation(self, mtf_result: MultiTimeframeResult) -> EntryCondition:
        """Evaluate multi-timeframe confirmation"""
        supporting_data = {
            'alignment_score': mtf_result.alignment_score,
            'trend_consistency': mtf_result.trend_consistency,
            'conflicting_timeframes': len(mtf_result.conflicting_timeframes),
            'total_timeframes': len(mtf_result.timeframe_results)
        }
        
        # Check alignment and consistency
        alignment_good = mtf_result.alignment_score >= 0.7
        consistency_good = mtf_result.trend_consistency >= 0.7
        conflicts_acceptable = len(mtf_result.conflicting_timeframes) <= 1
        
        overall_confirmation = (mtf_result.alignment_score + mtf_result.trend_consistency) / 2
        is_met = alignment_good and consistency_good and conflicts_acceptable
        
        return EntryCondition(
            condition_type=EntryConditionType.TIMEFRAME_CONFIRMATION,
            name="Timeframe Confirmation",
            is_met=is_met,
            confidence=overall_confirmation,
            weight=self.entry_condition_weights[EntryConditionType.TIMEFRAME_CONFIRMATION],
            description="Multi-timeframe alignment and trend consistency",
            supporting_data=supporting_data
        )
    
    def _calculate_weighted_entry_score(self, conditions: List[EntryCondition]) -> float:
        """Calculate weighted entry score from conditions"""
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for condition in conditions:
            if condition.is_met:
                score = condition.confidence
            else:
                score = condition.confidence * 0.5  # Penalty for unmet conditions
            
            total_weighted_score += score * condition.weight
            total_weight += condition.weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_optimal_entry(self,
                               signal: TradingSignal,
                               conditions: List[EntryCondition],
                               current_price: float) -> Tuple[SignalType, Optional[float]]:
        """Determine optimal entry type and price"""
        # Default to signal's recommended type
        entry_type = signal.signal_type
        optimal_price = signal.entry_price
        
        # Check if confluence suggests better entry level
        for condition in conditions:
            if (condition.condition_type == EntryConditionType.FIBONACCI_CONFLUENCE and 
                condition.is_met):
                confluence_levels = condition.supporting_data.get('confluence_levels', [])
                if confluence_levels:
                    # Use closest confluence level as optimal entry
                    closest_level = min(confluence_levels, key=lambda x: x.get('distance', float('inf')))
                    level_price = closest_level.get('target_price') or closest_level.get('level_price')
                    if level_price:
                        optimal_price = level_price
                        # Use limit order if optimal price is different from current
                        if abs(level_price - current_price) > current_price * 0.001:  # 0.1% difference
                            if signal.signal_type == SignalType.MARKET_BUY:
                                entry_type = SignalType.LIMIT_BUY
                            elif signal.signal_type == SignalType.MARKET_SELL:
                                entry_type = SignalType.LIMIT_SELL
        
        return entry_type, optimal_price
    
    def _calculate_position_size(self,
                               signal: TradingSignal,
                               conditions: List[EntryCondition],
                               confidence_score: Optional[ConfidenceScore]) -> float:
        """Calculate optimal position size based on conditions and confidence"""
        base_size = signal.position_size_pct or 0.02  # Default 2%
        
        # Adjust based on entry conditions confidence
        conditions_confidence = np.mean([c.confidence for c in conditions if c.is_met])
        
        # Adjust based on overall confidence score
        if confidence_score:
            overall_confidence = confidence_score.overall_score
        else:
            overall_confidence = signal.confidence
        
        # Calculate adjustment factor
        confidence_factor = (conditions_confidence + overall_confidence) / 2
        
        if confidence_factor > 0.8:
            size_multiplier = 1.2  # Increase size for high confidence
        elif confidence_factor > 0.6:
            size_multiplier = 1.0  # Standard size
        else:
            size_multiplier = 0.7  # Reduce size for low confidence
        
        adjusted_size = base_size * size_multiplier
        
        # Cap at maximum position size
        return min(adjusted_size, self.config['max_position_size_pct'])
    
    def _assess_entry_risk(self,
                         signal: TradingSignal,
                         conditions: List[EntryCondition],
                         unified_result: UnifiedAnalysisResult) -> Dict[str, Any]:
        """Assess entry risk factors"""
        risk_factors = []
        
        # Check for unmet critical conditions
        for condition in conditions:
            if not condition.is_met and condition.weight > 0.15:  # Critical conditions
                risk_factors.append(f"Critical condition not met: {condition.name}")
        
        # Check agreement score
        if unified_result.agreement_score < 0.5:
            risk_factors.append("Low agreement between analysis methods")
        
        # Check volatility (simplified)
        if len(unified_result.swing_points) > 10:  # High swing count = high volatility
            risk_factors.append("High market volatility detected")
        
        return {
            'risk_factors': risk_factors,
            'risk_level': 'HIGH' if len(risk_factors) > 2 else 'MEDIUM' if len(risk_factors) > 0 else 'LOW',
            'risk_score': len(risk_factors) / 5.0  # Normalize to 0-1
        }
    
    def _analyze_entry_timing(self,
                            signal: TradingSignal,
                            conditions: List[EntryCondition],
                            mtf_result: Optional[MultiTimeframeResult]) -> Dict[str, Any]:
        """Analyze entry timing quality"""
        timing_factors = {}
        
        # Check signal age
        signal_age_hours = (datetime.utcnow() - signal.generated_at).total_seconds() / 3600
        timing_factors['signal_age_hours'] = signal_age_hours
        timing_factors['signal_fresh'] = signal_age_hours < 2
        
        # Check urgency
        timing_factors['signal_urgency'] = signal.urgency.value
        timing_factors['high_urgency'] = signal.urgency in [SignalUrgency.IMMEDIATE, SignalUrgency.HIGH]
        
        # Check optimal entry timeframe (if MTF available)
        if mtf_result:
            timing_factors['optimal_entry_timeframe'] = mtf_result.optimal_entry_timeframe
            timing_factors['entry_confirmation_score'] = mtf_result.entry_confirmation_score
        
        # Overall timing assessment
        timing_score = 1.0
        if signal_age_hours > 4:
            timing_score -= 0.3  # Penalty for old signals
        if signal.urgency == SignalUrgency.LOW:
            timing_score -= 0.2  # Penalty for low urgency
        
        timing_factors['timing_score'] = max(0.0, timing_score)
        timing_factors['timing_quality'] = 'EXCELLENT' if timing_score > 0.8 else \
                                         'GOOD' if timing_score > 0.6 else \
                                         'FAIR' if timing_score > 0.4 else 'POOR'
        
        return timing_factors
    
    # Exit condition evaluation methods
    def _evaluate_take_profit_condition(self,
                                      signal: TradingSignal,
                                      current_price: float,
                                      current_position: Dict[str, Any]) -> ExitCondition:
        """Evaluate take profit hit condition"""
        supporting_data = {}
        is_triggered = False
        urgency = 0.0
        
        position_type = current_position.get('type', '').upper()
        
        # Check each take profit level
        tp_levels = [signal.take_profit_1, signal.take_profit_2, signal.take_profit_3]
        for i, tp_level in enumerate(tp_levels, 1):
            if tp_level:
                if position_type == 'BUY' and current_price >= tp_level:
                    is_triggered = True
                    urgency = 0.8 + (i * 0.1)  # Higher urgency for higher TP levels
                    supporting_data[f'tp_{i}_hit'] = True
                    supporting_data[f'tp_{i}_level'] = tp_level
                elif position_type == 'SELL' and current_price <= tp_level:
                    is_triggered = True
                    urgency = 0.8 + (i * 0.1)
                    supporting_data[f'tp_{i}_hit'] = True
                    supporting_data[f'tp_{i}_level'] = tp_level
        
        supporting_data.update({
            'current_price': current_price,
            'position_type': position_type
        })
        
        return ExitCondition(
            condition_type=ExitConditionType.TAKE_PROFIT_HIT,
            name="Take Profit Hit",
            is_triggered=is_triggered,
            urgency=urgency,
            weight=self.exit_condition_weights[ExitConditionType.TAKE_PROFIT_HIT],
            description="Take profit level has been reached",
            supporting_data=supporting_data
        )
    
    def _evaluate_stop_loss_condition(self,
                                    signal: TradingSignal,
                                    current_price: float,
                                    current_position: Dict[str, Any]) -> ExitCondition:
        """Evaluate stop loss hit condition"""
        supporting_data = {}
        is_triggered = False
        urgency = 0.0
        
        position_type = current_position.get('type', '').upper()
        stop_loss = signal.stop_loss
        
        if stop_loss:
            if position_type == 'BUY' and current_price <= stop_loss:
                is_triggered = True
                urgency = 1.0  # Maximum urgency for stop loss
            elif position_type == 'SELL' and current_price >= stop_loss:
                is_triggered = True
                urgency = 1.0
            
            # Calculate distance to stop loss for early warning
            if position_type == 'BUY':
                distance_pct = (current_price - stop_loss) / current_price
            else:
                distance_pct = (stop_loss - current_price) / current_price
            
            # Early warning if close to stop loss
            if distance_pct < 0.01:  # Within 1% of stop loss
                urgency = max(urgency, 0.7)
            
            supporting_data.update({
                'stop_loss_level': stop_loss,
                'distance_to_sl_pct': distance_pct,
                'close_to_sl': distance_pct < 0.01
            })
        
        supporting_data.update({
            'current_price': current_price,
            'position_type': position_type
        })
        
        return ExitCondition(
            condition_type=ExitConditionType.STOP_LOSS_HIT,
            name="Stop Loss Hit",
            is_triggered=is_triggered,
            urgency=urgency,
            weight=self.exit_condition_weights[ExitConditionType.STOP_LOSS_HIT],
            description="Stop loss level has been hit or approached",
            supporting_data=supporting_data
        )
    
    def _evaluate_pattern_invalidation(self,
                                     signal: TradingSignal,
                                     unified_result: UnifiedAnalysisResult,
                                     current_price: float) -> ExitCondition:
        """Evaluate pattern invalidation condition"""
        supporting_data = {}
        is_triggered = False
        urgency = 0.0
        
        # Check if Elliott Wave patterns are still valid
        pattern_confidence_drop = False
        if unified_result.elliott_patterns:
            avg_confidence = np.mean([p.confidence for p in unified_result.elliott_patterns])
            if avg_confidence < 0.4:  # Significant confidence drop
                pattern_confidence_drop = True
                urgency = 0.6
        
        # Check if trend has changed significantly
        trend_change = False
        if unified_result.combined_signal != signal.combined_signal:
            trend_change = True
            urgency = max(urgency, 0.7)
        
        # Check agreement score deterioration
        agreement_deterioration = False
        if unified_result.agreement_score < 0.3:  # Very low agreement
            agreement_deterioration = True
            urgency = max(urgency, 0.5)
        
        is_triggered = pattern_confidence_drop or trend_change or agreement_deterioration
        
        supporting_data.update({
            'pattern_confidence_drop': pattern_confidence_drop,
            'trend_change': trend_change,
            'agreement_deterioration': agreement_deterioration,
            'current_agreement_score': unified_result.agreement_score,
            'current_signal': unified_result.combined_signal,
            'original_signal': getattr(signal, 'original_signal', signal.combined_signal)
        })
        
        return ExitCondition(
            condition_type=ExitConditionType.PATTERN_INVALIDATION,
            name="Pattern Invalidation",
            is_triggered=is_triggered,
            urgency=urgency,
            weight=self.exit_condition_weights[ExitConditionType.PATTERN_INVALIDATION],
            description="Elliott Wave patterns or trend analysis invalidated",
            supporting_data=supporting_data
        )
    
    def _evaluate_trend_reversal(self,
                               signal: TradingSignal,
                               unified_result: UnifiedAnalysisResult,
                               mtf_result: Optional[MultiTimeframeResult]) -> ExitCondition:
        """Evaluate trend reversal condition"""
        supporting_data = {}
        is_triggered = False
        urgency = 0.0
        
        # Check Dow Theory trend reversal
        dow_reversal = False
        if unified_result.dow_trend:
            current_trend = unified_result.dow_trend.primary_trend.value.lower()
            original_signal = signal.combined_signal.lower()
            
            if (original_signal == 'buy' and current_trend == 'bearish') or \
               (original_signal == 'sell' and current_trend == 'bullish'):
                dow_reversal = True
                urgency = 0.7
        
        # Check multi-timeframe reversal
        mtf_reversal = False
        if mtf_result:
            if mtf_result.primary_signal != signal.combined_signal:
                mtf_reversal = True
                urgency = max(urgency, 0.6)
            
            # Check alignment deterioration
            if mtf_result.alignment_score < 0.4:
                urgency = max(urgency, 0.5)
        
        is_triggered = dow_reversal or mtf_reversal
        
        supporting_data.update({
            'dow_reversal': dow_reversal,
            'mtf_reversal': mtf_reversal,
            'current_dow_trend': unified_result.dow_trend.primary_trend.value if unified_result.dow_trend else None,
            'current_mtf_signal': mtf_result.primary_signal if mtf_result else None,
            'original_signal': signal.combined_signal
        })
        
        return ExitCondition(
            condition_type=ExitConditionType.TREND_REVERSAL,
            name="Trend Reversal",
            is_triggered=is_triggered,
            urgency=urgency,
            weight=self.exit_condition_weights[ExitConditionType.TREND_REVERSAL],
            description="Major trend reversal detected",
            supporting_data=supporting_data
        )
    
    def _evaluate_time_based_exit(self,
                                signal: TradingSignal,
                                current_position: Dict[str, Any]) -> ExitCondition:
        """Evaluate time-based exit condition"""
        supporting_data = {}
        is_triggered = False
        urgency = 0.0
        
        # Calculate position hold time
        entry_time = current_position.get('entry_time', signal.generated_at)
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time)
        
        hold_duration = datetime.utcnow() - entry_time
        hold_hours = hold_duration.total_seconds() / 3600
        
        # Time-based exit rules
        max_hold_hours = 24  # Maximum hold time
        warning_hours = 18   # Warning threshold
        
        if hold_hours >= max_hold_hours:
            is_triggered = True
            urgency = 0.8
        elif hold_hours >= warning_hours:
            urgency = 0.4  # Warning level
        
        supporting_data.update({
            'hold_hours': hold_hours,
            'max_hold_hours': max_hold_hours,
            'warning_hours': warning_hours,
            'entry_time': entry_time.isoformat()
        })
        
        return ExitCondition(
            condition_type=ExitConditionType.TIME_BASED_EXIT,
            name="Time-Based Exit",
            is_triggered=is_triggered,
            urgency=urgency,
            weight=self.exit_condition_weights[ExitConditionType.TIME_BASED_EXIT],
            description="Position held for maximum allowed time",
            supporting_data=supporting_data
        )
    
    def _evaluate_trailing_stop(self,
                              signal: TradingSignal,
                              current_price: float,
                              current_position: Dict[str, Any]) -> ExitCondition:
        """Evaluate trailing stop condition"""
        supporting_data = {}
        is_triggered = False
        urgency = 0.0
        
        position_type = current_position.get('type', '').upper()
        entry_price = current_position.get('entry_price', signal.entry_price)
        
        # Calculate current profit
        if position_type == 'BUY':
            current_profit_pct = (current_price - entry_price) / entry_price
        else:
            current_profit_pct = (entry_price - current_price) / entry_price
        
        # Activate trailing stop if in significant profit
        trailing_activation_pct = 0.02  # 2% profit
        trailing_distance_pct = 0.01    # 1% trailing distance
        
        if current_profit_pct > trailing_activation_pct:
            # Calculate trailing stop level
            if position_type == 'BUY':
                trailing_stop = current_price * (1 - trailing_distance_pct)
                is_triggered = current_price <= trailing_stop
            else:
                trailing_stop = current_price * (1 + trailing_distance_pct)
                is_triggered = current_price >= trailing_stop
            
            if is_triggered:
                urgency = 0.7
            
            supporting_data['trailing_stop_level'] = trailing_stop
            supporting_data['trailing_active'] = True
        else:
            supporting_data['trailing_active'] = False
        
        supporting_data.update({
            'current_profit_pct': current_profit_pct,
            'activation_threshold': trailing_activation_pct,
            'trailing_distance': trailing_distance_pct,
            'position_type': position_type
        })
        
        return ExitCondition(
            condition_type=ExitConditionType.TRAILING_STOP,
            name="Trailing Stop",
            is_triggered=is_triggered,
            urgency=urgency,
            weight=self.exit_condition_weights[ExitConditionType.TRAILING_STOP],
            description="Trailing stop loss activation",
            supporting_data=supporting_data
        )
    
    def _calculate_exit_urgency(self, conditions: List[ExitCondition]) -> float:
        """Calculate overall exit urgency"""
        urgencies = [c.urgency * c.weight for c in conditions if c.is_triggered]
        weights = [c.weight for c in conditions if c.is_triggered]
        
        if weights:
            return sum(urgencies) / sum(weights)
        else:
            return 0.0
    
    def _determine_optimal_exit(self,
                              conditions: List[ExitCondition],
                              current_price: float) -> Tuple[str, Optional[float]]:
        """Determine optimal exit type and price"""
        # Check for immediate exit conditions
        immediate_conditions = [
            ExitConditionType.STOP_LOSS_HIT,
            ExitConditionType.TAKE_PROFIT_HIT
        ]
        
        for condition in conditions:
            if condition.is_triggered and condition.condition_type in immediate_conditions:
                return "MARKET", current_price
        
        # Check for planned exit conditions
        return "MARKET", current_price  # Default to market exit
    
    def _analyze_partial_exit(self,
                            conditions: List[ExitCondition],
                            current_position: Dict[str, Any]) -> Tuple[bool, float]:
        """Analyze if partial exit is recommended"""
        # Check for take profit hits (suggest partial exit)
        for condition in conditions:
            if (condition.condition_type == ExitConditionType.TAKE_PROFIT_HIT and 
                condition.is_triggered):
                # Suggest 50% exit on first TP, 75% on second TP
                tp_data = condition.supporting_data
                if tp_data.get('tp_1_hit'):
                    return True, 0.5
                elif tp_data.get('tp_2_hit'):
                    return True, 0.75
        
        return False, 0.0
    
    def _analyze_hold_duration(self,
                             signal: TradingSignal,
                             current_position: Dict[str, Any],
                             conditions: List[ExitCondition]) -> Dict[str, Any]:
        """Analyze position hold duration"""
        entry_time = current_position.get('entry_time', signal.generated_at)
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time)
        
        hold_duration = datetime.utcnow() - entry_time
        hold_hours = hold_duration.total_seconds() / 3600
        
        return {
            'hold_duration_hours': hold_hours,
            'hold_duration_str': str(hold_duration),
            'entry_time': entry_time.isoformat(),
            'average_hold_time': 12,  # Example average
            'hold_classification': 'SHORT' if hold_hours < 4 else 
                                 'MEDIUM' if hold_hours < 12 else 'LONG'
        }
    
    def _create_default_entry_decision(self) -> EntryDecision:
        """Create default entry decision for error cases"""
        return EntryDecision(
            should_enter=False,
            entry_confidence=0.0,
            entry_conditions=[],
            conditions_met_pct=0.0,
            weighted_score=0.0,
            recommended_entry_type=SignalType.HOLD,
            optimal_entry_price=None,
            position_size_pct=0.0,
            risk_assessment={'risk_level': 'HIGH', 'risk_factors': ['Error in analysis']},
            timing_analysis={'timing_quality': 'POOR'}
        )
    
    def _create_default_exit_decision(self) -> ExitDecision:
        """Create default exit decision for error cases"""
        return ExitDecision(
            should_exit=False,
            exit_urgency=0.0,
            exit_conditions=[],
            conditions_triggered_pct=0.0,
            recommended_exit_type="HOLD",
            suggested_exit_price=None,
            partial_exit_recommended=False,
            partial_exit_percentage=0.0,
            hold_duration_analysis={'hold_classification': 'UNKNOWN'}
        )


# Create default entry/exit engine instance
entry_exit_engine = EntryExitEngine()