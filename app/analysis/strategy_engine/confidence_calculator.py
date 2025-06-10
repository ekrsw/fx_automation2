"""
Advanced Confidence Calculation System

Calculates sophisticated confidence scores for trading signals based on 
multiple analysis factors and market conditions.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from app.analysis.strategy_engine.unified_analyzer import UnifiedAnalysisResult
from app.analysis.strategy_engine.multi_timeframe_analyzer import MultiTimeframeResult
from app.analysis.strategy_engine.signal_generator import TradingSignal
from app.utils.logger import analysis_logger


class ConfidenceFactorType(Enum):
    """Types of confidence factors"""
    TECHNICAL_ANALYSIS = "technical_analysis"
    MARKET_STRUCTURE = "market_structure"
    TIMEFRAME_ALIGNMENT = "timeframe_alignment"
    RISK_REWARD = "risk_reward"
    HISTORICAL_PERFORMANCE = "historical_performance"
    MARKET_CONDITIONS = "market_conditions"
    VOLUME_CONFIRMATION = "volume_confirmation"


@dataclass
class ConfidenceFactor:
    """Individual confidence factor"""
    factor_type: ConfidenceFactorType
    name: str
    value: float  # 0.0 to 1.0
    weight: float  # Importance weight
    description: str
    supporting_data: Dict[str, Any]


@dataclass
class ConfidenceScore:
    """Comprehensive confidence score with breakdown"""
    overall_score: float  # Final confidence score (0.0 to 1.0)
    raw_score: float  # Score before adjustments
    
    # Individual factor scores
    technical_score: float
    structure_score: float
    timeframe_score: float
    risk_reward_score: float
    market_conditions_score: float
    
    # Factor breakdown
    factors: List[ConfidenceFactor]
    
    # Adjustments applied
    adjustments: Dict[str, float]
    
    # Metadata
    calculation_timestamp: datetime
    reliability_grade: str  # A, B, C, D rating
    
    
class ConfidenceCalculator:
    """
    Advanced confidence calculation engine
    
    Calculates comprehensive confidence scores by analyzing multiple
    factors including technical analysis quality, market structure,
    timeframe alignment, and historical performance.
    """
    
    def __init__(self):
        """Initialize confidence calculator"""
        
        # Factor weights (must sum to 1.0)
        self.factor_weights = {
            ConfidenceFactorType.TECHNICAL_ANALYSIS: 0.25,
            ConfidenceFactorType.MARKET_STRUCTURE: 0.20,
            ConfidenceFactorType.TIMEFRAME_ALIGNMENT: 0.20,
            ConfidenceFactorType.RISK_REWARD: 0.15,
            ConfidenceFactorType.HISTORICAL_PERFORMANCE: 0.10,
            ConfidenceFactorType.MARKET_CONDITIONS: 0.10
        }
        
        # Confidence calculation configuration
        self.config = {
            'enable_historical_adjustment': True,
            'enable_market_condition_adjustment': True,
            'enable_volatility_adjustment': True,
            'min_data_points_for_history': 20,
            'confidence_decay_hours': 24,  # How long confidence remains valid
            'volatility_adjustment_factor': 0.1,
            'news_impact_adjustment': 0.05
        }
        
        # Reliability grade thresholds
        self.reliability_grades = {
            'A': 0.8,   # Excellent confidence
            'B': 0.65,  # Good confidence  
            'C': 0.5,   # Moderate confidence
            'D': 0.0    # Poor confidence
        }
        
        # Historical performance tracking (would be loaded from database)
        self.historical_performance = {
            'dow_theory_accuracy': 0.72,
            'elliott_wave_accuracy': 0.68,
            'combined_accuracy': 0.75,
            'timeframe_reliability': {
                'D1': 0.78,
                'H4': 0.74,
                'H1': 0.70,
                'M30': 0.65,
                'M15': 0.60
            }
        }
    
    def _get_strength_numeric(self, strength) -> float:
        """Convert strength enum to numeric value"""
        if strength is None:
            return 0.5
        
        strength_mapping = {
            'very_strong': 1.0,
            'strong': 0.8,
            'moderate': 0.6,
            'weak': 0.4,
            'very_weak': 0.2
        }
        
        if hasattr(strength, 'value'):
            return strength_mapping.get(strength.value, 0.5)
        else:
            return strength_mapping.get(str(strength), 0.5)
    
    def calculate_unified_confidence(self, 
                                   unified_result: UnifiedAnalysisResult,
                                   signal: Optional[TradingSignal] = None,
                                   market_data: Optional[pd.DataFrame] = None) -> ConfidenceScore:
        """
        Calculate confidence score for unified analysis result
        
        Args:
            unified_result: Unified analysis result
            signal: Generated trading signal (optional)
            market_data: Additional market data for context
            
        Returns:
            Comprehensive confidence score
        """
        try:
            factors = []
            
            # Factor 1: Technical Analysis Quality
            tech_factor = self._calculate_technical_analysis_factor(unified_result)
            factors.append(tech_factor)
            
            # Factor 2: Market Structure Quality
            structure_factor = self._calculate_market_structure_factor(unified_result)
            factors.append(structure_factor)
            
            # Factor 3: Risk/Reward Assessment
            if signal:
                rr_factor = self._calculate_risk_reward_factor(signal)
                factors.append(rr_factor)
            
            # Factor 4: Historical Performance
            history_factor = self._calculate_historical_performance_factor(unified_result)
            factors.append(history_factor)
            
            # Factor 5: Market Conditions
            if market_data is not None:
                market_factor = self._calculate_market_conditions_factor(market_data)
                factors.append(market_factor)
            
            # Calculate weighted score
            raw_score = self._calculate_weighted_score(factors)
            
            # Apply adjustments
            adjustments = self._calculate_adjustments(unified_result, market_data)
            adjusted_score = self._apply_adjustments(raw_score, adjustments)
            
            # Create individual scores
            individual_scores = self._extract_individual_scores(factors)
            
            # Determine reliability grade
            reliability_grade = self._determine_reliability_grade(adjusted_score)
            
            confidence_score = ConfidenceScore(
                overall_score=adjusted_score,
                raw_score=raw_score,
                technical_score=individual_scores.get('technical', 0.0),
                structure_score=individual_scores.get('structure', 0.0),
                timeframe_score=individual_scores.get('timeframe', 0.0),
                risk_reward_score=individual_scores.get('risk_reward', 0.0),
                market_conditions_score=individual_scores.get('market_conditions', 0.0),
                factors=factors,
                adjustments=adjustments,
                calculation_timestamp=datetime.utcnow(),
                reliability_grade=reliability_grade
            )
            
            analysis_logger.debug(
                f"Calculated unified confidence: {adjusted_score:.3f} "
                f"(Grade: {reliability_grade}) for {unified_result.symbol}"
            )
            
            return confidence_score
            
        except Exception as e:
            analysis_logger.error(f"Error calculating unified confidence: {e}")
            # Return minimal confidence score on error
            return self._create_minimal_confidence_score()
    
    def calculate_mtf_confidence(self,
                               mtf_result: MultiTimeframeResult,
                               signal: Optional[TradingSignal] = None) -> ConfidenceScore:
        """
        Calculate confidence score for multi-timeframe analysis
        
        Args:
            mtf_result: Multi-timeframe analysis result
            signal: Generated trading signal (optional)
            
        Returns:
            Comprehensive confidence score
        """
        try:
            factors = []
            
            # Factor 1: Timeframe Alignment
            alignment_factor = self._calculate_timeframe_alignment_factor(mtf_result)
            factors.append(alignment_factor)
            
            # Factor 2: Trend Consistency
            trend_factor = self._calculate_trend_consistency_factor(mtf_result)
            factors.append(trend_factor)
            
            # Factor 3: Individual Timeframe Quality
            tf_quality_factor = self._calculate_timeframe_quality_factor(mtf_result)
            factors.append(tf_quality_factor)
            
            # Factor 4: Entry Timing Quality
            entry_factor = self._calculate_entry_timing_factor(mtf_result)
            factors.append(entry_factor)
            
            # Factor 5: Risk/Reward (if signal available)
            if signal:
                rr_factor = self._calculate_risk_reward_factor(signal)
                factors.append(rr_factor)
            
            # Calculate weighted score
            raw_score = self._calculate_weighted_score(factors)
            
            # Apply MTF-specific adjustments
            adjustments = self._calculate_mtf_adjustments(mtf_result)
            adjusted_score = self._apply_adjustments(raw_score, adjustments)
            
            # Create individual scores
            individual_scores = self._extract_individual_scores(factors)
            
            # Determine reliability grade
            reliability_grade = self._determine_reliability_grade(adjusted_score)
            
            confidence_score = ConfidenceScore(
                overall_score=adjusted_score,
                raw_score=raw_score,
                technical_score=individual_scores.get('technical', 0.0),
                structure_score=individual_scores.get('structure', 0.0),
                timeframe_score=individual_scores.get('timeframe', adjusted_score),  # Primary for MTF
                risk_reward_score=individual_scores.get('risk_reward', 0.0),
                market_conditions_score=individual_scores.get('market_conditions', 0.0),
                factors=factors,
                adjustments=adjustments,
                calculation_timestamp=datetime.utcnow(),
                reliability_grade=reliability_grade
            )
            
            analysis_logger.debug(
                f"Calculated MTF confidence: {adjusted_score:.3f} "
                f"(Grade: {reliability_grade}) for {mtf_result.symbol}"
            )
            
            return confidence_score
            
        except Exception as e:
            analysis_logger.error(f"Error calculating MTF confidence: {e}")
            return self._create_minimal_confidence_score()
    
    def _calculate_technical_analysis_factor(self, 
                                           unified_result: UnifiedAnalysisResult) -> ConfidenceFactor:
        """Calculate technical analysis quality factor"""
        scores = []
        supporting_data = {}
        
        # Dow Theory quality
        if unified_result.dow_confidence > 0:
            scores.append(unified_result.dow_confidence)
            supporting_data['dow_confidence'] = unified_result.dow_confidence
            
            if unified_result.dow_trend:
                supporting_data['dow_trend_strength'] = self._get_strength_numeric(
                    unified_result.dow_trend.strength if unified_result.dow_trend else None
                )
        
        # Elliott Wave quality
        if unified_result.elliott_confidence > 0:
            scores.append(unified_result.elliott_confidence)
            supporting_data['elliott_confidence'] = unified_result.elliott_confidence
            supporting_data['elliott_patterns'] = len(unified_result.elliott_patterns)
            supporting_data['elliott_predictions'] = len(unified_result.elliott_predictions)
        
        # Agreement between methods
        if unified_result.agreement_score > 0:
            scores.append(unified_result.agreement_score)
            supporting_data['agreement_score'] = unified_result.agreement_score
        
        # Calculate overall technical score
        tech_score = np.mean(scores) if scores else 0.0
        
        return ConfidenceFactor(
            factor_type=ConfidenceFactorType.TECHNICAL_ANALYSIS,
            name="Technical Analysis Quality",
            value=tech_score,
            weight=self.factor_weights[ConfidenceFactorType.TECHNICAL_ANALYSIS],
            description=f"Combined quality of Dow Theory and Elliott Wave analysis",
            supporting_data=supporting_data
        )
    
    def _calculate_market_structure_factor(self, 
                                         unified_result: UnifiedAnalysisResult) -> ConfidenceFactor:
        """Calculate market structure quality factor"""
        scores = []
        supporting_data = {}
        
        # Swing point quality
        swing_count = len(unified_result.swing_points)
        if swing_count >= 5:
            swing_score = 1.0
        elif swing_count >= 3:
            swing_score = 0.7
        else:
            swing_score = 0.3
        
        scores.append(swing_score)
        supporting_data['swing_point_count'] = swing_count
        
        # Pattern completeness
        complete_patterns = sum(1 for p in unified_result.elliott_patterns if p.is_complete)
        if unified_result.elliott_patterns:
            completeness_score = complete_patterns / len(unified_result.elliott_patterns)
            scores.append(completeness_score)
            supporting_data['pattern_completeness'] = completeness_score
        
        # Price target availability
        target_count = len([v for v in unified_result.price_targets.values() if v])
        if target_count >= 3:
            target_score = 1.0
        elif target_count >= 1:
            target_score = 0.6
        else:
            target_score = 0.2
        
        scores.append(target_score)
        supporting_data['price_targets_available'] = target_count
        
        structure_score = np.mean(scores) if scores else 0.0
        
        return ConfidenceFactor(
            factor_type=ConfidenceFactorType.MARKET_STRUCTURE,
            name="Market Structure Quality",
            value=structure_score,
            weight=self.factor_weights[ConfidenceFactorType.MARKET_STRUCTURE],
            description="Quality of market structure and pattern identification",
            supporting_data=supporting_data
        )
    
    def _calculate_timeframe_alignment_factor(self, 
                                            mtf_result: MultiTimeframeResult) -> ConfidenceFactor:
        """Calculate timeframe alignment factor for MTF analysis"""
        supporting_data = {
            'alignment_score': mtf_result.alignment_score,
            'trend_consistency': mtf_result.trend_consistency,
            'conflicting_timeframes': len(mtf_result.conflicting_timeframes),
            'total_timeframes': len(mtf_result.timeframe_results)
        }
        
        # Base alignment score
        alignment_component = mtf_result.alignment_score * 0.6
        
        # Trend consistency component
        trend_component = mtf_result.trend_consistency * 0.3
        
        # Penalty for conflicting timeframes
        conflict_ratio = len(mtf_result.conflicting_timeframes) / len(mtf_result.timeframe_results)
        conflict_penalty = conflict_ratio * 0.1
        
        timeframe_score = max(0.0, alignment_component + trend_component - conflict_penalty)
        
        return ConfidenceFactor(
            factor_type=ConfidenceFactorType.TIMEFRAME_ALIGNMENT,
            name="Timeframe Alignment",
            value=timeframe_score,
            weight=self.factor_weights[ConfidenceFactorType.TIMEFRAME_ALIGNMENT],
            description="Alignment and consistency across multiple timeframes",
            supporting_data=supporting_data
        )
    
    def _calculate_risk_reward_factor(self, signal: TradingSignal) -> ConfidenceFactor:
        """Calculate risk/reward quality factor"""
        supporting_data = {}
        
        if signal.risk_reward_ratio:
            # Excellent: RR > 3, Good: RR > 2, Fair: RR > 1.5, Poor: RR < 1.5
            if signal.risk_reward_ratio >= 3.0:
                rr_score = 1.0
            elif signal.risk_reward_ratio >= 2.0:
                rr_score = 0.8
            elif signal.risk_reward_ratio >= 1.5:
                rr_score = 0.6
            elif signal.risk_reward_ratio >= 1.0:
                rr_score = 0.4
            else:
                rr_score = 0.2
            
            supporting_data['risk_reward_ratio'] = signal.risk_reward_ratio
        else:
            rr_score = 0.0
        
        # Position size factor
        if signal.position_size_pct:
            if 0.01 <= signal.position_size_pct <= 0.03:  # Reasonable position size
                size_score = 1.0
            elif signal.position_size_pct <= 0.05:
                size_score = 0.7
            else:
                size_score = 0.3
            
            supporting_data['position_size_pct'] = signal.position_size_pct
        else:
            size_score = 0.5
        
        # Multiple take profits bonus
        tp_count = sum(1 for tp in [signal.take_profit_1, signal.take_profit_2, signal.take_profit_3] if tp)
        if tp_count >= 3:
            tp_bonus = 0.1
        elif tp_count >= 2:
            tp_bonus = 0.05
        else:
            tp_bonus = 0.0
        
        supporting_data['take_profit_levels'] = tp_count
        
        final_rr_score = min(1.0, (rr_score * 0.7 + size_score * 0.3) + tp_bonus)
        
        return ConfidenceFactor(
            factor_type=ConfidenceFactorType.RISK_REWARD,
            name="Risk/Reward Quality",
            value=final_rr_score,
            weight=self.factor_weights[ConfidenceFactorType.RISK_REWARD],
            description="Quality of risk management and reward potential",
            supporting_data=supporting_data
        )
    
    def _calculate_historical_performance_factor(self, 
                                               unified_result: UnifiedAnalysisResult) -> ConfidenceFactor:
        """Calculate historical performance factor"""
        supporting_data = {}
        
        # Method-specific performance
        dow_performance = self.historical_performance['dow_theory_accuracy']
        elliott_performance = self.historical_performance['elliott_wave_accuracy']
        combined_performance = self.historical_performance['combined_accuracy']
        
        # Weight by individual method confidences
        if unified_result.dow_confidence > 0 and unified_result.elliott_confidence > 0:
            # Both methods active
            weighted_performance = (
                dow_performance * unified_result.dow_confidence +
                elliott_performance * unified_result.elliott_confidence
            ) / (unified_result.dow_confidence + unified_result.elliott_confidence)
        elif unified_result.dow_confidence > 0:
            weighted_performance = dow_performance
        elif unified_result.elliott_confidence > 0:
            weighted_performance = elliott_performance
        else:
            weighted_performance = combined_performance
        
        supporting_data.update({
            'dow_theory_accuracy': dow_performance,
            'elliott_wave_accuracy': elliott_performance,
            'combined_accuracy': combined_performance,
            'weighted_performance': weighted_performance
        })
        
        return ConfidenceFactor(
            factor_type=ConfidenceFactorType.HISTORICAL_PERFORMANCE,
            name="Historical Performance",
            value=weighted_performance,
            weight=self.factor_weights[ConfidenceFactorType.HISTORICAL_PERFORMANCE],
            description="Historical accuracy of analysis methods",
            supporting_data=supporting_data
        )
    
    def _calculate_market_conditions_factor(self, market_data: pd.DataFrame) -> ConfidenceFactor:
        """Calculate market conditions factor"""
        supporting_data = {}
        
        if market_data is None or len(market_data) < 20:
            return ConfidenceFactor(
                factor_type=ConfidenceFactorType.MARKET_CONDITIONS,
                name="Market Conditions",
                value=0.5,  # Neutral when no data
                weight=self.factor_weights[ConfidenceFactorType.MARKET_CONDITIONS],
                description="Market volatility and conditions assessment",
                supporting_data={'insufficient_data': True}
            )
        
        # Calculate volatility
        returns = market_data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Optimal volatility range (not too low, not too high)
        if 0.1 <= volatility <= 0.3:  # 10-30% annual volatility
            volatility_score = 1.0
        elif 0.05 <= volatility <= 0.5:
            volatility_score = 0.7
        else:
            volatility_score = 0.4
        
        # Trend clarity (ADX-like measure)
        price_range = market_data['high'] - market_data['low']
        avg_range = price_range.rolling(14).mean()
        trend_clarity = min(1.0, avg_range.iloc[-1] / market_data['close'].iloc[-1] * 100)
        
        supporting_data.update({
            'volatility': volatility,
            'volatility_score': volatility_score,
            'trend_clarity': trend_clarity
        })
        
        conditions_score = (volatility_score * 0.6 + trend_clarity * 0.4)
        
        return ConfidenceFactor(
            factor_type=ConfidenceFactorType.MARKET_CONDITIONS,
            name="Market Conditions",
            value=conditions_score,
            weight=self.factor_weights[ConfidenceFactorType.MARKET_CONDITIONS],
            description="Market volatility and conditions assessment",
            supporting_data=supporting_data
        )
    
    def _calculate_trend_consistency_factor(self, mtf_result: MultiTimeframeResult) -> ConfidenceFactor:
        """Calculate trend consistency factor for MTF"""
        supporting_data = {
            'trend_consistency': mtf_result.trend_consistency,
            'primary_confidence': mtf_result.primary_confidence
        }
        
        # Combine trend consistency with primary confidence
        consistency_score = (mtf_result.trend_consistency * 0.7 + mtf_result.primary_confidence * 0.3)
        
        return ConfidenceFactor(
            factor_type=ConfidenceFactorType.TECHNICAL_ANALYSIS,
            name="Trend Consistency",
            value=consistency_score,
            weight=0.3,  # Custom weight for MTF
            description="Consistency of trend across timeframes",
            supporting_data=supporting_data
        )
    
    def _calculate_timeframe_quality_factor(self, mtf_result: MultiTimeframeResult) -> ConfidenceFactor:
        """Calculate average quality of individual timeframes"""
        timeframe_confidences = []
        supporting_data = {}
        
        for tf, result in mtf_result.timeframe_results.items():
            timeframe_confidences.append(result.combined_confidence)
            supporting_data[f'{tf}_confidence'] = result.combined_confidence
        
        avg_quality = np.mean(timeframe_confidences) if timeframe_confidences else 0.0
        supporting_data['average_timeframe_quality'] = avg_quality
        
        return ConfidenceFactor(
            factor_type=ConfidenceFactorType.TECHNICAL_ANALYSIS,
            name="Timeframe Quality",
            value=avg_quality,
            weight=0.25,  # Custom weight for MTF
            description="Average quality of individual timeframe analyses",
            supporting_data=supporting_data
        )
    
    def _calculate_entry_timing_factor(self, mtf_result: MultiTimeframeResult) -> ConfidenceFactor:
        """Calculate entry timing quality factor"""
        supporting_data = {
            'entry_confirmation_score': mtf_result.entry_confirmation_score,
            'optimal_entry_timeframe': mtf_result.optimal_entry_timeframe
        }
        
        # Entry timing score
        timing_score = mtf_result.entry_confirmation_score
        
        return ConfidenceFactor(
            factor_type=ConfidenceFactorType.TECHNICAL_ANALYSIS,
            name="Entry Timing",
            value=timing_score,
            weight=0.15,  # Custom weight for MTF
            description="Quality of entry timing and confirmation",
            supporting_data=supporting_data
        )
    
    def _calculate_weighted_score(self, factors: List[ConfidenceFactor]) -> float:
        """Calculate weighted average of confidence factors"""
        if not factors:
            return 0.0
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for factor in factors:
            total_weighted_score += factor.value * factor.weight
            total_weight += factor.weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_adjustments(self, 
                             unified_result: UnifiedAnalysisResult,
                             market_data: Optional[pd.DataFrame]) -> Dict[str, float]:
        """Calculate confidence adjustments"""
        adjustments = {}
        
        # Time decay adjustment
        if hasattr(unified_result, 'timestamp'):
            hours_old = (datetime.utcnow() - unified_result.timestamp).total_seconds() / 3600
            if hours_old > 0:
                decay_factor = max(0.5, 1.0 - (hours_old / self.config['confidence_decay_hours']))
                adjustments['time_decay'] = decay_factor - 1.0  # Negative adjustment
        
        # Agreement bonus/penalty
        if unified_result.agreement_score > 0.8:
            adjustments['high_agreement_bonus'] = 0.05
        elif unified_result.agreement_score < 0.4:
            adjustments['low_agreement_penalty'] = -0.1
        
        # Signal strength adjustment
        if unified_result.combined_confidence > 0.9:
            adjustments['high_confidence_bonus'] = 0.03
        elif unified_result.combined_confidence < 0.4:
            adjustments['low_confidence_penalty'] = -0.05
        
        return adjustments
    
    def _calculate_mtf_adjustments(self, mtf_result: MultiTimeframeResult) -> Dict[str, float]:
        """Calculate MTF-specific adjustments"""
        adjustments = {}
        
        # High alignment bonus
        if mtf_result.alignment_score > 0.85:
            adjustments['high_alignment_bonus'] = 0.08
        elif mtf_result.alignment_score < 0.5:
            adjustments['low_alignment_penalty'] = -0.15
        
        # Conflicting timeframes penalty
        conflict_ratio = len(mtf_result.conflicting_timeframes) / len(mtf_result.timeframe_results)
        if conflict_ratio > 0.3:
            adjustments['conflict_penalty'] = -0.1 * conflict_ratio
        
        # Trend consistency bonus
        if mtf_result.trend_consistency > 0.8:
            adjustments['trend_consistency_bonus'] = 0.05
        
        return adjustments
    
    def _apply_adjustments(self, base_score: float, adjustments: Dict[str, float]) -> float:
        """Apply adjustments to base confidence score"""
        adjusted_score = base_score
        
        for adjustment_name, adjustment_value in adjustments.items():
            adjusted_score += adjustment_value
        
        return max(0.0, min(1.0, adjusted_score))  # Clamp to [0, 1]
    
    def _extract_individual_scores(self, factors: List[ConfidenceFactor]) -> Dict[str, float]:
        """Extract individual factor scores"""
        scores = {}
        
        for factor in factors:
            if factor.factor_type == ConfidenceFactorType.TECHNICAL_ANALYSIS:
                scores['technical'] = factor.value
            elif factor.factor_type == ConfidenceFactorType.MARKET_STRUCTURE:
                scores['structure'] = factor.value
            elif factor.factor_type == ConfidenceFactorType.TIMEFRAME_ALIGNMENT:
                scores['timeframe'] = factor.value
            elif factor.factor_type == ConfidenceFactorType.RISK_REWARD:
                scores['risk_reward'] = factor.value
            elif factor.factor_type == ConfidenceFactorType.MARKET_CONDITIONS:
                scores['market_conditions'] = factor.value
        
        return scores
    
    def _determine_reliability_grade(self, confidence_score: float) -> str:
        """Determine reliability grade based on confidence score"""
        for grade, threshold in sorted(self.reliability_grades.items(), 
                                     key=lambda x: x[1], reverse=True):
            if confidence_score >= threshold:
                return grade
        return 'D'
    
    def _create_minimal_confidence_score(self) -> ConfidenceScore:
        """Create minimal confidence score for error cases"""
        return ConfidenceScore(
            overall_score=0.1,
            raw_score=0.1,
            technical_score=0.0,
            structure_score=0.0,
            timeframe_score=0.0,
            risk_reward_score=0.0,
            market_conditions_score=0.0,
            factors=[],
            adjustments={},
            calculation_timestamp=datetime.utcnow(),
            reliability_grade='D'
        )
    
    def get_confidence_breakdown(self, confidence_score: ConfidenceScore) -> Dict[str, Any]:
        """Get detailed breakdown of confidence calculation"""
        return {
            'overall_score': confidence_score.overall_score,
            'reliability_grade': confidence_score.reliability_grade,
            'component_scores': {
                'technical_analysis': confidence_score.technical_score,
                'market_structure': confidence_score.structure_score,
                'timeframe_alignment': confidence_score.timeframe_score,
                'risk_reward': confidence_score.risk_reward_score,
                'market_conditions': confidence_score.market_conditions_score
            },
            'factor_details': [
                {
                    'name': factor.name,
                    'type': factor.factor_type.value,
                    'value': factor.value,
                    'weight': factor.weight,
                    'contribution': factor.value * factor.weight,
                    'description': factor.description
                }
                for factor in confidence_score.factors
            ],
            'adjustments_applied': confidence_score.adjustments,
            'calculation_metadata': {
                'timestamp': confidence_score.calculation_timestamp.isoformat(),
                'raw_score': confidence_score.raw_score,
                'final_score': confidence_score.overall_score
            }
        }
    
    def update_historical_performance(self, 
                                    method: str, 
                                    accuracy: float,
                                    timeframe: Optional[str] = None):
        """Update historical performance data"""
        if method in self.historical_performance:
            # Simple exponential moving average update
            alpha = 0.1  # Learning rate
            current = self.historical_performance[method]
            self.historical_performance[method] = (1 - alpha) * current + alpha * accuracy
        
        if timeframe and timeframe in self.historical_performance['timeframe_reliability']:
            current = self.historical_performance['timeframe_reliability'][timeframe]
            self.historical_performance['timeframe_reliability'][timeframe] = (
                (1 - alpha) * current + alpha * accuracy
            )
        
        analysis_logger.info(f"Updated historical performance: {method} = {accuracy:.3f}")
    
    def set_factor_weights(self, weights: Dict[ConfidenceFactorType, float]):
        """Update factor weights"""
        # Ensure weights sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            normalized_weights = {k: v / total_weight for k, v in weights.items()}
            self.factor_weights.update(normalized_weights)
            analysis_logger.info(f"Updated confidence factor weights: {normalized_weights}")


# Create default confidence calculator instance
confidence_calculator = ConfidenceCalculator()