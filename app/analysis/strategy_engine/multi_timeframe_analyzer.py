"""
Multi-Timeframe Analysis Engine

Analyzes market across multiple timeframes to provide comprehensive signals.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from app.analysis.strategy_engine.unified_analyzer import UnifiedAnalyzer, UnifiedAnalysisResult
from app.utils.logger import analysis_logger


class TimeframeHierarchy(Enum):
    """Timeframe hierarchy for multi-timeframe analysis"""
    MONTHLY = "MN1"
    WEEKLY = "W1"  
    DAILY = "D1"
    H4 = "H4"
    H1 = "H1"
    M30 = "M30"
    M15 = "M15"
    M5 = "M5"
    M1 = "M1"


@dataclass
class TimeframeWeight:
    """Weight configuration for different timeframes"""
    timeframe: str
    weight: float
    influence_factor: float  # How much this timeframe influences lower timeframes


@dataclass
class MultiTimeframeResult:
    """Multi-timeframe analysis result"""
    symbol: str
    analysis_timestamp: datetime
    
    # Individual timeframe results
    timeframe_results: Dict[str, UnifiedAnalysisResult]
    
    # Aggregated signals
    primary_signal: str  # Overall signal based on all timeframes
    primary_confidence: float
    
    # Timeframe alignment
    alignment_score: float  # How well timeframes align
    conflicting_timeframes: List[str]
    
    # Weighted analysis
    weighted_signals: Dict[str, float]  # Signal strength by timeframe
    trend_consistency: float
    
    # Entry timing
    optimal_entry_timeframe: str
    entry_confirmation_score: float
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiTimeframeAnalyzer:
    """
    Multi-timeframe analysis engine
    
    Analyzes market conditions across multiple timeframes to generate
    comprehensive trading signals with proper timeframe hierarchy.
    """
    
    def __init__(self,
                 unified_analyzer: Optional[UnifiedAnalyzer] = None,
                 primary_timeframes: Optional[List[str]] = None):
        """
        Initialize multi-timeframe analyzer
        
        Args:
            unified_analyzer: Unified analyzer instance
            primary_timeframes: List of timeframes to analyze
        """
        self.unified_analyzer = unified_analyzer or UnifiedAnalyzer()
        
        # Default timeframe configuration
        self.primary_timeframes = primary_timeframes or ['D1', 'H4', 'H1', 'M30']
        
        # Timeframe weights (higher = more influential)
        self.timeframe_weights = {
            'MN1': TimeframeWeight('MN1', 1.0, 0.9),
            'W1': TimeframeWeight('W1', 0.9, 0.8),
            'D1': TimeframeWeight('D1', 0.8, 0.7),
            'H4': TimeframeWeight('H4', 0.7, 0.6),
            'H1': TimeframeWeight('H1', 0.6, 0.5),
            'M30': TimeframeWeight('M30', 0.5, 0.4),
            'M15': TimeframeWeight('M15', 0.4, 0.3),
            'M5': TimeframeWeight('M5', 0.3, 0.2),
            'M1': TimeframeWeight('M1', 0.2, 0.1)
        }
        
        # Analysis configuration
        self.config = {
            'alignment_threshold': 0.6,  # Minimum alignment for strong signals
            'max_conflicting_ratio': 0.3,  # Max ratio of conflicting timeframes
            'trend_consistency_weight': 0.4,
            'signal_strength_weight': 0.6,
            'require_higher_tf_confirmation': True,
            'entry_timeframe_preference': ['M30', 'M15', 'M5']  # Preferred for entry timing
        }
    
    def analyze_multiple_timeframes(self,
                                  price_data_dict: Dict[str, pd.DataFrame],
                                  symbol: str) -> MultiTimeframeResult:
        """
        Analyze market across multiple timeframes
        
        Args:
            price_data_dict: Dictionary of timeframe -> price data
            symbol: Trading symbol
            
        Returns:
            Multi-timeframe analysis result
        """
        try:
            analysis_timestamp = datetime.utcnow()
            timeframe_results = {}
            
            # Step 1: Analyze each timeframe individually
            for timeframe in self.primary_timeframes:
                if timeframe in price_data_dict:
                    price_data = price_data_dict[timeframe]
                    if len(price_data) > 50:  # Minimum data requirement
                        result = self.unified_analyzer.analyze(price_data, symbol, timeframe)
                        timeframe_results[timeframe] = result
                        
                        analysis_logger.debug(
                            f"Analyzed {symbol} {timeframe}: "
                            f"Signal={result.combined_signal}, "
                            f"Confidence={result.combined_confidence:.2f}"
                        )
            
            if not timeframe_results:
                raise ValueError("No valid timeframe data for analysis")
            
            # Step 2: Calculate timeframe alignment
            alignment_score, conflicting_timeframes = self._calculate_timeframe_alignment(
                timeframe_results
            )
            
            # Step 3: Generate weighted signals
            weighted_signals = self._calculate_weighted_signals(timeframe_results)
            
            # Step 4: Calculate trend consistency
            trend_consistency = self._calculate_trend_consistency(timeframe_results)
            
            # Step 5: Generate primary signal
            primary_signal, primary_confidence = self._generate_primary_signal(
                timeframe_results, weighted_signals, alignment_score, trend_consistency
            )
            
            # Step 6: Determine optimal entry timeframe
            optimal_entry_tf, entry_confirmation = self._determine_entry_timeframe(
                timeframe_results, primary_signal
            )
            
            # Create result
            result = MultiTimeframeResult(
                symbol=symbol,
                analysis_timestamp=analysis_timestamp,
                timeframe_results=timeframe_results,
                primary_signal=primary_signal,
                primary_confidence=primary_confidence,
                alignment_score=alignment_score,
                conflicting_timeframes=conflicting_timeframes,
                weighted_signals=weighted_signals,
                trend_consistency=trend_consistency,
                optimal_entry_timeframe=optimal_entry_tf,
                entry_confirmation_score=entry_confirmation,
                metadata={
                    'analyzed_timeframes': list(timeframe_results.keys()),
                    'config': self.config.copy(),
                    'weights_used': {tf: w.weight for tf, w in self.timeframe_weights.items()}
                }
            )
            
            analysis_logger.info(
                f"Multi-timeframe analysis complete: {symbol} - "
                f"Primary Signal: {primary_signal}, Confidence: {primary_confidence:.2f}, "
                f"Alignment: {alignment_score:.2f}, Entry TF: {optimal_entry_tf}"
            )
            
            return result
            
        except Exception as e:
            analysis_logger.error(f"Error in multi-timeframe analysis: {e}")
            raise
    
    def _calculate_timeframe_alignment(self, 
                                     timeframe_results: Dict[str, UnifiedAnalysisResult]) -> Tuple[float, List[str]]:
        """Calculate how well timeframes align with each other"""
        signals = {}
        for tf, result in timeframe_results.items():
            signals[tf] = result.combined_signal
        
        if not signals:
            return 0.0, []
        
        # Count signal distribution
        signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        weighted_signal_counts = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        
        for tf, signal in signals.items():
            signal_counts[signal] += 1
            weight = self.timeframe_weights.get(tf, TimeframeWeight(tf, 0.5, 0.5)).weight
            weighted_signal_counts[signal] += weight
        
        total_timeframes = len(signals)
        total_weight = sum(weighted_signal_counts.values())
        
        # Find dominant signal
        dominant_signal = max(weighted_signal_counts, key=weighted_signal_counts.get)
        
        # Calculate alignment score
        if total_weight > 0:
            alignment_score = weighted_signal_counts[dominant_signal] / total_weight
        else:
            alignment_score = 0.0
        
        # Find conflicting timeframes
        conflicting_timeframes = []
        for tf, signal in signals.items():
            if signal != dominant_signal and signal != 'HOLD':
                conflicting_timeframes.append(tf)
        
        return alignment_score, conflicting_timeframes
    
    def _calculate_weighted_signals(self, 
                                   timeframe_results: Dict[str, UnifiedAnalysisResult]) -> Dict[str, float]:
        """Calculate weighted signal strengths"""
        weighted_signals = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        total_weight = 0.0
        
        for tf, result in timeframe_results.items():
            weight = self.timeframe_weights.get(tf, TimeframeWeight(tf, 0.5, 0.5)).weight
            confidence = result.combined_confidence
            signal = result.combined_signal
            
            # Weight by both timeframe importance and signal confidence
            signal_strength = weight * confidence
            weighted_signals[signal] += signal_strength
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            for signal in weighted_signals:
                weighted_signals[signal] /= total_weight
        
        return weighted_signals
    
    def _calculate_trend_consistency(self, 
                                   timeframe_results: Dict[str, UnifiedAnalysisResult]) -> float:
        """Calculate trend consistency across timeframes"""
        trends = {}
        
        for tf, result in timeframe_results.items():
            # Extract trend direction from Dow Theory
            if result.dow_trend and hasattr(result.dow_trend, 'primary_trend'):
                trend = result.dow_trend.primary_trend.value.lower()
                trends[tf] = trend
        
        if not trends:
            return 0.5  # Neutral when no trend data
        
        # Calculate consistency
        trend_counts = {}
        weighted_trend_counts = {}
        
        for tf, trend in trends.items():
            trend_counts[trend] = trend_counts.get(trend, 0) + 1
            weight = self.timeframe_weights.get(tf, TimeframeWeight(tf, 0.5, 0.5)).weight
            weighted_trend_counts[trend] = weighted_trend_counts.get(trend, 0.0) + weight
        
        # Find dominant trend
        if weighted_trend_counts:
            total_weight = sum(weighted_trend_counts.values())
            max_trend_weight = max(weighted_trend_counts.values())
            consistency = max_trend_weight / total_weight if total_weight > 0 else 0.5
        else:
            consistency = 0.5
        
        return consistency
    
    def _generate_primary_signal(self,
                               timeframe_results: Dict[str, UnifiedAnalysisResult],
                               weighted_signals: Dict[str, float],
                               alignment_score: float,
                               trend_consistency: float) -> Tuple[str, float]:
        """Generate primary trading signal from multi-timeframe analysis"""
        try:
            # Find strongest weighted signal
            primary_signal = max(weighted_signals, key=weighted_signals.get)
            signal_strength = weighted_signals[primary_signal]
            
            # Calculate base confidence
            base_confidence = signal_strength
            
            # Apply alignment bonus/penalty
            alignment_factor = 0.5 + (alignment_score * 0.5)  # 0.5 to 1.0
            
            # Apply trend consistency factor
            trend_factor = 0.5 + (trend_consistency * 0.5)  # 0.5 to 1.0
            
            # Combine factors
            combined_factor = (
                alignment_factor * self.config['signal_strength_weight'] +
                trend_factor * self.config['trend_consistency_weight']
            )
            
            primary_confidence = base_confidence * combined_factor
            
            # Check for higher timeframe confirmation if required
            if self.config['require_higher_tf_confirmation']:
                higher_tf_confirmation = self._check_higher_timeframe_confirmation(
                    timeframe_results, primary_signal
                )
                if not higher_tf_confirmation:
                    primary_confidence *= 0.7  # Reduce confidence without higher TF confirmation
            
            # Apply thresholds
            if alignment_score < self.config['alignment_threshold']:
                primary_signal = "HOLD"  # Be conservative with low alignment
                primary_confidence = max(0.1, primary_confidence * 0.5)
            
            # Check conflicting timeframes ratio
            conflicting_ratio = len([tf for tf, result in timeframe_results.items() 
                                   if result.combined_signal != primary_signal and result.combined_signal != 'HOLD'])
            conflicting_ratio /= len(timeframe_results)
            
            if conflicting_ratio > self.config['max_conflicting_ratio']:
                primary_signal = "HOLD"
                primary_confidence = max(0.1, primary_confidence * 0.6)
            
            return primary_signal, min(1.0, primary_confidence)
            
        except Exception as e:
            analysis_logger.error(f"Error generating primary signal: {e}")
            return "HOLD", 0.1
    
    def _check_higher_timeframe_confirmation(self,
                                           timeframe_results: Dict[str, UnifiedAnalysisResult],
                                           primary_signal: str) -> bool:
        """Check if higher timeframes confirm the primary signal"""
        # Define timeframe hierarchy order
        hierarchy_order = ['MN1', 'W1', 'D1', 'H4', 'H1', 'M30', 'M15', 'M5', 'M1']
        
        # Get available timeframes in order
        available_tfs = [tf for tf in hierarchy_order if tf in timeframe_results]
        
        if len(available_tfs) < 2:
            return True  # Can't check with only one timeframe
        
        # Check if at least one higher timeframe confirms
        for i, tf in enumerate(available_tfs[:-1]):  # Exclude the lowest timeframe
            result = timeframe_results[tf]
            if result.combined_signal == primary_signal:
                return True
        
        return False
    
    def _determine_entry_timeframe(self,
                                 timeframe_results: Dict[str, UnifiedAnalysisResult],
                                 primary_signal: str) -> Tuple[str, float]:
        """Determine optimal timeframe for trade entry"""
        if primary_signal == "HOLD":
            return "M30", 0.0  # Default when no signal
        
        # Check preferred entry timeframes
        for tf in self.config['entry_timeframe_preference']:
            if tf in timeframe_results:
                result = timeframe_results[tf]
                
                # Check if this timeframe agrees with primary signal
                if result.combined_signal == primary_signal:
                    # Calculate entry confirmation score
                    confirmation_factors = []
                    
                    # Factor 1: Signal confidence
                    confirmation_factors.append(result.combined_confidence)
                    
                    # Factor 2: Agreement with primary signal
                    confirmation_factors.append(1.0 if result.combined_signal == primary_signal else 0.0)
                    
                    # Factor 3: Elliott Wave predictions (if available)
                    if result.elliott_predictions:
                        avg_prediction_conf = np.mean([
                            p.get('confidence', 0) for p in result.elliott_predictions
                        ])
                        confirmation_factors.append(avg_prediction_conf)
                    
                    confirmation_score = np.mean(confirmation_factors)
                    
                    if confirmation_score > 0.4:  # Minimum threshold for entry
                        return tf, confirmation_score
        
        # Fallback to any timeframe that agrees with primary signal
        for tf, result in timeframe_results.items():
            if result.combined_signal == primary_signal:
                return tf, result.combined_confidence
        
        return list(timeframe_results.keys())[0], 0.1  # Final fallback
    
    def get_timeframe_summary(self, result: MultiTimeframeResult) -> Dict[str, Any]:
        """Get summary of multi-timeframe analysis"""
        summary = {
            'symbol': result.symbol,
            'analysis_timestamp': result.analysis_timestamp.isoformat(),
            'primary_signal': result.primary_signal,
            'primary_confidence': result.primary_confidence,
            'alignment_score': result.alignment_score,
            'trend_consistency': result.trend_consistency,
            'optimal_entry_timeframe': result.optimal_entry_timeframe,
            'entry_confirmation_score': result.entry_confirmation_score,
            'timeframe_breakdown': {}
        }
        
        # Add individual timeframe details
        for tf, tf_result in result.timeframe_results.items():
            summary['timeframe_breakdown'][tf] = {
                'signal': tf_result.combined_signal,
                'confidence': tf_result.combined_confidence,
                'agreement_score': tf_result.agreement_score,
                'dow_trend': tf_result.dow_trend.primary_trend.value if tf_result.dow_trend else None,
                'elliott_patterns': len(tf_result.elliott_patterns),
                'elliott_predictions': len(tf_result.elliott_predictions)
            }
        
        # Add conflict analysis
        if result.conflicting_timeframes:
            summary['conflicts'] = {
                'conflicting_timeframes': result.conflicting_timeframes,
                'conflict_ratio': len(result.conflicting_timeframes) / len(result.timeframe_results)
            }
        
        return summary
    
    def set_timeframe_weights(self, weights: Dict[str, float]):
        """Update timeframe weights"""
        for tf, weight in weights.items():
            if tf in self.timeframe_weights:
                self.timeframe_weights[tf].weight = weight
        
        analysis_logger.info(f"Updated timeframe weights: {weights}")
    
    def set_config(self, config: Dict[str, Any]):
        """Update analysis configuration"""
        self.config.update(config)
        analysis_logger.info(f"Updated multi-timeframe config: {config}")


# Create default multi-timeframe analyzer instance
multi_timeframe_analyzer = MultiTimeframeAnalyzer()