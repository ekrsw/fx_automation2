"""
Unified Analysis Engine - Integrates Dow Theory and Elliott Wave Analysis
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field

from app.analysis.dow_theory import DowTheoryAnalyzer, TrendAnalysis
from app.analysis.elliott_wave import ElliottWaveAnalyzer, Wave, WavePattern, WaveType
from app.analysis.indicators.swing_detector import SwingPoint, SwingType
from app.utils.logger import analysis_logger


@dataclass
class UnifiedAnalysisResult:
    """Unified analysis result combining Dow Theory and Elliott Wave"""
    symbol: str
    timeframe: str
    timestamp: datetime
    
    # Dow Theory results
    dow_trend: TrendAnalysis
    dow_confidence: float
    
    # Elliott Wave results
    elliott_waves: List[Wave]
    elliott_patterns: List[WavePattern]
    elliott_predictions: List[Dict[str, Any]]
    elliott_confidence: float
    
    # Unified results
    combined_signal: str  # BUY, SELL, HOLD
    combined_confidence: float
    agreement_score: float  # How well Dow Theory and Elliott Wave agree
    
    # Supporting data
    swing_points: List[SwingPoint]
    price_targets: Dict[str, float]
    risk_levels: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedAnalyzer:
    """
    Unified analyzer combining Dow Theory and Elliott Wave analysis
    
    Provides comprehensive market analysis by integrating both methodologies
    and generating unified trading signals with confidence scoring.
    """
    
    def __init__(self,
                 dow_analyzer: Optional[DowTheoryAnalyzer] = None,
                 elliott_analyzer: Optional[ElliottWaveAnalyzer] = None,
                 agreement_threshold: float = 0.6,
                 min_confidence: float = 0.4):
        """
        Initialize unified analyzer
        
        Args:
            dow_analyzer: Dow Theory analyzer instance
            elliott_analyzer: Elliott Wave analyzer instance
            agreement_threshold: Minimum agreement score for strong signals
            min_confidence: Minimum confidence for signal generation
        """
        self.dow_analyzer = dow_analyzer or DowTheoryAnalyzer()
        self.elliott_analyzer = elliott_analyzer or ElliottWaveAnalyzer()
        self.agreement_threshold = agreement_threshold
        self.min_confidence = min_confidence
        
        # Analysis configuration
        self.analysis_config = {
            'enable_multi_timeframe': True,
            'require_both_confirmations': True,
            'weight_dow_theory': 0.4,
            'weight_elliott_wave': 0.6,
            'enable_prediction_targets': True,
            'max_lookback_periods': 500
        }
    
    def analyze(self, 
                price_data: pd.DataFrame,
                symbol: str,
                timeframe: str = 'H1') -> UnifiedAnalysisResult:
        """
        Perform unified analysis combining both methodologies
        
        Args:
            price_data: OHLCV price data
            symbol: Trading symbol
            timeframe: Analysis timeframe
            
        Returns:
            Unified analysis result
        """
        try:
            timestamp = datetime.utcnow()
            
            # Step 1: Generate swing points (shared by both analyzers)
            swing_points = self._generate_swing_points(price_data)
            
            # Step 2: Dow Theory analysis
            dow_result = self._analyze_dow_theory(price_data, swing_points)
            
            # Step 3: Elliott Wave analysis
            elliott_result = self._analyze_elliott_wave(swing_points)
            
            # Step 4: Calculate agreement and generate unified signal
            agreement_score = self._calculate_agreement(dow_result, elliott_result)
            combined_signal, combined_confidence = self._generate_unified_signal(
                dow_result, elliott_result, agreement_score
            )
            
            # Step 5: Calculate price targets and risk levels
            price_targets = self._calculate_unified_targets(dow_result, elliott_result)
            risk_levels = self._calculate_unified_risk_levels(dow_result, elliott_result)
            
            # Create unified result
            result = UnifiedAnalysisResult(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=timestamp,
                dow_trend=dow_result['trend_analysis'],
                dow_confidence=dow_result['confidence'],
                elliott_waves=elliott_result['waves'],
                elliott_patterns=elliott_result['patterns'],
                elliott_predictions=elliott_result['predictions'],
                elliott_confidence=elliott_result['confidence'],
                combined_signal=combined_signal,
                combined_confidence=combined_confidence,
                agreement_score=agreement_score,
                swing_points=swing_points,
                price_targets=price_targets,
                risk_levels=risk_levels,
                metadata={
                    'dow_details': dow_result.get('details', {}),
                    'elliott_details': elliott_result.get('details', {}),
                    'analysis_config': self.analysis_config.copy()
                }
            )
            
            analysis_logger.info(
                f"Unified analysis complete: {symbol} {timeframe} - "
                f"Signal: {combined_signal}, Confidence: {combined_confidence:.2f}, "
                f"Agreement: {agreement_score:.2f}"
            )
            
            return result
            
        except Exception as e:
            analysis_logger.error(f"Error in unified analysis: {e}")
            raise
    
    def _generate_swing_points(self, price_data: pd.DataFrame) -> List[SwingPoint]:
        """Generate swing points from price data"""
        # Use the swing detector from Dow Theory analyzer
        return self.dow_analyzer.swing_detector.detect_swings(price_data)
    
    def _analyze_dow_theory(self, 
                           price_data: pd.DataFrame, 
                           swing_points: List[SwingPoint]) -> Dict[str, Any]:
        """Perform Dow Theory analysis"""
        try:
            # Get trend analysis
            trend_analysis = self.dow_analyzer.analyze_trend(price_data)
            
            # Calculate confidence based on trend strength and confirmation
            confidence_factors = []
            
            # Factor 1: Trend strength
            if trend_analysis.strength.value in ['very_strong', 'strong']:
                confidence_factors.append(0.9)
            elif trend_analysis.strength.value == 'moderate':
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.4)
            
            # Factor 2: Volume confirmation
            if hasattr(trend_analysis, 'volume_confirmation') and trend_analysis.volume_confirmation:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.3)
            
            # Factor 3: Pattern quality
            if len(swing_points) >= 4:  # Sufficient swing points
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.4)
            
            dow_confidence = np.mean(confidence_factors)
            
            return {
                'trend_analysis': trend_analysis,
                'confidence': dow_confidence,
                'swing_points': swing_points,
                'details': {
                    'trend_direction': trend_analysis.direction.value,
                    'trend_strength': trend_analysis.strength.value,
                    'swing_point_count': len(swing_points),
                    'volume_confirmation': getattr(trend_analysis, 'volume_confirmation', False)
                }
            }
            
        except Exception as e:
            analysis_logger.error(f"Error in Dow Theory analysis: {e}")
            return {
                'trend_analysis': None,
                'confidence': 0.0,
                'swing_points': swing_points,
                'details': {'error': str(e)}
            }
    
    def _analyze_elliott_wave(self, swing_points: List[SwingPoint]) -> Dict[str, Any]:
        """Perform Elliott Wave analysis"""
        try:
            # Identify waves
            waves = self.elliott_analyzer.identify_waves(swing_points)
            
            # Find patterns
            patterns = self.elliott_analyzer.find_patterns(waves)
            
            # Find incomplete patterns and generate predictions
            incomplete_patterns = self.elliott_analyzer.find_incomplete_patterns(waves)
            predictions = []
            
            if incomplete_patterns and swing_points:
                current_price = swing_points[-1].price
                from app.analysis.elliott_wave import wave_predictor
                
                for pattern in incomplete_patterns[:3]:  # Top 3 patterns
                    prediction = wave_predictor.predict_next_wave(pattern, current_price)
                    if prediction.get('confidence', 0) > 0.3:
                        predictions.append(prediction)
            
            # Calculate overall Elliott Wave confidence
            pattern_confidences = [p.confidence for p in patterns if p.confidence > 0]
            if pattern_confidences:
                elliott_confidence = np.mean(pattern_confidences)
            elif predictions:
                elliott_confidence = np.mean([p.get('confidence', 0) for p in predictions])
            else:
                elliott_confidence = 0.0
            
            return {
                'waves': waves,
                'patterns': patterns,
                'predictions': predictions,
                'confidence': elliott_confidence,
                'details': {
                    'wave_count': len(waves),
                    'pattern_count': len(patterns),
                    'prediction_count': len(predictions),
                    'avg_pattern_confidence': elliott_confidence
                }
            }
            
        except Exception as e:
            analysis_logger.error(f"Error in Elliott Wave analysis: {e}")
            return {
                'waves': [],
                'patterns': [],
                'predictions': [],
                'confidence': 0.0,
                'details': {'error': str(e)}
            }
    
    def _calculate_agreement(self, 
                           dow_result: Dict[str, Any], 
                           elliott_result: Dict[str, Any]) -> float:
        """Calculate agreement score between Dow Theory and Elliott Wave"""
        agreement_factors = []
        
        try:
            # Factor 1: Directional agreement
            dow_trend = dow_result.get('trend_analysis')
            elliott_patterns = elliott_result.get('patterns', [])
            
            if dow_trend and elliott_patterns:
                # Check if Elliott Wave patterns align with Dow Theory trend
                dow_direction = dow_trend.direction.value.lower()
                
                # Count patterns that agree with Dow trend
                agreeing_patterns = 0
                for pattern in elliott_patterns:
                    if pattern.waves:
                        # Check overall pattern direction
                        pattern_direction = self._get_pattern_direction(pattern)
                        if pattern_direction == dow_direction:
                            agreeing_patterns += 1
                
                if len(elliott_patterns) > 0:
                    directional_agreement = agreeing_patterns / len(elliott_patterns)
                    agreement_factors.append(directional_agreement)
            
            # Factor 2: Confidence correlation
            dow_conf = dow_result.get('confidence', 0)
            elliott_conf = elliott_result.get('confidence', 0)
            
            if dow_conf > 0 and elliott_conf > 0:
                # Higher agreement when both confidences are similar and high
                conf_similarity = 1.0 - abs(dow_conf - elliott_conf)
                conf_average = (dow_conf + elliott_conf) / 2
                confidence_agreement = conf_similarity * conf_average
                agreement_factors.append(confidence_agreement)
            
            # Factor 3: Swing point consistency
            # Both methods should agree on major swing points
            if len(agreement_factors) == 0:
                agreement_factors.append(0.5)  # Neutral when no clear agreement
            
            return np.mean(agreement_factors)
            
        except Exception as e:
            analysis_logger.error(f"Error calculating agreement: {e}")
            return 0.5  # Neutral agreement on error
    
    def _get_pattern_direction(self, pattern: WavePattern) -> str:
        """Determine overall direction of an Elliott Wave pattern"""
        if not pattern.waves:
            return "neutral"
        
        start_price = pattern.waves[0].start_point.price
        end_price = pattern.waves[-1].end_point.price
        
        if end_price > start_price:
            return "bullish"
        elif end_price < start_price:
            return "bearish"
        else:
            return "neutral"
    
    def _generate_unified_signal(self, 
                               dow_result: Dict[str, Any], 
                               elliott_result: Dict[str, Any],
                               agreement_score: float) -> Tuple[str, float]:
        """Generate unified trading signal"""
        try:
            # Extract individual signals
            dow_signal = self._get_dow_signal(dow_result)
            elliott_signal = self._get_elliott_signal(elliott_result)
            
            # Calculate weighted confidence
            dow_weight = self.analysis_config['weight_dow_theory']
            elliott_weight = self.analysis_config['weight_elliott_wave']
            
            dow_conf = dow_result.get('confidence', 0)
            elliott_conf = elliott_result.get('confidence', 0)
            
            weighted_confidence = (dow_conf * dow_weight + elliott_conf * elliott_weight)
            
            # Apply agreement bonus/penalty
            agreement_multiplier = 0.5 + (agreement_score * 0.5)  # 0.5 to 1.0
            final_confidence = weighted_confidence * agreement_multiplier
            
            # Generate unified signal
            if agreement_score >= self.agreement_threshold:
                # High agreement - use stronger signal
                if dow_signal == elliott_signal and dow_signal != "HOLD":
                    unified_signal = dow_signal
                elif dow_signal != "HOLD" and elliott_signal == "HOLD":
                    unified_signal = dow_signal
                elif elliott_signal != "HOLD" and dow_signal == "HOLD":
                    unified_signal = elliott_signal
                else:
                    unified_signal = "HOLD"  # Conflicting signals
            else:
                # Low agreement - be conservative
                if dow_signal == elliott_signal:
                    unified_signal = dow_signal
                else:
                    unified_signal = "HOLD"  # Conflicting signals with low agreement
            
            # Apply minimum confidence filter
            if final_confidence < self.min_confidence:
                unified_signal = "HOLD"
                final_confidence = max(0.1, final_confidence)  # Minimum confidence floor
            
            return unified_signal, min(1.0, final_confidence)
            
        except Exception as e:
            analysis_logger.error(f"Error generating unified signal: {e}")
            return "HOLD", 0.1
    
    def _get_dow_signal(self, dow_result: Dict[str, Any]) -> str:
        """Extract trading signal from Dow Theory analysis"""
        trend_analysis = dow_result.get('trend_analysis')
        if not trend_analysis:
            return "HOLD"
        
        if trend_analysis.direction.value.lower() in ['uptrend', 'bullish']:
            return "BUY"
        elif trend_analysis.direction.value.lower() in ['downtrend', 'bearish']:
            return "SELL"
        else:
            return "HOLD"
    
    def _get_elliott_signal(self, elliott_result: Dict[str, Any]) -> str:
        """Extract trading signal from Elliott Wave analysis"""
        patterns = elliott_result.get('patterns', [])
        predictions = elliott_result.get('predictions', [])
        
        # Check for strong impulse patterns
        for pattern in patterns:
            if pattern.pattern_type == WaveType.IMPULSE and pattern.confidence > 0.6:
                if pattern.waves:
                    # Check if we're in a bullish or bearish impulse
                    start_price = pattern.waves[0].start_point.price
                    end_price = pattern.waves[-1].end_point.price
                    
                    if end_price > start_price:
                        return "BUY"
                    else:
                        return "SELL"
        
        # Check predictions for next wave direction
        for prediction in predictions:
            if prediction.get('confidence', 0) > 0.5:
                scenarios = prediction.get('scenarios', [])
                if scenarios:
                    # Use most probable scenario
                    best_scenario = max(scenarios, key=lambda x: x.get('probability', 0))
                    target_price = best_scenario.get('target_price')
                    current_price = prediction.get('current_price')
                    
                    if target_price and current_price:
                        if target_price > current_price:
                            return "BUY"
                        else:
                            return "SELL"
        
        return "HOLD"
    
    def _calculate_unified_targets(self, 
                                 dow_result: Dict[str, Any], 
                                 elliott_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate unified price targets"""
        targets = {}
        
        try:
            # Elliott Wave targets (primary)
            predictions = elliott_result.get('predictions', [])
            for i, prediction in enumerate(predictions):
                primary_targets = prediction.get('primary_targets', {})
                for target_type, price in primary_targets.items():
                    key = f"elliott_{target_type}_{i+1}"
                    targets[key] = price
            
            # Dow Theory targets (support/resistance levels)
            trend_analysis = dow_result.get('trend_analysis')
            if trend_analysis and hasattr(trend_analysis, 'support_levels'):
                for i, level in enumerate(trend_analysis.support_levels[:3]):
                    targets[f"dow_support_{i+1}"] = level
            
            if trend_analysis and hasattr(trend_analysis, 'resistance_levels'):
                for i, level in enumerate(trend_analysis.resistance_levels[:3]):
                    targets[f"dow_resistance_{i+1}"] = level
            
        except Exception as e:
            analysis_logger.error(f"Error calculating unified targets: {e}")
        
        return targets
    
    def _calculate_unified_risk_levels(self, 
                                     dow_result: Dict[str, Any], 
                                     elliott_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate unified risk levels"""
        risk_levels = {}
        
        try:
            # Elliott Wave risk levels
            predictions = elliott_result.get('predictions', [])
            for i, prediction in enumerate(predictions):
                pred_risk_levels = prediction.get('risk_levels', {})
                for risk_type, price in pred_risk_levels.items():
                    key = f"elliott_{risk_type}_{i+1}"
                    risk_levels[key] = price
            
            # Dow Theory risk levels (swing points)
            swing_points = dow_result.get('swing_points', [])
            if len(swing_points) >= 2:
                # Recent swing high/low as risk levels
                recent_swings = swing_points[-3:]
                
                highs = [sp.price for sp in recent_swings if sp.swing_type == SwingType.HIGH]
                lows = [sp.price for sp in recent_swings if sp.swing_type == SwingType.LOW]
                
                if highs:
                    risk_levels['dow_recent_high'] = max(highs)
                if lows:
                    risk_levels['dow_recent_low'] = min(lows)
            
        except Exception as e:
            analysis_logger.error(f"Error calculating unified risk levels: {e}")
        
        return risk_levels
    
    def set_analysis_config(self, config: Dict[str, Any]):
        """Update analysis configuration"""
        self.analysis_config.update(config)
        analysis_logger.info(f"Updated unified analysis config: {config}")
    
    def get_analysis_summary(self, result: UnifiedAnalysisResult) -> Dict[str, Any]:
        """Get comprehensive analysis summary"""
        return {
            'symbol': result.symbol,
            'timeframe': result.timeframe,
            'timestamp': result.timestamp.isoformat(),
            'unified_signal': {
                'signal': result.combined_signal,
                'confidence': result.combined_confidence,
                'agreement_score': result.agreement_score
            },
            'dow_theory': {
                'trend': result.dow_trend.direction.value if result.dow_trend else None,
                'confidence': result.dow_confidence,
                'strength': result.dow_trend.strength.value if result.dow_trend else None
            },
            'elliott_wave': {
                'wave_count': len(result.elliott_waves),
                'pattern_count': len(result.elliott_patterns),
                'prediction_count': len(result.elliott_predictions),
                'confidence': result.elliott_confidence
            },
            'targets': result.price_targets,
            'risk_levels': result.risk_levels,
            'swing_points': len(result.swing_points)
        }


# Create default unified analyzer instance
unified_analyzer = UnifiedAnalyzer()