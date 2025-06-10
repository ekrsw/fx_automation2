"""
Signal Generation Engine

Generates actionable trading signals from unified analysis results.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from app.analysis.strategy_engine.unified_analyzer import UnifiedAnalysisResult
from app.analysis.strategy_engine.multi_timeframe_analyzer import MultiTimeframeResult
from app.utils.logger import analysis_logger


class SignalType(Enum):
    """Types of trading signals"""
    MARKET_BUY = "MARKET_BUY"
    MARKET_SELL = "MARKET_SELL" 
    LIMIT_BUY = "LIMIT_BUY"
    LIMIT_SELL = "LIMIT_SELL"
    STOP_BUY = "STOP_BUY"
    STOP_SELL = "STOP_SELL"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"
    HOLD = "HOLD"


class SignalUrgency(Enum):
    """Signal urgency levels"""
    IMMEDIATE = "IMMEDIATE"  # Execute immediately
    HIGH = "HIGH"           # Execute within 5 minutes
    MEDIUM = "MEDIUM"       # Execute within 15 minutes
    LOW = "LOW"             # Execute within 1 hour
    WATCH = "WATCH"         # Monitor but don't execute


@dataclass
class TradingSignal:
    """Complete trading signal with all necessary information"""
    signal_id: str
    symbol: str
    signal_type: SignalType
    urgency: SignalUrgency
    
    # Price levels
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None
    take_profit_3: Optional[float] = None
    
    # Risk management
    risk_reward_ratio: Optional[float] = None
    max_risk_pct: float = 2.0  # Maximum risk percentage
    position_size_pct: Optional[float] = None
    
    # Signal quality
    confidence: float = 0.0
    strength: float = 0.0  # Signal strength (0-1)
    quality_score: float = 0.0  # Overall signal quality
    
    # Analysis details
    primary_timeframe: str = "H1"
    supporting_timeframes: List[str] = field(default_factory=list)
    dow_confirmation: bool = False
    elliott_confirmation: bool = False
    
    # Timing
    generated_at: datetime = field(default_factory=datetime.utcnow)
    valid_until: Optional[datetime] = None
    
    # Additional context
    analysis_summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SignalGenerator:
    """
    Advanced signal generation engine
    
    Converts analysis results into actionable trading signals with proper
    risk management, timing, and quality assessment.
    """
    
    def __init__(self,
                 default_risk_pct: float = 2.0,
                 min_risk_reward_ratio: float = 1.5,
                 signal_validity_hours: int = 4):
        """
        Initialize signal generator
        
        Args:
            default_risk_pct: Default risk percentage per trade
            min_risk_reward_ratio: Minimum acceptable risk/reward ratio
            signal_validity_hours: Hours until signal expires
        """
        self.default_risk_pct = default_risk_pct
        self.min_risk_reward_ratio = min_risk_reward_ratio
        self.signal_validity_hours = signal_validity_hours
        
        # Signal generation configuration
        self.config = {
            'min_confidence': 0.6,           # Minimum confidence for signal generation
            'min_strength': 0.5,             # Minimum signal strength
            'require_both_confirmations': False,  # Require both Dow + Elliott confirmation
            'enable_limit_orders': True,      # Generate limit orders for better entries
            'enable_multiple_targets': True,  # Multiple take profit levels
            'urgency_confidence_thresholds': {
                'IMMEDIATE': 0.9,
                'HIGH': 0.75,
                'MEDIUM': 0.6,
                'LOW': 0.45,
                'WATCH': 0.3
            }
        }
        
        # Risk management settings
        self.risk_config = {
            'max_risk_per_trade': 0.02,      # 2% maximum risk
            'conservative_risk': 0.01,       # 1% for lower confidence signals
            'aggressive_risk': 0.03,         # 3% for high confidence signals
            'default_stop_loss_atr': 2.0,    # Default stop loss in ATR multiples
            'trailing_stop_activation': 0.5  # When to activate trailing stops (% of target)
        }
    
    def generate_signal_from_unified(self,
                                   unified_result: UnifiedAnalysisResult,
                                   current_price: float) -> Optional[TradingSignal]:
        """
        Generate trading signal from unified analysis result
        
        Args:
            unified_result: Unified analysis result
            current_price: Current market price
            
        Returns:
            Trading signal or None if no signal generated
        """
        try:
            # Check if signal meets minimum requirements
            if not self._meets_signal_requirements(unified_result):
                return None
            
            # Generate base signal
            signal = self._create_base_signal(unified_result, current_price)
            
            # Calculate risk management levels
            self._calculate_risk_levels(signal, unified_result, current_price)
            
            # Determine signal urgency
            signal.urgency = self._determine_urgency(unified_result)
            
            # Calculate quality metrics
            self._calculate_quality_metrics(signal, unified_result)
            
            # Set validity period
            signal.valid_until = signal.generated_at + timedelta(hours=self.signal_validity_hours)
            
            # Add analysis summary
            signal.analysis_summary = self._create_analysis_summary(unified_result)
            
            analysis_logger.info(
                f"Generated signal: {signal.symbol} {signal.signal_type.value} "
                f"@ {signal.entry_price:.5f}, Confidence: {signal.confidence:.2f}, "
                f"RR: {signal.risk_reward_ratio:.2f}"
            )
            
            return signal
            
        except Exception as e:
            analysis_logger.error(f"Error generating signal from unified result: {e}")
            return None
    
    def generate_signal_from_multi_timeframe(self,
                                           mtf_result: MultiTimeframeResult,
                                           current_price: float) -> Optional[TradingSignal]:
        """
        Generate trading signal from multi-timeframe analysis
        
        Args:
            mtf_result: Multi-timeframe analysis result
            current_price: Current market price
            
        Returns:
            Trading signal or None if no signal generated
        """
        try:
            # Check if multi-timeframe analysis supports signal generation
            if not self._meets_mtf_signal_requirements(mtf_result):
                return None
            
            # Use the optimal entry timeframe result for signal generation
            entry_tf = mtf_result.optimal_entry_timeframe
            if entry_tf not in mtf_result.timeframe_results:
                return None
            
            unified_result = mtf_result.timeframe_results[entry_tf]
            
            # Generate base signal
            signal = self._create_base_signal(unified_result, current_price)
            
            # Enhance with multi-timeframe information
            self._enhance_with_mtf_data(signal, mtf_result)
            
            # Calculate risk management levels
            self._calculate_risk_levels(signal, unified_result, current_price)
            
            # Determine urgency based on multi-timeframe alignment
            signal.urgency = self._determine_mtf_urgency(mtf_result)
            
            # Calculate enhanced quality metrics
            self._calculate_mtf_quality_metrics(signal, mtf_result)
            
            # Set validity period (longer for multi-timeframe signals)
            signal.valid_until = signal.generated_at + timedelta(
                hours=self.signal_validity_hours * 2
            )
            
            # Add comprehensive analysis summary
            signal.analysis_summary = self._create_mtf_analysis_summary(mtf_result)
            
            analysis_logger.info(
                f"Generated MTF signal: {signal.symbol} {signal.signal_type.value} "
                f"@ {signal.entry_price:.5f}, Confidence: {signal.confidence:.2f}, "
                f"Alignment: {mtf_result.alignment_score:.2f}"
            )
            
            return signal
            
        except Exception as e:
            analysis_logger.error(f"Error generating signal from MTF result: {e}")
            return None
    
    def _meets_signal_requirements(self, result: UnifiedAnalysisResult) -> bool:
        """Check if unified result meets signal generation requirements"""
        # Basic requirements
        if result.combined_signal == "HOLD":
            return False
        
        if result.combined_confidence < self.config['min_confidence']:
            return False
        
        # Check for required confirmations
        if self.config['require_both_confirmations']:
            if not (result.dow_confidence > 0.4 and result.elliott_confidence > 0.4):
                return False
        
        return True
    
    def _meets_mtf_signal_requirements(self, mtf_result: MultiTimeframeResult) -> bool:
        """Check if multi-timeframe result meets signal generation requirements"""
        # Primary signal check
        if mtf_result.primary_signal == "HOLD":
            return False
        
        if mtf_result.primary_confidence < self.config['min_confidence']:
            return False
        
        # Alignment check
        if mtf_result.alignment_score < 0.5:  # Minimum alignment for MTF signals
            return False
        
        # Entry timeframe confirmation
        if mtf_result.entry_confirmation_score < 0.4:
            return False
        
        return True
    
    def _create_base_signal(self, 
                          unified_result: UnifiedAnalysisResult,
                          current_price: float) -> TradingSignal:
        """Create base trading signal"""
        signal_id = f"{unified_result.symbol}_{int(datetime.utcnow().timestamp())}"
        
        # Determine signal type
        if unified_result.combined_signal == "BUY":
            if self.config['enable_limit_orders'] and self._should_use_limit_order(unified_result):
                signal_type = SignalType.LIMIT_BUY
            else:
                signal_type = SignalType.MARKET_BUY
        elif unified_result.combined_signal == "SELL":
            if self.config['enable_limit_orders'] and self._should_use_limit_order(unified_result):
                signal_type = SignalType.LIMIT_SELL
            else:
                signal_type = SignalType.MARKET_SELL
        else:
            signal_type = SignalType.HOLD
        
        # Create signal
        signal = TradingSignal(
            signal_id=signal_id,
            symbol=unified_result.symbol,
            signal_type=signal_type,
            urgency=SignalUrgency.MEDIUM,  # Will be updated
            entry_price=current_price,  # Will be refined
            confidence=unified_result.combined_confidence,
            primary_timeframe=unified_result.timeframe if hasattr(unified_result, 'timeframe') else 'H1',
            dow_confirmation=unified_result.dow_confidence > 0.5,
            elliott_confirmation=unified_result.elliott_confidence > 0.5
        )
        
        return signal
    
    def _should_use_limit_order(self, result: UnifiedAnalysisResult) -> bool:
        """Determine if limit order should be used instead of market order"""
        # Use limit orders when:
        # 1. Elliott Wave suggests a pullback before continuation
        # 2. Multiple timeframes suggest better entry levels
        # 3. Market is not extremely urgent
        
        if result.elliott_predictions:
            for prediction in result.elliott_predictions:
                # Check if current pattern suggests a retracement
                scenarios = prediction.get('scenarios', [])
                for scenario in scenarios:
                    if 'retracement' in scenario.get('type', ''):
                        return True
        
        return False
    
    def _calculate_risk_levels(self, 
                             signal: TradingSignal,
                             unified_result: UnifiedAnalysisResult,
                             current_price: float):
        """Calculate stop loss and take profit levels"""
        try:
            # Get risk levels from analysis
            risk_levels = unified_result.risk_levels
            price_targets = unified_result.price_targets
            
            # Calculate stop loss
            signal.stop_loss = self._calculate_stop_loss(
                signal, unified_result, current_price, risk_levels
            )
            
            # Calculate take profit levels
            if self.config['enable_multiple_targets']:
                tp_levels = self._calculate_multiple_take_profits(
                    signal, unified_result, current_price, price_targets
                )
                signal.take_profit_1 = tp_levels.get('tp1')
                signal.take_profit_2 = tp_levels.get('tp2')
                signal.take_profit_3 = tp_levels.get('tp3')
            else:
                signal.take_profit_1 = self._calculate_single_take_profit(
                    signal, unified_result, current_price, price_targets
                )
            
            # Calculate risk/reward ratio
            if signal.stop_loss and signal.take_profit_1:
                risk = abs(current_price - signal.stop_loss)
                reward = abs(signal.take_profit_1 - current_price)
                signal.risk_reward_ratio = reward / risk if risk > 0 else 0.0
            
            # Calculate position size
            signal.position_size_pct = self._calculate_position_size(signal, unified_result)
            
        except Exception as e:
            analysis_logger.error(f"Error calculating risk levels: {e}")
    
    def _calculate_stop_loss(self,
                           signal: TradingSignal,
                           unified_result: UnifiedAnalysisResult,
                           current_price: float,
                           risk_levels: Dict[str, float]) -> Optional[float]:
        """Calculate stop loss level"""
        # Priority order for stop loss calculation:
        # 1. Elliott Wave risk levels
        # 2. Dow Theory swing points
        # 3. Default ATR-based stop
        
        candidates = []
        
        # Elliott Wave risk levels
        for key, level in risk_levels.items():
            if 'elliott' in key.lower() and level:
                candidates.append(level)
        
        # Dow Theory risk levels  
        for key, level in risk_levels.items():
            if 'dow' in key.lower() and level:
                candidates.append(level)
        
        if candidates:
            if signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
                # For buy signals, use highest low as stop loss
                return max([c for c in candidates if c < current_price], default=None)
            else:
                # For sell signals, use lowest high as stop loss
                return min([c for c in candidates if c > current_price], default=None)
        
        # Default ATR-based stop loss (simplified - would need ATR calculation)
        default_stop_distance = current_price * 0.01  # 1% default
        
        if signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
            return current_price - default_stop_distance
        else:
            return current_price + default_stop_distance
    
    def _calculate_multiple_take_profits(self,
                                       signal: TradingSignal,
                                       unified_result: UnifiedAnalysisResult,
                                       current_price: float,
                                       price_targets: Dict[str, float]) -> Dict[str, float]:
        """Calculate multiple take profit levels"""
        tp_levels = {}
        
        # Extract targets from Elliott Wave and Dow Theory
        targets = []
        for key, target in price_targets.items():
            if target and target != current_price:
                targets.append(target)
        
        # Sort targets by distance from current price
        if signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
            valid_targets = [t for t in targets if t > current_price]
            valid_targets.sort()
        else:
            valid_targets = [t for t in targets if t < current_price]
            valid_targets.sort(reverse=True)
        
        # Assign to TP levels
        if len(valid_targets) >= 1:
            tp_levels['tp1'] = valid_targets[0]
        if len(valid_targets) >= 2:
            tp_levels['tp2'] = valid_targets[1]
        if len(valid_targets) >= 3:
            tp_levels['tp3'] = valid_targets[2]
        
        # Fill missing levels with calculated targets
        if 'tp1' not in tp_levels:
            # Default first target
            distance = current_price * 0.02  # 2% default
            if signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
                tp_levels['tp1'] = current_price + distance
            else:
                tp_levels['tp1'] = current_price - distance
        
        return tp_levels
    
    def _calculate_single_take_profit(self,
                                    signal: TradingSignal,
                                    unified_result: UnifiedAnalysisResult,
                                    current_price: float,
                                    price_targets: Dict[str, float]) -> Optional[float]:
        """Calculate single take profit level"""
        # Use the most probable Elliott Wave target or best Dow Theory level
        best_target = None
        
        for key, target in price_targets.items():
            if target:
                if signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
                    if target > current_price:
                        if not best_target or target < best_target:  # Closest target
                            best_target = target
                else:
                    if target < current_price:
                        if not best_target or target > best_target:  # Closest target
                            best_target = target
        
        return best_target
    
    def _calculate_position_size(self,
                               signal: TradingSignal,
                               unified_result: UnifiedAnalysisResult) -> float:
        """Calculate position size as percentage of account"""
        # Base risk on confidence and signal quality
        confidence = unified_result.combined_confidence
        
        if confidence > 0.8:
            risk_pct = self.risk_config['aggressive_risk']
        elif confidence > 0.6:
            risk_pct = self.default_risk_pct
        else:
            risk_pct = self.risk_config['conservative_risk']
        
        # Adjust for agreement score
        risk_pct *= unified_result.agreement_score
        
        return min(risk_pct, self.risk_config['max_risk_per_trade'])
    
    def _determine_urgency(self, result: UnifiedAnalysisResult) -> SignalUrgency:
        """Determine signal urgency based on analysis confidence"""
        confidence = result.combined_confidence
        
        for urgency, threshold in self.config['urgency_confidence_thresholds'].items():
            if confidence >= threshold:
                return SignalUrgency(urgency)
        
        return SignalUrgency.WATCH
    
    def _determine_mtf_urgency(self, mtf_result: MultiTimeframeResult) -> SignalUrgency:
        """Determine urgency for multi-timeframe signals"""
        # Consider both confidence and alignment
        combined_score = (mtf_result.primary_confidence + mtf_result.alignment_score) / 2
        
        if combined_score >= 0.85:
            return SignalUrgency.IMMEDIATE
        elif combined_score >= 0.75:
            return SignalUrgency.HIGH
        elif combined_score >= 0.6:
            return SignalUrgency.MEDIUM
        elif combined_score >= 0.4:
            return SignalUrgency.LOW
        else:
            return SignalUrgency.WATCH
    
    def _calculate_quality_metrics(self, signal: TradingSignal, result: UnifiedAnalysisResult):
        """Calculate signal quality metrics"""
        quality_factors = []
        
        # Factor 1: Overall confidence
        quality_factors.append(result.combined_confidence * 0.4)
        
        # Factor 2: Agreement between methods
        quality_factors.append(result.agreement_score * 0.3)
        
        # Factor 3: Risk/reward ratio
        if signal.risk_reward_ratio:
            rr_score = min(1.0, signal.risk_reward_ratio / 3.0)  # Normalize to 3:1 max
            quality_factors.append(rr_score * 0.2)
        
        # Factor 4: Both confirmations
        confirmation_score = 0.0
        if signal.dow_confirmation:
            confirmation_score += 0.5
        if signal.elliott_confirmation:
            confirmation_score += 0.5
        quality_factors.append(confirmation_score * 0.1)
        
        signal.quality_score = sum(quality_factors)
        signal.strength = result.combined_confidence  # Use confidence as strength
    
    def _calculate_mtf_quality_metrics(self, signal: TradingSignal, mtf_result: MultiTimeframeResult):
        """Calculate quality metrics for multi-timeframe signals"""
        quality_factors = []
        
        # Factor 1: Primary confidence
        quality_factors.append(mtf_result.primary_confidence * 0.3)
        
        # Factor 2: Timeframe alignment
        quality_factors.append(mtf_result.alignment_score * 0.3)
        
        # Factor 3: Trend consistency
        quality_factors.append(mtf_result.trend_consistency * 0.2)
        
        # Factor 4: Entry confirmation
        quality_factors.append(mtf_result.entry_confirmation_score * 0.2)
        
        signal.quality_score = sum(quality_factors)
        signal.strength = mtf_result.primary_confidence
    
    def _enhance_with_mtf_data(self, signal: TradingSignal, mtf_result: MultiTimeframeResult):
        """Enhance signal with multi-timeframe information"""
        # Add supporting timeframes
        signal.supporting_timeframes = list(mtf_result.timeframe_results.keys())
        
        # Update confidence with multi-timeframe factors
        signal.confidence = mtf_result.primary_confidence
        
        # Add MTF metadata
        signal.metadata.update({
            'alignment_score': mtf_result.alignment_score,
            'trend_consistency': mtf_result.trend_consistency,
            'conflicting_timeframes': mtf_result.conflicting_timeframes,
            'entry_timeframe': mtf_result.optimal_entry_timeframe
        })
    
    def _create_analysis_summary(self, result: UnifiedAnalysisResult) -> Dict[str, Any]:
        """Create analysis summary for signal"""
        return {
            'combined_signal': result.combined_signal,
            'combined_confidence': result.combined_confidence,
            'agreement_score': result.agreement_score,
            'dow_trend': result.dow_trend.primary_trend.value if result.dow_trend else None,
            'dow_confidence': result.dow_confidence,
            'elliott_confidence': result.elliott_confidence,
            'elliott_patterns': len(result.elliott_patterns),
            'elliott_predictions': len(result.elliott_predictions),
            'swing_points': len(result.swing_points)
        }
    
    def _create_mtf_analysis_summary(self, mtf_result: MultiTimeframeResult) -> Dict[str, Any]:
        """Create comprehensive multi-timeframe analysis summary"""
        return {
            'primary_signal': mtf_result.primary_signal,
            'primary_confidence': mtf_result.primary_confidence,
            'alignment_score': mtf_result.alignment_score,
            'trend_consistency': mtf_result.trend_consistency,
            'analyzed_timeframes': list(mtf_result.timeframe_results.keys()),
            'optimal_entry_timeframe': mtf_result.optimal_entry_timeframe,
            'entry_confirmation_score': mtf_result.entry_confirmation_score,
            'conflicting_timeframes': mtf_result.conflicting_timeframes,
            'timeframe_breakdown': {
                tf: {
                    'signal': result.combined_signal,
                    'confidence': result.combined_confidence
                }
                for tf, result in mtf_result.timeframe_results.items()
            }
        }
    
    def validate_signal(self, signal: TradingSignal) -> Tuple[bool, List[str]]:
        """Validate signal before execution"""
        issues = []
        
        # Check required fields
        if not signal.entry_price:
            issues.append("Missing entry price")
        
        if not signal.stop_loss:
            issues.append("Missing stop loss")
        
        if not signal.take_profit_1:
            issues.append("Missing take profit")
        
        # Check risk/reward ratio
        if signal.risk_reward_ratio and signal.risk_reward_ratio < self.min_risk_reward_ratio:
            issues.append(f"Risk/reward ratio {signal.risk_reward_ratio:.2f} below minimum {self.min_risk_reward_ratio}")
        
        # Check signal expiry
        if signal.valid_until and datetime.utcnow() > signal.valid_until:
            issues.append("Signal has expired")
        
        # Check confidence
        if signal.confidence < self.config['min_confidence']:
            issues.append(f"Confidence {signal.confidence:.2f} below minimum {self.config['min_confidence']}")
        
        return len(issues) == 0, issues
    
    def set_config(self, config: Dict[str, Any]):
        """Update signal generation configuration"""
        self.config.update(config)
        analysis_logger.info(f"Updated signal generator config: {config}")
    
    def set_risk_config(self, risk_config: Dict[str, Any]):
        """Update risk management configuration"""
        self.risk_config.update(risk_config)
        analysis_logger.info(f"Updated risk config: {risk_config}")


# Create default signal generator instance
signal_generator = SignalGenerator()