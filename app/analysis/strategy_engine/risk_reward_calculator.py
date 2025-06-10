"""
Risk/Reward Ratio Calculator

Advanced risk/reward analysis system that calculates comprehensive
risk metrics, position sizing, and reward potential for trading signals.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from app.analysis.strategy_engine.signal_generator import TradingSignal, SignalType
from app.analysis.strategy_engine.stop_loss_calculator import StopLossRecommendation
from app.analysis.strategy_engine.take_profit_calculator import TakeProfitRecommendation
from app.utils.logger import analysis_logger


class RiskLevel(Enum):
    """Risk level classifications"""
    VERY_LOW = "very_low"      # < 1% risk
    LOW = "low"                # 1-2% risk
    MODERATE = "moderate"      # 2-3% risk
    HIGH = "high"              # 3-5% risk
    VERY_HIGH = "very_high"    # > 5% risk


class RewardPotential(Enum):
    """Reward potential classifications"""
    LIMITED = "limited"        # < 2% potential
    MODERATE = "moderate"      # 2-5% potential
    GOOD = "good"              # 5-10% potential
    EXCELLENT = "excellent"    # > 10% potential


@dataclass
class RiskMetrics:
    """Individual risk metrics calculation"""
    risk_amount: float          # Dollar amount at risk
    risk_percentage: float      # Percentage of account at risk
    stop_distance_pips: float   # Distance to stop loss in pips
    stop_distance_percent: float # Distance to stop loss as %
    risk_level: RiskLevel       # Risk classification
    max_position_size: float    # Maximum recommended position size
    position_value: float       # Total position value


@dataclass
class RewardMetrics:
    """Individual reward metrics calculation"""
    reward_amount: float        # Dollar amount of potential reward
    reward_percentage: float    # Percentage gain potential
    target_distance_pips: float # Distance to target in pips
    target_distance_percent: float # Distance to target as %
    reward_potential: RewardPotential # Reward classification
    success_probability: float  # Probability of reaching target


@dataclass
class RiskRewardRatio:
    """Individual risk/reward ratio calculation"""
    ratio: float                # Reward to risk ratio
    risk_metrics: RiskMetrics
    reward_metrics: RewardMetrics
    target_level: float         # Target price for this calculation
    expected_value: float       # Expected value of trade
    quality_score: float        # Overall quality score (0-1)


@dataclass
class RiskRewardAnalysis:
    """Comprehensive risk/reward analysis"""
    signal_id: str
    symbol: str
    
    # Primary metrics
    primary_ratio: float        # Best risk/reward ratio
    average_ratio: float        # Average across all targets
    weighted_ratio: float       # Probability-weighted ratio
    
    # Individual calculations
    ratio_calculations: List[RiskRewardRatio]
    
    # Portfolio metrics
    position_sizing: Dict[str, float]  # Different sizing recommendations
    portfolio_impact: Dict[str, float] # Impact on portfolio
    
    # Risk assessment
    overall_risk_level: RiskLevel
    risk_justification: str
    
    # Reward assessment
    overall_reward_potential: RewardPotential
    total_profit_potential: float
    
    # Trade quality
    trade_quality_score: float  # Overall trade quality (0-1)
    recommendation: str         # STRONG_BUY, BUY, HOLD, AVOID
    
    # Metadata
    calculation_timestamp: datetime
    account_balance: float
    max_risk_per_trade: float


class RiskRewardCalculator:
    """
    Advanced risk/reward calculation engine
    
    Analyzes risk and reward potential for trading signals and provides
    comprehensive risk management recommendations.
    """
    
    def __init__(self,
                 max_risk_per_trade: float = 0.02,  # 2% max risk
                 min_reward_risk_ratio: float = 1.5,
                 target_reward_risk_ratio: float = 2.0):
        """
        Initialize risk/reward calculator
        
        Args:
            max_risk_per_trade: Maximum risk per trade as decimal (0.02 = 2%)
            min_reward_risk_ratio: Minimum acceptable R:R ratio
            target_reward_risk_ratio: Target R:R ratio for quality trades
        """
        self.max_risk_per_trade = max_risk_per_trade
        self.min_reward_risk_ratio = min_reward_risk_ratio
        self.target_reward_risk_ratio = target_reward_risk_ratio
        
        # Configuration
        self.config = {
            'risk_levels': {
                RiskLevel.VERY_LOW: 0.01,
                RiskLevel.LOW: 0.02,
                RiskLevel.MODERATE: 0.03,
                RiskLevel.HIGH: 0.05,
                RiskLevel.VERY_HIGH: 1.0
            },
            'reward_levels': {
                RewardPotential.LIMITED: 0.02,
                RewardPotential.MODERATE: 0.05,
                RewardPotential.GOOD: 0.10,
                RewardPotential.EXCELLENT: 1.0
            },
            'position_sizing_methods': ['fixed_risk', 'kelly', 'conservative', 'aggressive'],
            'quality_thresholds': {
                'excellent': 0.8,
                'good': 0.6,
                'fair': 0.4,
                'poor': 0.0
            },
            'pip_values': {
                'MAJOR_PAIRS': 0.0001,
                'JPY_PAIRS': 0.01,
                'MICRO_PAIRS': 0.00001
            }
        }
    
    def calculate_risk_reward(self,
                            signal: TradingSignal,
                            stop_loss_rec: StopLossRecommendation,
                            take_profit_rec: TakeProfitRecommendation,
                            account_balance: float = 10000.0) -> RiskRewardAnalysis:
        """
        Calculate comprehensive risk/reward analysis
        
        Args:
            signal: Trading signal
            stop_loss_rec: Stop loss recommendation
            take_profit_rec: Take profit recommendation
            account_balance: Account balance for calculations
            
        Returns:
            Complete risk/reward analysis
        """
        try:
            current_price = signal.entry_price or 1.0
            ratio_calculations = []
            
            # Calculate R:R for each take profit target
            for target in take_profit_rec.primary_targets:
                rr_calc = self._calculate_individual_ratio(
                    signal, current_price, stop_loss_rec.primary_stop, 
                    target, account_balance, take_profit_rec
                )
                ratio_calculations.append(rr_calc)
            
            # If no targets, create a default calculation
            if not ratio_calculations:
                default_target = self._calculate_default_target(current_price, signal)
                rr_calc = self._calculate_individual_ratio(
                    signal, current_price, stop_loss_rec.primary_stop,
                    default_target, account_balance, take_profit_rec
                )
                ratio_calculations.append(rr_calc)
            
            # Calculate aggregate metrics
            primary_ratio = max(calc.ratio for calc in ratio_calculations)
            average_ratio = np.mean([calc.ratio for calc in ratio_calculations])
            
            # Calculate probability-weighted ratio
            weighted_ratio = self._calculate_weighted_ratio(
                ratio_calculations, take_profit_rec
            )
            
            # Calculate position sizing recommendations
            position_sizing = self._calculate_position_sizing(
                ratio_calculations[0].risk_metrics, account_balance, signal
            )
            
            # Calculate portfolio impact
            portfolio_impact = self._calculate_portfolio_impact(
                ratio_calculations, position_sizing, account_balance
            )
            
            # Assess overall risk and reward
            overall_risk_level = self._assess_overall_risk(ratio_calculations)
            overall_reward_potential = self._assess_overall_reward(ratio_calculations)
            
            # Calculate trade quality score
            trade_quality_score = self._calculate_trade_quality_score(
                ratio_calculations, stop_loss_rec, take_profit_rec
            )
            
            # Generate recommendation
            recommendation = self._generate_recommendation(
                trade_quality_score, primary_ratio, overall_risk_level
            )
            
            analysis = RiskRewardAnalysis(
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                primary_ratio=primary_ratio,
                average_ratio=average_ratio,
                weighted_ratio=weighted_ratio,
                ratio_calculations=ratio_calculations,
                position_sizing=position_sizing,
                portfolio_impact=portfolio_impact,
                overall_risk_level=overall_risk_level,
                risk_justification=self._generate_risk_justification(ratio_calculations),
                overall_reward_potential=overall_reward_potential,
                total_profit_potential=portfolio_impact.get('max_profit_potential', 0.0),
                trade_quality_score=trade_quality_score,
                recommendation=recommendation,
                calculation_timestamp=datetime.utcnow(),
                account_balance=account_balance,
                max_risk_per_trade=self.max_risk_per_trade
            )
            
            analysis_logger.info(
                f"Risk/Reward Analysis: {signal.symbol} - "
                f"Primary R:R: {primary_ratio:.2f}, "
                f"Quality: {trade_quality_score:.2f}, "
                f"Recommendation: {recommendation}"
            )
            
            return analysis
            
        except Exception as e:
            analysis_logger.error(f"Error calculating risk/reward: {e}")
            return self._create_default_analysis(signal, account_balance)
    
    def _calculate_individual_ratio(self,
                                  signal: TradingSignal,
                                  current_price: float,
                                  stop_loss: float,
                                  target: float,
                                  account_balance: float,
                                  take_profit_rec: TakeProfitRecommendation) -> RiskRewardRatio:
        """Calculate risk/reward ratio for individual target"""
        # Calculate risk metrics
        risk_distance = abs(current_price - stop_loss)
        risk_percentage = risk_distance / current_price
        risk_amount = account_balance * self.max_risk_per_trade
        
        # Calculate position size based on risk
        position_value = risk_amount / risk_percentage if risk_percentage > 0 else 0
        max_position_size = min(position_value, account_balance * 0.1)  # Max 10% of account
        
        risk_level = self._classify_risk_level(risk_percentage)
        stop_distance_pips = self._calculate_pips(current_price, stop_loss, signal.symbol)
        
        risk_metrics = RiskMetrics(
            risk_amount=risk_amount,
            risk_percentage=risk_percentage * 100,
            stop_distance_pips=stop_distance_pips,
            stop_distance_percent=risk_percentage * 100,
            risk_level=risk_level,
            max_position_size=max_position_size,
            position_value=position_value
        )
        
        # Calculate reward metrics
        reward_distance = abs(target - current_price)
        reward_percentage = reward_distance / current_price
        reward_amount = (reward_percentage * position_value) if position_value > 0 else 0
        
        reward_potential = self._classify_reward_potential(reward_percentage)
        target_distance_pips = self._calculate_pips(current_price, target, signal.symbol)
        
        # Get success probability from take profit recommendation
        success_probability = self._get_target_probability(target, take_profit_rec)
        
        reward_metrics = RewardMetrics(
            reward_amount=reward_amount,
            reward_percentage=reward_percentage * 100,
            target_distance_pips=target_distance_pips,
            target_distance_percent=reward_percentage * 100,
            reward_potential=reward_potential,
            success_probability=success_probability
        )
        
        # Calculate ratio and expected value
        ratio = reward_distance / risk_distance if risk_distance > 0 else 0.0
        expected_value = (reward_amount * success_probability) - (risk_amount * (1 - success_probability))
        
        # Calculate quality score
        quality_score = self._calculate_ratio_quality_score(
            ratio, success_probability, risk_level, reward_potential
        )
        
        return RiskRewardRatio(
            ratio=ratio,
            risk_metrics=risk_metrics,
            reward_metrics=reward_metrics,
            target_level=target,
            expected_value=expected_value,
            quality_score=quality_score
        )
    
    def _calculate_weighted_ratio(self,
                                ratio_calculations: List[RiskRewardRatio],
                                take_profit_rec: TakeProfitRecommendation) -> float:
        """Calculate probability-weighted risk/reward ratio"""
        if not ratio_calculations:
            return 0.0
        
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for calc in ratio_calculations:
            probability = calc.reward_metrics.success_probability
            weighted_sum += calc.ratio * probability
            weight_sum += probability
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0
    
    def _calculate_position_sizing(self,
                                 risk_metrics: RiskMetrics,
                                 account_balance: float,
                                 signal: TradingSignal) -> Dict[str, float]:
        """Calculate different position sizing recommendations"""
        sizing = {}
        
        # Fixed risk method
        fixed_risk_size = (account_balance * self.max_risk_per_trade) / (risk_metrics.risk_percentage / 100)
        sizing['fixed_risk'] = min(fixed_risk_size, account_balance * 0.1)
        
        # Conservative method (half of max risk)
        conservative_size = fixed_risk_size * 0.5
        sizing['conservative'] = conservative_size
        
        # Aggressive method (higher risk for high confidence)
        confidence_multiplier = signal.confidence if signal.confidence > 0.7 else 0.5
        aggressive_size = fixed_risk_size * (1 + confidence_multiplier * 0.5)
        sizing['aggressive'] = min(aggressive_size, account_balance * 0.15)
        
        # Kelly criterion (simplified)
        win_rate = signal.confidence  # Using confidence as proxy for win rate
        avg_win = risk_metrics.position_value * 0.02  # Assume 2% average win
        avg_loss = risk_metrics.risk_amount
        
        if avg_loss > 0:
            kelly_f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_size = account_balance * max(0, min(kelly_f, 0.25))  # Cap at 25%
            sizing['kelly'] = kelly_size
        else:
            sizing['kelly'] = sizing['conservative']
        
        return sizing
    
    def _calculate_portfolio_impact(self,
                                  ratio_calculations: List[RiskRewardRatio],
                                  position_sizing: Dict[str, float],
                                  account_balance: float) -> Dict[str, float]:
        """Calculate impact on portfolio"""
        if not ratio_calculations:
            return {}
        
        best_calc = max(ratio_calculations, key=lambda x: x.ratio)
        
        # Use conservative position sizing for calculations
        position_size = position_sizing.get('conservative', 0)
        
        return {
            'position_size_percent': (position_size / account_balance) * 100,
            'max_loss_amount': best_calc.risk_metrics.risk_amount,
            'max_loss_percent': (best_calc.risk_metrics.risk_amount / account_balance) * 100,
            'max_profit_potential': best_calc.reward_metrics.reward_amount,
            'max_profit_percent': (best_calc.reward_metrics.reward_amount / account_balance) * 100,
            'expected_value': best_calc.expected_value,
            'expected_value_percent': (best_calc.expected_value / account_balance) * 100
        }
    
    def _assess_overall_risk(self, ratio_calculations: List[RiskRewardRatio]) -> RiskLevel:
        """Assess overall risk level"""
        if not ratio_calculations:
            return RiskLevel.HIGH
        
        # Use the highest risk level among calculations
        risk_levels = [calc.risk_metrics.risk_level for calc in ratio_calculations]
        
        # Convert to numeric values for comparison
        risk_values = {
            RiskLevel.VERY_LOW: 1,
            RiskLevel.LOW: 2,
            RiskLevel.MODERATE: 3,
            RiskLevel.HIGH: 4,
            RiskLevel.VERY_HIGH: 5
        }
        
        max_risk_value = max(risk_values[level] for level in risk_levels)
        
        # Convert back to enum
        for level, value in risk_values.items():
            if value == max_risk_value:
                return level
        
        return RiskLevel.MODERATE
    
    def _assess_overall_reward(self, ratio_calculations: List[RiskRewardRatio]) -> RewardPotential:
        """Assess overall reward potential"""
        if not ratio_calculations:
            return RewardPotential.LIMITED
        
        # Use the best reward potential among calculations
        reward_potentials = [calc.reward_metrics.reward_potential for calc in ratio_calculations]
        
        # Convert to numeric values for comparison
        reward_values = {
            RewardPotential.LIMITED: 1,
            RewardPotential.MODERATE: 2,
            RewardPotential.GOOD: 3,
            RewardPotential.EXCELLENT: 4
        }
        
        max_reward_value = max(reward_values[potential] for potential in reward_potentials)
        
        # Convert back to enum
        for potential, value in reward_values.items():
            if value == max_reward_value:
                return potential
        
        return RewardPotential.MODERATE
    
    def _calculate_trade_quality_score(self,
                                     ratio_calculations: List[RiskRewardRatio],
                                     stop_loss_rec: StopLossRecommendation,
                                     take_profit_rec: TakeProfitRecommendation) -> float:
        """Calculate overall trade quality score"""
        if not ratio_calculations:
            return 0.1
        
        quality_factors = []
        
        # Factor 1: Best risk/reward ratio (40%)
        best_ratio = max(calc.ratio for calc in ratio_calculations)
        if best_ratio >= 3.0:
            ratio_score = 1.0
        elif best_ratio >= 2.0:
            ratio_score = 0.8
        elif best_ratio >= 1.5:
            ratio_score = 0.6
        else:
            ratio_score = 0.3
        quality_factors.append(ratio_score * 0.4)
        
        # Factor 2: Success probability (25%)
        avg_probability = np.mean([calc.reward_metrics.success_probability for calc in ratio_calculations])
        quality_factors.append(avg_probability * 0.25)
        
        # Factor 3: Stop loss confidence (15%)
        sl_confidence_score = stop_loss_rec.confidence
        quality_factors.append(sl_confidence_score * 0.15)
        
        # Factor 4: Take profit confidence (15%)
        tp_confidence_score = take_profit_rec.confidence
        quality_factors.append(tp_confidence_score * 0.15)
        
        # Factor 5: Risk level appropriateness (5%)
        risk_score = 1.0 - (self._get_risk_level_penalty(ratio_calculations[0].risk_metrics.risk_level))
        quality_factors.append(risk_score * 0.05)
        
        return sum(quality_factors)
    
    def _generate_recommendation(self,
                               quality_score: float,
                               primary_ratio: float,
                               risk_level: RiskLevel) -> str:
        """Generate trading recommendation"""
        if quality_score >= 0.8 and primary_ratio >= 2.5 and risk_level in [RiskLevel.VERY_LOW, RiskLevel.LOW]:
            return "STRONG_BUY"
        elif quality_score >= 0.6 and primary_ratio >= 2.0 and risk_level in [RiskLevel.VERY_LOW, RiskLevel.LOW, RiskLevel.MODERATE]:
            return "BUY"
        elif quality_score >= 0.4 and primary_ratio >= 1.5:
            return "HOLD"
        else:
            return "AVOID"
    
    def _generate_risk_justification(self, ratio_calculations: List[RiskRewardRatio]) -> str:
        """Generate risk justification text"""
        if not ratio_calculations:
            return "Insufficient data for risk assessment"
        
        calc = ratio_calculations[0]
        risk_pct = calc.risk_metrics.risk_percentage
        
        if risk_pct < 1.0:
            return f"Very low risk trade with {risk_pct:.1f}% account risk"
        elif risk_pct < 2.0:
            return f"Low risk trade with {risk_pct:.1f}% account risk"
        elif risk_pct < 3.0:
            return f"Moderate risk trade with {risk_pct:.1f}% account risk"
        elif risk_pct < 5.0:
            return f"High risk trade with {risk_pct:.1f}% account risk"
        else:
            return f"Very high risk trade with {risk_pct:.1f}% account risk - consider reducing position size"
    
    def _classify_risk_level(self, risk_percentage: float) -> RiskLevel:
        """Classify risk level based on percentage"""
        for level, threshold in self.config['risk_levels'].items():
            if risk_percentage <= threshold:
                return level
        return RiskLevel.VERY_HIGH
    
    def _classify_reward_potential(self, reward_percentage: float) -> RewardPotential:
        """Classify reward potential based on percentage"""
        for potential, threshold in self.config['reward_levels'].items():
            if reward_percentage <= threshold:
                return potential
        return RewardPotential.EXCELLENT
    
    def _get_target_probability(self, target: float, take_profit_rec: TakeProfitRecommendation) -> float:
        """Get success probability for a specific target"""
        # Find matching calculation
        for calc in take_profit_rec.calculations:
            if abs(calc.level - target) < 0.0001:  # Close enough
                return calc.probability
        
        # Default probability based on distance from entry
        return 0.6  # Default 60% probability
    
    def _calculate_ratio_quality_score(self,
                                     ratio: float,
                                     success_probability: float,
                                     risk_level: RiskLevel,
                                     reward_potential: RewardPotential) -> float:
        """Calculate quality score for individual ratio"""
        # Base score from ratio
        if ratio >= 3.0:
            ratio_score = 1.0
        elif ratio >= 2.0:
            ratio_score = 0.8
        elif ratio >= 1.5:
            ratio_score = 0.6
        else:
            ratio_score = 0.3
        
        # Adjust for probability
        prob_factor = success_probability
        
        # Penalty for high risk
        risk_penalty = self._get_risk_level_penalty(risk_level)
        
        # Bonus for high reward potential
        reward_bonus = self._get_reward_potential_bonus(reward_potential)
        
        quality = (ratio_score * 0.6 + prob_factor * 0.4) * (1 - risk_penalty) * (1 + reward_bonus)
        return max(0.0, min(1.0, quality))
    
    def _get_risk_level_penalty(self, risk_level: RiskLevel) -> float:
        """Get penalty factor for risk level"""
        penalties = {
            RiskLevel.VERY_LOW: 0.0,
            RiskLevel.LOW: 0.05,
            RiskLevel.MODERATE: 0.10,
            RiskLevel.HIGH: 0.20,
            RiskLevel.VERY_HIGH: 0.40
        }
        return penalties.get(risk_level, 0.20)
    
    def _get_reward_potential_bonus(self, reward_potential: RewardPotential) -> float:
        """Get bonus factor for reward potential"""
        bonuses = {
            RewardPotential.LIMITED: 0.0,
            RewardPotential.MODERATE: 0.05,
            RewardPotential.GOOD: 0.10,
            RewardPotential.EXCELLENT: 0.20
        }
        return bonuses.get(reward_potential, 0.0)
    
    def _calculate_pips(self, price1: float, price2: float, symbol: str) -> float:
        """Calculate distance in pips between two prices"""
        if 'JPY' in symbol.upper():
            pip_value = self.config['pip_values']['JPY_PAIRS']
        else:
            pip_value = self.config['pip_values']['MAJOR_PAIRS']
        
        return abs(price1 - price2) / pip_value
    
    def _calculate_default_target(self, current_price: float, signal: TradingSignal) -> float:
        """Calculate default target when none provided"""
        # Simple 2% target
        if signal.signal_type in [SignalType.MARKET_BUY, SignalType.LIMIT_BUY]:
            return current_price * 1.02
        else:
            return current_price * 0.98
    
    def _create_default_analysis(self, signal: TradingSignal, account_balance: float) -> RiskRewardAnalysis:
        """Create default analysis for error cases"""
        return RiskRewardAnalysis(
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            primary_ratio=1.0,
            average_ratio=1.0,
            weighted_ratio=1.0,
            ratio_calculations=[],
            position_sizing={'conservative': account_balance * 0.01},
            portfolio_impact={'max_loss_percent': 1.0},
            overall_risk_level=RiskLevel.MODERATE,
            risk_justification="Default risk assessment due to calculation error",
            overall_reward_potential=RewardPotential.MODERATE,
            total_profit_potential=2.0,
            trade_quality_score=0.3,
            recommendation="AVOID",
            calculation_timestamp=datetime.utcnow(),
            account_balance=account_balance,
            max_risk_per_trade=self.max_risk_per_trade
        )
    
    def get_risk_reward_summary(self, analysis: RiskRewardAnalysis) -> Dict[str, Any]:
        """Get detailed summary of risk/reward analysis"""
        return {
            'symbol': analysis.symbol,
            'recommendation': analysis.recommendation,
            'ratios': {
                'primary_ratio': analysis.primary_ratio,
                'average_ratio': analysis.average_ratio,
                'weighted_ratio': analysis.weighted_ratio
            },
            'risk_assessment': {
                'overall_risk_level': analysis.overall_risk_level.value,
                'risk_justification': analysis.risk_justification,
                'max_loss_amount': analysis.portfolio_impact.get('max_loss_amount', 0),
                'max_loss_percent': analysis.portfolio_impact.get('max_loss_percent', 0)
            },
            'reward_assessment': {
                'overall_reward_potential': analysis.overall_reward_potential.value,
                'total_profit_potential': analysis.total_profit_potential,
                'max_profit_amount': analysis.portfolio_impact.get('max_profit_potential', 0),
                'max_profit_percent': analysis.portfolio_impact.get('max_profit_percent', 0)
            },
            'position_sizing': analysis.position_sizing,
            'trade_quality': {
                'score': analysis.trade_quality_score,
                'grade': self._get_quality_grade(analysis.trade_quality_score)
            },
            'detailed_calculations': [
                {
                    'target_level': calc.target_level,
                    'ratio': calc.ratio,
                    'expected_value': calc.expected_value,
                    'quality_score': calc.quality_score,
                    'success_probability': calc.reward_metrics.success_probability
                }
                for calc in analysis.ratio_calculations
            ],
            'metadata': {
                'calculation_timestamp': analysis.calculation_timestamp.isoformat(),
                'account_balance': analysis.account_balance,
                'max_risk_per_trade': analysis.max_risk_per_trade
            }
        }
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade"""
        if score >= self.config['quality_thresholds']['excellent']:
            return 'A'
        elif score >= self.config['quality_thresholds']['good']:
            return 'B'
        elif score >= self.config['quality_thresholds']['fair']:
            return 'C'
        else:
            return 'D'
    
    def compare_multiple_signals(self, analyses: List[RiskRewardAnalysis]) -> Dict[str, Any]:
        """Compare multiple risk/reward analyses for ranking"""
        if not analyses:
            return {}
        
        # Sort by trade quality score
        sorted_analyses = sorted(analyses, key=lambda x: x.trade_quality_score, reverse=True)
        
        return {
            'best_signal': {
                'symbol': sorted_analyses[0].symbol,
                'quality_score': sorted_analyses[0].trade_quality_score,
                'primary_ratio': sorted_analyses[0].primary_ratio,
                'recommendation': sorted_analyses[0].recommendation
            },
            'rankings': [
                {
                    'rank': i + 1,
                    'symbol': analysis.symbol,
                    'quality_score': analysis.trade_quality_score,
                    'primary_ratio': analysis.primary_ratio,
                    'recommendation': analysis.recommendation
                }
                for i, analysis in enumerate(sorted_analyses)
            ],
            'summary_stats': {
                'total_signals': len(analyses),
                'strong_buy_count': sum(1 for a in analyses if a.recommendation == 'STRONG_BUY'),
                'buy_count': sum(1 for a in analyses if a.recommendation == 'BUY'),
                'hold_count': sum(1 for a in analyses if a.recommendation == 'HOLD'),
                'avoid_count': sum(1 for a in analyses if a.recommendation == 'AVOID'),
                'average_quality': np.mean([a.trade_quality_score for a in analyses]),
                'average_ratio': np.mean([a.primary_ratio for a in analyses])
            }
        }


# Create default risk/reward calculator instance
risk_reward_calculator = RiskRewardCalculator()