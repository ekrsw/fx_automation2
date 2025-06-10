"""
Momentum indicators for market analysis
"""

from typing import Optional, Union, List, Dict, Any
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from app.analysis.indicators.atr import IndicatorBase
from app.utils.logger import analysis_logger


class MomentumIndicator(IndicatorBase):
    """
    Basic Momentum Indicator
    
    Calculates the rate of change in price over a specified period.
    Momentum = Current Price - Price N periods ago
    """
    
    def __init__(self, period: int = 14, use_percentage: bool = False):
        """
        Initialize Momentum indicator
        
        Args:
            period: Number of periods for momentum calculation
            use_percentage: Whether to calculate percentage momentum
        """
        super().__init__(period)
        self.use_percentage = use_percentage
    
    def get_required_columns(self) -> List[str]:
        """Required columns"""
        return ['close']
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Momentum
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Series with Momentum values
        """
        if not self.validate_data(data):
            raise ValueError(f"Data missing required columns: {self.get_required_columns()}")
        
        close = data['close']
        
        if self.use_percentage:
            # Percentage momentum: (Current / Previous - 1) * 100
            momentum = ((close / close.shift(self.period)) - 1) * 100
        else:
            # Absolute momentum: Current - Previous
            momentum = close - close.shift(self.period)
        
        momentum.name = f'Momentum_{self.period}{"_pct" if self.use_percentage else ""}'
        return momentum


class RSIIndicator(IndicatorBase):
    """
    Relative Strength Index (RSI)
    
    Measures the speed and magnitude of price changes.
    RSI oscillates between 0 and 100.
    """
    
    def __init__(self, period: int = 14, smoothing_method: str = 'wilder'):
        """
        Initialize RSI indicator
        
        Args:
            period: Number of periods for RSI calculation
            smoothing_method: Method for smoothing ('sma', 'ema', 'wilder')
        """
        super().__init__(period)
        self.smoothing_method = smoothing_method.lower()
        
        if self.smoothing_method not in ['sma', 'ema', 'wilder']:
            raise ValueError("smoothing_method must be 'sma', 'ema', or 'wilder'")
    
    def get_required_columns(self) -> List[str]:
        """Required columns"""
        return ['close']
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate RSI
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Series with RSI values
        """
        if not self.validate_data(data):
            raise ValueError(f"Data missing required columns: {self.get_required_columns()}")
        
        try:
            close = data['close']
            
            # Calculate price changes
            delta = close.diff()
            
            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses
            if self.smoothing_method == 'sma':
                avg_gains = gains.rolling(window=self.period).mean()
                avg_losses = losses.rolling(window=self.period).mean()
            
            elif self.smoothing_method == 'ema':
                avg_gains = gains.ewm(span=self.period).mean()
                avg_losses = losses.ewm(span=self.period).mean()
            
            elif self.smoothing_method == 'wilder':
                avg_gains = self._wilder_smoothing(gains)
                avg_losses = self._wilder_smoothing(losses)
            
            # Calculate RS and RSI
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            rsi.name = f'RSI_{self.period}'
            
            analysis_logger.debug(f"Calculated RSI with period {self.period}")
            return rsi
            
        except Exception as e:
            analysis_logger.error(f"Error calculating RSI: {e}")
            raise
    
    def _wilder_smoothing(self, data: pd.Series) -> pd.Series:
        """Apply Wilder's smoothing method"""
        result = pd.Series(index=data.index, dtype=float)
        
        # First value is simple average of first 'period' values
        first_avg = data.iloc[1:self.period + 1].mean()  # Skip first NaN
        result.iloc[self.period] = first_avg
        
        # Apply Wilder's smoothing for subsequent values
        for i in range(self.period + 1, len(data)):
            if pd.notna(data.iloc[i]):
                prev_avg = result.iloc[i - 1]
                current_value = data.iloc[i]
                result.iloc[i] = (prev_avg * (self.period - 1) + current_value) / self.period
        
        return result
    
    def get_signal(self, current_rsi: float) -> str:
        """
        Get RSI signal interpretation
        
        Args:
            current_rsi: Current RSI value
            
        Returns:
            Signal interpretation
        """
        if current_rsi >= 70:
            return 'overbought'
        elif current_rsi <= 30:
            return 'oversold'
        elif current_rsi > 50:
            return 'bullish'
        else:
            return 'bearish'


class MACDIndicator(IndicatorBase):
    """
    Moving Average Convergence Divergence (MACD)
    
    Shows the relationship between two moving averages of price.
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        Initialize MACD indicator
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
        """
        super().__init__(period=slow_period)  # Use slow period as main period
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def get_required_columns(self) -> List[str]:
        """Required columns"""
        return ['close']
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with MACD, Signal, and Histogram
        """
        if not self.validate_data(data):
            raise ValueError(f"Data missing required columns: {self.get_required_columns()}")
        
        try:
            close = data['close']
            
            # Calculate EMAs
            ema_fast = close.ewm(span=self.fast_period).mean()
            ema_slow = close.ewm(span=self.slow_period).mean()
            
            # Calculate MACD line
            macd_line = ema_fast - ema_slow
            
            # Calculate Signal line
            signal_line = macd_line.ewm(span=self.signal_period).mean()
            
            # Calculate Histogram
            histogram = macd_line - signal_line
            
            # Create result DataFrame
            result = pd.DataFrame(index=data.index)
            result['MACD'] = macd_line
            result['Signal'] = signal_line
            result['Histogram'] = histogram
            
            analysis_logger.debug(f"Calculated MACD with periods {self.fast_period}/{self.slow_period}/{self.signal_period}")
            return result
            
        except Exception as e:
            analysis_logger.error(f"Error calculating MACD: {e}")
            raise
    
    def get_signal(self, macd_data: pd.DataFrame) -> str:
        """
        Get MACD signal interpretation
        
        Args:
            macd_data: DataFrame with MACD data
            
        Returns:
            Signal interpretation
        """
        if len(macd_data) < 2:
            return 'insufficient_data'
        
        current_macd = macd_data['MACD'].iloc[-1]
        current_signal = macd_data['Signal'].iloc[-1]
        current_histogram = macd_data['Histogram'].iloc[-1]
        
        prev_macd = macd_data['MACD'].iloc[-2]
        prev_signal = macd_data['Signal'].iloc[-2]
        prev_histogram = macd_data['Histogram'].iloc[-2]
        
        # Check for crossovers
        if prev_macd <= prev_signal and current_macd > current_signal:
            return 'bullish_crossover'
        elif prev_macd >= prev_signal and current_macd < current_signal:
            return 'bearish_crossover'
        
        # Check histogram direction
        if current_histogram > prev_histogram and current_histogram > 0:
            return 'strengthening_bullish'
        elif current_histogram < prev_histogram and current_histogram < 0:
            return 'strengthening_bearish'
        elif current_histogram > 0:
            return 'weakening_bullish'
        else:
            return 'weakening_bearish'


class StochasticIndicator(IndicatorBase):
    """
    Stochastic Oscillator
    
    Compares a closing price to its price range over a given time period.
    """
    
    def __init__(self, k_period: int = 14, d_period: int = 3, smooth_k: int = 1):
        """
        Initialize Stochastic indicator
        
        Args:
            k_period: Period for %K calculation
            d_period: Period for %D smoothing
            smooth_k: Period for %K smoothing
        """
        super().__init__(period=k_period)
        self.k_period = k_period
        self.d_period = d_period
        self.smooth_k = smooth_k
    
    def get_required_columns(self) -> List[str]:
        """Required columns"""
        return ['high', 'low', 'close']
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Stochastic
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            DataFrame with %K and %D values
        """
        if not self.validate_data(data):
            raise ValueError(f"Data missing required columns: {self.get_required_columns()}")
        
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate rolling highest high and lowest low
            rolling_high = high.rolling(window=self.k_period).max()
            rolling_low = low.rolling(window=self.k_period).min()
            
            # Calculate %K
            k_percent = 100 * ((close - rolling_low) / (rolling_high - rolling_low))
            
            # Smooth %K if requested
            if self.smooth_k > 1:
                k_percent = k_percent.rolling(window=self.smooth_k).mean()
            
            # Calculate %D (smoothed %K)
            d_percent = k_percent.rolling(window=self.d_period).mean()
            
            # Create result DataFrame
            result = pd.DataFrame(index=data.index)
            result['%K'] = k_percent
            result['%D'] = d_percent
            
            analysis_logger.debug(f"Calculated Stochastic with periods K:{self.k_period}, D:{self.d_period}")
            return result
            
        except Exception as e:
            analysis_logger.error(f"Error calculating Stochastic: {e}")
            raise
    
    def get_signal(self, stoch_data: pd.DataFrame) -> str:
        """
        Get Stochastic signal interpretation
        
        Args:
            stoch_data: DataFrame with Stochastic data
            
        Returns:
            Signal interpretation
        """
        if len(stoch_data) < 2:
            return 'insufficient_data'
        
        current_k = stoch_data['%K'].iloc[-1]
        current_d = stoch_data['%D'].iloc[-1]
        
        prev_k = stoch_data['%K'].iloc[-2]
        prev_d = stoch_data['%D'].iloc[-2]
        
        # Check for overbought/oversold
        if current_k >= 80 and current_d >= 80:
            return 'overbought'
        elif current_k <= 20 and current_d <= 20:
            return 'oversold'
        
        # Check for crossovers
        if prev_k <= prev_d and current_k > current_d:
            return 'bullish_crossover'
        elif prev_k >= prev_d and current_k < current_d:
            return 'bearish_crossover'
        
        # General trend
        if current_k > 50 and current_d > 50:
            return 'bullish'
        else:
            return 'bearish'


class MomentumAnalyzer:
    """
    Comprehensive momentum analysis using multiple indicators
    """
    
    def __init__(self,
                 rsi_period: int = 14,
                 momentum_period: int = 14,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 stoch_k: int = 14,
                 stoch_d: int = 3):
        """
        Initialize momentum analyzer
        
        Args:
            rsi_period: RSI calculation period
            momentum_period: Momentum calculation period
            macd_fast: MACD fast EMA period
            macd_slow: MACD slow EMA period
            macd_signal: MACD signal period
            stoch_k: Stochastic %K period
            stoch_d: Stochastic %D period
        """
        self.rsi = RSIIndicator(period=rsi_period)
        self.momentum = MomentumIndicator(period=momentum_period, use_percentage=True)
        self.macd = MACDIndicator(fast_period=macd_fast, slow_period=macd_slow, signal_period=macd_signal)
        self.stochastic = StochasticIndicator(k_period=stoch_k, d_period=stoch_d)
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive momentum analysis
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Calculate all indicators
            rsi_values = self.rsi.calculate(data)
            momentum_values = self.momentum.calculate(data)
            macd_data = self.macd.calculate(data)
            stoch_data = self.stochastic.calculate(data)
            
            # Get current values
            current_rsi = rsi_values.iloc[-1] if not rsi_values.empty else None
            current_momentum = momentum_values.iloc[-1] if not momentum_values.empty else None
            
            # Get signals
            rsi_signal = self.rsi.get_signal(current_rsi) if current_rsi is not None else 'unknown'
            macd_signal = self.macd.get_signal(macd_data) if not macd_data.empty else 'unknown'
            stoch_signal = self.stochastic.get_signal(stoch_data) if not stoch_data.empty else 'unknown'
            
            # Calculate composite momentum score
            momentum_score = self._calculate_momentum_score(
                rsi_signal, macd_signal, stoch_signal, current_momentum
            )
            
            return {
                'rsi': {
                    'value': current_rsi,
                    'signal': rsi_signal
                },
                'momentum': {
                    'value': current_momentum,
                    'interpretation': 'bullish' if current_momentum and current_momentum > 0 else 'bearish'
                },
                'macd': {
                    'signal': macd_signal,
                    'data': macd_data.tail(1).to_dict('records')[0] if not macd_data.empty else {}
                },
                'stochastic': {
                    'signal': stoch_signal,
                    'data': stoch_data.tail(1).to_dict('records')[0] if not stoch_data.empty else {}
                },
                'composite_score': momentum_score,
                'overall_signal': self._get_overall_signal(momentum_score)
            }
            
        except Exception as e:
            analysis_logger.error(f"Error in momentum analysis: {e}")
            return {
                'error': str(e),
                'composite_score': 0.0,
                'overall_signal': 'unknown'
            }
    
    def _calculate_momentum_score(self, rsi_signal: str, macd_signal: str, 
                                stoch_signal: str, momentum_value: Optional[float]) -> float:
        """Calculate composite momentum score from -1 to 1"""
        score = 0.0
        
        # RSI contribution
        rsi_scores = {
            'overbought': -0.3,
            'oversold': 0.3,
            'bullish': 0.2,
            'bearish': -0.2
        }
        score += rsi_scores.get(rsi_signal, 0.0)
        
        # MACD contribution
        macd_scores = {
            'bullish_crossover': 0.4,
            'bearish_crossover': -0.4,
            'strengthening_bullish': 0.3,
            'strengthening_bearish': -0.3,
            'weakening_bullish': 0.1,
            'weakening_bearish': -0.1
        }
        score += macd_scores.get(macd_signal, 0.0)
        
        # Stochastic contribution
        stoch_scores = {
            'overbought': -0.2,
            'oversold': 0.2,
            'bullish_crossover': 0.3,
            'bearish_crossover': -0.3,
            'bullish': 0.1,
            'bearish': -0.1
        }
        score += stoch_scores.get(stoch_signal, 0.0)
        
        # Raw momentum contribution
        if momentum_value is not None:
            # Normalize momentum value (clamp between -0.2 and 0.2)
            momentum_contrib = max(-0.2, min(0.2, momentum_value / 100))
            score += momentum_contrib
        
        # Clamp final score between -1 and 1
        return max(-1.0, min(1.0, score))
    
    def _get_overall_signal(self, momentum_score: float) -> str:
        """Get overall signal based on momentum score"""
        if momentum_score >= 0.5:
            return 'strong_bullish'
        elif momentum_score >= 0.2:
            return 'bullish'
        elif momentum_score >= -0.2:
            return 'neutral'
        elif momentum_score >= -0.5:
            return 'bearish'
        else:
            return 'strong_bearish'


# Convenience functions
def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate RSI"""
    rsi = RSIIndicator(period=period)
    return rsi.calculate(data)


def calculate_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Calculate MACD"""
    macd = MACDIndicator(fast_period=fast, slow_period=slow, signal_period=signal)
    return macd.calculate(data)


# Create default instances
rsi_14 = RSIIndicator(period=14)
momentum_14 = MomentumIndicator(period=14, use_percentage=True)
macd_default = MACDIndicator()
stochastic_default = StochasticIndicator()
momentum_analyzer = MomentumAnalyzer()