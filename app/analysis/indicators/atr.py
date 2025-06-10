"""
Average True Range (ATR) indicator implementation
"""

from typing import Optional, Union, List
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from app.utils.logger import analysis_logger


class IndicatorBase(ABC):
    """Base class for all technical indicators"""
    
    def __init__(self, period: int = 14):
        self.period = period
        self.name = self.__class__.__name__
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate the indicator values"""
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data format"""
        required_columns = self.get_required_columns()
        return all(col in data.columns for col in required_columns)
    
    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """Get list of required columns"""
        pass


class ATRIndicator(IndicatorBase):
    """
    Average True Range (ATR) Indicator
    
    Measures market volatility by calculating the average of true ranges
    over a specified period. True Range is the maximum of:
    1. Current High - Current Low
    2. Current High - Previous Close (absolute value)
    3. Current Low - Previous Close (absolute value)
    """
    
    def __init__(self, period: int = 14, smoothing_method: str = 'ema'):
        """
        Initialize ATR indicator
        
        Args:
            period: Number of periods for ATR calculation
            smoothing_method: Method for smoothing ('sma', 'ema', 'wilder')
        """
        super().__init__(period)
        self.smoothing_method = smoothing_method.lower()
        
        if self.smoothing_method not in ['sma', 'ema', 'wilder']:
            raise ValueError("smoothing_method must be 'sma', 'ema', or 'wilder'")
    
    def get_required_columns(self) -> List[str]:
        """Required OHLC columns"""
        return ['high', 'low', 'close']
    
    def calculate_true_range(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate True Range values
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            Series with True Range values
        """
        if not self.validate_data(data):
            raise ValueError(f"Data missing required columns: {self.get_required_columns()}")
        
        # Current High - Current Low
        hl = data['high'] - data['low']
        
        # Current High - Previous Close (absolute)
        hc = (data['high'] - data['close'].shift(1)).abs()
        
        # Current Low - Previous Close (absolute)
        lc = (data['low'] - data['close'].shift(1)).abs()
        
        # True Range is the maximum of the three
        true_range = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        
        return true_range
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate ATR values
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            Series with ATR values
        """
        try:
            true_range = self.calculate_true_range(data)
            
            if self.smoothing_method == 'sma':
                atr = true_range.rolling(window=self.period).mean()
            
            elif self.smoothing_method == 'ema':
                atr = true_range.ewm(span=self.period).mean()
            
            elif self.smoothing_method == 'wilder':
                # Wilder's smoothing (used in original ATR)
                atr = self._wilder_smoothing(true_range)
            
            # Set name for the series
            atr.name = f'ATR_{self.period}'
            
            analysis_logger.debug(f"Calculated ATR with period {self.period}, method {self.smoothing_method}")
            return atr
            
        except Exception as e:
            analysis_logger.error(f"Error calculating ATR: {e}")
            raise
    
    def _wilder_smoothing(self, data: pd.Series) -> pd.Series:
        """
        Apply Wilder's smoothing method
        
        ATR[i] = (ATR[i-1] * (period-1) + TR[i]) / period
        """
        result = pd.Series(index=data.index, dtype=float)
        
        # First ATR value is simple average of first 'period' values
        first_atr = data.iloc[:self.period].mean()
        result.iloc[self.period - 1] = first_atr
        
        # Apply Wilder's smoothing for subsequent values
        for i in range(self.period, len(data)):
            prev_atr = result.iloc[i - 1]
            current_tr = data.iloc[i]
            result.iloc[i] = (prev_atr * (self.period - 1) + current_tr) / self.period
        
        return result
    
    def get_volatility_level(self, current_atr: float, data: pd.DataFrame, 
                           lookback_periods: int = 50) -> str:
        """
        Determine volatility level compared to recent history
        
        Args:
            current_atr: Current ATR value
            data: Historical OHLC data
            lookback_periods: Number of periods to look back for comparison
            
        Returns:
            Volatility level: 'low', 'normal', 'high', 'extreme'
        """
        if len(data) < lookback_periods + self.period:
            return 'unknown'
        
        # Calculate ATR for the lookback period
        recent_data = data.tail(lookback_periods + self.period)
        atr_series = self.calculate(recent_data).dropna()
        
        if len(atr_series) < lookback_periods:
            return 'unknown'
        
        # Get percentiles for classification
        percentiles = atr_series.quantile([0.25, 0.5, 0.75, 0.9])
        
        if current_atr <= percentiles[0.25]:
            return 'low'
        elif current_atr <= percentiles[0.5]:
            return 'normal'
        elif current_atr <= percentiles[0.9]:
            return 'high'
        else:
            return 'extreme'
    
    def get_atr_bands(self, data: pd.DataFrame, close_price: float, 
                      multiplier: float = 2.0) -> dict:
        """
        Calculate ATR-based support and resistance bands
        
        Args:
            data: Historical OHLC data
            close_price: Current close price
            multiplier: ATR multiplier for band calculation
            
        Returns:
            Dictionary with upper and lower bands
        """
        atr_series = self.calculate(data)
        current_atr = atr_series.iloc[-1] if not atr_series.empty else 0
        
        return {
            'upper_band': close_price + (current_atr * multiplier),
            'lower_band': close_price - (current_atr * multiplier),
            'atr_value': current_atr,
            'band_width': current_atr * multiplier * 2
        }


def calculate_atr(data: pd.DataFrame, period: int = 14, 
                  method: str = 'wilder') -> pd.Series:
    """
    Convenience function to calculate ATR
    
    Args:
        data: DataFrame with OHLC data
        period: ATR period
        method: Smoothing method ('sma', 'ema', 'wilder')
        
    Returns:
        Series with ATR values
    """
    atr_indicator = ATRIndicator(period=period, smoothing_method=method)
    return atr_indicator.calculate(data)


# Create default ATR instance
atr_14 = ATRIndicator(period=14, smoothing_method='wilder')
atr_20 = ATRIndicator(period=20, smoothing_method='wilder')