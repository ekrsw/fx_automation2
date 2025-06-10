"""
Data validation utilities for FX trading system
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

from app.utils.logger import db_logger


class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_ohlc_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate OHLC data integrity
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if df.empty:
            errors.append("DataFrame is empty")
            return False, errors
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
            return False, errors
        
        # Check for null values
        null_counts = df[required_columns].isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                errors.append(f"Column '{col}' has {count} null values")
        
        # Check for negative prices
        for col in required_columns:
            negative_count = (df[col] <= 0).sum()
            if negative_count > 0:
                errors.append(f"Column '{col}' has {negative_count} non-positive values")
        
        # Check OHLC relationships
        # High should be >= Open, Low, Close
        high_violations = ((df['high'] < df['open']) | 
                          (df['high'] < df['low']) | 
                          (df['high'] < df['close'])).sum()
        if high_violations > 0:
            errors.append(f"High price violations: {high_violations} records")
        
        # Low should be <= Open, High, Close  
        low_violations = ((df['low'] > df['open']) | 
                         (df['low'] > df['high']) | 
                         (df['low'] > df['close'])).sum()
        if low_violations > 0:
            errors.append(f"Low price violations: {low_violations} records")
        
        # Check for extreme price movements (potential data errors)
        if len(df) > 1:
            price_changes = df['close'].pct_change().abs()
            extreme_changes = (price_changes > 0.1).sum()  # 10% change threshold
            if extreme_changes > 0:
                errors.append(f"Extreme price movements detected: {extreme_changes} records")
        
        # Check for duplicate timestamps
        if 'timestamp' in df.columns or df.index.name == 'timestamp':
            timestamp_col = df['timestamp'] if 'timestamp' in df.columns else df.index
            duplicates = timestamp_col.duplicated().sum()
            if duplicates > 0:
                errors.append(f"Duplicate timestamps: {duplicates} records")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    @staticmethod
    def validate_trade_data(trade_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate trade data
        
        Args:
            trade_data: Dictionary with trade information
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Required fields
        required_fields = ['symbol', 'trade_type', 'volume', 'open_price', 'open_time']
        missing_fields = [field for field in required_fields if field not in trade_data]
        if missing_fields:
            errors.append(f"Missing required fields: {missing_fields}")
            return False, errors
        
        # Validate volume
        volume = trade_data.get('volume')
        if volume is not None:
            if not isinstance(volume, (int, float)) or volume <= 0:
                errors.append("Volume must be a positive number")
        
        # Validate prices
        price_fields = ['open_price', 'close_price', 'stop_loss', 'take_profit']
        for field in price_fields:
            price = trade_data.get(field)
            if price is not None:
                if not isinstance(price, (int, float)) or price <= 0:
                    errors.append(f"{field} must be a positive number")
        
        # Validate trade type
        trade_type = trade_data.get('trade_type')
        if trade_type and trade_type not in ['BUY', 'SELL']:
            errors.append("trade_type must be 'BUY' or 'SELL'")
        
        # Validate symbol format
        symbol = trade_data.get('symbol')
        if symbol and (not isinstance(symbol, str) or len(symbol) < 3):
            errors.append("symbol must be a valid string (at least 3 characters)")
        
        # Validate profit/loss logic for closed trades
        if trade_data.get('status') == 'CLOSED':
            if 'close_price' not in trade_data:
                errors.append("Closed trades must have close_price")
            if 'close_time' not in trade_data:
                errors.append("Closed trades must have close_time")
        
        # Validate stop loss and take profit relationships
        open_price = trade_data.get('open_price')
        stop_loss = trade_data.get('stop_loss')
        take_profit = trade_data.get('take_profit')
        
        if all([open_price, stop_loss, take_profit, trade_type]):
            if trade_type == 'BUY':
                if stop_loss >= open_price:
                    errors.append("For BUY trades, stop_loss should be below open_price")
                if take_profit <= open_price:
                    errors.append("For BUY trades, take_profit should be above open_price")
            elif trade_type == 'SELL':
                if stop_loss <= open_price:
                    errors.append("For SELL trades, stop_loss should be above open_price")
                if take_profit >= open_price:
                    errors.append("For SELL trades, take_profit should be below open_price")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    @staticmethod
    def validate_signal_data(signal_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate signal data
        
        Args:
            signal_data: Dictionary with signal information
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Required fields
        required_fields = ['symbol', 'signal_type', 'strategy', 'confidence', 'timestamp']
        missing_fields = [field for field in required_fields if field not in signal_data]
        if missing_fields:
            errors.append(f"Missing required fields: {missing_fields}")
            return False, errors
        
        # Validate signal type
        signal_type = signal_data.get('signal_type')
        if signal_type and signal_type not in ['BUY', 'SELL', 'CLOSE']:
            errors.append("signal_type must be 'BUY', 'SELL', or 'CLOSE'")
        
        # Validate confidence
        confidence = signal_data.get('confidence')
        if confidence is not None:
            if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                errors.append("confidence must be a number between 0 and 1")
        
        # Validate prices
        price_fields = ['entry_price', 'stop_loss', 'take_profit']
        for field in price_fields:
            price = signal_data.get(field)
            if price is not None:
                if not isinstance(price, (int, float)) or price <= 0:
                    errors.append(f"{field} must be a positive number")
        
        # Validate risk-reward ratio
        rr_ratio = signal_data.get('rr_ratio')
        if rr_ratio is not None:
            if not isinstance(rr_ratio, (int, float)) or rr_ratio <= 0:
                errors.append("rr_ratio must be a positive number")
        
        # Validate Fibonacci level
        fibonacci_level = signal_data.get('fibonacci_level')
        if fibonacci_level is not None:
            valid_fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618, 2.618]
            if not any(abs(fibonacci_level - level) < 0.01 for level in valid_fib_levels):
                errors.append(f"fibonacci_level {fibonacci_level} is not a standard Fibonacci ratio")
        
        # Validate Elliott wave
        elliott_wave = signal_data.get('elliott_wave')
        if elliott_wave:
            valid_waves = ['Wave1', 'Wave2', 'Wave3', 'Wave4', 'Wave5', 'WaveA', 'WaveB', 'WaveC']
            if elliott_wave not in valid_waves:
                errors.append(f"elliott_wave must be one of: {valid_waves}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    @staticmethod
    def validate_price_data_consistency(df: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Comprehensive validation of price data consistency
        
        Args:
            df: DataFrame with price data
            symbol: Trading symbol
            timeframe: Data timeframe
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'symbol': symbol,
            'timeframe': timeframe,
            'total_records': len(df),
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'quality_score': 100.0
        }
        
        if df.empty:
            validation_result['is_valid'] = False
            validation_result['errors'].append("No data to validate")
            validation_result['quality_score'] = 0.0
            return validation_result
        
        # Basic OHLC validation
        is_valid, errors = DataValidator.validate_ohlc_data(df)
        if not is_valid:
            validation_result['is_valid'] = False
            validation_result['errors'].extend(errors)
            validation_result['quality_score'] -= 30
        
        # Check data completeness
        if 'timestamp' in df.columns or hasattr(df.index, 'freq'):
            # Check for gaps in data
            if 'timestamp' in df.columns:
                time_diffs = df['timestamp'].diff().dropna()
            else:
                time_diffs = pd.Series(df.index).diff().dropna()
            
            # Expected interval based on timeframe
            expected_intervals = {
                'M1': pd.Timedelta(minutes=1),
                'M5': pd.Timedelta(minutes=5), 
                'M15': pd.Timedelta(minutes=15),
                'M30': pd.Timedelta(minutes=30),
                'H1': pd.Timedelta(hours=1),
                'H4': pd.Timedelta(hours=4),
                'D1': pd.Timedelta(days=1)
            }
            
            expected_interval = expected_intervals.get(timeframe)
            if expected_interval:
                gaps = time_diffs[time_diffs > expected_interval * 1.5]
                if len(gaps) > 0:
                    validation_result['warnings'].append(f"Data gaps detected: {len(gaps)} instances")
                    validation_result['quality_score'] -= min(20, len(gaps) * 2)
        
        # Check for outliers
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers > 0:
                    validation_result['warnings'].append(f"Outliers in {col}: {outliers} values")
                    validation_result['quality_score'] -= min(10, outliers * 0.5)
        
        # Check volatility consistency
        if len(df) > 1:
            price_changes = df['close'].pct_change().abs()
            avg_volatility = price_changes.mean()
            high_volatility_count = (price_changes > avg_volatility * 5).sum()
            
            if high_volatility_count > len(df) * 0.05:  # More than 5% of data
                validation_result['warnings'].append(f"High volatility periods: {high_volatility_count} instances")
                validation_result['quality_score'] -= 10
        
        # Final quality assessment
        validation_result['quality_score'] = max(0, validation_result['quality_score'])
        
        if validation_result['quality_score'] >= 90:
            validation_result['quality_rating'] = 'Excellent'
        elif validation_result['quality_score'] >= 75:
            validation_result['quality_rating'] = 'Good'
        elif validation_result['quality_score'] >= 60:
            validation_result['quality_rating'] = 'Fair'
        else:
            validation_result['quality_rating'] = 'Poor'
        
        return validation_result


# Global instance
data_validator = DataValidator()