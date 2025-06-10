"""
Data service for managing price data collection and storage
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import pandas as pd

from app.mt5.data_fetcher import get_data_fetcher
from app.mt5.connection import get_mt5_connection
from app.db.crud.price_data import price_data_crud
from app.db.database import SessionLocal
from app.utils.logger import trading_logger


class DataService:
    """Service for managing price data operations"""
    
    def __init__(self):
        self.data_fetcher = get_data_fetcher()
        self.mt5_connection = get_mt5_connection()
        
    async def fetch_and_store_live_prices(self, symbols: List[str]) -> Dict[str, Any]:
        """Fetch live prices and store to database"""
        try:
            # Get live prices from MT5
            live_prices = await self.data_fetcher.get_live_prices(symbols)
            
            if not live_prices:
                trading_logger.warning("No live prices received")
                return {'status': 'no_data', 'symbols_processed': 0}
            
            # Store to database
            stored_count = 0
            db = SessionLocal()
            
            try:
                for symbol, tick_data in live_prices.items():
                    # Create price data record from tick data
                    price_data = {
                        'symbol': symbol,
                        'timeframe': 'TICK',
                        'timestamp': tick_data.time,
                        'open': tick_data.last,
                        'high': tick_data.last,
                        'low': tick_data.last,
                        'close': tick_data.last,
                        'volume': tick_data.volume,
                        'spread': tick_data.ask - tick_data.bid
                    }
                    
                    # Check if record already exists
                    if not price_data_crud.exists(
                        db, symbol, 'TICK', tick_data.time
                    ):
                        price_data_crud.create(db, **price_data)
                        stored_count += 1
                
                trading_logger.info(f"Stored {stored_count} live price records")
                return {
                    'status': 'success', 
                    'symbols_processed': len(live_prices),
                    'records_stored': stored_count
                }
                
            finally:
                db.close()
                
        except Exception as e:
            trading_logger.error(f"Error fetching/storing live prices: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def fetch_and_store_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Fetch historical data and store to database"""
        try:
            # Get historical data from MT5
            df = await self.data_fetcher.get_historical_data(
                symbol, timeframe, start_date, end_date
            )
            
            if df.empty:
                trading_logger.warning(f"No historical data received for {symbol} {timeframe}")
                return {'status': 'no_data', 'records_stored': 0}
            
            # Store to database
            db = SessionLocal()
            stored_count = 0
            
            try:
                price_data_list = []
                
                for index, row in df.iterrows():
                    price_data = {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'timestamp': row['time'] if 'time' in row else index,
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row.get('volume', 0)
                    }
                    
                    # Check if record already exists
                    timestamp = price_data['timestamp']
                    if not price_data_crud.exists(db, symbol, timeframe, timestamp):
                        price_data_list.append(price_data)
                
                if price_data_list:
                    stored_count = price_data_crud.create_bulk(db, price_data_list)
                
                trading_logger.info(f"Stored {stored_count} historical records for {symbol} {timeframe}")
                return {
                    'status': 'success',
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'records_fetched': len(df),
                    'records_stored': stored_count
                }
                
            finally:
                db.close()
                
        except Exception as e:
            trading_logger.error(f"Error fetching/storing historical data: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def fetch_recent_data(
        self,
        symbol: str,
        timeframe: str,
        count: int = 100
    ) -> Dict[str, Any]:
        """Fetch recent OHLC data and store to database"""
        try:
            # Get recent data from MT5
            df = await self.data_fetcher.get_ohlc_data(symbol, timeframe, count)
            
            if df.empty:
                trading_logger.warning(f"No recent data received for {symbol} {timeframe}")
                return {'status': 'no_data', 'records_stored': 0}
            
            # Store to database
            db = SessionLocal()
            stored_count = 0
            
            try:
                price_data_list = []
                
                for index, row in df.iterrows():
                    price_data = {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'timestamp': row['time'] if 'time' in row else index,
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row.get('volume', 0)
                    }
                    
                    # Check if record already exists
                    timestamp = price_data['timestamp']
                    if not price_data_crud.exists(db, symbol, timeframe, timestamp):
                        price_data_list.append(price_data)
                
                if price_data_list:
                    stored_count = price_data_crud.create_bulk(db, price_data_list)
                
                trading_logger.info(f"Stored {stored_count} recent records for {symbol} {timeframe}")
                return {
                    'status': 'success',
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'records_fetched': len(df),
                    'records_stored': stored_count
                }
                
            finally:
                db.close()
                
        except Exception as e:
            trading_logger.error(f"Error fetching/storing recent data: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_price_data_as_dataframe(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Get stored price data as DataFrame"""
        db = SessionLocal()
        try:
            return price_data_crud.get_as_dataframe(db, symbol, timeframe, limit)
        finally:
            db.close()
    
    def get_latest_price(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Get latest price for symbol/timeframe"""
        db = SessionLocal()
        try:
            latest = price_data_crud.get_latest(db, symbol, timeframe)
            return latest.to_dict() if latest else None
        finally:
            db.close()
    
    def get_price_statistics(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get price data statistics"""
        db = SessionLocal()
        try:
            return price_data_crud.get_statistics(db, symbol, timeframe)
        finally:
            db.close()
    
    async def update_all_symbols(
        self, 
        symbols: List[str], 
        timeframes: List[str],
        count: int = 100
    ) -> Dict[str, Any]:
        """Update data for all symbols and timeframes"""
        results = {
            'total_symbols': len(symbols),
            'total_timeframes': len(timeframes),
            'successful_updates': 0,
            'failed_updates': 0,
            'details': []
        }
        
        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    result = await self.fetch_recent_data(symbol, timeframe, count)
                    
                    if result['status'] == 'success':
                        results['successful_updates'] += 1
                    else:
                        results['failed_updates'] += 1
                    
                    results['details'].append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'status': result['status'],
                        'records_stored': result.get('records_stored', 0)
                    })
                    
                    # Small delay to avoid overwhelming the system
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    results['failed_updates'] += 1
                    results['details'].append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'status': 'error',
                        'error': str(e)
                    })
                    trading_logger.error(f"Error updating {symbol} {timeframe}: {e}")
        
        trading_logger.info(f"Bulk update completed: {results['successful_updates']} successful, {results['failed_updates']} failed")
        return results
    
    def cleanup_old_data(self, days: int = 365) -> int:
        """Clean up old price data"""
        db = SessionLocal()
        try:
            deleted_count = price_data_crud.delete_old_data(db, days)
            trading_logger.info(f"Cleaned up {deleted_count} old price data records")
            return deleted_count
        finally:
            db.close()


# Global instance
data_service = DataService()