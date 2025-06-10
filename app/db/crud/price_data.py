"""
CRUD operations for price data
"""

from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, asc
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd

from app.db.models.price_data import PriceData
from app.utils.logger import db_logger


class PriceDataCRUD:
    """CRUD operations for price data"""
    
    def create(self, db: Session, **kwargs) -> PriceData:
        """Create new price data record"""
        try:
            db_obj = PriceData(**kwargs)
            db.add(db_obj)
            db.commit()
            db.refresh(db_obj)
            return db_obj
        except Exception as e:
            db.rollback()
            db_logger.error(f"Error creating price data: {e}")
            raise
    
    def create_bulk(self, db: Session, price_data_list: List[Dict[str, Any]]) -> int:
        """Create multiple price data records"""
        try:
            objects = [PriceData(**data) for data in price_data_list]
            db.add_all(objects)
            db.commit()
            db_logger.info(f"Created {len(objects)} price data records")
            return len(objects)
        except Exception as e:
            db.rollback()
            db_logger.error(f"Error creating bulk price data: {e}")
            raise
    
    def get_by_id(self, db: Session, price_id: int) -> Optional[PriceData]:
        """Get price data by ID"""
        return db.query(PriceData).filter(PriceData.id == price_id).first()
    
    def get_latest(self, db: Session, symbol: str, timeframe: str) -> Optional[PriceData]:
        """Get latest price data for symbol and timeframe"""
        return db.query(PriceData)\
            .filter(and_(PriceData.symbol == symbol, PriceData.timeframe == timeframe))\
            .order_by(desc(PriceData.timestamp))\
            .first()
    
    def get_by_symbol_timeframe(
        self, 
        db: Session, 
        symbol: str, 
        timeframe: str,
        limit: int = 1000
    ) -> List[PriceData]:
        """Get price data by symbol and timeframe"""
        return db.query(PriceData)\
            .filter(and_(PriceData.symbol == symbol, PriceData.timeframe == timeframe))\
            .order_by(desc(PriceData.timestamp))\
            .limit(limit)\
            .all()
    
    def get_by_date_range(
        self,
        db: Session,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[PriceData]:
        """Get price data by date range"""
        return db.query(PriceData)\
            .filter(and_(
                PriceData.symbol == symbol,
                PriceData.timeframe == timeframe,
                PriceData.timestamp >= start_date,
                PriceData.timestamp <= end_date
            ))\
            .order_by(asc(PriceData.timestamp))\
            .all()
    
    def get_recent(
        self,
        db: Session,
        symbol: str,
        timeframe: str,
        hours: int = 24
    ) -> List[PriceData]:
        """Get recent price data within specified hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return db.query(PriceData)\
            .filter(and_(
                PriceData.symbol == symbol,
                PriceData.timeframe == timeframe,
                PriceData.timestamp >= cutoff_time
            ))\
            .order_by(asc(PriceData.timestamp))\
            .all()
    
    def get_as_dataframe(
        self,
        db: Session,
        symbol: str,
        timeframe: str,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Get price data as pandas DataFrame"""
        data = self.get_by_symbol_timeframe(db, symbol, timeframe, limit)
        
        if not data:
            return pd.DataFrame()
        
        # Convert to list of dictionaries
        records = [record.to_dict() for record in data]
        df = pd.DataFrame(records)
        
        # Convert timestamp to datetime index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
        
        return df
    
    def update(self, db: Session, price_id: int, **kwargs) -> Optional[PriceData]:
        """Update price data record"""
        try:
            db_obj = self.get_by_id(db, price_id)
            if not db_obj:
                return None
            
            for key, value in kwargs.items():
                if hasattr(db_obj, key):
                    setattr(db_obj, key, value)
            
            db.commit()
            db.refresh(db_obj)
            return db_obj
        except Exception as e:
            db.rollback()
            db_logger.error(f"Error updating price data: {e}")
            raise
    
    def delete(self, db: Session, price_id: int) -> bool:
        """Delete price data record"""
        try:
            db_obj = self.get_by_id(db, price_id)
            if not db_obj:
                return False
            
            db.delete(db_obj)
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            db_logger.error(f"Error deleting price data: {e}")
            raise
    
    def delete_old_data(self, db: Session, days: int = 365) -> int:
        """Delete price data older than specified days"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            deleted_count = db.query(PriceData)\
                .filter(PriceData.timestamp < cutoff_date)\
                .delete()
            db.commit()
            
            if deleted_count > 0:
                db_logger.info(f"Deleted {deleted_count} old price data records")
            
            return deleted_count
        except Exception as e:
            db.rollback()
            db_logger.error(f"Error deleting old price data: {e}")
            raise
    
    def get_statistics(self, db: Session, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get statistics for price data"""
        try:
            # Get total count
            total_count = db.query(PriceData)\
                .filter(and_(PriceData.symbol == symbol, PriceData.timeframe == timeframe))\
                .count()
            
            if total_count == 0:
                return {
                    'total_count': 0,
                    'date_range': None,
                    'latest_price': None
                }
            
            # Get date range
            oldest = db.query(PriceData)\
                .filter(and_(PriceData.symbol == symbol, PriceData.timeframe == timeframe))\
                .order_by(asc(PriceData.timestamp))\
                .first()
            
            latest = db.query(PriceData)\
                .filter(and_(PriceData.symbol == symbol, PriceData.timeframe == timeframe))\
                .order_by(desc(PriceData.timestamp))\
                .first()
            
            return {
                'total_count': total_count,
                'date_range': {
                    'start': oldest.timestamp.isoformat() if oldest else None,
                    'end': latest.timestamp.isoformat() if latest else None
                },
                'latest_price': latest.close if latest else None
            }
        except Exception as e:
            db_logger.error(f"Error getting price data statistics: {e}")
            return {'error': str(e)}
    
    def exists(self, db: Session, symbol: str, timeframe: str, timestamp: datetime) -> bool:
        """Check if price data exists for given parameters"""
        return db.query(PriceData)\
            .filter(and_(
                PriceData.symbol == symbol,
                PriceData.timeframe == timeframe,
                PriceData.timestamp == timestamp
            )).first() is not None


# Global instance
price_data_crud = PriceDataCRUD()