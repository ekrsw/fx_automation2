"""
CRUD operations for trades
"""

from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, asc, func
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from app.db.models.trades import Trade, TradeType, TradeStatus
from app.utils.logger import db_logger


class TradeCRUD:
    """CRUD operations for trades"""
    
    def create(self, db: Session, **kwargs) -> Trade:
        """Create new trade record"""
        try:
            db_obj = Trade(**kwargs)
            db.add(db_obj)
            db.commit()
            db.refresh(db_obj)
            db_logger.info(f"Created trade: {db_obj.ticket}")
            return db_obj
        except Exception as e:
            db.rollback()
            db_logger.error(f"Error creating trade: {e}")
            raise
    
    def get_by_id(self, db: Session, trade_id: int) -> Optional[Trade]:
        """Get trade by ID"""
        return db.query(Trade).filter(Trade.id == trade_id).first()
    
    def get_by_ticket(self, db: Session, ticket: int) -> Optional[Trade]:
        """Get trade by MT5 ticket number"""
        return db.query(Trade).filter(Trade.ticket == ticket).first()
    
    def get_open_trades(self, db: Session, symbol: Optional[str] = None) -> List[Trade]:
        """Get all open trades, optionally filtered by symbol"""
        query = db.query(Trade).filter(Trade.status == TradeStatus.OPEN)
        
        if symbol:
            query = query.filter(Trade.symbol == symbol)
        
        return query.order_by(desc(Trade.open_time)).all()
    
    def get_closed_trades(
        self, 
        db: Session, 
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Trade]:
        """Get closed trades, optionally filtered by symbol"""
        query = db.query(Trade).filter(Trade.status == TradeStatus.CLOSED)
        
        if symbol:
            query = query.filter(Trade.symbol == symbol)
        
        return query.order_by(desc(Trade.close_time)).limit(limit).all()
    
    def get_trades_by_strategy(
        self, 
        db: Session, 
        strategy_id: str,
        limit: int = 100
    ) -> List[Trade]:
        """Get trades by strategy ID"""
        return db.query(Trade)\
            .filter(Trade.strategy_id == strategy_id)\
            .order_by(desc(Trade.open_time))\
            .limit(limit)\
            .all()
    
    def get_trades_in_date_range(
        self,
        db: Session,
        start_date: datetime,
        end_date: datetime,
        symbol: Optional[str] = None
    ) -> List[Trade]:
        """Get trades within date range"""
        query = db.query(Trade).filter(
            and_(
                Trade.open_time >= start_date,
                Trade.open_time <= end_date
            )
        )
        
        if symbol:
            query = query.filter(Trade.symbol == symbol)
        
        return query.order_by(asc(Trade.open_time)).all()
    
    def get_recent_trades(self, db: Session, hours: int = 24) -> List[Trade]:
        """Get recent trades within specified hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return db.query(Trade)\
            .filter(Trade.open_time >= cutoff_time)\
            .order_by(desc(Trade.open_time))\
            .all()
    
    def update(self, db: Session, trade_id: int, **kwargs) -> Optional[Trade]:
        """Update trade record"""
        try:
            db_obj = self.get_by_id(db, trade_id)
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
            db_logger.error(f"Error updating trade: {e}")
            raise
    
    def close_trade(
        self,
        db: Session,
        trade_id: int,
        close_price: float,
        close_time: Optional[datetime] = None,
        profit: Optional[float] = None
    ) -> Optional[Trade]:
        """Close a trade"""
        try:
            trade = self.get_by_id(db, trade_id)
            if not trade or trade.status != TradeStatus.OPEN:
                return None
            
            trade.close_price = close_price
            trade.close_time = close_time or datetime.utcnow()
            trade.status = TradeStatus.CLOSED
            
            if profit is not None:
                trade.profit = profit
            
            db.commit()
            db.refresh(trade)
            db_logger.info(f"Closed trade: {trade.ticket}")
            return trade
        except Exception as e:
            db.rollback()
            db_logger.error(f"Error closing trade: {e}")
            raise
    
    def delete(self, db: Session, trade_id: int) -> bool:
        """Delete trade record"""
        try:
            db_obj = self.get_by_id(db, trade_id)
            if not db_obj:
                return False
            
            db.delete(db_obj)
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            db_logger.error(f"Error deleting trade: {e}")
            raise
    
    def get_performance_statistics(
        self, 
        db: Session, 
        symbol: Optional[str] = None,
        strategy_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get performance statistics for trades"""
        try:
            # Base query for closed trades
            query = db.query(Trade).filter(Trade.status == TradeStatus.CLOSED)
            
            if symbol:
                query = query.filter(Trade.symbol == symbol)
            
            if strategy_id:
                query = query.filter(Trade.strategy_id == strategy_id)
            
            trades = query.all()
            
            if not trades:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0.0,
                    'total_profit': 0.0,
                    'average_profit': 0.0,
                    'max_profit': 0.0,
                    'max_loss': 0.0
                }
            
            # Calculate statistics
            total_trades = len(trades)
            winning_trades = sum(1 for trade in trades if trade.is_profitable())
            losing_trades = total_trades - winning_trades
            
            profits = [trade.calculate_total_profit() for trade in trades]
            total_profit = sum(profits)
            average_profit = total_profit / total_trades if total_trades > 0 else 0
            
            winning_profits = [p for p in profits if p > 0]
            losing_profits = [p for p in profits if p < 0]
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
                'total_profit': round(total_profit, 2),
                'average_profit': round(average_profit, 2),
                'max_profit': round(max(profits), 2) if profits else 0,
                'max_loss': round(min(profits), 2) if profits else 0,
                'average_winning_trade': round(sum(winning_profits) / len(winning_profits), 2) if winning_profits else 0,
                'average_losing_trade': round(sum(losing_profits) / len(losing_profits), 2) if losing_profits else 0,
                'profit_factor': abs(sum(winning_profits) / sum(losing_profits)) if losing_profits and sum(losing_profits) != 0 else 0
            }
        except Exception as e:
            db_logger.error(f"Error calculating performance statistics: {e}")
            return {'error': str(e)}
    
    def get_daily_performance(self, db: Session, days: int = 30) -> List[Dict[str, Any]]:
        """Get daily performance for the last N days"""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # Get trades closed in the date range
            trades = db.query(Trade)\
                .filter(and_(
                    Trade.status == TradeStatus.CLOSED,
                    Trade.close_time >= start_date
                ))\
                .all()
            
            # Group by date
            daily_stats = {}
            for trade in trades:
                date_key = trade.close_time.date().isoformat()
                if date_key not in daily_stats:
                    daily_stats[date_key] = {
                        'date': date_key,
                        'trades': 0,
                        'profit': 0.0,
                        'winning_trades': 0
                    }
                
                daily_stats[date_key]['trades'] += 1
                daily_stats[date_key]['profit'] += trade.calculate_total_profit()
                if trade.is_profitable():
                    daily_stats[date_key]['winning_trades'] += 1
            
            # Convert to list and sort by date
            result = list(daily_stats.values())
            result.sort(key=lambda x: x['date'])
            
            return result
        except Exception as e:
            db_logger.error(f"Error getting daily performance: {e}")
            return []
    
    def get_open_positions_count(self, db: Session, symbol: Optional[str] = None) -> int:
        """Get count of open positions"""
        query = db.query(Trade).filter(Trade.status == TradeStatus.OPEN)
        
        if symbol:
            query = query.filter(Trade.symbol == symbol)
        
        return query.count()
    
    def get_total_exposure(self, db: Session, symbol: Optional[str] = None) -> float:
        """Get total exposure (volume) of open positions"""
        query = db.query(func.sum(Trade.volume)).filter(Trade.status == TradeStatus.OPEN)
        
        if symbol:
            query = query.filter(Trade.symbol == symbol)
        
        result = query.scalar()
        return result or 0.0


# Global instance
trade_crud = TradeCRUD()