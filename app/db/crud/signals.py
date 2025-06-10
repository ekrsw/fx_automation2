"""
CRUD operations for signals
"""

from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, asc, func
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from app.db.models.signals import Signal, SignalType
from app.utils.logger import db_logger


class SignalCRUD:
    """CRUD operations for signals"""
    
    def create(self, db: Session, **kwargs) -> Signal:
        """Create new signal record"""
        try:
            db_obj = Signal(**kwargs)
            db.add(db_obj)
            db.commit()
            db.refresh(db_obj)
            db_logger.info(f"Created signal: {db_obj.symbol} {db_obj.signal_type.value} {db_obj.strategy}")
            return db_obj
        except Exception as e:
            db.rollback()
            db_logger.error(f"Error creating signal: {e}")
            raise
    
    def get_by_id(self, db: Session, signal_id: int) -> Optional[Signal]:
        """Get signal by ID"""
        return db.query(Signal).filter(Signal.id == signal_id).first()
    
    def get_pending_signals(self, db: Session, symbol: Optional[str] = None) -> List[Signal]:
        """Get all pending (unexecuted) signals"""
        query = db.query(Signal).filter(Signal.executed == False)
        
        if symbol:
            query = query.filter(Signal.symbol == symbol)
        
        return query.order_by(desc(Signal.timestamp)).all()
    
    def get_executed_signals(
        self, 
        db: Session, 
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Signal]:
        """Get executed signals"""
        query = db.query(Signal).filter(Signal.executed == True)
        
        if symbol:
            query = query.filter(Signal.symbol == symbol)
        
        return query.order_by(desc(Signal.timestamp)).limit(limit).all()
    
    def get_signals_by_strategy(
        self, 
        db: Session, 
        strategy: str,
        limit: int = 100
    ) -> List[Signal]:
        """Get signals by strategy"""
        return db.query(Signal)\
            .filter(Signal.strategy == strategy)\
            .order_by(desc(Signal.timestamp))\
            .limit(limit)\
            .all()
    
    def get_signals_in_date_range(
        self,
        db: Session,
        start_date: datetime,
        end_date: datetime,
        symbol: Optional[str] = None
    ) -> List[Signal]:
        """Get signals within date range"""
        query = db.query(Signal).filter(
            and_(
                Signal.timestamp >= start_date,
                Signal.timestamp <= end_date
            )
        )
        
        if symbol:
            query = query.filter(Signal.symbol == symbol)
        
        return query.order_by(asc(Signal.timestamp)).all()
    
    def get_recent_signals(self, db: Session, hours: int = 24) -> List[Signal]:
        """Get recent signals within specified hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return db.query(Signal)\
            .filter(Signal.timestamp >= cutoff_time)\
            .order_by(desc(Signal.timestamp))\
            .all()
    
    def get_high_confidence_signals(
        self, 
        db: Session, 
        min_confidence: float = 0.7,
        symbol: Optional[str] = None
    ) -> List[Signal]:
        """Get high confidence signals"""
        query = db.query(Signal).filter(Signal.confidence >= min_confidence)
        
        if symbol:
            query = query.filter(Signal.symbol == symbol)
        
        return query.order_by(desc(Signal.confidence), desc(Signal.timestamp)).all()
    
    def get_signals_by_wave(
        self, 
        db: Session, 
        elliott_wave: str,
        symbol: Optional[str] = None
    ) -> List[Signal]:
        """Get signals by Elliott wave pattern"""
        query = db.query(Signal).filter(Signal.elliott_wave == elliott_wave)
        
        if symbol:
            query = query.filter(Signal.symbol == symbol)
        
        return query.order_by(desc(Signal.timestamp)).all()
    
    def update(self, db: Session, signal_id: int, **kwargs) -> Optional[Signal]:
        """Update signal record"""
        try:
            db_obj = self.get_by_id(db, signal_id)
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
            db_logger.error(f"Error updating signal: {e}")
            raise
    
    def mark_as_executed(self, db: Session, signal_id: int, trade_id: Optional[int] = None) -> Optional[Signal]:
        """Mark signal as executed"""
        try:
            signal = self.get_by_id(db, signal_id)
            if not signal:
                return None
            
            signal.executed = True
            if trade_id:
                signal.trade_id = trade_id
            
            db.commit()
            db.refresh(signal)
            db_logger.info(f"Marked signal {signal_id} as executed")
            return signal
        except Exception as e:
            db.rollback()
            db_logger.error(f"Error marking signal as executed: {e}")
            raise
    
    def delete(self, db: Session, signal_id: int) -> bool:
        """Delete signal record"""
        try:
            db_obj = self.get_by_id(db, signal_id)
            if not db_obj:
                return False
            
            db.delete(db_obj)
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            db_logger.error(f"Error deleting signal: {e}")
            raise
    
    def delete_old_signals(self, db: Session, days: int = 30) -> int:
        """Delete signals older than specified days"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            deleted_count = db.query(Signal)\
                .filter(Signal.timestamp < cutoff_date)\
                .delete()
            db.commit()
            
            if deleted_count > 0:
                db_logger.info(f"Deleted {deleted_count} old signals")
            
            return deleted_count
        except Exception as e:
            db.rollback()
            db_logger.error(f"Error deleting old signals: {e}")
            raise
    
    def get_signal_statistics(
        self, 
        db: Session, 
        symbol: Optional[str] = None,
        strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get signal statistics"""
        try:
            # Base query
            query = db.query(Signal)
            
            if symbol:
                query = query.filter(Signal.symbol == symbol)
            
            if strategy:
                query = query.filter(Signal.strategy == strategy)
            
            signals = query.all()
            
            if not signals:
                return {
                    'total_signals': 0,
                    'executed_signals': 0,
                    'pending_signals': 0,
                    'execution_rate': 0.0,
                    'average_confidence': 0.0,
                    'signal_types': {}
                }
            
            # Calculate statistics
            total_signals = len(signals)
            executed_signals = sum(1 for s in signals if s.executed)
            pending_signals = total_signals - executed_signals
            
            # Signal type distribution
            signal_types = {}
            for signal in signals:
                signal_type = signal.signal_type.value
                signal_types[signal_type] = signal_types.get(signal_type, 0) + 1
            
            # Strategy distribution
            strategies = {}
            for signal in signals:
                strategy_name = signal.strategy
                strategies[strategy_name] = strategies.get(strategy_name, 0) + 1
            
            # Confidence statistics
            confidences = [s.confidence for s in signals if s.confidence is not None]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'total_signals': total_signals,
                'executed_signals': executed_signals,
                'pending_signals': pending_signals,
                'execution_rate': (executed_signals / total_signals * 100) if total_signals > 0 else 0,
                'average_confidence': round(avg_confidence, 3),
                'signal_types': signal_types,
                'strategies': strategies,
                'high_confidence_signals': sum(1 for s in signals if s.confidence and s.confidence >= 0.8)
            }
        except Exception as e:
            db_logger.error(f"Error getting signal statistics: {e}")
            return {'error': str(e)}
    
    def get_signal_performance(self, db: Session, days: int = 30) -> List[Dict[str, Any]]:
        """Get signal performance analysis"""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # Get executed signals with linked trades
            signals = db.query(Signal)\
                .filter(and_(
                    Signal.executed == True,
                    Signal.timestamp >= start_date,
                    Signal.trade_id.isnot(None)
                ))\
                .all()
            
            performance_data = []
            for signal in signals:
                if signal.trades:  # Check if trade exists
                    trade = signal.trades[0] if isinstance(signal.trades, list) else signal.trades
                    
                    # Calculate signal effectiveness
                    was_profitable = trade.is_profitable() if hasattr(trade, 'is_profitable') else False
                    
                    performance_data.append({
                        'signal_id': signal.id,
                        'symbol': signal.symbol,
                        'strategy': signal.strategy,
                        'confidence': signal.confidence,
                        'signal_type': signal.signal_type.value,
                        'elliott_wave': signal.elliott_wave,
                        'was_profitable': was_profitable,
                        'profit': trade.calculate_total_profit() if hasattr(trade, 'calculate_total_profit') else 0,
                        'rr_ratio': signal.rr_ratio,
                        'timestamp': signal.timestamp.isoformat()
                    })
            
            return performance_data
        except Exception as e:
            db_logger.error(f"Error getting signal performance: {e}")
            return []
    
    def get_strategy_effectiveness(self, db: Session) -> Dict[str, Dict[str, Any]]:
        """Get effectiveness metrics by strategy"""
        try:
            # Get all executed signals with trades
            signals = db.query(Signal)\
                .filter(and_(
                    Signal.executed == True,
                    Signal.trade_id.isnot(None)
                ))\
                .all()
            
            strategy_stats = {}
            
            for signal in signals:
                strategy = signal.strategy
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {
                        'total_signals': 0,
                        'profitable_signals': 0,
                        'total_profit': 0.0,
                        'confidence_sum': 0.0,
                        'rr_ratio_sum': 0.0,
                        'rr_count': 0
                    }
                
                stats = strategy_stats[strategy]
                stats['total_signals'] += 1
                stats['confidence_sum'] += signal.confidence or 0
                
                if signal.rr_ratio:
                    stats['rr_ratio_sum'] += signal.rr_ratio
                    stats['rr_count'] += 1
                
                # Check if trade was profitable
                if signal.trades:
                    trade = signal.trades[0] if isinstance(signal.trades, list) else signal.trades
                    if hasattr(trade, 'is_profitable') and trade.is_profitable():
                        stats['profitable_signals'] += 1
                    
                    if hasattr(trade, 'calculate_total_profit'):
                        stats['total_profit'] += trade.calculate_total_profit()
            
            # Calculate final metrics
            result = {}
            for strategy, stats in strategy_stats.items():
                total = stats['total_signals']
                result[strategy] = {
                    'total_signals': total,
                    'win_rate': (stats['profitable_signals'] / total * 100) if total > 0 else 0,
                    'total_profit': round(stats['total_profit'], 2),
                    'average_confidence': round(stats['confidence_sum'] / total, 3) if total > 0 else 0,
                    'average_rr_ratio': round(stats['rr_ratio_sum'] / stats['rr_count'], 2) if stats['rr_count'] > 0 else 0
                }
            
            return result
        except Exception as e:
            db_logger.error(f"Error getting strategy effectiveness: {e}")
            return {}


# Global instance
signal_crud = SignalCRUD()