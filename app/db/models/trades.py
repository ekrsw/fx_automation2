"""
Trade model for storing trading transactions
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Enum, Boolean, ForeignKey, Index
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from app.db.database import Base


class TradeType(enum.Enum):
    """Trade type enumeration"""
    BUY = "BUY"
    SELL = "SELL"


class TradeStatus(enum.Enum):
    """Trade status enumeration"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


class Trade(Base):
    """
    Trade model for storing trading transactions
    """
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    ticket = Column(Integer, unique=True, index=True)  # MT5チケット番号
    symbol = Column(String(10), nullable=False, index=True)
    trade_type = Column(Enum(TradeType), nullable=False)
    volume = Column(Float, nullable=False)
    open_price = Column(Float, nullable=False)
    close_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    open_time = Column(DateTime, nullable=False, index=True)
    close_time = Column(DateTime)
    profit = Column(Float, default=0.0)
    commission = Column(Float, default=0.0)
    swap = Column(Float, default=0.0)
    status = Column(Enum(TradeStatus), default=TradeStatus.OPEN, index=True)
    strategy_id = Column(String(50))  # どの戦略での取引か
    signal_id = Column(Integer, ForeignKey("signals.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    signal = relationship("Signal", back_populates="trades")
    
    # Indexes
    __table_args__ = (
        Index('idx_symbol_status', 'symbol', 'status'),
        Index('idx_open_time_desc', 'open_time', postgresql_using='btree'),
        Index('idx_strategy_status', 'strategy_id', 'status'),
    )
    
    def __repr__(self):
        return f"<Trade(ticket={self.ticket}, symbol='{self.symbol}', type='{self.trade_type.value}', status='{self.status.value}')>"
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'ticket': self.ticket,
            'symbol': self.symbol,
            'trade_type': self.trade_type.value if self.trade_type else None,
            'volume': self.volume,
            'open_price': self.open_price,
            'close_price': self.close_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'open_time': self.open_time.isoformat() if self.open_time else None,
            'close_time': self.close_time.isoformat() if self.close_time else None,
            'profit': self.profit,
            'commission': self.commission,
            'swap': self.swap,
            'status': self.status.value if self.status else None,
            'strategy_id': self.strategy_id,
            'signal_id': self.signal_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def calculate_total_profit(self):
        """Calculate total profit including commission and swap"""
        return (self.profit or 0.0) + (self.commission or 0.0) + (self.swap or 0.0)
    
    def is_profitable(self):
        """Check if trade is profitable"""
        return self.calculate_total_profit() > 0
    
    def get_duration_seconds(self):
        """Get trade duration in seconds"""
        if self.open_time and self.close_time:
            return (self.close_time - self.open_time).total_seconds()
        return None