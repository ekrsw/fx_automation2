"""
Signal model for storing trading signals
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Enum, Boolean, ForeignKey, Index
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from app.db.database import Base


class SignalType(enum.Enum):
    """Signal type enumeration"""
    BUY = "BUY"
    SELL = "SELL"
    CLOSE = "CLOSE"


class Signal(Base):
    """
    Signal model for storing trading signals
    """
    __tablename__ = "signals"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    signal_type = Column(Enum(SignalType), nullable=False)
    strategy = Column(String(50), nullable=False, index=True)  # DOW_ELLIOTT, DOW_ONLY, etc.
    confidence = Column(Float, nullable=False)  # 0.0-1.0
    entry_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    dow_trend = Column(String(20))  # UPTREND, DOWNTREND, SIDEWAYS
    elliott_wave = Column(String(10))  # Wave1, Wave2, etc.
    fibonacci_level = Column(Float)
    rr_ratio = Column(Float)  # Risk-Reward Ratio
    timestamp = Column(DateTime, nullable=False, index=True)
    executed = Column(Boolean, default=False, index=True)
    trade_id = Column(Integer, ForeignKey("trades.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    trades = relationship("Trade", back_populates="signal")
    
    # Indexes
    __table_args__ = (
        Index('idx_timestamp_desc', 'timestamp', postgresql_using='btree'),
        Index('idx_strategy_executed', 'strategy', 'executed'),
        Index('idx_symbol_strategy', 'symbol', 'strategy'),
        Index('idx_signal_type_timestamp', 'signal_type', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<Signal(symbol='{self.symbol}', type='{self.signal_type.value}', strategy='{self.strategy}', confidence={self.confidence})>"
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'signal_type': self.signal_type.value if self.signal_type else None,
            'strategy': self.strategy,
            'confidence': self.confidence,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'dow_trend': self.dow_trend,
            'elliott_wave': self.elliott_wave,
            'fibonacci_level': self.fibonacci_level,
            'rr_ratio': self.rr_ratio,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'executed': self.executed,
            'trade_id': self.trade_id,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def is_valid_rr_ratio(self, min_ratio: float = 1.5):
        """Check if risk-reward ratio is acceptable"""
        if self.rr_ratio is None:
            return False
        return self.rr_ratio >= min_ratio
    
    def calculate_position_size(self, account_balance: float, risk_percentage: float = 0.02):
        """Calculate position size based on risk management"""
        if not self.entry_price or not self.stop_loss:
            return None
            
        risk_amount = account_balance * risk_percentage
        pip_value = abs(self.entry_price - self.stop_loss)
        
        if pip_value == 0:
            return None
            
        # Simple position size calculation (would need symbol-specific pip value in real implementation)
        position_size = risk_amount / pip_value
        return round(position_size, 2)
    
    def get_signal_quality_score(self):
        """Calculate overall signal quality score"""
        score = self.confidence * 100
        
        # Bonus for good risk-reward ratio
        if self.is_valid_rr_ratio():
            score += 10
            
        # Bonus for specific wave patterns
        if self.elliott_wave in ['Wave3', 'Wave5']:
            score += 15
            
        # Bonus for strong trend confirmation
        if self.dow_trend in ['UPTREND', 'DOWNTREND']:
            score += 5
            
        return min(score, 100)  # Cap at 100