"""
System settings model for dynamic configuration
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Index
from datetime import datetime
import json

from app.db.database import Base


class SystemSettings(Base):
    """
    System settings model for storing dynamic configuration
    """
    __tablename__ = "system_settings"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, nullable=False, index=True)
    value = Column(Text, nullable=False)
    value_type = Column(String(20), default="string")  # string, int, float, bool, json
    description = Column(Text)
    category = Column(String(50), index=True)  # trading, risk_management, strategy, etc.
    is_active = Column(String(10), default="true")  # true, false as string
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_category_active', 'category', 'is_active'),
        Index('idx_key_active', 'key', 'is_active'),
    )
    
    def __repr__(self):
        return f"<SystemSettings(key='{self.key}', value='{self.value}', type='{self.value_type}')>"
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'key': self.key,
            'value': self.get_typed_value(),
            'value_type': self.value_type,
            'description': self.description,
            'category': self.category,
            'is_active': self.is_active == "true",
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def get_typed_value(self):
        """Get value converted to appropriate type"""
        if self.value is None:
            return None
            
        try:
            if self.value_type == "int":
                return int(self.value)
            elif self.value_type == "float":
                return float(self.value)
            elif self.value_type == "bool":
                return self.value.lower() in ('true', '1', 'yes', 'on')
            elif self.value_type == "json":
                return json.loads(self.value)
            else:  # string or unknown
                return self.value
        except (ValueError, json.JSONDecodeError):
            # If conversion fails, return as string
            return self.value
    
    def set_typed_value(self, value, value_type=None):
        """Set value with automatic type conversion"""
        if value_type:
            self.value_type = value_type
            
        if self.value_type == "json":
            self.value = json.dumps(value)
        elif self.value_type == "bool":
            self.value = "true" if value else "false"
        else:
            self.value = str(value)
    
    @classmethod
    def create_default_settings(cls):
        """Create default system settings"""
        default_settings = [
            # Trading settings
            {
                'key': 'trading.default_volume',
                'value': '0.1',
                'value_type': 'float',
                'description': 'Default trading volume',
                'category': 'trading'
            },
            {
                'key': 'trading.max_positions',
                'value': '3',
                'value_type': 'int',
                'description': 'Maximum number of open positions',
                'category': 'trading'
            },
            {
                'key': 'trading.symbols',
                'value': '["USDJPY", "EURJPY", "GBPJPY"]',
                'value_type': 'json',
                'description': 'List of trading symbols',
                'category': 'trading'
            },
            
            # Risk management settings
            {
                'key': 'risk.max_daily_loss',
                'value': '0.05',
                'value_type': 'float',
                'description': 'Maximum daily loss as percentage of account',
                'category': 'risk_management'
            },
            {
                'key': 'risk.max_drawdown',
                'value': '0.15',
                'value_type': 'float',
                'description': 'Maximum allowable drawdown',
                'category': 'risk_management'
            },
            {
                'key': 'risk.risk_per_trade',
                'value': '0.02',
                'value_type': 'float',
                'description': 'Risk percentage per trade',
                'category': 'risk_management'
            },
            
            # Strategy settings
            {
                'key': 'strategy.dow.atr_multiplier',
                'value': '0.5',
                'value_type': 'float',
                'description': 'ATR multiplier for Dow theory analysis',
                'category': 'strategy'
            },
            {
                'key': 'strategy.dow.swing_period',
                'value': '5',
                'value_type': 'int',
                'description': 'Period for swing point detection',
                'category': 'strategy'
            },
            {
                'key': 'strategy.elliott.fibonacci_tolerance',
                'value': '0.1',
                'value_type': 'float',
                'description': 'Tolerance for Fibonacci level matching',
                'category': 'strategy'
            },
            {
                'key': 'strategy.elliott.min_wave_size',
                'value': '0.001',
                'value_type': 'float',
                'description': 'Minimum wave size for Elliott wave analysis',
                'category': 'strategy'
            },
            
            # System settings
            {
                'key': 'system.auto_trading_enabled',
                'value': 'false',
                'value_type': 'bool',
                'description': 'Enable automatic trading',
                'category': 'system'
            },
            {
                'key': 'system.data_update_interval',
                'value': '60',
                'value_type': 'int',
                'description': 'Data update interval in seconds',
                'category': 'system'
            },
            {
                'key': 'system.signal_generation_interval',
                'value': '300',
                'value_type': 'int',
                'description': 'Signal generation interval in seconds',
                'category': 'system'
            }
        ]
        
        return [cls(**setting) for setting in default_settings]