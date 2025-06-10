"""
CRUD operations for system settings
"""

from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
from typing import List, Optional, Dict, Any, Union
import json

from app.db.models.settings import SystemSettings
from app.utils.logger import db_logger


class SettingsCRUD:
    """CRUD operations for system settings"""
    
    def create(self, db: Session, **kwargs) -> SystemSettings:
        """Create new setting record"""
        try:
            db_obj = SystemSettings(**kwargs)
            db.add(db_obj)
            db.commit()
            db.refresh(db_obj)
            db_logger.info(f"Created setting: {db_obj.key}")
            return db_obj
        except Exception as e:
            db.rollback()
            db_logger.error(f"Error creating setting: {e}")
            raise
    
    def get_by_id(self, db: Session, setting_id: int) -> Optional[SystemSettings]:
        """Get setting by ID"""
        return db.query(SystemSettings).filter(SystemSettings.id == setting_id).first()
    
    def get_by_key(self, db: Session, key: str) -> Optional[SystemSettings]:
        """Get setting by key"""
        return db.query(SystemSettings).filter(SystemSettings.key == key).first()
    
    def get_all_active(self, db: Session) -> List[SystemSettings]:
        """Get all active settings"""
        return db.query(SystemSettings)\
            .filter(SystemSettings.is_active == "true")\
            .order_by(SystemSettings.category, SystemSettings.key)\
            .all()
    
    def get_by_category(self, db: Session, category: str) -> List[SystemSettings]:
        """Get settings by category"""
        return db.query(SystemSettings)\
            .filter(and_(
                SystemSettings.category == category,
                SystemSettings.is_active == "true"
            ))\
            .order_by(SystemSettings.key)\
            .all()
    
    def get_value(self, db: Session, key: str, default: Any = None) -> Any:
        """Get setting value with type conversion"""
        setting = self.get_by_key(db, key)
        if not setting or setting.is_active != "true":
            return default
        
        return setting.get_typed_value()
    
    def set_value(
        self, 
        db: Session, 
        key: str, 
        value: Any, 
        value_type: Optional[str] = None,
        description: Optional[str] = None,
        category: Optional[str] = None
    ) -> SystemSettings:
        """Set setting value, create if doesn't exist"""
        try:
            setting = self.get_by_key(db, key)
            
            if setting:
                # Update existing setting
                setting.set_typed_value(value, value_type)
                if description:
                    setting.description = description
                if category:
                    setting.category = category
            else:
                # Create new setting
                setting = SystemSettings(
                    key=key,
                    description=description or f"Setting for {key}",
                    category=category or "general"
                )
                setting.set_typed_value(value, value_type)
                db.add(setting)
            
            db.commit()
            db.refresh(setting)
            db_logger.info(f"Set setting {key} = {value}")
            return setting
        except Exception as e:
            db.rollback()
            db_logger.error(f"Error setting value for {key}: {e}")
            raise
    
    def update(self, db: Session, setting_id: int, **kwargs) -> Optional[SystemSettings]:
        """Update setting record"""
        try:
            db_obj = self.get_by_id(db, setting_id)
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
            db_logger.error(f"Error updating setting: {e}")
            raise
    
    def delete(self, db: Session, setting_id: int) -> bool:
        """Delete setting record"""
        try:
            db_obj = self.get_by_id(db, setting_id)
            if not db_obj:
                return False
            
            db.delete(db_obj)
            db.commit()
            db_logger.info(f"Deleted setting: {db_obj.key}")
            return True
        except Exception as e:
            db.rollback()
            db_logger.error(f"Error deleting setting: {e}")
            raise
    
    def deactivate(self, db: Session, key: str) -> bool:
        """Deactivate setting (soft delete)"""
        try:
            setting = self.get_by_key(db, key)
            if not setting:
                return False
            
            setting.is_active = "false"
            db.commit()
            db_logger.info(f"Deactivated setting: {key}")
            return True
        except Exception as e:
            db.rollback()
            db_logger.error(f"Error deactivating setting {key}: {e}")
            raise
    
    def activate(self, db: Session, key: str) -> bool:
        """Activate setting"""
        try:
            setting = self.get_by_key(db, key)
            if not setting:
                return False
            
            setting.is_active = "true"
            db.commit()
            db_logger.info(f"Activated setting: {key}")
            return True
        except Exception as e:
            db.rollback()
            db_logger.error(f"Error activating setting {key}: {e}")
            raise
    
    def get_as_dict(self, db: Session, category: Optional[str] = None) -> Dict[str, Any]:
        """Get settings as a dictionary"""
        if category:
            settings = self.get_by_category(db, category)
        else:
            settings = self.get_all_active(db)
        
        result = {}
        for setting in settings:
            # Create nested dictionary structure from dot notation
            keys = setting.key.split('.')
            current = result
            
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            current[keys[-1]] = setting.get_typed_value()
        
        return result
    
    def get_trading_config(self, db: Session) -> Dict[str, Any]:
        """Get trading-specific configuration"""
        return self.get_as_dict(db, "trading")
    
    def get_risk_config(self, db: Session) -> Dict[str, Any]:
        """Get risk management configuration"""
        return self.get_as_dict(db, "risk_management")
    
    def get_strategy_config(self, db: Session) -> Dict[str, Any]:
        """Get strategy configuration"""
        return self.get_as_dict(db, "strategy")
    
    def get_system_config(self, db: Session) -> Dict[str, Any]:
        """Get system configuration"""
        return self.get_as_dict(db, "system")
    
    def update_batch(self, db: Session, settings: Dict[str, Any]) -> int:
        """Update multiple settings at once"""
        try:
            updated_count = 0
            
            for key, value in settings.items():
                # Determine value type
                value_type = "string"
                if isinstance(value, bool):
                    value_type = "bool"
                elif isinstance(value, int):
                    value_type = "int"
                elif isinstance(value, float):
                    value_type = "float"
                elif isinstance(value, (list, dict)):
                    value_type = "json"
                
                self.set_value(db, key, value, value_type)
                updated_count += 1
            
            db_logger.info(f"Updated {updated_count} settings in batch")
            return updated_count
        except Exception as e:
            db_logger.error(f"Error updating settings batch: {e}")
            raise
    
    def initialize_default_settings(self, db: Session) -> int:
        """Initialize default settings if they don't exist"""
        try:
            default_settings = SystemSettings.create_default_settings()
            created_count = 0
            
            for setting in default_settings:
                existing = self.get_by_key(db, setting.key)
                if not existing:
                    db.add(setting)
                    created_count += 1
            
            if created_count > 0:
                db.commit()
                db_logger.info(f"Initialized {created_count} default settings")
            
            return created_count
        except Exception as e:
            db.rollback()
            db_logger.error(f"Error initializing default settings: {e}")
            raise
    
    def export_settings(self, db: Session, category: Optional[str] = None) -> Dict[str, Any]:
        """Export settings for backup/configuration"""
        settings = self.get_by_category(db, category) if category else self.get_all_active(db)
        
        export_data = {
            'settings': [],
            'export_timestamp': db_logger.handlers[0].formatter.formatTime(db_logger.makeRecord("", 0, "", 0, "", (), None)) if db_logger.handlers else None,
            'category': category
        }
        
        for setting in settings:
            export_data['settings'].append({
                'key': setting.key,
                'value': setting.value,
                'value_type': setting.value_type,
                'description': setting.description,
                'category': setting.category
            })
        
        return export_data
    
    def import_settings(self, db: Session, settings_data: Dict[str, Any]) -> int:
        """Import settings from backup/configuration"""
        try:
            imported_count = 0
            
            if 'settings' not in settings_data:
                raise ValueError("Invalid settings data format")
            
            for setting_data in settings_data['settings']:
                self.set_value(
                    db,
                    setting_data['key'],
                    setting_data['value'],
                    setting_data.get('value_type', 'string'),
                    setting_data.get('description'),
                    setting_data.get('category')
                )
                imported_count += 1
            
            db_logger.info(f"Imported {imported_count} settings")
            return imported_count
        except Exception as e:
            db_logger.error(f"Error importing settings: {e}")
            raise


# Global instance
settings_crud = SettingsCRUD()