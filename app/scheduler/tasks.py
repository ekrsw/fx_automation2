"""
Scheduled tasks for FX trading system
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from app.services.data_service import data_service
from app.mt5.connection import get_mt5_connection
from app.db.crud.settings import settings_crud
from app.db.database import SessionLocal
from app.config import get_settings
from app.utils.logger import main_logger


class TradingScheduler:
    """Scheduler for trading system tasks"""
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.settings = get_settings()
        self.is_running = False
        
    async def start(self):
        """Start the scheduler"""
        if self.is_running:
            main_logger.warning("Scheduler is already running")
            return
        
        try:
            # Schedule data collection tasks
            await self._schedule_data_tasks()
            
            # Schedule maintenance tasks
            await self._schedule_maintenance_tasks()
            
            # Schedule health check tasks
            await self._schedule_health_tasks()
            
            # Start the scheduler
            self.scheduler.start()
            self.is_running = True
            
            main_logger.info("Trading scheduler started successfully")
            
        except Exception as e:
            main_logger.error(f"Error starting scheduler: {e}")
            raise
    
    async def stop(self):
        """Stop the scheduler"""
        if not self.is_running:
            return
        
        try:
            self.scheduler.shutdown()
            self.is_running = False
            main_logger.info("Trading scheduler stopped")
        except Exception as e:
            main_logger.error(f"Error stopping scheduler: {e}")
    
    async def _schedule_data_tasks(self):
        """Schedule data collection tasks"""
        # Get data update interval from settings
        db = SessionLocal()
        try:
            update_interval = settings_crud.get_value(db, 'system.data_update_interval', 60)
        finally:
            db.close()
        
        # Schedule price data collection
        self.scheduler.add_job(
            self._collect_price_data,
            trigger=IntervalTrigger(seconds=update_interval),
            id='price_data_collection',
            name='Price Data Collection',
            max_instances=1,
            coalesce=True
        )
        
        # Schedule historical data update (daily)
        self.scheduler.add_job(
            self._update_historical_data,
            trigger=CronTrigger(hour=1, minute=0),  # 1 AM daily
            id='historical_data_update',
            name='Historical Data Update',
            max_instances=1
        )
        
        main_logger.info(f"Scheduled data collection tasks (interval: {update_interval}s)")
    
    async def _schedule_maintenance_tasks(self):
        """Schedule maintenance tasks"""
        # Data cleanup (weekly)
        self.scheduler.add_job(
            self._cleanup_old_data,
            trigger=CronTrigger(day_of_week=0, hour=2, minute=0),  # Sunday 2 AM
            id='data_cleanup',
            name='Data Cleanup',
            max_instances=1
        )
        
        # Database optimization (monthly)
        self.scheduler.add_job(
            self._optimize_database,
            trigger=CronTrigger(day=1, hour=3, minute=0),  # 1st of month, 3 AM
            id='database_optimization',
            name='Database Optimization',
            max_instances=1
        )
        
        main_logger.info("Scheduled maintenance tasks")
    
    async def _schedule_health_tasks(self):
        """Schedule health monitoring tasks"""
        # System health check
        self.scheduler.add_job(
            self._system_health_check,
            trigger=IntervalTrigger(seconds=30),
            id='system_health_check',
            name='System Health Check',
            max_instances=1,
            coalesce=True
        )
        
        # MT5 connection monitoring
        self.scheduler.add_job(
            self._monitor_mt5_connection,
            trigger=IntervalTrigger(seconds=60),
            id='mt5_connection_monitor',
            name='MT5 Connection Monitor',
            max_instances=1,
            coalesce=True
        )
        
        main_logger.info("Scheduled health monitoring tasks")
    
    async def _collect_price_data(self):
        """Collect current price data"""
        try:
            # Get trading symbols from settings
            db = SessionLocal()
            try:
                symbols = settings_crud.get_value(db, 'trading.symbols', ['USDJPY', 'EURJPY', 'GBPJPY'])
            finally:
                db.close()
            
            # Collect live prices
            result = await data_service.fetch_and_store_live_prices(symbols)
            
            if result['status'] == 'success':
                main_logger.debug(f"Collected price data for {result['symbols_processed']} symbols")
            else:
                main_logger.warning(f"Price data collection issue: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            main_logger.error(f"Error in price data collection: {e}")
    
    async def _update_historical_data(self):
        """Update historical data for all symbols"""
        try:
            db = SessionLocal()
            try:
                symbols = settings_crud.get_value(db, 'trading.symbols', ['USDJPY', 'EURJPY', 'GBPJPY'])
                timeframes = ['H1', 'H4', 'D1']
            finally:
                db.close()
            
            # Update recent data for all symbols and timeframes
            result = await data_service.update_all_symbols(symbols, timeframes, count=100)
            
            main_logger.info(f"Historical data update completed: {result['successful_updates']} successful, {result['failed_updates']} failed")
            
        except Exception as e:
            main_logger.error(f"Error in historical data update: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old data"""
        try:
            # Clean up price data older than 1 year
            deleted_count = data_service.cleanup_old_data(days=365)
            
            # Clean up old signals (30 days)
            db = SessionLocal()
            try:
                from app.db.crud.signals import signal_crud
                deleted_signals = signal_crud.delete_old_signals(db, days=30)
            finally:
                db.close()
            
            main_logger.info(f"Data cleanup completed: {deleted_count} price records, {deleted_signals} signals deleted")
            
        except Exception as e:
            main_logger.error(f"Error in data cleanup: {e}")
    
    async def _optimize_database(self):
        """Optimize database performance"""
        try:
            # For SQLite, we can run VACUUM and ANALYZE commands
            from app.db.database import engine
            
            with engine.connect() as connection:
                connection.execute("VACUUM")
                connection.execute("ANALYZE")
            
            main_logger.info("Database optimization completed")
            
        except Exception as e:
            main_logger.error(f"Error in database optimization: {e}")
    
    async def _system_health_check(self):
        """Perform system health check"""
        try:
            # Check if all critical components are running
            health_status = {
                'timestamp': datetime.utcnow().isoformat(),
                'scheduler_running': self.is_running,
                'mt5_connected': False,
                'database_accessible': False
            }
            
            # Check MT5 connection
            mt5_connection = get_mt5_connection()
            health_status['mt5_connected'] = await mt5_connection.health_check()
            
            # Check database
            try:
                db = SessionLocal()
                db.execute("SELECT 1")
                health_status['database_accessible'] = True
                db.close()
            except:
                health_status['database_accessible'] = False
            
            # Log warnings for any issues
            if not health_status['mt5_connected']:
                main_logger.warning("MT5 connection health check failed")
            
            if not health_status['database_accessible']:
                main_logger.error("Database health check failed")
            
        except Exception as e:
            main_logger.error(f"Error in system health check: {e}")
    
    async def _monitor_mt5_connection(self):
        """Monitor MT5 connection and attempt reconnection if needed"""
        try:
            mt5_connection = get_mt5_connection()
            
            if not mt5_connection.is_connected():
                main_logger.warning("MT5 disconnected, attempting reconnection...")
                
                # Attempt reconnection
                success = await mt5_connection.reconnect()
                
                if success:
                    main_logger.info("MT5 reconnection successful")
                else:
                    main_logger.error("MT5 reconnection failed")
            
        except Exception as e:
            main_logger.error(f"Error in MT5 connection monitoring: {e}")
    
    def get_job_status(self) -> List[Dict[str, Any]]:
        """Get status of all scheduled jobs"""
        jobs = []
        
        if not self.is_running:
            return jobs
        
        for job in self.scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'name': job.name,
                'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None,
                'trigger': str(job.trigger),
                'max_instances': job.max_instances,
                'coalesce': job.coalesce
            })
        
        return jobs
    
    async def run_job_manually(self, job_id: str) -> bool:
        """Run a scheduled job manually"""
        try:
            job = self.scheduler.get_job(job_id)
            if not job:
                main_logger.error(f"Job {job_id} not found")
                return False
            
            # Execute the job function
            if job_id == 'price_data_collection':
                await self._collect_price_data()
            elif job_id == 'historical_data_update':
                await self._update_historical_data()
            elif job_id == 'data_cleanup':
                await self._cleanup_old_data()
            elif job_id == 'database_optimization':
                await self._optimize_database()
            elif job_id == 'system_health_check':
                await self._system_health_check()
            elif job_id == 'mt5_connection_monitor':
                await self._monitor_mt5_connection()
            else:
                main_logger.error(f"Unknown job ID: {job_id}")
                return False
            
            main_logger.info(f"Manually executed job: {job_id}")
            return True
            
        except Exception as e:
            main_logger.error(f"Error running job {job_id} manually: {e}")
            return False


# Global scheduler instance
trading_scheduler = TradingScheduler()