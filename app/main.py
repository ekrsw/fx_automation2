"""
FastAPI application for FX Auto Trading System
"""

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os

from app.config import get_settings
from app.dependencies import get_settings_cached
from app.db.database import create_tables

# Initialize FastAPI app
app = FastAPI(
    title="FX Auto Trading System",
    description="Automated FX trading system implementing Dow Theory and Elliott Wave analysis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if os.path.exists("frontend/static"):
    app.mount("/static", StaticFiles(directory="frontend/static"), name="static")


@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    settings = get_settings()
    
    # Create database tables
    create_tables()
    
    print(f"Starting {settings.app_name}...")
    print(f"Debug mode: {settings.debug}")
    print(f"Database: {settings.database_url}")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    print("Shutting down FX Auto Trading System...")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint"""
    return """
    <html>
        <head>
            <title>FX Auto Trading System</title>
        </head>
        <body>
            <h1>FX Auto Trading System</h1>
            <p>Welcome to the FX Auto Trading System</p>
            <ul>
                <li><a href="/docs">API Documentation</a></li>
                <li><a href="/health">Health Check</a></li>
            </ul>
        </body>
    </html>
    """


@app.get("/health")
async def health_check(settings: get_settings_cached = Depends()):
    """Health check endpoint"""
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": "1.0.0",
        "debug": settings.debug
    }


@app.get("/info")
async def app_info(settings: get_settings_cached = Depends()):
    """Application information"""
    return {
        "app_name": settings.app_name,
        "version": "1.0.0",
        "description": "Automated FX trading system implementing Dow Theory and Elliott Wave analysis",
        "trading_symbols": settings.trading_symbols,
        "debug_mode": settings.debug
    }


# Include API routers
from app.api import dashboard
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])
# Additional routers will be added in later phases
# from app.api import trading, analysis, settings as api_settings
# app.include_router(trading.router, prefix="/api/trading", tags=["trading"])
# app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
# app.include_router(api_settings.router, prefix="/api/settings", tags=["settings"])


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )