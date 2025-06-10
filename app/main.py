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
    
    # Initialize WebSocket integration
    from app.integrations.websocket_integration import websocket_integration
    websocket_integration.initialize()
    
    # Initialize MT5 connection if credentials are available
    if all([settings.mt5_login, settings.mt5_password, settings.mt5_server]):
        try:
            from app.mt5.connection import get_mt5_connection
            mt5_connection = get_mt5_connection()
            
            success = await mt5_connection.connect(
                login=settings.mt5_login,
                password=settings.mt5_password,
                server=settings.mt5_server
            )
            
            if success:
                print(f"✅ MT5 Connection: Successfully connected to {settings.mt5_server}")
            else:
                print(f"❌ MT5 Connection: Failed to connect to {settings.mt5_server}")
                
        except Exception as e:
            print(f"❌ MT5 Connection Error: {e}")
    else:
        print("⚠️  MT5 Connection: No credentials configured")
    
    print(f"Starting {settings.app_name}...")
    print(f"Debug mode: {settings.debug}")
    print(f"Database: {settings.database_url}")
    print(f"WebSocket enabled: {websocket_integration.enabled}")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    print("Shutting down FX Auto Trading System...")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - redirect to dashboard"""
    return """
    <html>
        <head>
            <title>FX Auto Trading System - Phase 7.3</title>
            <meta http-equiv="refresh" content="0; url=/ui/dashboard">
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    margin: 40px; 
                    text-align: center;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    flex-direction: column;
                }
                .container {
                    background: rgba(255, 255, 255, 0.1);
                    padding: 3rem;
                    border-radius: 15px;
                    backdrop-filter: blur(10px);
                }
                h1 { margin-bottom: 1rem; }
                p { margin-bottom: 2rem; }
                .btn {
                    display: inline-block;
                    padding: 12px 24px;
                    background: #28a745;
                    color: white;
                    text-decoration: none;
                    border-radius: 6px;
                    transition: background 0.3s;
                }
                .btn:hover {
                    background: #218838;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 FX Auto Trading System - Phase 7.3</h1>
                <p>Real-time Trading Dashboard with WebSocket Integration</p>
                <p>Redirecting to dashboard...</p>
                <a href="/ui/dashboard" class="btn">Go to Dashboard</a>
            </div>
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
from app.api import dashboard, trading, analysis, settings as api_settings, dashboard_ui, mt5_control
from app.api.websockets import websocket_endpoint
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])
app.include_router(trading.router, prefix="/api/trading", tags=["trading"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(api_settings.router, prefix="/api/settings", tags=["settings"])
app.include_router(mt5_control.router, prefix="/api/mt5", tags=["mt5"])
app.include_router(dashboard_ui.router, prefix="/ui", tags=["ui"])

# WebSocket endpoint
app.websocket("/ws")(websocket_endpoint)


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )