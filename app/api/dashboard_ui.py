"""
Dashboard UI API endpoints
Phase 7.3 Implementation - Real-time trading monitoring interface
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os

router = APIRouter()

# Setup templates
templates_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "frontend", "templates")
templates = Jinja2Templates(directory=templates_dir)


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """
    Serve the real-time trading dashboard page
    
    Returns:
        HTML dashboard interface
    """
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "title": "FX Auto Trading System - Real-time Dashboard"
    })


@router.get("/", response_class=HTMLResponse)
async def index_redirect(request: Request):
    """
    Redirect root to dashboard
    
    Returns:
        Redirect to dashboard
    """
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "title": "FX Auto Trading System - Real-time Dashboard"
    })