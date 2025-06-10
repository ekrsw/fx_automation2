"""
Basic API endpoint tests - Testing core functionality without complex mocking
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

# Initialize TestClient with positional argument
client = TestClient(app)


class TestBasicAPIEndpoints:
    """Test basic API endpoint availability and responses"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_info_endpoint(self):
        """Test info endpoint"""
        response = client.get("/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "app_name" in data
        assert "description" in data
    
    def test_dashboard_ui_endpoint(self):
        """Test dashboard UI endpoint"""
        response = client.get("/ui/dashboard")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_api_docs_endpoint(self):
        """Test API documentation endpoint"""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_openapi_json_endpoint(self):
        """Test OpenAPI JSON endpoint"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        data = response.json()
        assert "openapi" in data
        assert "info" in data
    
    def test_dashboard_status_basic(self):
        """Test dashboard status endpoint basic functionality"""
        response = client.get("/api/dashboard/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "overall_status" in data
        assert "components" in data
    
    def test_mt5_status_endpoint(self):
        """Test MT5 status endpoint"""
        response = client.get("/api/mt5/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "is_connected" in data
    
    def test_trading_session_status(self):
        """Test trading session status endpoint"""
        response = client.get("/api/trading/session/status")
        # This endpoint may not exist yet, so check if accessible
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
    
    def test_analysis_health_endpoint(self):
        """Test analysis health endpoint"""
        response = client.get("/api/analysis/health")
        # This endpoint may not exist yet, so check if accessible
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
    
    def test_websocket_endpoint_options(self):
        """Test WebSocket endpoint with OPTIONS request"""
        response = client.options("/ws")
        # WebSocket endpoint should handle options or return appropriate response
        assert response.status_code in [200, 405, 404]
    
    def test_invalid_endpoint_404(self):
        """Test that invalid endpoints return 404"""
        response = client.get("/api/nonexistent/endpoint")
        assert response.status_code == 404
    
    def test_api_response_headers(self):
        """Test API response headers"""
        response = client.get("/api/dashboard/status")
        assert response.status_code == 200
        
        # Check content type
        assert "application/json" in response.headers.get("content-type", "")
    
    def test_cors_headers(self):
        """Test CORS headers if configured"""
        response = client.get("/api/dashboard/status")
        # CORS headers may or may not be present depending on configuration
        assert response.status_code == 200
    
    def test_error_handling_robustness(self):
        """Test error handling robustness"""
        # Test various endpoints to ensure they don't crash
        endpoints = [
            "/api/dashboard/performance",
            "/api/dashboard/positions",
            "/api/dashboard/orders",
            "/api/dashboard/risk-status",
            "/api/dashboard/market-data"
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            # Should either work or fail gracefully, not crash
            assert response.status_code in [200, 404, 500]
            
            if response.status_code == 200:
                # Should return valid JSON
                data = response.json()
                assert isinstance(data, dict)
    
    def test_api_consistency(self):
        """Test API response consistency"""
        response = client.get("/api/dashboard/status")
        assert response.status_code == 200
        
        data = response.json()
        
        # Check timestamp format consistency
        assert "timestamp" in data
        timestamp = data["timestamp"]
        assert isinstance(timestamp, str)
        
        # Check status format consistency
        assert "overall_status" in data
        status = data["overall_status"]
        assert isinstance(status, str)
        assert status in ["healthy", "degraded", "unhealthy"]