/**
 * FX Auto Trading System - Real-time Dashboard
 * Phase 7.3 Implementation - WebSocket Integration
 */

class TradingDashboard {
    constructor() {
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 5000;
        this.priceChart = null;
        this.priceData = {};
        
        this.init();
    }
    
    init() {
        this.setupWebSocket();
        this.setupEventHandlers();
        this.setupChart();
        this.loadInitialData();
    }
    
    setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        try {
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = (event) => {
                console.log('WebSocket connected');
                this.reconnectAttempts = 0;
                this.updateConnectionStatus(true);
                this.subscribeToAll();
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };
            
            this.ws.onclose = (event) => {
                console.log('WebSocket disconnected');
                this.updateConnectionStatus(false);
                this.scheduleReconnect();
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus(false);
            };
            
        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
            this.updateConnectionStatus(false);
        }
    }
    
    scheduleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect in ${this.reconnectDelay / 1000} seconds... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
            setTimeout(() => {
                this.setupWebSocket();
            }, this.reconnectDelay);
        } else {
            console.error('Max reconnection attempts reached');
        }
    }
    
    subscribeToAll() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            // Subscribe to all available data types
            const subscriptions = [
                'market_data',
                'signals', 
                'trading_events',
                'system_status',
                'positions',
                'orders',
                'risk_alerts'
            ];
            
            subscriptions.forEach(type => {
                this.ws.send(JSON.stringify({
                    action: 'subscribe',
                    subscription_type: type,
                    symbols: ['EURUSD', 'USDJPY', 'GBPUSD', 'USDCHF']
                }));
            });
        }
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'market_data':
                this.handleMarketData(data.data);
                break;
            case 'signal':
                this.handleSignal(data.data);
                break;
            case 'trading_event':
                this.handleTradingEvent(data.data);
                break;
            case 'position_update':
                this.handlePositionUpdate(data.data);
                break;
            case 'order_update':
                this.handleOrderUpdate(data.data);
                break;
            case 'system_status':
                this.handleSystemStatus(data.data);
                break;
            case 'risk_alert':
                this.handleRiskAlert(data.data);
                break;
            default:
                console.log('Unknown message type:', data.type);
        }
    }
    
    handleMarketData(data) {
        if (data.symbol && data.bid && data.ask) {
            this.updateMarketDataTable(data);
            this.updatePriceChart(data);
        }
    }
    
    handleSignal(data) {
        this.addSignalToTable(data);
        this.addToTradingLog(`📊 Signal: ${data.signal_type} for ${data.symbol} @ ${data.entry_price}`, 'info');
    }
    
    handleTradingEvent(data) {
        let logMessage = '';
        let logType = 'info';
        
        switch (data.event_type) {
            case 'session_started':
                logMessage = `🟢 Trading session started in ${data.mode} mode`;
                logType = 'success';
                break;
            case 'session_stopped':
                logMessage = '🔴 Trading session stopped';
                logType = 'warning';
                break;
            case 'session_paused':
                logMessage = '⏸️ Trading paused';
                logType = 'warning';
                break;
            case 'session_resumed':
                logMessage = '▶️ Trading resumed';
                logType = 'success';
                break;
            case 'order_placed':
                logMessage = `📋 Order placed: ${data.symbol} ${data.order_type} ${data.volume}`;
                break;
            case 'order_cancelled':
                logMessage = `❌ Order cancelled: ${data.order_id}`;
                break;
            case 'position_opened':
                logMessage = `📈 Position opened: ${data.symbol} ${data.volume}`;
                logType = 'success';
                break;
            case 'position_closed':
                logMessage = `📉 Position closed: ${data.symbol} P&L: ${data.realized_pnl}`;
                logType = data.realized_pnl > 0 ? 'success' : 'danger';
                break;
            case 'position_modified':
                logMessage = `⚙️ Position modified: ${data.position_id}`;
                break;
        }
        
        if (logMessage) {
            this.addToTradingLog(logMessage, logType);
        }
    }
    
    handlePositionUpdate(data) {
        this.updatePositionsTable();
    }
    
    handleOrderUpdate(data) {
        this.updateOrderCount();
    }
    
    handleSystemStatus(data) {
        this.updateSystemStatus(data);
    }
    
    handleRiskAlert(data) {
        const alertClass = data.severity === 'critical' ? 'danger' : 
                          data.severity === 'high' ? 'warning' : 'info';
        this.addToTradingLog(`⚠️ Risk Alert: ${data.alert_type} - ${data.reason || ''}`, alertClass);
        
        if (data.severity === 'critical') {
            this.showAlert('Critical Risk Alert', data.alert_type, 'danger');
        }
    }
    
    updateConnectionStatus(connected) {
        const connectedEl = document.getElementById('wsConnected');
        const disconnectedEl = document.getElementById('wsDisconnected');
        
        if (connected) {
            connectedEl.classList.remove('d-none');
            disconnectedEl.classList.add('d-none');
        } else {
            connectedEl.classList.add('d-none');
            disconnectedEl.classList.remove('d-none');
        }
    }
    
    updateMarketDataTable(data) {
        const tableBody = document.getElementById('marketDataTable');
        let row = document.querySelector(`tr[data-symbol="${data.symbol}"]`);
        
        if (!row) {
            row = document.createElement('tr');
            row.setAttribute('data-symbol', data.symbol);
            tableBody.appendChild(row);
            
            // Clear "loading" message if this is first data
            if (tableBody.children.length === 2) {
                tableBody.querySelector('td[colspan]')?.parentElement.remove();
            }
        }
        
        const spread = (data.ask - data.bid).toFixed(5);
        const change = data.change || 0;
        const changeClass = change >= 0 ? 'price-up' : 'price-down';
        const changeIcon = change >= 0 ? '↗' : '↘';
        
        row.innerHTML = `
            <td><strong>${data.symbol}</strong></td>
            <td>${data.bid.toFixed(5)}</td>
            <td>${data.ask.toFixed(5)}</td>
            <td>${spread}</td>
            <td class="${changeClass}">${changeIcon} ${Math.abs(change).toFixed(5)}</td>
        `;
    }
    
    updatePriceChart(data) {
        if (!this.priceChart) return;
        
        const selectedSymbol = document.getElementById('symbolSelect').value;
        if (data.symbol !== selectedSymbol) return;
        
        const now = new Date();
        const price = (data.bid + data.ask) / 2;
        
        // Keep only last 50 data points
        if (this.priceChart.data.labels.length >= 50) {
            this.priceChart.data.labels.shift();
            this.priceChart.data.datasets[0].data.shift();
        }
        
        this.priceChart.data.labels.push(now.toLocaleTimeString());
        this.priceChart.data.datasets[0].data.push(price);
        this.priceChart.update('none');
    }
    
    addSignalToTable(data) {
        const tableBody = document.getElementById('signalsTable');
        
        // Clear "no signals" message
        const noDataRow = tableBody.querySelector('td[colspan]');
        if (noDataRow) {
            noDataRow.parentElement.remove();
        }
        
        const row = document.createElement('tr');
        const time = new Date(data.processed_at || data.timestamp).toLocaleTimeString();
        const confidence = Math.round(data.confidence * 100);
        
        row.innerHTML = `
            <td>${time}</td>
            <td><strong>${data.symbol}</strong></td>
            <td><span class="badge bg-primary">${data.signal_type}</span></td>
            <td>${data.entry_price}</td>
            <td>
                <div class="progress" style="height: 20px;">
                    <div class="progress-bar" style="width: ${confidence}%">${confidence}%</div>
                </div>
            </td>
            <td><span class="badge bg-success">Processed</span></td>
        `;
        
        // Insert at top
        tableBody.insertBefore(row, tableBody.firstChild);
        
        // Keep only last 10 signals
        while (tableBody.children.length > 10) {
            tableBody.removeChild(tableBody.lastChild);
        }
    }
    
    addToTradingLog(message, type = 'info') {
        const logContainer = document.getElementById('tradingLog');
        const time = new Date().toLocaleTimeString();
        
        const logEntry = document.createElement('div');
        logEntry.className = `mb-2 p-2 border-start border-3 border-${type}`;
        logEntry.innerHTML = `
            <small class="text-muted">${time}</small><br>
            ${message}
        `;
        
        // Clear "waiting" message
        if (logContainer.querySelector('.text-muted')?.textContent === 'Waiting for trading events...') {
            logContainer.innerHTML = '';
        }
        
        logContainer.insertBefore(logEntry, logContainer.firstChild);
        
        // Keep only last 20 entries
        while (logContainer.children.length > 20) {
            logContainer.removeChild(logContainer.lastChild);
        }
    }
    
    setupChart() {
        const ctx = document.getElementById('priceChart').getContext('2d');
        this.priceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Price',
                    data: [],
                    borderColor: '#007bff',
                    backgroundColor: 'rgba(0, 123, 255, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Price'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                },
                animation: {
                    duration: 0
                }
            }
        });
    }
    
    setupEventHandlers() {
        // MT5 connection buttons
        document.getElementById('mt5ConnectBtn').addEventListener('click', () => {
            this.connectMT5();
        });
        
        document.getElementById('mt5DisconnectBtn').addEventListener('click', () => {
            this.disconnectMT5();
        });
        
        // Trading control buttons
        document.getElementById('startTradingBtn').addEventListener('click', () => {
            this.startTrading();
        });
        
        document.getElementById('pauseTradingBtn').addEventListener('click', () => {
            this.pauseTrading();
        });
        
        document.getElementById('stopTradingBtn').addEventListener('click', () => {
            this.stopTrading();
        });
        
        document.getElementById('emergencyStopBtn').addEventListener('click', () => {
            this.emergencyStop();
        });
        
        // Symbol selection
        document.getElementById('symbolSelect').addEventListener('change', (e) => {
            this.clearChart();
        });
    }
    
    async loadInitialData() {
        try {
            // Load MT5 status
            const mt5StatusResponse = await fetch('/api/mt5/status');
            const mt5StatusData = await mt5StatusResponse.json();
            this.updateMT5Status(mt5StatusData);
            
            // Load system status
            const statusResponse = await fetch('/api/dashboard/status');
            const statusData = await statusResponse.json();
            this.updateSystemStatus(statusData);
            
            // Load positions
            const positionsResponse = await fetch('/api/dashboard/positions');
            const positionsData = await positionsResponse.json();
            this.updatePositionsTable(positionsData);
            
            // Load performance metrics
            const performanceResponse = await fetch('/api/dashboard/performance');
            const performanceData = await performanceResponse.json();
            this.updatePerformanceMetrics(performanceData);
            
        } catch (error) {
            console.error('Error loading initial data:', error);
        }
    }
    
    updateSystemStatus(data) {
        const statusElement = document.getElementById('tradingStatus');
        const status = data.trading_status || 'stopped';
        
        // Update status text and indicator
        const statusText = status.charAt(0).toUpperCase() + status.slice(1);
        const indicatorClass = `status-${status}`;
        
        statusElement.innerHTML = `
            <span class="status-indicator ${indicatorClass}"></span>
            ${statusText}
        `;
    }
    
    async updatePositionsTable(data) {
        const tableBody = document.getElementById('positionsTable');
        
        if (!data || !data.positions || data.positions.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">No open positions</td></tr>';
            document.getElementById('openPositions').textContent = '0';
            return;
        }
        
        tableBody.innerHTML = '';
        document.getElementById('openPositions').textContent = data.positions.length;
        
        data.positions.forEach(position => {
            const row = document.createElement('tr');
            const pnlClass = position.unrealized_pnl >= 0 ? 'text-success' : 'text-danger';
            
            row.innerHTML = `
                <td><strong>${position.symbol}</strong></td>
                <td><span class="badge ${position.position_type === 'BUY' ? 'bg-success' : 'bg-danger'}">${position.position_type}</span></td>
                <td>${position.volume}</td>
                <td>${position.open_price}</td>
                <td class="${pnlClass}">$${position.unrealized_pnl.toFixed(2)}</td>
                <td>
                    <button class="btn btn-sm btn-outline-warning" onclick="dashboard.modifyPosition('${position.position_id}')">
                        <i class="fas fa-edit"></i>
                    </button>
                    <button class="btn btn-sm btn-outline-danger" onclick="dashboard.closePosition('${position.position_id}')">
                        <i class="fas fa-times"></i>
                    </button>
                </td>
            `;
            tableBody.appendChild(row);
        });
    }
    
    updatePerformanceMetrics(data) {
        if (data.total_pnl !== undefined) {
            const pnlElement = document.getElementById('totalPnl');
            pnlElement.textContent = `$${data.total_pnl.toFixed(2)}`;
            pnlElement.className = data.total_pnl >= 0 ? 'mb-0 text-success' : 'mb-0 text-danger';
        }
        
        if (data.active_orders !== undefined) {
            document.getElementById('activeOrders').textContent = data.active_orders;
        }
    }
    
    async startTrading() {
        try {
            const response = await fetch('/api/trading/session/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ mode: 'demo' })
            });
            
            const result = await response.json();
            if (result.success) {
                this.addToTradingLog('🟢 Trading session started', 'success');
            } else {
                this.showAlert('Error', 'Failed to start trading session', 'danger');
            }
        } catch (error) {
            console.error('Error starting trading:', error);
            this.showAlert('Error', 'Failed to start trading session', 'danger');
        }
    }
    
    async pauseTrading() {
        try {
            const response = await fetch('/api/trading/session/pause', {
                method: 'POST'
            });
            
            const result = await response.json();
            if (result.success) {
                this.addToTradingLog('⏸️ Trading paused', 'warning');
            }
        } catch (error) {
            console.error('Error pausing trading:', error);
        }
    }
    
    async stopTrading() {
        try {
            const response = await fetch('/api/trading/session/stop', {
                method: 'POST'
            });
            
            const result = await response.json();
            if (result.success) {
                this.addToTradingLog('🔴 Trading session stopped', 'warning');
            }
        } catch (error) {
            console.error('Error stopping trading:', error);
        }
    }
    
    async emergencyStop() {
        if (!confirm('Are you sure you want to activate emergency stop? This will close all positions and stop all trading.')) {
            return;
        }
        
        try {
            const response = await fetch('/api/trading/risk/emergency-stop', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ reason: 'Manual emergency stop from dashboard' })
            });
            
            const result = await response.json();
            if (result.success) {
                this.addToTradingLog('🚨 EMERGENCY STOP ACTIVATED', 'danger');
            }
        } catch (error) {
            console.error('Error activating emergency stop:', error);
        }
    }
    
    clearChart() {
        if (this.priceChart) {
            this.priceChart.data.labels = [];
            this.priceChart.data.datasets[0].data = [];
            this.priceChart.update();
        }
    }
    
    showAlert(title, message, type = 'info') {
        // Simple alert for now - could be enhanced with modal
        alert(`${title}: ${message}`);
    }
    
    async modifyPosition(positionId) {
        // Placeholder for position modification
        console.log('Modify position:', positionId);
    }
    
    async closePosition(positionId) {
        if (!confirm('Are you sure you want to close this position?')) {
            return;
        }
        
        try {
            const response = await fetch(`/api/trading/positions/${positionId}/close`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ reason: 'Manual close from dashboard' })
            });
            
            const result = await response.json();
            if (result.success) {
                this.addToTradingLog(`📉 Position closed: ${positionId}`, 'info');
            }
        } catch (error) {
            console.error('Error closing position:', error);
        }
    }
    
    updateOrderCount() {
        // Refresh order count from API
        fetch('/api/dashboard/orders')
            .then(response => response.json())
            .then(data => {
                document.getElementById('activeOrders').textContent = data.active_orders || 0;
            })
            .catch(error => console.error('Error updating order count:', error));
    }
    
    updateMT5Status(data) {
        const statusElement = document.getElementById('mt5Status');
        const isConnected = data.is_connected || false;
        
        if (isConnected) {
            statusElement.innerHTML = `
                <span class="status-indicator status-active"></span>
                Connected
            `;
            
            // Update connection info if available
            if (data.account_info) {
                this.addToTradingLog(`🔌 MT5 Connected: ${data.account_info.server} (Login: ${data.account_info.login})`, 'success');
            }
        } else {
            statusElement.innerHTML = `
                <span class="status-indicator status-stopped"></span>
                Disconnected
            `;
        }
    }
    
    async connectMT5() {
        try {
            this.addToTradingLog('🔌 Connecting to MT5...', 'info');
            
            const response = await fetch('/api/mt5/connect', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.addToTradingLog('✅ Successfully connected to MT5', 'success');
                this.updateMT5Status({ is_connected: true, account_info: result.account_info });
                
                if (result.account_info) {
                    this.addToTradingLog(`📊 Account: ${result.account_info.name} | Balance: $${result.account_info.balance}`, 'info');
                }
            } else {
                this.addToTradingLog('❌ Failed to connect to MT5', 'danger');
                this.showAlert('Connection Error', 'Failed to connect to MT5', 'danger');
            }
        } catch (error) {
            console.error('Error connecting to MT5:', error);
            this.addToTradingLog('❌ MT5 connection error', 'danger');
            this.showAlert('Error', 'Failed to connect to MT5', 'danger');
        }
    }
    
    async disconnectMT5() {
        try {
            this.addToTradingLog('🔌 Disconnecting from MT5...', 'info');
            
            const response = await fetch('/api/mt5/disconnect', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.addToTradingLog('✅ Disconnected from MT5', 'warning');
                this.updateMT5Status({ is_connected: false });
            } else {
                this.addToTradingLog('❌ Failed to disconnect from MT5', 'danger');
            }
        } catch (error) {
            console.error('Error disconnecting from MT5:', error);
            this.addToTradingLog('❌ MT5 disconnection error', 'danger');
        }
    }
}

// Initialize dashboard when page loads
let dashboard;
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new TradingDashboard();
});