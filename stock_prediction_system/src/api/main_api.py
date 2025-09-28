"""Main FastAPI application for the trading system"""

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import asyncio
from datetime import datetime

from ..execution.signal_engine import SignalEngine, TradingSignal, SignalType, TimeFrame
from ..utils.logging_config import get_logger


# Pydantic models for API
class SignalResponse(BaseModel):
    symbol: str
    signal_type: str
    confidence: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    timeframe: str
    timestamp: datetime
    reasons: List[str]
    risk_score: float
    position_size: float
    technical_score: float
    fundamental_score: float
    sentiment_score: float
    macro_score: float
    momentum_score: float


class SystemStatusResponse(BaseModel):
    status: str
    uptime: float
    active_symbols: int
    signals_generated: int
    last_signal_time: Optional[datetime]


class SymbolRequest(BaseModel):
    symbol: str


class WebSocketManager:
    """WebSocket connection manager"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.logger = get_logger("websocket")
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        self.logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            self.logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                self.logger.error(f"Error broadcasting message: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for conn in disconnected:
            self.disconnect(conn)


def create_app(config, signal_engine: SignalEngine) -> FastAPI:
    """Create and configure FastAPI app"""
    
    app = FastAPI(
        title="Stock Market Prediction System",
        description="Real-time stock market prediction and trading signals for Indian markets",
        version="1.0.0"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # WebSocket manager
    websocket_manager = WebSocketManager()
    
    # Logger
    logger = get_logger("api")
    
    # Application state
    app_state = {
        "start_time": datetime.now(),
        "signals_generated": 0,
        "last_signal_time": None
    }
    
    # Signal callback to update WebSocket clients
    async def signal_callback(signal: TradingSignal):
        app_state["signals_generated"] += 1
        app_state["last_signal_time"] = signal.timestamp
        
        # Convert signal to dict for JSON serialization
        signal_dict = {
            "symbol": signal.symbol,
            "signal_type": signal.signal_type.value,
            "confidence": signal.confidence,
            "target_price": signal.target_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "timeframe": signal.timeframe.value,
            "timestamp": signal.timestamp.isoformat(),
            "reasons": signal.reasons,
            "risk_score": signal.risk_score,
            "position_size": signal.position_size,
            "technical_score": signal.technical_score,
            "fundamental_score": signal.fundamental_score,
            "sentiment_score": signal.sentiment_score,
            "macro_score": signal.macro_score,
            "momentum_score": signal.momentum_score
        }
        
        # Broadcast to WebSocket clients
        await websocket_manager.broadcast(json.dumps({
            "type": "signal",
            "data": signal_dict
        }))
    
    # Add signal callback
    signal_engine.add_signal_callback(signal_callback)
    
    # API Routes
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "Stock Market Prediction System API",
            "version": "1.0.0",
            "status": "active",
            "endpoints": {
                "signals": "/signals",
                "status": "/status",
                "websocket": "/ws",
                "docs": "/docs"
            }
        }
    
    @app.get("/status", response_model=SystemStatusResponse)
    async def get_system_status():
        """Get system status"""
        uptime = (datetime.now() - app_state["start_time"]).total_seconds()
        
        latest_signals = signal_engine.get_all_latest_signals()
        
        return SystemStatusResponse(
            status="running",
            uptime=uptime,
            active_symbols=len(latest_signals),
            signals_generated=app_state["signals_generated"],
            last_signal_time=app_state["last_signal_time"]
        )
    
    @app.get("/signals", response_model=List[SignalResponse])
    async def get_all_signals():
        """Get all latest signals"""
        latest_signals = signal_engine.get_all_latest_signals()
        
        response = []
        for symbol, signal in latest_signals.items():
            response.append(SignalResponse(
                symbol=signal.symbol,
                signal_type=signal.signal_type.value,
                confidence=signal.confidence,
                target_price=signal.target_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                timeframe=signal.timeframe.value,
                timestamp=signal.timestamp,
                reasons=signal.reasons,
                risk_score=signal.risk_score,
                position_size=signal.position_size,
                technical_score=signal.technical_score,
                fundamental_score=signal.fundamental_score,
                sentiment_score=signal.sentiment_score,
                macro_score=signal.macro_score,
                momentum_score=signal.momentum_score
            ))
        
        return response
    
    @app.get("/signals/live", response_model=List[SignalResponse])
    async def get_live_signals():
        """Get all live/latest signals - same as /signals but with explicit live endpoint"""
        return await get_all_signals()
    
    @app.get("/signals/{symbol}", response_model=SignalResponse)
    async def get_signal_for_symbol(symbol: str):
        """Get latest signal for a specific symbol"""
        signal = signal_engine.get_latest_signal(symbol.upper())
        
        if not signal:
            raise HTTPException(
                status_code=404,
                detail=f"No signal found for symbol {symbol}"
            )
        
        return SignalResponse(
            symbol=signal.symbol,
            signal_type=signal.signal_type.value,
            confidence=signal.confidence,
            target_price=signal.target_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            timeframe=signal.timeframe.value,
            timestamp=signal.timestamp,
            reasons=signal.reasons,
            risk_score=signal.risk_score,
            position_size=signal.position_size,
            technical_score=signal.technical_score,
            fundamental_score=signal.fundamental_score,
            sentiment_score=signal.sentiment_score,
            macro_score=signal.macro_score,
            momentum_score=signal.momentum_score
        )
    
    @app.post("/signals/subscribe")
    async def subscribe_to_symbol(request: SymbolRequest):
        """Subscribe to signals for a new symbol"""
        symbol = request.symbol.upper()
        
        # Add symbol to data orchestrator tracking
        # This would need to be implemented in the data orchestrator
        
        return {
            "message": f"Subscribed to signals for {symbol}",
            "symbol": symbol
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "trading-system-api"
        }
    
    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        """Trading dashboard with live signals"""
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Stock Market Prediction Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f0f0f0; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                .signals-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                .signal-card { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .signal-buy { border-left: 5px solid #27ae60; }
                .signal-sell { border-left: 5px solid #e74c3c; }
                .signal-hold { border-left: 5px solid #f39c12; }
                .confidence { font-weight: bold; font-size: 1.2em; }
                .high-confidence { color: #27ae60; }
                .medium-confidence { color: #f39c12; }
                .low-confidence { color: #e74c3c; }
                .refresh-btn { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
                .status { margin-bottom: 20px; padding: 15px; background: white; border-radius: 8px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 Stock Market Prediction System</h1>
                    <p>Real-time Indian Stock Market Analysis & Trading Signals</p>
                </div>
                
                <div class="status" id="status">
                    <h3>System Status</h3>
                    <p id="system-info">Loading...</p>
                    <button class="refresh-btn" onclick="refreshData()">Refresh Data</button>
                </div>
                
                <div id="signals-container">
                    <h3>Live Trading Signals</h3>
                    <div class="signals-grid" id="signals-grid">
                        Loading signals...
                    </div>
                </div>
            </div>
            
            <script>
                async function loadStatus() {
                    try {
                        const response = await fetch('/status');
                        const data = await response.json();
                        document.getElementById('system-info').innerHTML = `
                            <strong>Status:</strong> ${data.status} | 
                            <strong>Uptime:</strong> ${Math.round(data.uptime)}s | 
                            <strong>Active Symbols:</strong> ${data.active_symbols} | 
                            <strong>Signals Generated:</strong> ${data.signals_generated}
                        `;
                    } catch (error) {
                        document.getElementById('system-info').innerHTML = 'Error loading status';
                    }
                }
                
                async function loadSignals() {
                    try {
                        const response = await fetch('/signals');
                        const signals = await response.json();
                        const grid = document.getElementById('signals-grid');
                        
                        if (signals.length === 0) {
                            grid.innerHTML = '<p>No signals available. System is still analyzing market data...</p>';
                            return;
                        }
                        
                        grid.innerHTML = signals.map(signal => `
                            <div class="signal-card signal-${signal.signal_type.toLowerCase()}">
                                <h4>${signal.symbol}</h4>
                                <div class="confidence ${getConfidenceClass(signal.confidence)}">
                                    ${signal.signal_type.toUpperCase()} - ${(signal.confidence * 100).toFixed(1)}%
                                </div>
                                <p><strong>Target:</strong> ₹${signal.target_price?.toFixed(2) || 'N/A'}</p>
                                <p><strong>Stop Loss:</strong> ₹${signal.stop_loss?.toFixed(2) || 'N/A'}</p>
                                <p><strong>Risk Score:</strong> ${signal.risk_score?.toFixed(2) || 'N/A'}</p>
                                <p><strong>Timeframe:</strong> ${signal.timeframe}</p>
                                <small><strong>Last Updated:</strong> ${new Date(signal.timestamp).toLocaleString()}</small>
                                ${signal.reasons ? `<p><strong>Reasons:</strong> ${signal.reasons.join(', ')}</p>` : ''}
                            </div>
                        `).join('');
                    } catch (error) {
                        document.getElementById('signals-grid').innerHTML = 'Error loading signals';
                    }
                }
                
                function getConfidenceClass(confidence) {
                    if (confidence >= 0.7) return 'high-confidence';
                    if (confidence >= 0.5) return 'medium-confidence';
                    return 'low-confidence';
                }
                
                function refreshData() {
                    loadStatus();
                    loadSignals();
                }
                
                // Initial load
                refreshData();
                
                // Auto-refresh every 30 seconds
                setInterval(refreshData, 30000);
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time signals"""
        await websocket_manager.connect(websocket)
        
        try:
            # Send welcome message
            await websocket_manager.send_personal_message(
                json.dumps({
                    "type": "welcome",
                    "message": "Connected to Trading System WebSocket",
                    "timestamp": datetime.now().isoformat()
                }),
                websocket
            )
            
            # Keep connection alive and handle incoming messages
            while True:
                try:
                    # Wait for incoming messages
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Handle different message types
                    if message.get("type") == "ping":
                        await websocket_manager.send_personal_message(
                            json.dumps({
                                "type": "pong",
                                "timestamp": datetime.now().isoformat()
                            }),
                            websocket
                        )
                    elif message.get("type") == "subscribe":
                        symbol = message.get("symbol", "").upper()
                        if symbol:
                            # Send latest signal for symbol if available
                            latest_signal = signal_engine.get_latest_signal(symbol)
                            if latest_signal:
                                signal_dict = {
                                    "symbol": latest_signal.symbol,
                                    "signal_type": latest_signal.signal_type.value,
                                    "confidence": latest_signal.confidence,
                                    "target_price": latest_signal.target_price,
                                    "stop_loss": latest_signal.stop_loss,
                                    "take_profit": latest_signal.take_profit,
                                    "timeframe": latest_signal.timeframe.value,
                                    "timestamp": latest_signal.timestamp.isoformat(),
                                    "reasons": latest_signal.reasons,
                                    "risk_score": latest_signal.risk_score,
                                    "position_size": latest_signal.position_size,
                                    "technical_score": latest_signal.technical_score,
                                    "fundamental_score": latest_signal.fundamental_score,
                                    "sentiment_score": latest_signal.sentiment_score,
                                    "macro_score": latest_signal.macro_score,
                                    "momentum_score": latest_signal.momentum_score
                                }
                                
                                await websocket_manager.send_personal_message(
                                    json.dumps({
                                        "type": "signal",
                                        "data": signal_dict
                                    }),
                                    websocket
                                )
                
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {e}")
                    break
                    
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            websocket_manager.disconnect(websocket)
    
    @app.get("/dashboard", response_class=HTMLResponse)
    async def get_dashboard():
        """Simple dashboard HTML page"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading System Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .signal { border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px; }
                .buy { background-color: #d4edda; }
                .sell { background-color: #f8d7da; }
                .hold { background-color: #fff3cd; }
                .status { background-color: #e2e3e5; padding: 10px; border-radius: 5px; margin-bottom: 20px; }
                .ws-status { color: red; }
                .ws-connected { color: green; }
            </style>
        </head>
        <body>
            <h1>Stock Market Prediction System Dashboard</h1>
            
            <div id="status" class="status">
                <h3>System Status</h3>
                <p>WebSocket: <span id="ws-status" class="ws-status">Disconnected</span></p>
                <p>Signals Received: <span id="signal-count">0</span></p>
                <p>Last Update: <span id="last-update">Never</span></p>
            </div>
            
            <div id="signals">
                <h3>Real-time Signals</h3>
                <div id="signal-list">No signals received yet...</div>
            </div>
            
            <script>
                let signalCount = 0;
                const ws = new WebSocket('ws://localhost:8000/ws');
                
                ws.onopen = function(event) {
                    document.getElementById('ws-status').textContent = 'Connected';
                    document.getElementById('ws-status').className = 'ws-connected';
                    
                    // Send ping every 30 seconds
                    setInterval(() => {
                        ws.send(JSON.stringify({type: 'ping'}));
                    }, 30000);
                };
                
                ws.onmessage = function(event) {
                    const message = JSON.parse(event.data);
                    
                    if (message.type === 'signal') {
                        signalCount++;
                        document.getElementById('signal-count').textContent = signalCount;
                        document.getElementById('last-update').textContent = new Date().toLocaleString();
                        
                        const signal = message.data;
                        addSignalToList(signal);
                    }
                };
                
                ws.onclose = function(event) {
                    document.getElementById('ws-status').textContent = 'Disconnected';
                    document.getElementById('ws-status').className = 'ws-status';
                };
                
                function addSignalToList(signal) {
                    const signalDiv = document.createElement('div');
                    signalDiv.className = `signal ${signal.signal_type.toLowerCase()}`;
                    
                    signalDiv.innerHTML = `
                        <h4>${signal.symbol} - ${signal.signal_type}</h4>
                        <p><strong>Confidence:</strong> ${(signal.confidence * 100).toFixed(1)}%</p>
                        <p><strong>Target Price:</strong> ₹${signal.target_price?.toFixed(2) || 'N/A'}</p>
                        <p><strong>Stop Loss:</strong> ₹${signal.stop_loss?.toFixed(2) || 'N/A'}</p>
                        <p><strong>Position Size:</strong> ${(signal.position_size * 100).toFixed(1)}%</p>
                        <p><strong>Risk Score:</strong> ${(signal.risk_score * 100).toFixed(1)}%</p>
                        <p><strong>Reasons:</strong> ${signal.reasons.join(', ')}</p>
                        <p><strong>Time:</strong> ${new Date(signal.timestamp).toLocaleString()}</p>
                    `;
                    
                    const signalList = document.getElementById('signal-list');
                    signalList.insertBefore(signalDiv, signalList.firstChild);
                    
                    // Keep only last 10 signals
                    while (signalList.children.length > 10) {
                        signalList.removeChild(signalList.lastChild);
                    }
                }
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    
    return app