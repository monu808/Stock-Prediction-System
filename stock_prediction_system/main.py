#!/usr/bin/env python3
"""
Stock Market Prediction System - Main Entry Point
Real-time prediction system for Indian stock markets (NSE/BSE)
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.api.main_api import create_app
from src.data_ingestion.data_orchestrator import DataOrchestrator
from src.execution.signal_engine import SignalEngine
from src.utils.config import Config
from src.utils.logging_config import setup_logging


class TradingSystem:
    """Main trading system orchestrator"""
    
    def __init__(self):
        self.config = Config()
        self.logger = setup_logging()
        self.data_orchestrator = None
        self.signal_engine = None
        self.app = None
        self.shutdown_event = asyncio.Event()
        
    async def initialize(self):
        """Initialize all system components"""
        self.logger.info("Initializing Trading System...")
        
        try:
            # Initialize data orchestrator
            self.data_orchestrator = DataOrchestrator(self.config)
            await self.data_orchestrator.initialize()
            
            # Initialize signal engine
            self.signal_engine = SignalEngine(self.config)
            await self.signal_engine.initialize()
            
            # Create FastAPI app
            self.app = create_app(self.config, self.signal_engine)
            
            self.logger.info("Trading System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize trading system: {e}")
            raise
    
    async def start(self):
        """Start all system components"""
        self.logger.info("Starting Trading System...")
        
        try:
            # Start data ingestion
            data_task = asyncio.create_task(
                self.data_orchestrator.start_ingestion()
            )
            
            # Start signal generation
            signal_task = asyncio.create_task(
                self.signal_engine.start_processing()
            )
            
            # Start API server
            import uvicorn
            api_config = uvicorn.Config(
                self.app,
                host=self.config.api.host,
                port=self.config.api.port,
                log_level="info"
            )
            server = uvicorn.Server(api_config)
            api_task = asyncio.create_task(server.serve())
            
            self.logger.info("All components started successfully")
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            # Graceful shutdown
            self.logger.info("Shutting down Trading System...")
            data_task.cancel()
            signal_task.cancel()
            server.should_exit = True
            
            await asyncio.gather(
                data_task, signal_task, api_task,
                return_exceptions=True
            )
            
        except Exception as e:
            self.logger.error(f"Error in trading system: {e}")
            raise
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown_event.set()


async def main():
    """Main entry point"""
    system = TradingSystem()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, system.signal_handler)
    signal.signal(signal.SIGTERM, system.signal_handler)
    
    try:
        await system.initialize()
        await system.start()
    except KeyboardInterrupt:
        system.logger.info("Received keyboard interrupt")
    except Exception as e:
        system.logger.error(f"System error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)