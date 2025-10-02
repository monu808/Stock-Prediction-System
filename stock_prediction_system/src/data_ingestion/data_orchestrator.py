"""Data orchestrator to coordinate all data ingestion components"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from .market_data import MarketDataCollector, MarketTick
from .news_sentiment import NewsAggregator, NewsArticle
from ..utils.logging_config import LoggingMixin


class DataOrchestrator(LoggingMixin):
    """Main data orchestrator coordinating all data sources"""
    
    def __init__(self, config):
        self.config = config
        self.market_collector: Optional[MarketDataCollector] = None
        self.news_aggregator: Optional[NewsAggregator] = None
        self.is_running = False
        
        # Data storage callbacks
        self.market_data_callbacks = []
        self.news_data_callbacks = []
        
        # Default symbols to track
        self.tracked_symbols = [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
            'SBIN', 'BHARTIARTL', 'ITC', 'KOTAKBANK', 'LT',
            'ASIANPAINT', 'MARUTI', 'AXISBANK', 'BAJFINANCE', 'WIPRO'
        ]
    
    async def initialize(self) -> None:
        """Initialize all data collection components"""
        self.logger.info("Initializing data orchestrator...")
        
        try:
            # Initialize market data collector
            if self.config.data_sources:
                data_sources_dict = self.config.data_sources.__dict__
                self.logger.info(f"Creating MarketDataCollector with config: {data_sources_dict}")
                self.market_collector = MarketDataCollector(data_sources_dict)
                await self.market_collector.initialize()
                
                # Add callback for market data
                self.logger.info(f"Adding market data callback: {self._handle_market_data}")
                self.market_collector.add_data_callback(self._handle_market_data)
                self.logger.info(f"Market collector now has {len(self.market_collector.data_callbacks)} callbacks registered")
            
            # Initialize news aggregator
            if self.config.data_sources:
                self.news_aggregator = NewsAggregator(self.config.data_sources.__dict__)
                await self.news_aggregator.initialize()
                
                # Add callback for news data
                self.news_aggregator.add_news_callback(self._handle_news_data)
            
            self.logger.info("Data orchestrator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize data orchestrator: {e}")
            raise
    
    async def start_ingestion(self) -> None:
        """Start all data ingestion processes"""
        self.logger.info("Starting data ingestion...")
        self.is_running = True
        
        tasks = []
        
        # Start market data collection
        if self.market_collector:
            self.logger.info(f"Starting market data collection for {len(self.tracked_symbols)} symbols...")
            market_task = asyncio.create_task(
                self.market_collector.start_collection(
                    symbols=self.tracked_symbols,
                    interval=1.0  # 1 second interval
                )
            )
            tasks.append(market_task)
        else:
            self.logger.warning("Market collector is None, skipping market data collection")
        
        # Start news collection
        if self.news_aggregator:
            self.logger.info("Starting news collection...")
            news_task = asyncio.create_task(
                self.news_aggregator.start_news_collection(
                    interval=300  # 5 minutes interval
                )
            )
            tasks.append(news_task)
        
        # Wait for all tasks
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _handle_market_data(self, tick: MarketTick, provider: str) -> None:
        """Handle incoming market data"""
        try:
            # Log the data
            self.logger.info(
                f"Orchestrator received market data - {tick.symbol}: {tick.price} "
                f"(Vol: {tick.volume}) from {provider}, forwarding to {len(self.market_data_callbacks)} callbacks"
            )
            
            # Call all registered callbacks
            for callback in self.market_data_callbacks:
                self.logger.info(f"Calling callback: {callback}")
                await callback(tick, provider)
                
        except Exception as e:
            self.logger.error(f"Error handling market data: {e}", exc_info=True)
    
    async def _handle_news_data(self, article: NewsArticle) -> None:
        """Handle incoming news data"""
        try:
            # Log the news
            self.logger.info(
                f"News - {article.source}: {article.title[:100]}... "
                f"(Sentiment: {article.sentiment_label})"
            )
            
            # Call all registered callbacks
            for callback in self.news_data_callbacks:
                await callback(article)
                
        except Exception as e:
            self.logger.error(f"Error handling news data: {e}")
    
    def add_market_data_callback(self, callback) -> None:
        """Add callback for market data"""
        self.market_data_callbacks.append(callback)
    
    def add_news_data_callback(self, callback) -> None:
        """Add callback for news data"""
        self.news_data_callbacks.append(callback)
    
    def add_symbol(self, symbol: str) -> None:
        """Add symbol to tracking list"""
        if symbol not in self.tracked_symbols:
            self.tracked_symbols.append(symbol)
            self.logger.info(f"Added symbol {symbol} to tracking list")
    
    def remove_symbol(self, symbol: str) -> None:
        """Remove symbol from tracking list"""
        if symbol in self.tracked_symbols:
            self.tracked_symbols.remove(symbol)
            self.logger.info(f"Removed symbol {symbol} from tracking list")
    
    def get_tracked_symbols(self) -> List[str]:
        """Get list of tracked symbols"""
        return self.tracked_symbols.copy()
    
    async def stop_ingestion(self) -> None:
        """Stop all data ingestion"""
        self.logger.info("Stopping data ingestion...")
        self.is_running = False
        
        if self.market_collector:
            await self.market_collector.stop_collection()
        
        if self.news_aggregator:
            await self.news_aggregator.stop_news_collection()
    
    async def close(self) -> None:
        """Close all data sources"""
        if self.market_collector:
            await self.market_collector.close()
        
        if self.news_aggregator:
            await self.news_aggregator.close()
        
        self.logger.info("Data orchestrator closed")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of data ingestion"""
        return {
            'is_running': self.is_running,
            'tracked_symbols': len(self.tracked_symbols),
            'symbols': self.tracked_symbols,
            'market_collector_active': self.market_collector is not None,
            'news_aggregator_active': self.news_aggregator is not None,
            'market_callbacks': len(self.market_data_callbacks),
            'news_callbacks': len(self.news_data_callbacks)
        }