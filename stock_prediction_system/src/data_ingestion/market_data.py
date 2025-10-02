"""Real-time market data ingestion from NSE/BSE and other sources"""

import asyncio
import aiohttp
import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..utils.logging_config import LoggingMixin


@dataclass
class MarketTick:
    """Market tick data structure"""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    exchange: str
    tick_id: Optional[str] = None


@dataclass
class OrderBookEntry:
    """Order book entry"""
    price: float
    quantity: int
    order_count: int


@dataclass
class OrderBook:
    """Order book data structure"""
    symbol: str
    timestamp: datetime
    bids: List[OrderBookEntry]
    asks: List[OrderBookEntry]
    exchange: str


class NSEDataProvider(LoggingMixin):
    """NSE real-time data provider"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_running = False
        self.subscribed_symbols = set()
        
        # NSE API endpoints
        self.base_url = "https://www.nseindia.com/api"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate',
            'Accept-Language': 'en-US,en;q=0.9',
        }
    
    async def initialize(self) -> None:
        """Initialize NSE data provider"""
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=30)
        )
        self.logger.info("NSE data provider initialized")
    
    async def subscribe_symbol(self, symbol: str) -> None:
        """Subscribe to symbol updates"""
        self.subscribed_symbols.add(symbol)
        self.logger.info(f"Subscribed to NSE symbol: {symbol}")
    
    async def fetch_quote(self, symbol: str) -> Optional[MarketTick]:
        """Fetch real-time quote for a symbol"""
        self.logger.debug(f"Attempting to fetch quote for symbol: {symbol}")
        try:
            url = f"{self.base_url}/quote-equity?symbol={symbol}"
            self.logger.debug(f"Fetching URL: {url}")
            async with self.session.get(url) as response:
                self.logger.debug(f"Response status for {symbol}: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    self.logger.debug(f"Received data for {symbol}: {str(data)[:200]}")
                    return self._parse_nse_quote(data, symbol)
                else:
                    self.logger.warning(f"Failed to fetch NSE quote for {symbol}: {response.status}")
                    return None
        except Exception as e:
            self.logger.error(f"Error fetching NSE quote for {symbol}: {e}")
            return None
    
    async def fetch_order_book(self, symbol: str) -> Optional[OrderBook]:
        """Fetch order book data"""
        try:
            url = f"{self.base_url}/quote-equity?symbol={symbol}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_nse_order_book(data, symbol)
                else:
                    self.logger.warning(f"Failed to fetch NSE order book for {symbol}: {response.status}")
                    return None
        except Exception as e:
            self.logger.error(f"Error fetching NSE order book for {symbol}: {e}")
            return None
    
    async def fetch_market_depth(self, symbol: str) -> Dict[str, Any]:
        """Fetch market depth data"""
        try:
            url = f"{self.base_url}/quote-equity?symbol={symbol}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('marketDeptOrderBook', {})
                else:
                    return {}
        except Exception as e:
            self.logger.error(f"Error fetching market depth for {symbol}: {e}")
            return {}
    
    def _parse_nse_quote(self, data: Dict[str, Any], symbol: str) -> MarketTick:
        """Parse NSE quote data to MarketTick"""
        price_info = data.get('priceInfo', {})
        
        return MarketTick(
            symbol=symbol,
            timestamp=datetime.now(),
            price=float(price_info.get('lastPrice', 0)),
            volume=int(data.get('marketDeptOrderBook', {}).get('totalTradedVolume', 0)),
            bid_price=float(price_info.get('intraDayHighLow', {}).get('min', 0)),
            ask_price=float(price_info.get('intraDayHighLow', {}).get('max', 0)),
            bid_size=0,
            ask_size=0,
            exchange="NSE"
        )
    
    def _parse_nse_order_book(self, data: Dict[str, Any], symbol: str) -> OrderBook:
        """Parse NSE order book data"""
        market_depth = data.get('marketDeptOrderBook', {})
        
        bids = []
        asks = []
        
        # Parse bid data
        bid_data = market_depth.get('bid', [])
        for bid in bid_data:
            bids.append(OrderBookEntry(
                price=float(bid.get('price', 0)),
                quantity=int(bid.get('quantity', 0)),
                order_count=int(bid.get('orders', 0))
            ))
        
        # Parse ask data
        ask_data = market_depth.get('ask', [])
        for ask in ask_data:
            asks.append(OrderBookEntry(
                price=float(ask.get('price', 0)),
                quantity=int(ask.get('quantity', 0)),
                order_count=int(ask.get('orders', 0))
            ))
        
        return OrderBook(
            symbol=symbol,
            timestamp=datetime.now(),
            bids=bids,
            asks=asks,
            exchange="NSE"
        )
    
    async def close(self) -> None:
        """Close NSE data provider"""
        if self.session:
            await self.session.close()
        self.logger.info("NSE data provider closed")


class BSEDataProvider(LoggingMixin):
    """BSE real-time data provider"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_running = False
        self.subscribed_symbols = set()
        
        # BSE API endpoints
        self.base_url = "https://api.bseindia.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        }
    
    async def initialize(self) -> None:
        """Initialize BSE data provider"""
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=30)
        )
        self.logger.info("BSE data provider initialized")
    
    async def subscribe_symbol(self, symbol: str) -> None:
        """Subscribe to symbol updates"""
        self.subscribed_symbols.add(symbol)
        self.logger.info(f"Subscribed to BSE symbol: {symbol}")
    
    async def fetch_quote(self, symbol: str) -> Optional[MarketTick]:
        """Fetch real-time quote for a symbol"""
        try:
            # BSE implementation would go here
            # For now, return a mock tick
            return MarketTick(
                symbol=symbol,
                timestamp=datetime.now(),
                price=0.0,
                volume=0,
                bid_price=0.0,
                ask_price=0.0,
                bid_size=0,
                ask_size=0,
                exchange="BSE"
            )
        except Exception as e:
            self.logger.error(f"Error fetching BSE quote for {symbol}: {e}")
            return None
    
    async def close(self) -> None:
        """Close BSE data provider"""
        if self.session:
            await self.session.close()
        self.logger.info("BSE data provider closed")


class YahooFinanceProvider(LoggingMixin):
    """Yahoo Finance data provider for fallback data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.base_url = "https://query1.finance.yahoo.com/v8/finance/chart"
        self.subscribed_symbols = set()
    
    async def initialize(self) -> None:
        """Initialize Yahoo Finance provider"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        self.logger.info("Yahoo Finance provider initialized")
    
    async def subscribe_symbol(self, symbol: str) -> None:
        """Subscribe to symbol updates"""
        self.subscribed_symbols.add(symbol)
        self.logger.debug(f"Yahoo subscribed to symbol: {symbol}")
    
    async def fetch_quote(self, symbol: str) -> Optional[MarketTick]:
        """Fetch quote from Yahoo Finance"""
        try:
            # Convert NSE symbol to Yahoo format (e.g., RELIANCE.NS)
            yahoo_symbol = f"{symbol}.NS" if not symbol.endswith('.NS') else symbol
            
            url = f"{self.base_url}/{yahoo_symbol}"
            params = {
                'interval': '1m',
                'range': '1d',
                'includePrePost': 'true'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_yahoo_quote(data, symbol)
                else:
                    return None
        except Exception as e:
            self.logger.error(f"Error fetching Yahoo quote for {symbol}: {e}")
            return None
    
    def _parse_yahoo_quote(self, data: Dict[str, Any], symbol: str) -> Optional[MarketTick]:
        """Parse Yahoo Finance data"""
        try:
            result = data['chart']['result'][0]
            meta = result['meta']
            
            return MarketTick(
                symbol=symbol,
                timestamp=datetime.now(),
                price=float(meta.get('regularMarketPrice', 0)),
                volume=int(meta.get('regularMarketVolume', 0)),
                bid_price=0.0,
                ask_price=0.0,
                bid_size=0,
                ask_size=0,
                exchange="NSE"
            )
        except Exception as e:
            self.logger.error(f"Error parsing Yahoo data for {symbol}: {e}")
            return None
    
    async def close(self) -> None:
        """Close Yahoo Finance provider"""
        if self.session:
            await self.session.close()
        self.logger.info("Yahoo Finance provider closed")


class MockProvider(LoggingMixin):
    """In-memory mock provider that emits synthetic ticks for testing."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_running = False
        self.subscribed_symbols = set()
        # Maintain last prices per symbol for a simple random-walk
        self._last_prices: Dict[str, float] = {}

    async def initialize(self) -> None:
        """No-op initialize for mock provider"""
        # seed some plausible starting prices if provided
        seed_prices = self.config.get('seed_prices', {})
        for sym, p in seed_prices.items():
            try:
                self._last_prices[sym] = float(p)
            except Exception:
                self._last_prices[sym] = 100.0

        self.logger.info("Mock provider initialized")

    async def subscribe_symbol(self, symbol: str) -> None:
        self.subscribed_symbols.add(symbol)
        # seed default price if not present
        if symbol not in self._last_prices:
            self._last_prices[symbol] = float(self.config.get('default_price', 100.0))
        self.logger.debug(f"Mock subscribed to symbol: {symbol}")

    async def fetch_quote(self, symbol: str) -> Optional[MarketTick]:
        """Return a synthetic MarketTick using a small random-walk around last price"""
        try:
            base = self._last_prices.get(symbol, float(self.config.get('default_price', 100.0)))
            # Larger random walk step to create more varied signals (increased from 0.002 to 0.01)
            step = random.uniform(-2.0, 2.0) * max(0.1, base * 0.01)
            price = max(0.01, base + step)
            self._last_prices[symbol] = price

            tick = MarketTick(
                symbol=symbol,
                timestamp=datetime.now(),
                price=round(price, 2),
                volume=int(random.uniform(100, 10000)),
                bid_price=round(price - 0.05, 2),
                ask_price=round(price + 0.05, 2),
                bid_size=int(random.uniform(1, 50)),
                ask_size=int(random.uniform(1, 50)),
                exchange="MOCK",
            )

            self.logger.debug(f"Mock emitted tick for {symbol}: {tick}")
            return tick
        except Exception as e:
            self.logger.error(f"Error generating mock tick for {symbol}: {e}")
            return None

    async def close(self) -> None:
        self.logger.info("Mock provider closed")


class MarketDataCollector(LoggingMixin):
    """Main market data collector orchestrating multiple providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers = {}
        self.is_running = False
        self.data_callbacks = []
        
        self.logger.info(f"MarketDataCollector initializing with config keys: {list(config.keys())}")
        
        # Initialize providers
        if config.get('nse', {}).get('enabled', False):
            self.providers['nse'] = NSEDataProvider(config['nse'])
            self.logger.info("NSE provider will be initialized")
        
        if config.get('bse', {}).get('enabled', False):
            self.providers['bse'] = BSEDataProvider(config['bse'])
            self.logger.info("BSE provider will be initialized")
        
        if config.get('yahoo_finance', {}).get('enabled', False):
            self.providers['yahoo'] = YahooFinanceProvider(config['yahoo_finance'])
            self.logger.info("Yahoo Finance provider will be initialized")
        
        # Temporary/mock provider useful for testing and when external sources are blocked
        if config.get('mock_provider', {}).get('enabled', False):
            self.providers['mock'] = MockProvider(config.get('mock_provider', {}))
            self.logger.info("Mock provider will be initialized")
        
        if not self.providers:
            self.logger.warning("No data providers enabled! Check your config.yaml data_sources section")
    
    async def initialize(self) -> None:
        """Initialize all providers"""
        for name, provider in self.providers.items():
            await provider.initialize()
            self.logger.info(f"Initialized {name} provider")
    
    async def subscribe_symbols(self, symbols: List[str]) -> None:
        """Subscribe to multiple symbols across all providers"""
        for symbol in symbols:
            for provider in self.providers.values():
                await provider.subscribe_symbol(symbol)
    
    async def start_collection(self, symbols: List[str], interval: float = 1.0) -> None:
        """Start real-time data collection"""
        try:
            self.logger.info(f"start_collection called with {len(symbols)} symbols and {len(self.providers)} providers")
            self.is_running = True
            await self.subscribe_symbols(symbols)
            
            self.logger.info(f"Starting data collection for {len(symbols)} symbols")
            
            while self.is_running:
                start_time = time.time()
                
                # Collect data from all providers
                tasks = []
                for symbol in symbols:
                    for provider_name, provider in self.providers.items():
                        task = asyncio.create_task(
                            self._fetch_and_process(provider_name, provider, symbol)
                        )
                        tasks.append(task)
                
                # Wait for all tasks to complete
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                # Maintain collection interval
                elapsed = time.time() - start_time
                if elapsed < interval:
                    await asyncio.sleep(interval - elapsed)
        except Exception as e:
            self.logger.error(f"Error in start_collection: {e}", exc_info=True)
    
    async def _fetch_and_process(self, provider_name: str, provider: Any, symbol: str) -> None:
        """Fetch data from provider and process it"""
        self.logger.debug(f"Fetching data for {symbol} from provider {provider_name}")
        try:
            tick = await provider.fetch_quote(symbol)
            if tick:
                self.logger.debug(f"Fetched tick for {symbol} from {provider_name}: {tick}")
                # Process the tick data
                await self._process_tick(tick, provider_name)
            else:
                self.logger.debug(f"No tick data returned for {symbol} from {provider_name}")
        except Exception as e:
            self.logger.error(f"Error fetching data from {provider_name} for {symbol}: {e}")
    
    async def _process_tick(self, tick: MarketTick, provider_name: str) -> None:
        """Process incoming tick data"""
        self.logger.debug(f"_process_tick called for {tick.symbol}, forwarding to {len(self.data_callbacks)} callbacks")
        # Call all registered callbacks
        for callback in self.data_callbacks:
            try:
                self.logger.debug(f"Calling callback: {callback}")
                await callback(tick, provider_name)
            except Exception as e:
                self.logger.error(f"Error in data callback: {e}", exc_info=True)
    
    def add_data_callback(self, callback) -> None:
        """Add a callback function for processed data"""
        self.data_callbacks.append(callback)
    
    async def stop_collection(self) -> None:
        """Stop data collection"""
        self.is_running = False
        self.logger.info("Stopping data collection")
    
    async def close(self) -> None:
        """Close all providers"""
        for provider in self.providers.values():
            await provider.close()
        self.logger.info("Market data collector closed")