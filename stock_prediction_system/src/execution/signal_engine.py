"""Trading signal generation engine combining multiple prediction models"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json

from ..data_ingestion.market_data import MarketTick
from ..data_ingestion.news_sentiment import NewsArticle
from ..utils.logging_config import LoggingMixin


class SignalType(Enum):
    """Types of trading signals"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"


class TimeFrame(Enum):
    """Trading timeframes"""
    MILLISECOND = "1ms"
    SECOND = "1s"
    MINUTE = "1m"
    HOUR = "1h"
    DAILY = "1d"
    WEEKLY = "1w"


@dataclass
class TradingSignal:
    """Trading signal data structure"""
    symbol: str
    signal_type: SignalType
    confidence: float  # 0.0 to 1.0
    target_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    timeframe: TimeFrame
    timestamp: datetime
    reasons: List[str]  # Reasons for the signal
    risk_score: float  # 0.0 to 1.0
    position_size: float  # Recommended position size (0.0 to 1.0)
    
    # Additional metadata
    technical_score: float = 0.0
    fundamental_score: float = 0.0
    sentiment_score: float = 0.0
    macro_score: float = 0.0
    momentum_score: float = 0.0


class TechnicalAnalyzer(LoggingMixin):
    """Technical analysis component"""
    
    def __init__(self):
        self.price_history = {}  # symbol -> list of prices
        self.volume_history = {}  # symbol -> list of volumes
        
    def update_data(self, tick: MarketTick) -> None:
        """Update with new market data"""
        symbol = tick.symbol
        
        # Update price history
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append({
            'timestamp': tick.timestamp,
            'price': tick.price,
            'volume': tick.volume,
            'bid': tick.bid_price,
            'ask': tick.ask_price
        })
        
        # Keep only recent data (last 1000 points)
        if len(self.price_history[symbol]) > 1000:
            self.price_history[symbol] = self.price_history[symbol][-1000:]
    
    def calculate_sma(self, symbol: str, period: int = 20) -> Optional[float]:
        """Calculate Simple Moving Average"""
        if symbol not in self.price_history:
            return None
        
        prices = [p['price'] for p in self.price_history[symbol]]
        if len(prices) < period:
            return None
        
        return sum(prices[-period:]) / period
    
    def calculate_ema(self, symbol: str, period: int = 20) -> Optional[float]:
        """Calculate Exponential Moving Average"""
        if symbol not in self.price_history:
            return None
        
        prices = [p['price'] for p in self.price_history[symbol]]
        if len(prices) < period:
            return None
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def calculate_rsi(self, symbol: str, period: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index"""
        if symbol not in self.price_history:
            return None
        
        prices = [p['price'] for p in self.price_history[symbol]]
        if len(prices) < period + 1:
            return None
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) < period:
            return None
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, symbol: str) -> Optional[Dict[str, float]]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_12 = self.calculate_ema(symbol, 12)
        ema_26 = self.calculate_ema(symbol, 26)
        
        if ema_12 is None or ema_26 is None:
            return None
        
        macd_line = ema_12 - ema_26
        # Simplified signal line (would need more sophisticated calculation)
        signal_line = macd_line * 0.9  # Approximation
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': macd_line - signal_line
        }
    
    def calculate_bollinger_bands(self, symbol: str, period: int = 20) -> Optional[Dict[str, float]]:
        """Calculate Bollinger Bands"""
        sma = self.calculate_sma(symbol, period)
        if sma is None:
            return None
        
        prices = [p['price'] for p in self.price_history[symbol]]
        if len(prices) < period:
            return None
        
        recent_prices = prices[-period:]
        
        # Calculate standard deviation
        variance = sum([(price - sma) ** 2 for price in recent_prices]) / period
        std_dev = variance ** 0.5
        
        return {
            'upper': sma + (2 * std_dev),
            'middle': sma,
            'lower': sma - (2 * std_dev)
        }
    
    def generate_technical_score(self, symbol: str) -> float:
        """Generate overall technical analysis score"""
        scores = []
        
        # RSI scoring
        rsi = self.calculate_rsi(symbol)
        if rsi is not None:
            if rsi < 30:  # Oversold
                scores.append(0.8)
            elif rsi > 70:  # Overbought
                scores.append(0.2)
            else:
                scores.append(0.5)
        
        # MACD scoring
        macd = self.calculate_macd(symbol)
        if macd is not None:
            if macd['macd'] > macd['signal']:  # Bullish
                scores.append(0.7)
            else:  # Bearish
                scores.append(0.3)
        
        # Bollinger Bands scoring
        bb = self.calculate_bollinger_bands(symbol)
        if bb is not None and symbol in self.price_history:
            current_price = self.price_history[symbol][-1]['price']
            if current_price < bb['lower']:  # Oversold
                scores.append(0.8)
            elif current_price > bb['upper']:  # Overbought
                scores.append(0.2)
            else:
                scores.append(0.5)
        
        return sum(scores) / len(scores) if scores else 0.5


class SentimentAnalyzer(LoggingMixin):
    """Sentiment analysis component"""
    
    def __init__(self):
        self.news_history = {}  # symbol -> list of news
        self.sentiment_scores = {}  # symbol -> current sentiment
    
    def update_news(self, article: NewsArticle) -> None:
        """Update with new news article"""
        for symbol in article.symbols_mentioned:
            if symbol not in self.news_history:
                self.news_history[symbol] = []
            
            self.news_history[symbol].append({
                'timestamp': article.published_at,
                'sentiment': article.sentiment_score,
                'importance': article.importance_score,
                'title': article.title
            })
            
            # Keep only recent news (last 100 articles per symbol)
            if len(self.news_history[symbol]) > 100:
                self.news_history[symbol] = self.news_history[symbol][-100:]
    
    def calculate_sentiment_score(self, symbol: str, hours_back: int = 24) -> float:
        """Calculate sentiment score for a symbol"""
        if symbol not in self.news_history:
            return 0.5  # Neutral
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        recent_news = [
            news for news in self.news_history[symbol]
            if news['timestamp'] > cutoff_time
        ]
        
        if not recent_news:
            return 0.5
        
        # Weighted sentiment score
        total_weight = 0
        weighted_sentiment = 0
        
        for news in recent_news:
            weight = news['importance']
            sentiment = (news['sentiment'] + 1) / 2  # Convert from [-1,1] to [0,1]
            
            weighted_sentiment += sentiment * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        return weighted_sentiment / total_weight


class RiskManager(LoggingMixin):
    """Risk management component"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_position_size = config.get('max_position_size', 0.05)
        self.max_portfolio_risk = config.get('max_portfolio_risk', 0.02)
        self.stop_loss_pct = config.get('stop_loss_percentage', 0.03)
        self.take_profit_pct = config.get('take_profit_percentage', 0.06)
        
        # Current positions (symbol -> position_info)
        self.current_positions = {}
        self.portfolio_value = 1000000  # Starting portfolio value
    
    def calculate_position_size(
        self,
        symbol: str,
        signal_confidence: float,
        current_price: float,
        volatility: float
    ) -> float:
        """Calculate recommended position size"""
        # Base position size based on confidence
        base_size = self.max_position_size * signal_confidence
        
        # Adjust for volatility
        volatility_adjustment = max(0.1, 1.0 - volatility)
        adjusted_size = base_size * volatility_adjustment
        
        # Ensure we don't exceed maximum position size
        return min(adjusted_size, self.max_position_size)
    
    def calculate_stop_loss(self, entry_price: float, signal_type: SignalType) -> float:
        """Calculate stop loss price"""
        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            return entry_price * (1 - self.stop_loss_pct)
        else:
            return entry_price * (1 + self.stop_loss_pct)
    
    def calculate_take_profit(self, entry_price: float, signal_type: SignalType) -> float:
        """Calculate take profit price"""
        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            return entry_price * (1 + self.take_profit_pct)
        else:
            return entry_price * (1 - self.take_profit_pct)
    
    def calculate_risk_score(
        self,
        symbol: str,
        signal_confidence: float,
        volatility: float,
        correlation_with_portfolio: float
    ) -> float:
        """Calculate risk score for a signal"""
        risk_score = 0.0
        
        # Confidence risk (lower confidence = higher risk)
        risk_score += (1 - signal_confidence) * 0.3
        
        # Volatility risk
        risk_score += volatility * 0.4
        
        # Correlation risk
        risk_score += correlation_with_portfolio * 0.3
        
        return min(risk_score, 1.0)


class SignalEngine(LoggingMixin):
    """Main signal generation engine"""
    
    def __init__(self, config):
        self.config = config
        self.signal_config = config.signals
        self.risk_config = config.risk_management
        
        # Components
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.risk_manager = RiskManager(self.risk_config.__dict__)
        
        # Signal generation
        self.is_running = False
        self.signal_callbacks = []
        self.latest_signals = {}  # symbol -> latest signal
        
        # Market data buffer
        self.market_data_buffer = {}
        
    async def initialize(self) -> None:
        """Initialize signal engine"""
        self.logger.info("Initializing signal engine...")
        # Any initialization logic here
        self.logger.info("Signal engine initialized")
    
    async def process_market_data(self, tick: MarketTick, provider: str) -> None:
        """Process incoming market data"""
        try:
            # Update technical analyzer
            self.technical_analyzer.update_data(tick)
            
            # Store in buffer
            self.market_data_buffer[tick.symbol] = tick
            
            # Generate signal if conditions are met
            await self._maybe_generate_signal(tick.symbol)
            
        except Exception as e:
            self.logger.error(f"Error processing market data for {tick.symbol}: {e}")
    
    async def process_news_data(self, article: NewsArticle) -> None:
        """Process incoming news data"""
        try:
            # Update sentiment analyzer
            self.sentiment_analyzer.update_news(article)
            
            # Generate signals for mentioned symbols
            for symbol in article.symbols_mentioned:
                await self._maybe_generate_signal(symbol)
                
        except Exception as e:
            self.logger.error(f"Error processing news data: {e}")
    
    async def _maybe_generate_signal(self, symbol: str) -> None:
        """Generate signal if conditions are met"""
        try:
            # Check if we have enough data
            if symbol not in self.market_data_buffer:
                return
            
            current_tick = self.market_data_buffer[symbol]
            
            # Calculate component scores
            technical_score = self.technical_analyzer.generate_technical_score(symbol)
            sentiment_score = self.sentiment_analyzer.calculate_sentiment_score(symbol)
            
            # Ensemble scoring
            ensemble_weights = self.signal_config.ensemble_weights
            
            composite_score = (
                technical_score * ensemble_weights.get('technical', 0.3) +
                sentiment_score * ensemble_weights.get('sentiment', 0.2) +
                0.5 * ensemble_weights.get('fundamental', 0.2) +  # Placeholder
                0.5 * ensemble_weights.get('macro', 0.15) +      # Placeholder
                0.5 * ensemble_weights.get('momentum', 0.15)     # Placeholder
            )
            
            # Determine signal type
            confidence_threshold = self.signal_config.confidence_threshold
            
            if composite_score > 0.8:
                signal_type = SignalType.STRONG_BUY
                confidence = composite_score
            elif composite_score > 0.6:
                signal_type = SignalType.BUY
                confidence = composite_score
            elif composite_score < 0.2:
                signal_type = SignalType.STRONG_SELL
                confidence = 1 - composite_score
            elif composite_score < 0.4:
                signal_type = SignalType.SELL
                confidence = 1 - composite_score
            else:
                signal_type = SignalType.HOLD
                confidence = 0.5
            
            # Only generate signal if confidence is above threshold
            if confidence >= confidence_threshold and signal_type != SignalType.HOLD:
                signal = await self._create_trading_signal(
                    symbol, signal_type, confidence, current_tick,
                    technical_score, sentiment_score
                )
                
                # Store latest signal
                self.latest_signals[symbol] = signal
                
                # Send to callbacks
                for callback in self.signal_callbacks:
                    await callback(signal)
                
                self.logger.info(
                    f"Generated signal for {symbol}: {signal_type.value} "
                    f"(Confidence: {confidence:.2f})"
                )
                
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {e}")
    
    async def _create_trading_signal(
        self,
        symbol: str,
        signal_type: SignalType,
        confidence: float,
        current_tick: MarketTick,
        technical_score: float,
        sentiment_score: float
    ) -> TradingSignal:
        """Create a complete trading signal"""
        
        current_price = current_tick.price
        
        # Calculate position size
        volatility = 0.02  # Placeholder - would calculate from price history
        position_size = self.risk_manager.calculate_position_size(
            symbol, confidence, current_price, volatility
        )
        
        # Calculate stop loss and take profit
        stop_loss = self.risk_manager.calculate_stop_loss(current_price, signal_type)
        take_profit = self.risk_manager.calculate_take_profit(current_price, signal_type)
        
        # Calculate risk score
        correlation = 0.3  # Placeholder
        risk_score = self.risk_manager.calculate_risk_score(
            symbol, confidence, volatility, correlation
        )
        
        # Generate reasons
        reasons = []
        if technical_score > 0.6:
            reasons.append("Strong technical indicators")
        if sentiment_score > 0.6:
            reasons.append("Positive sentiment")
        if technical_score < 0.4:
            reasons.append("Weak technical indicators")
        if sentiment_score < 0.4:
            reasons.append("Negative sentiment")
        
        # Target price (simplified)
        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            target_price = current_price * 1.03  # 3% target
        else:
            target_price = current_price * 0.97  # 3% target
        
        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            target_price=target_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timeframe=TimeFrame.MINUTE,
            timestamp=datetime.now(),
            reasons=reasons,
            risk_score=risk_score,
            position_size=position_size,
            technical_score=technical_score,
            fundamental_score=0.5,  # Placeholder
            sentiment_score=sentiment_score,
            macro_score=0.5,        # Placeholder
            momentum_score=0.5      # Placeholder
        )
    
    async def start_processing(self) -> None:
        """Start signal processing"""
        self.is_running = True
        self.logger.info("Signal engine processing started")
        
        # Main processing loop
        while self.is_running:
            # Processing happens through data callbacks
            await asyncio.sleep(1)
    
    def add_signal_callback(self, callback) -> None:
        """Add callback for generated signals"""
        self.signal_callbacks.append(callback)
    
    def get_latest_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Get latest signal for a symbol"""
        return self.latest_signals.get(symbol)
    
    def get_all_latest_signals(self) -> Dict[str, TradingSignal]:
        """Get all latest signals"""
        return self.latest_signals.copy()
    
    async def stop_processing(self) -> None:
        """Stop signal processing"""
        self.is_running = False
        self.logger.info("Signal engine processing stopped")