"""Database models and schema for the trading system"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class MarketData(Base):
    """Market data table"""
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    price = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    bid_price = Column(Float)
    ask_price = Column(Float)
    bid_size = Column(Integer)
    ask_size = Column(Integer)
    exchange = Column(String(10), nullable=False)
    provider = Column(String(20))
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_timestamp_symbol', 'timestamp', 'symbol'),
    )


class NewsData(Base):
    """News articles table"""
    __tablename__ = 'news_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(Text, nullable=False)
    content = Column(Text)
    source = Column(String(50), nullable=False)
    url = Column(Text)
    published_at = Column(DateTime, nullable=False, index=True)
    sentiment_score = Column(Float)
    sentiment_label = Column(String(20))
    category = Column(String(50))
    importance_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to symbols mentioned
    symbols_mentioned = relationship("NewsSymbol", back_populates="news")


class NewsSymbol(Base):
    """Junction table for news articles and symbols mentioned"""
    __tablename__ = 'news_symbols'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    news_id = Column(Integer, ForeignKey('news_data.id'), nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    
    # Relationships
    news = relationship("NewsData", back_populates="symbols_mentioned")
    
    __table_args__ = (
        Index('idx_news_symbol', 'news_id', 'symbol'),
    )


class TradingSignals(Base):
    """Trading signals table"""
    __tablename__ = 'trading_signals'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    signal_type = Column(String(20), nullable=False)  # BUY, SELL, HOLD, etc.
    confidence = Column(Float, nullable=False)
    target_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    reasons = Column(Text)  # JSON string of reasons
    risk_score = Column(Float)
    position_size = Column(Float)
    
    # Component scores
    technical_score = Column(Float)
    fundamental_score = Column(Float)
    sentiment_score = Column(Float)
    macro_score = Column(Float)
    momentum_score = Column(Float)
    
    # Signal status
    status = Column(String(20), default='active')  # active, expired, executed
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_signal_status', 'status', 'timestamp'),
    )


class Symbols(Base):
    """Symbols master table"""
    __tablename__ = 'symbols'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), unique=True, nullable=False, index=True)
    company_name = Column(String(200))
    sector = Column(String(100))
    industry = Column(String(100))
    exchange = Column(String(10), nullable=False)
    is_active = Column(Boolean, default=True)
    market_cap = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TechnicalIndicators(Base):
    """Technical indicators table"""
    __tablename__ = 'technical_indicators'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Moving Averages
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    sma_200 = Column(Float)
    ema_12 = Column(Float)
    ema_26 = Column(Float)
    
    # Momentum Indicators
    rsi = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_histogram = Column(Float)
    
    # Bollinger Bands
    bb_upper = Column(Float)
    bb_middle = Column(Float)
    bb_lower = Column(Float)
    bb_width = Column(Float)
    
    # Volume Indicators
    volume_sma = Column(Float)
    volume_ratio = Column(Float)
    
    # Volatility
    atr = Column(Float)  # Average True Range
    volatility = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
    )


class ModelPredictions(Base):
    """Model predictions table"""
    __tablename__ = 'model_predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    model_name = Column(String(50), nullable=False)
    prediction_timestamp = Column(DateTime, nullable=False, index=True)
    target_timestamp = Column(DateTime, nullable=False)  # When prediction is for
    
    # Predictions
    predicted_price = Column(Float)
    predicted_direction = Column(String(10))  # UP, DOWN, FLAT
    confidence = Column(Float)
    
    # Actual values (for backtesting)
    actual_price = Column(Float)
    actual_direction = Column(String(10))
    prediction_error = Column(Float)
    
    # Model metadata
    model_version = Column(String(20))
    feature_importance = Column(Text)  # JSON string
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_symbol_model_timestamp', 'symbol', 'model_name', 'prediction_timestamp'),
    )


class Portfolio(Base):
    """Portfolio positions table"""
    __tablename__ = 'portfolio'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    quantity = Column(Integer, nullable=False)
    average_price = Column(Float, nullable=False)
    current_price = Column(Float)
    unrealized_pnl = Column(Float)
    realized_pnl = Column(Float, default=0.0)
    
    # Position metadata
    entry_date = Column(DateTime, nullable=False)
    entry_signal_id = Column(Integer, ForeignKey('trading_signals.id'))
    status = Column(String(20), default='open')  # open, closed, partial
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Backtests(Base):
    """Backtesting results table"""
    __tablename__ = 'backtests'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    test_name = Column(String(100), nullable=False)
    strategy_name = Column(String(100), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    
    # Performance metrics
    initial_capital = Column(Float, nullable=False)
    final_capital = Column(Float, nullable=False)
    total_return = Column(Float)
    annualized_return = Column(Float)
    max_drawdown = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    
    # Trade statistics
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    win_rate = Column(Float)
    avg_win = Column(Float)
    avg_loss = Column(Float)
    profit_factor = Column(Float)
    
    # Configuration
    config = Column(Text)  # JSON string of backtest configuration
    results = Column(Text)  # JSON string of detailed results
    
    created_at = Column(DateTime, default=datetime.utcnow)


class SystemLogs(Base):
    """System logs table"""
    __tablename__ = 'system_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True, default=datetime.utcnow)
    level = Column(String(20), nullable=False, index=True)
    component = Column(String(50), nullable=False, index=True)
    message = Column(Text, nullable=False)
    details = Column(Text)  # JSON string for additional details
    
    __table_args__ = (
        Index('idx_timestamp_level', 'timestamp', 'level'),
        Index('idx_component_timestamp', 'component', 'timestamp'),
    )