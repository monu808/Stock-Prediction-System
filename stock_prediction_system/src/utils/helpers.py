"""Utility functions and helpers for the trading system"""

import asyncio
import time
import hashlib
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from dataclasses import asdict


def calculate_returns(prices: List[float]) -> List[float]:
    """Calculate percentage returns from price series"""
    if len(prices) < 2:
        return []
    
    returns = []
    for i in range(1, len(prices)):
        ret = (prices[i] - prices[i-1]) / prices[i-1]
        returns.append(ret)
    
    return returns


def calculate_volatility(returns: List[float], periods: int = 252) -> float:
    """Calculate annualized volatility from returns"""
    if len(returns) < 2:
        return 0.0
    
    std_dev = np.std(returns)
    return std_dev * np.sqrt(periods)


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.06) -> float:
    """Calculate Sharpe ratio"""
    if len(returns) < 2:
        return 0.0
    
    excess_returns = [r - risk_free_rate/252 for r in returns]  # Daily risk-free rate
    avg_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns)
    
    if std_excess_return == 0:
        return 0.0
    
    return (avg_excess_return / std_excess_return) * np.sqrt(252)


def calculate_max_drawdown(prices: List[float]) -> float:
    """Calculate maximum drawdown from price series"""
    if len(prices) < 2:
        return 0.0
    
    peak = prices[0]
    max_dd = 0.0
    
    for price in prices[1:]:
        if price > peak:
            peak = price
        else:
            drawdown = (peak - price) / peak
            max_dd = max(max_dd, drawdown)
    
    return max_dd


def normalize_symbol(symbol: str) -> str:
    """Normalize stock symbol to standard format"""
    return symbol.upper().strip()


def generate_hash(data: Any) -> str:
    """Generate hash for data"""
    if isinstance(data, dict):
        data_str = json.dumps(data, sort_keys=True)
    else:
        data_str = str(data)
    
    return hashlib.md5(data_str.encode()).hexdigest()


def serialize_dataclass(obj: Any) -> Dict[str, Any]:
    """Serialize dataclass to dictionary"""
    if hasattr(obj, '__dataclass_fields__'):
        result = asdict(obj)
        
        # Convert datetime objects to ISO strings
        for key, value in result.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
        
        return result
    else:
        return obj


def batch_list(items: List[Any], batch_size: int) -> List[List[Any]]:
    """Split list into batches"""
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    return batches


def retry_async(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying async functions"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
                    continue
            
            raise last_exception
        
        return wrapper
    return decorator


def format_currency(amount: float, currency: str = "INR") -> str:
    """Format currency amount"""
    if currency == "INR":
        if amount >= 10000000:  # 1 Crore
            return f"₹{amount/10000000:.2f}Cr"
        elif amount >= 100000:  # 1 Lakh
            return f"₹{amount/100000:.2f}L"
        else:
            return f"₹{amount:,.2f}"
    else:
        return f"{currency} {amount:,.2f}"


def format_percentage(value: float, decimal_places: int = 2) -> str:
    """Format percentage"""
    return f"{value * 100:.{decimal_places}f}%"


def is_market_hours(exchange: str = "NSE") -> bool:
    """Check if market is currently open"""
    now = datetime.now()
    
    # Check if it's a weekday
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    if exchange == "NSE":
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    return False


def get_next_trading_day(current_date: datetime = None) -> datetime:
    """Get next trading day"""
    if current_date is None:
        current_date = datetime.now()
    
    next_day = current_date + timedelta(days=1)
    
    # Skip weekends
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    
    return next_day


def calculate_portfolio_metrics(positions: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate portfolio-level metrics"""
    total_value = sum(pos.get('market_value', 0) for pos in positions)
    total_pnl = sum(pos.get('unrealized_pnl', 0) for pos in positions)
    
    if total_value == 0:
        return {
            'total_value': 0.0,
            'total_pnl': 0.0,
            'pnl_percentage': 0.0,
            'position_count': 0
        }
    
    return {
        'total_value': total_value,
        'total_pnl': total_pnl,
        'pnl_percentage': total_pnl / (total_value - total_pnl),
        'position_count': len(positions)
    }


def validate_symbol(symbol: str) -> bool:
    """Validate if symbol is in correct format"""
    if not symbol or not isinstance(symbol, str):
        return False
    
    symbol = symbol.strip().upper()
    
    # Basic validation - alphanumeric characters only
    if not symbol.isalnum():
        return False
    
    # Length check
    if len(symbol) < 2 or len(symbol) > 20:
        return False
    
    return True


def rate_limiter(calls_per_second: float):
    """Rate limiting decorator"""
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            
            if left_to_wait > 0:
                await asyncio.sleep(left_to_wait)
            
            ret = await func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        
        return wrapper
    return decorator


class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"{self.name} took {duration:.4f} seconds")
    
    @property
    def duration(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


class CircuitBreaker:
    """Circuit breaker for handling failures"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        """Call function with circuit breaker protection"""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker"""
        if self.last_failure_time is None:
            return False
        
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


def create_correlation_matrix(price_data: Dict[str, List[float]]) -> pd.DataFrame:
    """Create correlation matrix from price data"""
    # Convert to DataFrame
    df = pd.DataFrame(price_data)
    
    # Calculate returns
    returns_df = df.pct_change().dropna()
    
    # Calculate correlation matrix
    correlation_matrix = returns_df.corr()
    
    return correlation_matrix


def detect_outliers(data: List[float], method: str = "iqr") -> List[int]:
    """Detect outliers in data"""
    if len(data) < 4:
        return []
    
    data_array = np.array(data)
    outlier_indices = []
    
    if method == "iqr":
        Q1 = np.percentile(data_array, 25)
        Q3 = np.percentile(data_array, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_indices = [
            i for i, value in enumerate(data_array)
            if value < lower_bound or value > upper_bound
        ]
    
    elif method == "zscore":
        mean = np.mean(data_array)
        std = np.std(data_array)
        
        if std > 0:
            z_scores = np.abs((data_array - mean) / std)
            outlier_indices = [
                i for i, z_score in enumerate(z_scores)
                if z_score > 3
            ]
    
    return outlier_indices