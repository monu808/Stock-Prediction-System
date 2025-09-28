"""Logging configuration for the trading system"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    log_format: Optional[str] = None,
    max_file_size: str = "100MB",
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup logging configuration for the trading system
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        log_format: Log message format
        max_file_size: Maximum size of log files
        backup_count: Number of backup log files to keep
    
    Returns:
        Configured logger instance
    """
    
    # Default log directory
    if log_dir is None:
        log_dir = Path(__file__).parent.parent.parent / "logs"
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Default log format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    
    # Convert level string to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create root logger
    logger = logging.getLogger("trading_system")
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler for general logs
    general_log_file = log_dir / f"trading_system_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        general_log_file,
        maxBytes=_parse_size(max_file_size),
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Error log file handler
    error_log_file = log_dir / f"trading_system_errors_{datetime.now().strftime('%Y%m%d')}.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=_parse_size(max_file_size),
        backupCount=backup_count,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    
    # Trading signals log file
    signals_log_file = log_dir / f"trading_signals_{datetime.now().strftime('%Y%m%d')}.log"
    signals_handler = logging.handlers.RotatingFileHandler(
        signals_log_file,
        maxBytes=_parse_size(max_file_size),
        backupCount=backup_count,
        encoding='utf-8'
    )
    signals_handler.setLevel(logging.INFO)
    signals_handler.setFormatter(formatter)
    
    # Create signals logger
    signals_logger = logging.getLogger("trading_system.signals")
    signals_logger.addHandler(signals_handler)
    signals_logger.setLevel(logging.INFO)
    signals_logger.propagate = False
    
    # Market data log file
    market_data_log_file = log_dir / f"market_data_{datetime.now().strftime('%Y%m%d')}.log"
    market_data_handler = logging.handlers.RotatingFileHandler(
        market_data_log_file,
        maxBytes=_parse_size(max_file_size),
        backupCount=backup_count,
        encoding='utf-8'
    )
    market_data_handler.setLevel(logging.DEBUG)
    market_data_handler.setFormatter(formatter)
    
    # Create market data logger
    market_data_logger = logging.getLogger("trading_system.market_data")
    market_data_logger.addHandler(market_data_handler)
    # Ensure debug messages for market data are captured regardless of root logger level
    market_data_logger.setLevel(logging.DEBUG)
    market_data_logger.propagate = False
    
    # Performance log file
    performance_log_file = log_dir / f"performance_{datetime.now().strftime('%Y%m%d')}.log"
    performance_handler = logging.handlers.RotatingFileHandler(
        performance_log_file,
        maxBytes=_parse_size(max_file_size),
        backupCount=backup_count,
        encoding='utf-8'
    )
    performance_handler.setLevel(logging.INFO)
    performance_handler.setFormatter(formatter)
    
    # Create performance logger
    performance_logger = logging.getLogger("trading_system.performance")
    performance_logger.addHandler(performance_handler)
    performance_logger.setLevel(logging.INFO)
    performance_logger.propagate = False
    
    logger.info(f"Logging configured - Level: {level}, Log Dir: {log_dir}")
    
    return logger


def _parse_size(size_str: str) -> int:
    """Parse size string to bytes"""
    size_str = size_str.upper()
    if size_str.endswith('KB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        return int(size_str)


def get_logger(name: str) -> logging.Logger:
    """Get logger instance"""
    return logging.getLogger(f"trading_system.{name}")


class LoggingMixin:
    """Mixin class to add logging capability"""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        # Use the module name (last part) to map to the logger names created in setup_logging
        # e.g., src.data_ingestion.market_data -> 'market_data' logger
        try:
            module_name = self.__class__.__module__.split('.')[-1]
            return get_logger(module_name)
        except Exception:
            # Fallback to class-name based logger
            return get_logger(self.__class__.__name__.lower())