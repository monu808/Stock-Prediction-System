"""Configuration management for the trading system"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class APIConfig:
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list = field(default_factory=lambda: ["*"])
    jwt_secret: str = "default-secret-key"
    rate_limit: int = 1000


@dataclass
class DatabaseConfig:
    """Database configuration"""
    influxdb: Dict[str, Any] = field(default_factory=dict)
    postgres: Dict[str, Any] = field(default_factory=dict)
    redis: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KafkaConfig:
    """Kafka configuration"""
    bootstrap_servers: list = field(default_factory=lambda: ["localhost:9092"])
    topics: Dict[str, str] = field(default_factory=dict)


@dataclass
class DataSourceConfig:
    """Data source configuration"""
    nse: Dict[str, Any] = field(default_factory=dict)
    bse: Dict[str, Any] = field(default_factory=dict)
    yahoo_finance: Dict[str, Any] = field(default_factory=dict)
    alpha_vantage: Dict[str, Any] = field(default_factory=dict)
    news_sources: Dict[str, Any] = field(default_factory=dict)
    twitter: Dict[str, Any] = field(default_factory=dict)
    mock_provider: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Model configuration"""
    lstm_price: Dict[str, Any] = field(default_factory=dict)
    transformer_sentiment: Dict[str, Any] = field(default_factory=dict)
    gnn_sector: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_position_size: float = 0.05
    max_portfolio_risk: float = 0.02
    stop_loss_percentage: float = 0.03
    take_profit_percentage: float = 0.06
    correlation_threshold: float = 0.7
    circuit_breakers: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalConfig:
    """Signal generation configuration"""
    confidence_threshold: float = 0.6
    ensemble_weights: Dict[str, float] = field(default_factory=dict)
    timeframes: Dict[str, bool] = field(default_factory=dict)


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    prometheus: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)
    alerts: Dict[str, Any] = field(default_factory=dict)


class Config:
    """Main configuration class"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self._config_data = self._load_config()
        
        # Initialize configuration sections
        self.api = self._create_api_config()
        self.databases = self._create_database_config()
        self.kafka = self._create_kafka_config()
        self.data_sources = self._create_data_source_config()
        self.models = self._create_model_config()
        self.risk_management = self._create_risk_config()
        self.signals = self._create_signal_config()
        self.monitoring = self._create_monitoring_config()
        
        # Additional configurations
        self.backtesting = self._config_data.get("backtesting", {})
        self.performance = self._config_data.get("performance", {})
    
    def _find_config_file(self) -> str:
        """Find configuration file"""
        possible_paths = [
            Path(__file__).parent.parent / "config" / "config.yaml",
            Path(__file__).parent.parent / "config" / "config.example.yaml",
            Path("config.yaml"),
            Path("config/config.yaml")
        ]
        
        for path in possible_paths:
            if path.exists():
                print(f"Found config file: {path}")
                return str(path)
        
        # Create default config if none found
        default_path = Path(__file__).parent.parent / "config" / "config.yaml"
        print(f"No config file found, creating default at: {default_path}")
        self._create_default_config(default_path)
        return str(default_path)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            print(f"Loaded config from {self.config_path}")
            print(f"data_sources section: {config.get('data_sources', {})}")
                
            # Environment variable substitution
            config = self._substitute_env_vars(config)
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    def _substitute_env_vars(self, config: Any) -> Any:
        """Substitute environment variables in configuration"""
        if isinstance(config, dict):
            return {key: self._substitute_env_vars(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        else:
            return config
    
    def _create_default_config(self, path: Path) -> None:
        """Create default configuration file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        default_config = {
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "cors_origins": ["*"],
                "jwt_secret": "default-secret-key",
                "rate_limit": 1000
            },
            "databases": {
                "influxdb": {
                    "url": "http://localhost:8086",
                    "token": "your-token-here",
                    "org": "trading_org",
                    "bucket": "market_data"
                },
                "postgres": {
                    "host": "localhost",
                    "port": 5432,
                    "database": "trading_db",
                    "username": "admin",
                    "password": "password123"
                },
                "redis": {
                    "host": "localhost",
                    "port": 6379,
                    "password": None,
                    "db": 0
                }
            }
        }
        
        with open(path, 'w') as file:
            yaml.dump(default_config, file, default_flow_style=False)
    
    def _create_api_config(self) -> APIConfig:
        """Create API configuration"""
        api_data = self._config_data.get("api", {})
        return APIConfig(
            host=api_data.get("host", "0.0.0.0"),
            port=api_data.get("port", 8000),
            cors_origins=api_data.get("cors_origins", ["*"]),
            jwt_secret=api_data.get("jwt_secret", "default-secret-key"),
            rate_limit=api_data.get("rate_limit", 1000)
        )
    
    def _create_database_config(self) -> DatabaseConfig:
        """Create database configuration"""
        db_data = self._config_data.get("databases", {})
        return DatabaseConfig(
            influxdb=db_data.get("influxdb", {}),
            postgres=db_data.get("postgres", {}),
            redis=db_data.get("redis", {})
        )
    
    def _create_kafka_config(self) -> KafkaConfig:
        """Create Kafka configuration"""
        kafka_data = self._config_data.get("kafka", {})
        return KafkaConfig(
            bootstrap_servers=kafka_data.get("bootstrap_servers", ["localhost:9092"]),
            topics=kafka_data.get("topics", {})
        )
    
    def _create_data_source_config(self) -> DataSourceConfig:
        """Create data source configuration"""
        data_sources = self._config_data.get("data_sources", {})
        return DataSourceConfig(
            nse=data_sources.get("nse", {}),
            bse=data_sources.get("bse", {}),
            yahoo_finance=data_sources.get("yahoo_finance", {}),
            alpha_vantage=data_sources.get("alpha_vantage", {}),
            news_sources=data_sources.get("news_sources", {}),
            twitter=data_sources.get("twitter", {}),
            mock_provider=data_sources.get("mock_provider", {})
        )
    
    def _create_model_config(self) -> ModelConfig:
        """Create model configuration"""
        models = self._config_data.get("models", {})
        return ModelConfig(
            lstm_price=models.get("lstm_price", {}),
            transformer_sentiment=models.get("transformer_sentiment", {}),
            gnn_sector=models.get("gnn_sector", {})
        )
    
    def _create_risk_config(self) -> RiskConfig:
        """Create risk management configuration"""
        risk = self._config_data.get("risk_management", {})
        return RiskConfig(
            max_position_size=risk.get("max_position_size", 0.05),
            max_portfolio_risk=risk.get("max_portfolio_risk", 0.02),
            stop_loss_percentage=risk.get("stop_loss_percentage", 0.03),
            take_profit_percentage=risk.get("take_profit_percentage", 0.06),
            correlation_threshold=risk.get("correlation_threshold", 0.7),
            circuit_breakers=risk.get("circuit_breakers", {})
        )
    
    def _create_signal_config(self) -> SignalConfig:
        """Create signal configuration"""
        signals = self._config_data.get("signals", {})
        return SignalConfig(
            confidence_threshold=signals.get("confidence_threshold", 0.6),
            ensemble_weights=signals.get("ensemble_weights", {}),
            timeframes=signals.get("timeframes", {})
        )
    
    def _create_monitoring_config(self) -> MonitoringConfig:
        """Create monitoring configuration"""
        monitoring = self._config_data.get("monitoring", {})
        return MonitoringConfig(
            prometheus=monitoring.get("prometheus", {}),
            logging=monitoring.get("logging", {}),
            alerts=monitoring.get("alerts", {})
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        keys = key.split(".")
        value = self._config_data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def reload(self) -> None:
        """Reload configuration"""
        self._config_data = self._load_config()
        self.__init__(self.config_path)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self._config_data