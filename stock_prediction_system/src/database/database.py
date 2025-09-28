"""Database initialization and management"""

import asyncio
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from .models import Base, Symbols
from ..utils.logging_config import LoggingMixin


class DatabaseManager(LoggingMixin):
    """Database manager for handling connections and operations"""
    
    def __init__(self, config: dict):
        self.config = config
        self.engine = None
        self.async_engine = None
        self.session_factory = None
        self.async_session_factory = None
        
        # Connection strings
        self.postgres_url = self._build_postgres_url()
        self.async_postgres_url = self._build_async_postgres_url()
    
    def _build_postgres_url(self) -> str:
        """Build PostgreSQL connection URL"""
        pg_config = self.config['postgres']
        return (
            f"postgresql://{pg_config['username']}:{pg_config['password']}"
            f"@{pg_config['host']}:{pg_config['port']}/{pg_config['database']}"
        )
    
    def _build_async_postgres_url(self) -> str:
        """Build async PostgreSQL connection URL"""
        pg_config = self.config['postgres']
        return (
            f"postgresql+asyncpg://{pg_config['username']}:{pg_config['password']}"
            f"@{pg_config['host']}:{pg_config['port']}/{pg_config['database']}"
        )
    
    async def initialize(self) -> None:
        """Initialize database connections"""
        try:
            # Create sync engine
            self.engine = create_engine(
                self.postgres_url,
                pool_size=20,
                max_overflow=30,
                pool_pre_ping=True,
                echo=False
            )
            
            # Create async engine
            self.async_engine = create_async_engine(
                self.async_postgres_url,
                pool_size=20,
                max_overflow=30,
                pool_pre_ping=True,
                echo=False
            )
            
            # Create session factories
            self.session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False
            )
            
            self.async_session_factory = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                autocommit=False,
                autoflush=False
            )
            
            self.logger.info("Database connections initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def create_tables(self) -> None:
        """Create all database tables"""
        try:
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            self.logger.info("Database tables created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create tables: {e}")
            raise
    
    async def drop_tables(self) -> None:
        """Drop all database tables"""
        try:
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            
            self.logger.info("Database tables dropped successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to drop tables: {e}")
            raise
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session"""
        async with self.async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    def get_sync_session(self):
        """Get sync database session"""
        return self.session_factory()
    
    async def close(self) -> None:
        """Close database connections"""
        if self.async_engine:
            await self.async_engine.dispose()
        
        if self.engine:
            self.engine.dispose()
        
        self.logger.info("Database connections closed")
    
    async def initialize_master_data(self) -> None:
        """Initialize master data (symbols, etc.)"""
        try:
            async with self.get_async_session() as session:
                # Check if symbols already exist
                existing_count = await session.execute(
                    "SELECT COUNT(*) FROM symbols"
                )
                count = existing_count.scalar()
                
                if count == 0:
                    # Insert default symbols
                    default_symbols = [
                        {'symbol': 'RELIANCE', 'company_name': 'Reliance Industries Ltd', 'exchange': 'NSE', 'sector': 'Oil & Gas'},
                        {'symbol': 'TCS', 'company_name': 'Tata Consultancy Services Ltd', 'exchange': 'NSE', 'sector': 'IT'},
                        {'symbol': 'HDFCBANK', 'company_name': 'HDFC Bank Ltd', 'exchange': 'NSE', 'sector': 'Banking'},
                        {'symbol': 'INFY', 'company_name': 'Infosys Ltd', 'exchange': 'NSE', 'sector': 'IT'},
                        {'symbol': 'ICICIBANK', 'company_name': 'ICICI Bank Ltd', 'exchange': 'NSE', 'sector': 'Banking'},
                        {'symbol': 'SBIN', 'company_name': 'State Bank of India', 'exchange': 'NSE', 'sector': 'Banking'},
                        {'symbol': 'BHARTIARTL', 'company_name': 'Bharti Airtel Ltd', 'exchange': 'NSE', 'sector': 'Telecom'},
                        {'symbol': 'ITC', 'company_name': 'ITC Ltd', 'exchange': 'NSE', 'sector': 'FMCG'},
                        {'symbol': 'KOTAKBANK', 'company_name': 'Kotak Mahindra Bank Ltd', 'exchange': 'NSE', 'sector': 'Banking'},
                        {'symbol': 'LT', 'company_name': 'Larsen & Toubro Ltd', 'exchange': 'NSE', 'sector': 'Construction'},
                        {'symbol': 'ASIANPAINT', 'company_name': 'Asian Paints Ltd', 'exchange': 'NSE', 'sector': 'Consumer Goods'},
                        {'symbol': 'MARUTI', 'company_name': 'Maruti Suzuki India Ltd', 'exchange': 'NSE', 'sector': 'Automotive'},
                        {'symbol': 'AXISBANK', 'company_name': 'Axis Bank Ltd', 'exchange': 'NSE', 'sector': 'Banking'},
                        {'symbol': 'BAJFINANCE', 'company_name': 'Bajaj Finance Ltd', 'exchange': 'NSE', 'sector': 'Financial Services'},
                        {'symbol': 'WIPRO', 'company_name': 'Wipro Ltd', 'exchange': 'NSE', 'sector': 'IT'},
                    ]
                    
                    for symbol_data in default_symbols:
                        symbol_obj = Symbols(**symbol_data)
                        session.add(symbol_obj)
                    
                    await session.commit()
                    self.logger.info(f"Initialized {len(default_symbols)} symbols")
                else:
                    self.logger.info(f"Found {count} existing symbols")
                    
        except Exception as e:
            self.logger.error(f"Failed to initialize master data: {e}")
            raise


# Global database manager instance
db_manager: DatabaseManager = None


async def init_database(config: dict) -> DatabaseManager:
    """Initialize database manager"""
    global db_manager
    
    db_manager = DatabaseManager(config)
    await db_manager.initialize()
    await db_manager.create_tables()
    await db_manager.initialize_master_data()
    
    return db_manager


async def get_database() -> DatabaseManager:
    """Get database manager instance"""
    if db_manager is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    return db_manager


async def close_database():
    """Close database connections"""
    global db_manager
    
    if db_manager:
        await db_manager.close()
        db_manager = None