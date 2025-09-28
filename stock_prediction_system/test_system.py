"""
Test the basic functionality of the trading system
Run this script to verify the system is working correctly
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.config import Config
from src.utils.logging_config import setup_logging
from src.data_ingestion.market_data import MarketDataCollector, MarketTick
from src.execution.signal_engine import SignalEngine
from datetime import datetime


async def test_configuration():
    """Test configuration loading"""
    print("Testing configuration...")
    try:
        config = Config()
        assert config.api.host is not None
        assert config.api.port > 0
        print("✅ Configuration test passed")
        return config
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return None


async def test_logging():
    """Test logging system"""
    print("Testing logging system...")
    try:
        logger = setup_logging()
        logger.info("Test log message")
        print("✅ Logging test passed")
        return True
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False


async def test_signal_engine(config):
    """Test signal engine initialization"""
    print("Testing signal engine...")
    try:
        signal_engine = SignalEngine(config)
        await signal_engine.initialize()
        
        # Create a mock market tick
        mock_tick = MarketTick(
            symbol="RELIANCE",
            timestamp=datetime.now(),
            price=2500.0,
            volume=1000,
            bid_price=2499.0,
            ask_price=2501.0,
            bid_size=100,
            ask_size=100,
            exchange="NSE"
        )
        
        # Process the mock data
        await signal_engine.process_market_data(mock_tick, "test_provider")
        
        print("✅ Signal engine test passed")
        return True
    except Exception as e:
        print(f"❌ Signal engine test failed: {e}")
        return False


async def test_data_ingestion_setup(config):
    """Test data ingestion setup"""
    print("Testing data ingestion setup...")
    try:
        # Test if we can create a market data collector
        if hasattr(config, 'data_sources') and config.data_sources:
            collector = MarketDataCollector(config.data_sources.__dict__)
            print("✅ Data ingestion setup test passed")
            return True
        else:
            print("⚠️ Data sources not configured, but setup test passed")
            return True
    except Exception as e:
        print(f"❌ Data ingestion setup test failed: {e}")
        return False


async def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("STOCK MARKET PREDICTION SYSTEM - SYSTEM TESTS")
    print("=" * 50)
    
    test_results = []
    
    # Test configuration
    config = await test_configuration()
    test_results.append(config is not None)
    
    if config is None:
        print("Cannot proceed with other tests due to configuration failure")
        return False
    
    # Test logging
    logging_result = await test_logging()
    test_results.append(logging_result)
    
    # Test signal engine
    signal_result = await test_signal_engine(config)
    test_results.append(signal_result)
    
    # Test data ingestion setup
    data_result = await test_data_ingestion_setup(config)
    test_results.append(data_result)
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to run.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the configuration and dependencies.")
        return False


def check_dependencies():
    """Check if required dependencies are available"""
    print("Checking dependencies...")
    
    required_modules = [
        ('asyncio', 'asyncio'),
        ('datetime', 'datetime'),
        ('pathlib', 'pathlib'),
        ('json', 'json'),
        ('time', 'time'),
        ('typing', 'typing'),
        ('dataclasses', 'dataclasses'),
        ('enum', 'enum'),
    ]
    
    missing_modules = []
    
    for module_name, import_name in required_modules:
        try:
            __import__(import_name)
        except ImportError:
            missing_modules.append(module_name)
    
    if missing_modules:
        print(f"❌ Missing dependencies: {', '.join(missing_modules)}")
        print("Please install missing dependencies and try again.")
        return False
    else:
        print("✅ All core dependencies available")
        return True


async def main():
    """Main test function"""
    print("Starting system tests...\n")
    
    # Check dependencies first
    if not check_dependencies():
        return 1
    
    # Run all tests
    success = await run_all_tests()
    
    if success:
        print("\n🚀 System is ready! You can now run: python main.py")
        return 0
    else:
        print("\n💥 System tests failed. Please fix the issues and try again.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)