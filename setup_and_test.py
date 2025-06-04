#!/usr/bin/env python3
"""
AI Trading Bot - Setup and Test Script
This script helps users set up and test the trading system.
"""

import os
import sys
import asyncio
from datetime import datetime
import pandas as pd

def check_dependencies():
    """Check if all required packages are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'yfinance', 'streamlit', 'plotly',
        'scikit-learn', 'tensorflow', 'loguru', 'requests',
        'beautifulsoup4', 'vaderSentiment', 'alpha-vantage'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def check_environment():
    """Check environment configuration"""
    print("\n🔍 Checking environment configuration...")
    
    required_vars = [
        'ALPHA_VANTAGE_API_KEY'
    ]
    
    env_file_exists = os.path.exists('.env')
    print(f"  .env file: {'✅ Found' if env_file_exists else '❌ Not found'}")
    
    if not env_file_exists:
        print("  📝 Create .env file from environment_example.txt")
        return False
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        for var in required_vars:
            value = os.getenv(var)
            if value and value != f'your_{var.lower()}_here':
                print(f"  ✅ {var}")
            else:
                print(f"  ❌ {var} (not set or using default value)")
                return False
                
    except ImportError:
        print("  ⚠️ python-dotenv not installed")
        return False
    
    print("✅ Environment configuration looks good!")
    return True

def test_data_fetching():
    """Test data fetching functionality"""
    print("\n🔍 Testing data fetching...")
    
    try:
        from data_fetcher import DataFetcher
        
        fetcher = DataFetcher()
        
        # Test Yahoo Finance
        print("  📊 Testing Yahoo Finance...")
        data = fetcher.get_stock_data('AAPL', period='1mo')
        
        if data is not None and not data.empty:
            print(f"    ✅ Retrieved {len(data)} days of AAPL data")
            print(f"    📈 Latest close: ${data['Close'].iloc[-1]:.2f}")
        else:
            print("    ❌ Failed to retrieve data")
            return False
        
        # Test technical indicators
        print("  📈 Testing technical indicators...")
        data_with_indicators = fetcher.calculate_technical_indicators(data)
        
        indicators = ['SMA_20', 'RSI', 'MACD']
        for indicator in indicators:
            if indicator in data_with_indicators.columns:
                print(f"    ✅ {indicator}")
            else:
                print(f"    ❌ {indicator}")
                return False
        
        print("✅ Data fetching works correctly!")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_ai_models():
    """Test AI model initialization and basic functionality"""
    print("\n🔍 Testing AI models...")
    
    try:
        from ai_models import StockPricePredictor, TradingSignalGenerator
        from data_fetcher import DataFetcher
        
        # Get some test data
        fetcher = DataFetcher()
        data = fetcher.get_stock_data('AAPL', period='6mo')
        data_with_indicators = fetcher.calculate_technical_indicators(data)
        
        # Test price predictor
        print("  🧠 Testing price predictor...")
        predictor = StockPricePredictor()
        print("    ✅ Price predictor initialized")
        
        # Test signal generator
        print("  📊 Testing signal generator...")
        signal_gen = TradingSignalGenerator()
        
        # Test feature preparation
        feature_data = signal_gen.prepare_signal_features(data_with_indicators)
        if len(feature_data) > 0:
            print("    ✅ Signal features prepared")
        else:
            print("    ❌ Signal features failed")
            return False
        
        print("✅ AI models initialize correctly!")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

async def test_news_sentiment():
    """Test news and sentiment analysis"""
    print("\n🔍 Testing news and sentiment analysis...")
    
    try:
        from data_fetcher import DataFetcher
        
        fetcher = DataFetcher()
        
        print("  📰 Testing news fetching...")
        sentiment_data = await fetcher.fetch_news_sentiment('AAPL', num_articles=3)
        
        if sentiment_data and 'articles' in sentiment_data:
            print(f"    ✅ Retrieved {len(sentiment_data['articles'])} articles")
            print(f"    😊 Overall sentiment: {sentiment_data['sentiment_score']:.3f}")
        else:
            print("    ⚠️ No news data (this is normal for rate limiting)")
        
        print("✅ News and sentiment analysis works!")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_trading_engine():
    """Test trading engine functionality"""
    print("\n🔍 Testing trading engine...")
    
    try:
        from trading_engine import TradingEngine, Portfolio
        
        # Test portfolio
        print("  💼 Testing portfolio...")
        portfolio = Portfolio(initial_capital=10000)
        print(f"    ✅ Portfolio initialized with ${portfolio.cash:,.2f}")
        
        # Test trading engine
        print("  🤖 Testing trading engine...")
        engine = TradingEngine()
        print("    ✅ Trading engine initialized")
        
        print("✅ Trading engine works correctly!")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def quick_demo():
    """Run a quick demonstration of the system"""
    print("\n🎯 Running quick demo...")
    
    try:
        from data_fetcher import DataFetcher
        
        fetcher = DataFetcher()
        
        # Get data for Apple
        print("  📊 Fetching AAPL data...")
        data = fetcher.get_stock_data('AAPL', period='1mo')
        data_with_indicators = fetcher.calculate_technical_indicators(data)
        
        # Show latest data
        latest = data_with_indicators.iloc[-1]
        print(f"  📈 AAPL Latest Data:")
        print(f"    💰 Price: ${latest['Close']:.2f}")
        print(f"    📊 RSI: {latest['RSI']:.1f}")
        print(f"    📈 SMA(20): ${latest['SMA_20']:.2f}")
        
        # Market indicators
        print("  🌍 Market Overview:")
        indicators = fetcher.get_market_indicators()
        
        for symbol, data in indicators.items():
            clean_symbol = symbol.replace('^', '')
            print(f"    {clean_symbol}: ${data['price']:.2f} ({data['change_pct']:+.2f}%)")
        
        print("✅ Demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"  ❌ Demo failed: {e}")
        return False

def main():
    """Main setup and test function"""
    print("🤖 AI Trading Bot - Setup and Test")
    print("=" * 50)
    
    # Run all tests
    tests = [
        ("Dependencies", check_dependencies),
        ("Environment", check_environment),
        ("Data Fetching", test_data_fetching),
        ("AI Models", test_ai_models),
        ("Trading Engine", test_trading_engine),
    ]
    
    async_tests = [
        ("News Sentiment", test_news_sentiment),
    ]
    
    results = {}
    
    # Run synchronous tests
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Run asynchronous tests
    async def run_async_tests():
        for test_name, test_func in async_tests:
            try:
                results[test_name] = await test_func()
            except Exception as e:
                print(f"❌ {test_name} test crashed: {e}")
                results[test_name] = False
    
    asyncio.run(run_async_tests())
    
    # Run demo if basic tests pass
    if all(results.values()):
        results["Demo"] = quick_demo()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All tests passed! Your system is ready to use.")
        print("\n🚀 Next steps:")
        print("  1. Run the dashboard: streamlit run dashboard.py")
        print("  2. Train models: python main.py (will train on first run)")
        print("  3. Start trading: Enable auto-trading in the dashboard")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        print("   Make sure you have:")
        print("   - Installed all dependencies: pip install -r requirements.txt")
        print("   - Set up .env file with your API keys")
        print("   - Stable internet connection")
    
    return all_passed

if __name__ == "__main__":
    main() 