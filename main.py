#!/usr/bin/env python3
"""
AI Stock Trading Bot - Main Entry Point
This is the main orchestrator for the AI-powered stock trading system.
"""

import asyncio
import schedule
import time
from datetime import datetime, timedelta
from loguru import logger
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_engine import TradingEngine
from data_fetcher import DataFetcher
from ai_models import StockPricePredictor, TradingSignalGenerator
from config import Config

class TradingBot:
    def __init__(self):
        logger.info("Initializing AI Trading Bot...")
        
        self.trading_engine = TradingEngine()
        self.data_fetcher = DataFetcher()
        self.price_predictor = StockPricePredictor()
        self.signal_generator = TradingSignalGenerator()
        
        self.is_market_open = False
        self.last_training_date = None
        self.training_data = {}
        
        logger.info("AI Trading Bot initialized successfully!")
    
    async def check_market_status(self):
        """Check if the market is currently open"""
        now = datetime.now()
        
        # Simple market hours check (9:30 AM - 4:00 PM EST, Monday-Friday)
        if now.weekday() >= 5:  # Weekend
            self.is_market_open = False
            return False
        
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        self.is_market_open = market_open <= now <= market_close
        return self.is_market_open
    
    async def train_models(self, symbols=None):
        """Train or retrain AI models with latest data"""
        if symbols is None:
            symbols = Config.DEFAULT_SYMBOLS
        
        logger.info("Starting model training...")
        
        for symbol in symbols:
            try:
                logger.info(f"Training models for {symbol}")
                
                # Fetch training data (2 years for better model performance)
                data = self.data_fetcher.get_stock_data(symbol, period='2y')
                if data is None or len(data) < Config.LSTM_LOOKBACK_DAYS + 50:
                    logger.warning(f"Insufficient data for {symbol}, skipping...")
                    continue
                
                # Add technical indicators
                data_with_indicators = self.data_fetcher.calculate_technical_indicators(data)
                
                # Train price prediction model
                logger.info(f"Training LSTM model for {symbol}")
                if self.price_predictor.train_lstm_model(data_with_indicators):
                    logger.success(f"LSTM model trained successfully for {symbol}")
                else:
                    logger.error(f"Failed to train LSTM model for {symbol}")
                
                # Train signal generation models
                logger.info(f"Training signal models for {symbol}")
                if self.signal_generator.train_signal_models(data_with_indicators):
                    logger.success(f"Signal models trained successfully for {symbol}")
                else:
                    logger.error(f"Failed to train signal models for {symbol}")
                
                # Store training data for reference
                self.training_data[symbol] = {
                    'last_training': datetime.now(),
                    'data_points': len(data_with_indicators),
                    'date_range': f"{data_with_indicators.index[0]} to {data_with_indicators.index[-1]}"
                }
                
            except Exception as e:
                logger.error(f"Error training models for {symbol}: {e}")
                continue
        
        self.last_training_date = datetime.now()
        logger.info("Model training completed!")
    
    async def run_market_analysis(self):
        """Run comprehensive market analysis"""
        logger.info("Running market analysis...")
        
        try:
            # Get market indicators
            market_indicators = self.data_fetcher.get_market_indicators()
            
            if market_indicators:
                logger.info("Market Overview:")
                for symbol, data in market_indicators.items():
                    logger.info(f"{symbol}: ${data['price']:.2f} ({data['change_pct']:+.2f}%)")
            
            # Analyze each stock in our watchlist
            analysis_results = {}
            
            for symbol in Config.DEFAULT_SYMBOLS:
                try:
                    analysis = await self.trading_engine.analyze_stock(symbol)
                    if analysis:
                        analysis_results[symbol] = analysis
                        
                        # Log key insights
                        signal = analysis.get('signal', {})
                        sentiment = analysis.get('sentiment', {})
                        
                        logger.info(f"{symbol} Analysis:")
                        logger.info(f"  Current Price: ${analysis['current_price']:.2f}")
                        
                        if signal:
                            logger.info(f"  Signal: {signal['recommendation']} (confidence: {signal['confidence']:.2f})")
                        
                        if sentiment:
                            logger.info(f"  Sentiment: {sentiment['sentiment_score']:.3f}")
                            
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    continue
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            return {}
    
    async def run_trading_session(self):
        """Execute a complete trading session"""
        logger.info("=== Starting Trading Session ===")
        
        # Check if market is open
        if not await self.check_market_status():
            logger.info("Market is closed, skipping trading session")
            return
        
        try:
            # Run market analysis
            analysis_results = await self.run_market_analysis()
            
            if not analysis_results:
                logger.warning("No analysis results available, skipping trading")
                return
            
            # Execute trading decisions
            decisions = await self.trading_engine.run_trading_session(Config.DEFAULT_SYMBOLS)
            
            if decisions:
                logger.success(f"Executed {len(decisions)} trades:")
                for decision in decisions:
                    logger.info(f"  {decision['action']} {decision['shares']} shares of {decision['symbol']} at ${decision['price']:.2f}")
            else:
                logger.info("No trades executed this session")
            
            # Portfolio summary
            portfolio = self.trading_engine.portfolio
            current_prices = {}
            for symbol in portfolio.positions.keys():
                data = self.data_fetcher.get_stock_data(symbol, period='1d')
                if data is not None and not data.empty:
                    current_prices[symbol] = data['Close'].iloc[-1]
            
            total_value = portfolio.get_portfolio_value(current_prices)
            
            logger.info(f"Portfolio Summary:")
            logger.info(f"  Total Value: ${total_value:,.2f}")
            logger.info(f"  Cash: ${portfolio.cash:,.2f}")
            logger.info(f"  Positions: {len(portfolio.positions)}")
            logger.info(f"  Total Trades: {len(portfolio.trade_history)}")
            
        except Exception as e:
            logger.error(f"Error in trading session: {e}")
        
        logger.info("=== Trading Session Complete ===")
    
    async def health_check(self):
        """Perform system health check"""
        logger.info("Performing health check...")
        
        checks = {
            'data_fetcher': False,
            'price_predictor': False,
            'signal_generator': False,
            'market_data': False
        }
        
        try:
            # Test data fetching
            test_data = self.data_fetcher.get_stock_data('AAPL', period='5d')
            checks['data_fetcher'] = test_data is not None and not test_data.empty
            
            # Test AI models
            checks['price_predictor'] = self.price_predictor.load_model()
            checks['signal_generator'] = self.signal_generator.load_signal_models()
            
            # Test market data
            market_data = self.data_fetcher.get_market_indicators()
            checks['market_data'] = bool(market_data)
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
        
        # Report results
        for component, status in checks.items():
            status_text = "✅ OK" if status else "❌ FAILED"
            logger.info(f"  {component}: {status_text}")
        
        all_healthy = all(checks.values())
        logger.info(f"Overall System Health: {'✅ HEALTHY' if all_healthy else '⚠️ ISSUES DETECTED'}")
        
        return all_healthy
    
    def schedule_tasks(self):
        """Schedule periodic tasks"""
        logger.info("Setting up scheduled tasks...")
        
        # Market analysis every 30 minutes during market hours
        schedule.every(30).minutes.do(lambda: asyncio.run(self.run_market_analysis()))
        
        # Trading session every hour during market hours
        schedule.every().hour.do(lambda: asyncio.run(self.run_trading_session()))
        
        # Model retraining every week
        schedule.every().sunday.at("02:00").do(lambda: asyncio.run(self.train_models()))
        
        # Health check every 6 hours
        schedule.every(6).hours.do(lambda: asyncio.run(self.health_check()))
        
        # Daily portfolio summary
        schedule.every().day.at("16:30").do(self.daily_summary)
        
        logger.info("Scheduled tasks configured")
    
    def daily_summary(self):
        """Generate daily portfolio summary"""
        logger.info("=== Daily Portfolio Summary ===")
        
        portfolio = self.trading_engine.portfolio
        
        # Get current prices
        current_prices = {}
        for symbol in portfolio.positions.keys():
            data = self.data_fetcher.get_stock_data(symbol, period='1d')
            if data is not None and not data.empty:
                current_prices[symbol] = data['Close'].iloc[-1]
        
        total_value = portfolio.get_portfolio_value(current_prices)
        daily_pnl = total_value - portfolio.initial_capital
        daily_pnl_pct = (daily_pnl / portfolio.initial_capital) * 100
        
        logger.info(f"Portfolio Value: ${total_value:,.2f}")
        logger.info(f"P&L: ${daily_pnl:,.2f} ({daily_pnl_pct:+.2f}%)")
        logger.info(f"Cash: ${portfolio.cash:,.2f}")
        logger.info(f"Active Positions: {len(portfolio.positions)}")
        
        # Today's trades
        today = datetime.now().date()
        today_trades = [t for t in portfolio.trade_history 
                       if t['timestamp'].date() == today]
        
        if today_trades:
            logger.info(f"Today's Trades: {len(today_trades)}")
            for trade in today_trades:
                logger.info(f"  {trade['action']} {trade['shares']} {trade['symbol']} @ ${trade['price']:.2f}")
        else:
            logger.info("No trades today")
        
        logger.info("=== End Daily Summary ===")
    
    async def run(self):
        """Main bot execution loop"""
        logger.info("🚀 Starting AI Trading Bot...")
        
        try:
            # Initial health check
            if not await self.health_check():
                logger.warning("System health check failed, but continuing...")
            
            # Check if models need training
            if self.last_training_date is None:
                logger.info("No previous training detected, training models...")
                await self.train_models()
            
            # Set up scheduling
            self.schedule_tasks()
            
            # Initial trading session
            await self.run_trading_session()
            
            # Main execution loop
            logger.info("Bot is now running. Press Ctrl+C to stop.")
            
            while True:
                schedule.run_pending()
                await asyncio.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
        finally:
            logger.info("AI Trading Bot shutting down...")

def main():
    """Entry point for the trading bot"""
    
    # Set up logging
    logger.remove()
    logger.add(
        Config.LOG_FILE,
        level=Config.LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        rotation="1 day",
        retention="30 days"
    )
    logger.add(sys.stderr, level=Config.LOG_LEVEL)
    
    # Create bot and run
    bot = TradingBot()
    asyncio.run(bot.run())

if __name__ == "__main__":
    main() 