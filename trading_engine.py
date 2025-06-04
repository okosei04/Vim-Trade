import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
from config import Config
from data_fetcher import DataFetcher
from ai_models import StockPricePredictor, TradingSignalGenerator
import asyncio

class Portfolio:
    def __init__(self, initial_capital=Config.INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trade_history = []
        self.daily_returns = []
        
    def get_portfolio_value(self, current_prices):
        """Calculate total portfolio value"""
        portfolio_value = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                portfolio_value += position['shares'] * current_prices[symbol]
        
        return portfolio_value
    
    def get_position_value(self, symbol, current_price):
        """Get value of a specific position"""
        if symbol in self.positions:
            return self.positions[symbol]['shares'] * current_price
        return 0
    
    def add_trade(self, trade):
        """Add trade to history"""
        self.trade_history.append(trade)
        logger.info(f"Trade executed: {trade}")

class TradingEngine:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.price_predictor = StockPricePredictor()
        self.signal_generator = TradingSignalGenerator()
        self.portfolio = Portfolio()
        self.risk_manager = RiskManager(self.portfolio)
        
    async def analyze_stock(self, symbol):
        """Comprehensive stock analysis"""
        try:
            # Fetch data
            data = self.data_fetcher.get_stock_data(symbol, period='2y')
            if data is None or data.empty:
                return None
            
            # Add technical indicators
            data_with_indicators = self.data_fetcher.calculate_technical_indicators(data)
            
            # Get sentiment analysis
            sentiment_data = await self.data_fetcher.fetch_news_sentiment(symbol)
            
            # Generate AI predictions
            price_predictions = self.price_predictor.predict_price(data_with_indicators)
            trading_signal = self.signal_generator.generate_trading_signal(data_with_indicators)
            
            # Calculate metrics
            current_price = data_with_indicators['Close'].iloc[-1]
            
            analysis = {
                'symbol': symbol,
                'current_price': current_price,
                'predictions': price_predictions,
                'signal': trading_signal,
                'sentiment': sentiment_data,
                'technical_data': data_with_indicators.tail(5).to_dict('records'),
                'analysis_time': datetime.now()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def calculate_position_size(self, symbol, signal_strength, current_price):
        """Calculate optimal position size based on signal and risk"""
        portfolio_value = self.portfolio.get_portfolio_value({symbol: current_price})
        
        # Base position size on signal strength and available capital
        base_size = portfolio_value * Config.MAX_POSITION_SIZE
        
        # Adjust based on signal confidence
        if 'confidence' in signal_strength:
            adjusted_size = base_size * signal_strength['confidence']
        else:
            adjusted_size = base_size * 0.5  # Conservative default
        
        # Calculate number of shares
        max_shares = int(adjusted_size / current_price)
        
        # Risk management check
        if max_shares * current_price > self.portfolio.cash:
            max_shares = int(self.portfolio.cash / current_price)
        
        return max_shares
    
    def execute_trade(self, symbol, action, shares, current_price, reason=""):
        """Execute a trade"""
        try:
            if action.upper() == 'BUY':
                cost = shares * current_price
                if cost <= self.portfolio.cash:
                    self.portfolio.cash -= cost
                    
                    if symbol in self.portfolio.positions:
                        self.portfolio.positions[symbol]['shares'] += shares
                        # Update average price
                        total_value = self.portfolio.positions[symbol]['avg_price'] * self.portfolio.positions[symbol]['shares'] + cost
                        total_shares = self.portfolio.positions[symbol]['shares']
                        self.portfolio.positions[symbol]['avg_price'] = total_value / total_shares
                    else:
                        self.portfolio.positions[symbol] = {
                            'shares': shares,
                            'avg_price': current_price,
                            'entry_date': datetime.now()
                        }
                    
                    trade = {
                        'symbol': symbol,
                        'action': 'BUY',
                        'shares': shares,
                        'price': current_price,
                        'value': cost,
                        'timestamp': datetime.now(),
                        'reason': reason
                    }
                    
                    self.portfolio.add_trade(trade)
                    return True
                else:
                    logger.warning(f"Insufficient funds for {symbol} purchase")
                    return False
            
            elif action.upper() == 'SELL':
                if symbol in self.portfolio.positions and self.portfolio.positions[symbol]['shares'] >= shares:
                    revenue = shares * current_price
                    self.portfolio.cash += revenue
                    self.portfolio.positions[symbol]['shares'] -= shares
                    
                    if self.portfolio.positions[symbol]['shares'] == 0:
                        del self.portfolio.positions[symbol]
                    
                    trade = {
                        'symbol': symbol,
                        'action': 'SELL',
                        'shares': shares,
                        'price': current_price,
                        'value': revenue,
                        'timestamp': datetime.now(),
                        'reason': reason
                    }
                    
                    self.portfolio.add_trade(trade)
                    return True
                else:
                    logger.warning(f"Insufficient shares for {symbol} sale")
                    return False
                    
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            return False
    
    async def make_trading_decision(self, analysis):
        """Make trading decision based on analysis"""
        if not analysis or not analysis['signal']:
            return None
        
        symbol = analysis['symbol']
        current_price = analysis['current_price']
        signal = analysis['signal']
        
        # Check if we should trade
        recommendation = signal['recommendation']
        confidence = signal['confidence']
        
        # Risk management checks
        if not self.risk_manager.can_trade(symbol, current_price):
            logger.info(f"Risk management blocked trade for {symbol}")
            return None
        
        decision = None
        
        if recommendation in ['STRONG_BUY', 'BUY'] and confidence > 0.6:
            # Calculate position size
            shares = self.calculate_position_size(symbol, signal, current_price)
            
            if shares > 0:
                success = self.execute_trade(
                    symbol, 'BUY', shares, current_price,
                    f"AI Signal: {recommendation} (confidence: {confidence:.2f})"
                )
                
                if success:
                    decision = {
                        'action': 'BUY',
                        'symbol': symbol,
                        'shares': shares,
                        'price': current_price,
                        'confidence': confidence
                    }
        
        elif recommendation in ['STRONG_SELL', 'SELL'] and confidence > 0.6:
            # Sell existing position
            if symbol in self.portfolio.positions:
                shares = self.portfolio.positions[symbol]['shares']
                
                success = self.execute_trade(
                    symbol, 'SELL', shares, current_price,
                    f"AI Signal: {recommendation} (confidence: {confidence:.2f})"
                )
                
                if success:
                    decision = {
                        'action': 'SELL',
                        'symbol': symbol,
                        'shares': shares,
                        'price': current_price,
                        'confidence': confidence
                    }
        
        return decision
    
    async def run_trading_session(self, symbols=None):
        """Run a complete trading session"""
        if symbols is None:
            symbols = Config.DEFAULT_SYMBOLS
        
        logger.info(f"Starting trading session for {len(symbols)} symbols")
        
        decisions = []
        
        for symbol in symbols:
            try:
                # Analyze stock
                analysis = await self.analyze_stock(symbol)
                
                if analysis:
                    # Make trading decision
                    decision = await self.make_trading_decision(analysis)
                    
                    if decision:
                        decisions.append(decision)
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in trading session for {symbol}: {e}")
                continue
        
        logger.info(f"Trading session complete. Made {len(decisions)} trading decisions")
        return decisions

class RiskManager:
    def __init__(self, portfolio):
        self.portfolio = portfolio
        
    def can_trade(self, symbol, current_price):
        """Check if trade passes risk management rules"""
        
        # Check daily loss limit
        if self.check_daily_loss():
            return False
        
        # Check position concentration
        if self.check_position_concentration(symbol, current_price):
            return False
        
        # Check portfolio risk
        if self.check_portfolio_risk():
            return False
        
        return True
    
    def check_daily_loss(self):
        """Check if daily loss limit exceeded"""
        # This would need to be implemented with daily P&L tracking
        return False
    
    def check_position_concentration(self, symbol, current_price):
        """Check if position would be too concentrated"""
        current_positions = {s: p['shares'] * current_price 
                           for s, p in self.portfolio.positions.items()}
        
        total_value = self.portfolio.cash + sum(current_positions.values())
        
        # Check if single position would exceed limit
        if symbol in current_positions:
            position_ratio = current_positions[symbol] / total_value
            return position_ratio > Config.MAX_POSITION_SIZE
        
        return False
    
    def check_portfolio_risk(self):
        """Check overall portfolio risk"""
        # This would implement more sophisticated risk metrics
        return False

# Example usage
if __name__ == "__main__":
    async def main():
        engine = TradingEngine()
        
        # Run analysis on a single stock
        analysis = await engine.analyze_stock('AAPL')
        if analysis:
            print("Analysis completed successfully")
            print(f"Current price: ${analysis['current_price']:.2f}")
            if analysis['signal']:
                print(f"Trading signal: {analysis['signal']['recommendation']}")
    
    asyncio.run(main()) 