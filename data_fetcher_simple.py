import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from loguru import logger
from config import Config
import asyncio
import aiohttp
from datetime import datetime, timedelta

class SimpleDataFetcher:
    """
    Simplified data fetcher that doesn't require TA-Lib or Alpha Vantage
    Uses only yfinance and basic pandas calculations
    """
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    def get_stock_data(self, symbol, period='1y', interval='1d'):
        """Fetch stock data using yfinance"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period, interval=interval)
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_multiple_stocks(self, symbols, period='1y'):
        """Fetch data for multiple stocks"""
        all_data = {}
        for symbol in symbols:
            data = self.get_stock_data(symbol, period)
            if data is not None:
                all_data[symbol] = data
        return all_data
    
    def get_company_info(self, symbol):
        """Get company information and fundamentals"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            return info
        except Exception as e:
            logger.error(f"Error fetching company info for {symbol}: {e}")
            return None
    
    async def fetch_news_sentiment(self, symbol, num_articles=10):
        """Fetch and analyze news sentiment for a stock"""
        try:
            # Search for news articles
            search_query = f"{symbol} stock news"
            news_data = await self._scrape_financial_news(search_query, num_articles)
            
            if not news_data:
                return {'sentiment_score': 0, 'articles': []}
            
            # Analyze sentiment
            sentiment_scores = []
            analyzed_articles = []
            
            for article in news_data:
                sentiment = self._analyze_text_sentiment(article['content'])
                sentiment_scores.append(sentiment['compound'])
                analyzed_articles.append({
                    'title': article['title'],
                    'url': article['url'],
                    'sentiment': sentiment,
                    'published': article.get('published', 'N/A')
                })
            
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
            
            return {
                'sentiment_score': avg_sentiment,
                'articles': analyzed_articles,
                'total_articles': len(analyzed_articles)
            }
            
        except Exception as e:
            logger.error(f"Error fetching news sentiment for {symbol}: {e}")
            return {'sentiment_score': 0, 'articles': []}
    
    async def _scrape_financial_news(self, query, num_articles):
        """Scrape financial news from various sources"""
        articles = []
        
        # Yahoo Finance News
        try:
            yahoo_url = f"https://finance.yahoo.com/quote/{query.split()[0]}/news"
            async with aiohttp.ClientSession() as session:
                async with session.get(yahoo_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        soup = BeautifulSoup(content, 'html.parser')
                        
                        # Extract news articles
                        news_items = soup.find_all('h3')[:num_articles]
                        
                        for item in news_items:
                            try:
                                title_elem = item.find('a')
                                if title_elem:
                                    title = title_elem.get_text(strip=True)
                                    url = title_elem.get('href', '')
                                    if url.startswith('/'):
                                        url = f"https://finance.yahoo.com{url}"
                                    
                                    articles.append({
                                        'title': title,
                                        'url': url,
                                        'content': title,  # Using title as content for sentiment
                                        'source': 'Yahoo Finance'
                                    })
                            except Exception as e:
                                continue
        except Exception as e:
            logger.warning(f"Could not fetch Yahoo Finance news: {e}")
        
        return articles[:num_articles]
    
    def _analyze_text_sentiment(self, text):
        """Analyze sentiment of text using VADER"""
        return self.sentiment_analyzer.polarity_scores(text)
    
    def get_market_indicators(self):
        """Get major market indicators"""
        indicators = {}
        market_symbols = ['^GSPC', '^DJI', '^IXIC', '^VIX']  # S&P 500, Dow, Nasdaq, VIX
        
        for symbol in market_symbols:
            data = self.get_stock_data(symbol, period='5d', interval='1d')
            if data is not None and not data.empty:
                latest = data.iloc[-1]
                prev = data.iloc[-2] if len(data) > 1 else latest
                
                indicators[symbol] = {
                    'price': latest['Close'],
                    'change': latest['Close'] - prev['Close'],
                    'change_pct': ((latest['Close'] - prev['Close']) / prev['Close']) * 100,
                    'volume': latest['Volume']
                }
        
        return indicators
    
    def calculate_simple_technical_indicators(self, data):
        """Calculate basic technical indicators using only pandas (no TA-Lib required)"""
        df = data.copy()
        
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Additional simple indicators
        # Stochastic Oscillator
        df['Lowest_Low'] = df['Low'].rolling(window=14).min()
        df['Highest_High'] = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * (df['Close'] - df['Lowest_Low']) / (df['Highest_High'] - df['Lowest_Low'])
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Williams %R
        df['Williams_R'] = -100 * (df['Highest_High'] - df['Close']) / (df['Highest_High'] - df['Lowest_Low'])
        
        # Price Rate of Change
        df['ROC'] = ((df['Close'] - df['Close'].shift(12)) / df['Close'].shift(12)) * 100
        
        # Average True Range (ATR)
        df['High_Low'] = df['High'] - df['Low']
        df['High_Close'] = abs(df['High'] - df['Close'].shift(1))
        df['Low_Close'] = abs(df['Low'] - df['Close'].shift(1))
        df['True_Range'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
        df['ATR'] = df['True_Range'].rolling(window=14).mean()
        
        # Clean up temporary columns
        df = df.drop(['Lowest_Low', 'Highest_High', 'High_Low', 'High_Close', 'Low_Close', 'True_Range'], axis=1, errors='ignore')
        
        return df
    
    # Backward compatibility
    def calculate_technical_indicators(self, data):
        """Backward compatibility method"""
        return self.calculate_simple_technical_indicators(data)

# Backward compatibility alias
DataFetcher = SimpleDataFetcher

# Example usage
if __name__ == "__main__":
    fetcher = SimpleDataFetcher()
    
    # Test data fetching
    data = fetcher.get_stock_data('AAPL')
    if data is not None:
        print("Stock data fetched successfully")
        print(data.tail())
        
        # Add technical indicators
        technical_data = fetcher.calculate_simple_technical_indicators(data)
        print("\nTechnical indicators added:")
        print(technical_data[['Close', 'SMA_20', 'RSI', 'MACD']].tail()) 