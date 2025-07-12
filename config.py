import os
from dotenv import load_dotenv
import yaml

load_dotenv()

class Config:
    # API Keys
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'your_alpha_vantage_key')
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'your_telegram_bot_token')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'your_chat_id')
    
    # Trading Parameters
    INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', 10000))
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', 0.1))  # 10% of portfolio
    STOP_LOSS_PERCENTAGE = float(os.getenv('STOP_LOSS_PERCENTAGE', 0.05))  # 5%
    TAKE_PROFIT_PERCENTAGE = float(os.getenv('TAKE_PROFIT_PERCENTAGE', 0.15))  # 15%
    
    # AI Model Parameters
    LSTM_LOOKBACK_DAYS = int(os.getenv('LSTM_LOOKBACK_DAYS', 60))
    PREDICTION_DAYS = int(os.getenv('PREDICTION_DAYS', 5))
    MODEL_RETRAIN_FREQUENCY = int(os.getenv('MODEL_RETRAIN_FREQUENCY', 7))  # days
    
    # Risk Management
    MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', 0.02))  # 2%
    MAX_PORTFOLIO_RISK = float(os.getenv('MAX_PORTFOLIO_RISK', 0.1))  # 10%
    
    # Data Sources
    DEFAULT_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'SPY']
    
    
    NEWS_SOURCES = [
        'https://finance.yahoo.com/news/',
        'https://www.marketwatch.com/',
        'https://www.cnbc.com/markets/'
    ]
    
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///trading_bot.db')
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'trading_bot.log') 

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config() 