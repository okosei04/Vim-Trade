#!/usr/bin/env python3
"""
Quick Setup Script for AI Trading Bot
This script helps you set up your .env file interactively.
"""

import os
import sys

def print_banner():
    print("""
🚀 AI Trading Bot - Quick Setup
===============================
This script will help you create your .env file with API keys.
""")

def get_alpha_vantage_key():
    print("\n📊 ALPHA VANTAGE API KEY (Required)")
    print("=" * 40)
    print("This provides free stock market data.")
    print("\n📝 How to get it:")
    print("1. Go to: https://www.alphavantage.co/support/#api-key")
    print("2. Click 'GET YOUR FREE API KEY TODAY'")
    print("3. Fill out the simple form")
    print("4. Copy the key they give you")
    print("\n💡 The key looks like: ABCD1234EFGH5678")
    
    while True:
        key = input("\n🔑 Enter your Alpha Vantage API key (or 'skip' for now): ").strip()
        
        if key.lower() == 'skip':
            return 'your_alpha_vantage_key_here'
        elif len(key) > 10 and not ' ' in key:
            return key
        else:
            print("❌ That doesn't look like a valid API key. Try again or type 'skip'.")

def get_telegram_setup():
    print("\n📱 TELEGRAM SETUP (Optional)")
    print("=" * 30)
    print("This lets the bot send you trading notifications.")
    
    setup_telegram = input("\n❓ Do you want to set up Telegram notifications? (y/n): ").lower().strip()
    
    if setup_telegram != 'y':
        return 'your_telegram_bot_token_here', 'your_chat_id_here'
    
    print("\n🤖 Setting up Telegram Bot...")
    print("1. Open Telegram and search for @BotFather")
    print("2. Send: /start")
    print("3. Send: /newbot")
    print("4. Choose a name: 'My Trading Bot'")
    print("5. Choose username ending in 'bot': 'mytrading_bot'")
    print("6. Copy the token you receive")
    
    bot_token = input("\n🔑 Enter your Telegram Bot Token (or press Enter to skip): ").strip()
    
    if not bot_token:
        return 'your_telegram_bot_token_here', 'your_chat_id_here'
    
    print("\n💬 Getting Chat ID...")
    print("1. Send any message to your bot in Telegram")
    print("2. Open this URL in your browser:")
    print(f"   https://api.telegram.org/bot{bot_token}/getUpdates")
    print("3. Look for '\"chat\":{\"id\":12345678' in the response")
    print("4. Copy the number after 'id':")
    
    chat_id = input("\n🆔 Enter your Chat ID (or press Enter to skip): ").strip()
    
    if not chat_id:
        chat_id = 'your_chat_id_here'
    
    return bot_token, chat_id

def get_trading_settings():
    print("\n💰 TRADING SETTINGS")
    print("=" * 20)
    print("You can modify these later in the .env file")
    
    capital = input(f"\n💵 Starting capital (default: $10,000): ").strip()
    if not capital:
        capital = "10000"
    
    print("📋 Using these settings:")
    print(f"   • Starting capital: ${capital}")
    print("   • Max position size: 10% of portfolio")
    print("   • Stop loss: 5%")
    print("   • Take profit: 15%")
    
    return capital

def create_env_file(alpha_key, telegram_token, telegram_chat, capital):
    env_content = f"""# Copy this file to .env and fill in your actual values

# API Keys
ALPHA_VANTAGE_API_KEY={alpha_key}
TELEGRAM_BOT_TOKEN={telegram_token}
TELEGRAM_CHAT_ID={telegram_chat}

# Trading Parameters
INITIAL_CAPITAL={capital}
MAX_POSITION_SIZE=0.1
STOP_LOSS_PERCENTAGE=0.05
TAKE_PROFIT_PERCENTAGE=0.15

# AI Model Parameters
LSTM_LOOKBACK_DAYS=60
PREDICTION_DAYS=5
MODEL_RETRAIN_FREQUENCY=7

# Risk Management
MAX_DAILY_LOSS=0.02
MAX_PORTFOLIO_RISK=0.1

# Database
DATABASE_URL=sqlite:///trading_bot.db

# Logging
LOG_LEVEL=INFO
LOG_FILE=trading_bot.log
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("\n✅ Created .env file successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Error creating .env file: {e}")
        return False

def test_setup():
    print("\n🧪 TESTING YOUR SETUP")
    print("=" * 25)
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        # Test Alpha Vantage key
        alpha_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if alpha_key and alpha_key != 'your_alpha_vantage_key_here':
            print("✅ Alpha Vantage API key loaded")
            
            # Quick test
            print("📊 Testing Alpha Vantage connection...")
            try:
                import yfinance as yf
                data = yf.Ticker('AAPL').history(period='5d')
                if not data.empty:
                    print(f"✅ Successfully retrieved AAPL data")
                    print(f"   Latest price: ${data['Close'].iloc[-1]:.2f}")
                else:
                    print("⚠️ No data retrieved (might be market hours)")
            except Exception as e:
                print(f"⚠️ Data test failed: {e}")
        else:
            print("⚠️ Alpha Vantage API key not set")
        
        # Test Telegram
        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        if telegram_token and telegram_token != 'your_telegram_bot_token_here':
            print("✅ Telegram bot token loaded")
        else:
            print("ℹ️ Telegram not configured (optional)")
        
        print("\n🎉 Setup complete!")
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error testing setup: {e}")

def main():
    print_banner()
    
    # Check if .env already exists
    if os.path.exists('.env'):
        overwrite = input("⚠️ .env file already exists. Overwrite? (y/n): ").lower().strip()
        if overwrite != 'y':
            print("👋 Setup cancelled. Your existing .env file is unchanged.")
            return
    
    # Get API keys
    alpha_key = get_alpha_vantage_key()
    telegram_token, telegram_chat = get_telegram_setup()
    capital = get_trading_settings()
    
    # Create .env file
    if create_env_file(alpha_key, telegram_token, telegram_chat, capital):
        print("\n📋 Your .env file has been created with these settings:")
        print(f"   🔑 Alpha Vantage: {'✅ Set' if alpha_key != 'your_alpha_vantage_key_here' else '⚠️ Placeholder'}")
        print(f"   📱 Telegram: {'✅ Set' if telegram_token != 'your_telegram_bot_token_here' else '⚠️ Skipped'}")
        print(f"   💰 Capital: ${capital}")
        
        # Test the setup
        test_setup()
        
        print("\n🚀 NEXT STEPS:")
        print("1. Run the full test: python setup_and_test.py")
        print("2. Start the dashboard: streamlit run dashboard.py")
        print("3. Or run the trading bot: python main.py")
        
        if alpha_key == 'your_alpha_vantage_key_here':
            print("\n⚠️ IMPORTANT: You still need to get your Alpha Vantage API key!")
            print("   Visit: https://www.alphavantage.co/support/#api-key")
            print("   Then edit your .env file to add the real key.")
    
    print("\n" + "=" * 50)
    print("Setup complete! Happy trading! 📈🤖")

if __name__ == "__main__":
    main() 