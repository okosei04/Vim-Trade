#!/usr/bin/env python3
"""
Verify Setup Script - Check if your .env file is working properly
"""

import os
import sys
from datetime import datetime

def print_header():
    print("""
🔍 AI Trading Bot - Setup Verification
=====================================
This script will verify your .env file and API keys are working.
""")

def check_env_file():
    """Check if .env file exists and load it"""
    print("📁 CHECKING .ENV FILE")
    print("=" * 25)
    
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("   Create it by copying environment_example.txt to .env")
        return False
    
    print("✅ .env file found")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Environment variables loaded")
        return True
    except ImportError:
        print("❌ python-dotenv not installed")
        print("   Run: pip install python-dotenv")
        return False
    except Exception as e:
        print(f"❌ Error loading .env: {e}")
        return False

def verify_api_keys():
    """Verify all API keys are properly set"""
    print("\n🔑 CHECKING API KEYS")
    print("=" * 20)
    
    # Check Alpha Vantage (Required)
    alpha_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    print(f"📊 Alpha Vantage API Key:")
    
    if not alpha_key:
        print("   ❌ Not set in .env file")
        return False
    elif alpha_key == 'your_alpha_vantage_key_here':
        print("   ⚠️ Still using placeholder value")
        print("   🔗 Get your key at: https://www.alphavantage.co/support/#api-key")
        return False
    elif len(alpha_key) < 10:
        print("   ⚠️ Key looks too short - check you copied it completely")
        return False
    else:
        print(f"   ✅ Set (Key: {alpha_key[:8]}...{alpha_key[-4:]})")
    
    # Check Telegram (Optional)
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    telegram_chat = os.getenv('TELEGRAM_CHAT_ID')
    
    print(f"\n📱 Telegram Bot Token:")
    if not telegram_token or telegram_token == 'your_telegram_bot_token_here':
        print("   ℹ️ Not configured (optional)")
    else:
        print(f"   ✅ Set (Token: {telegram_token[:10]}...)")
    
    print(f"📱 Telegram Chat ID:")
    if not telegram_chat or telegram_chat == 'your_chat_id_here':
        print("   ℹ️ Not configured (optional)")
    else:
        print(f"   ✅ Set (Chat ID: {telegram_chat})")
    
    return True

def test_data_connection():
    """Test if we can actually fetch data with the API key"""
    print("\n📊 TESTING DATA CONNECTION")
    print("=" * 30)
    
    try:
        print("🔄 Testing Yahoo Finance (free backup)...")
        import yfinance as yf
        
        # Test basic data fetch
        ticker = yf.Ticker('AAPL')
        data = ticker.history(period='5d')
        
        if not data.empty:
            latest_price = data['Close'].iloc[-1]
            print(f"   ✅ Yahoo Finance working")
            print(f"   📈 AAPL latest price: ${latest_price:.2f}")
        else:
            print("   ⚠️ No data retrieved (might be market hours/weekend)")
        
        # Test Alpha Vantage if key is set
        alpha_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if alpha_key and alpha_key != 'your_alpha_vantage_key_here':
            print("\n🔄 Testing Alpha Vantage API...")
            try:
                from alpha_vantage.timeseries import TimeSeries
                ts = TimeSeries(key=alpha_key, output_format='pandas')
                
                # This will test the API key without using too many requests
                print("   🔄 Making test API call...")
                data, _ = ts.get_daily('AAPL', outputsize='compact')
                
                if data is not None and not data.empty:
                    print("   ✅ Alpha Vantage API working!")
                    print(f"   📊 Retrieved {len(data)} days of data")
                else:
                    print("   ⚠️ Alpha Vantage returned no data")
                    
            except Exception as e:
                if "rate limit" in str(e).lower():
                    print("   ⚠️ Rate limit hit (normal with free tier)")
                elif "invalid" in str(e).lower():
                    print(f"   ❌ Invalid API key: {e}")
                    return False
                else:
                    print(f"   ⚠️ Alpha Vantage error: {e}")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Missing package: {e}")
        print("   Run: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"   ❌ Connection test failed: {e}")
        return False

def test_telegram_connection():
    """Test Telegram bot if configured"""
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    
    if not telegram_token or telegram_token == 'your_telegram_bot_token_here':
        print("\n📱 TELEGRAM TEST")
        print("=" * 15)
        print("   ℹ️ Telegram not configured - skipping test")
        return True
    
    print("\n📱 TESTING TELEGRAM BOT")
    print("=" * 25)
    
    try:
        import requests
        
        # Test bot token
        print("🔄 Testing bot token...")
        url = f"https://api.telegram.org/bot{telegram_token}/getMe"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            bot_info = response.json()
            if bot_info.get('ok'):
                bot_name = bot_info['result'].get('username', 'Unknown')
                print(f"   ✅ Bot token valid (Bot: @{bot_name})")
            else:
                print("   ❌ Bot token invalid")
                return False
        else:
            print(f"   ❌ HTTP error: {response.status_code}")
            return False
        
        # Test chat ID if provided
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        if chat_id and chat_id != 'your_chat_id_here':
            print("🔄 Testing chat ID...")
            url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': f'🤖 AI Trading Bot test message - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            }
            
            response = requests.post(url, data=data, timeout=10)
            if response.status_code == 200:
                result = response.json()
                if result.get('ok'):
                    print("   ✅ Test message sent successfully!")
                    print("   📱 Check your Telegram for the test message")
                else:
                    print(f"   ❌ Failed to send message: {result.get('description', 'Unknown error')}")
                    return False
            else:
                print(f"   ❌ HTTP error sending message: {response.status_code}")
                return False
        
        return True
        
    except ImportError:
        print("   ❌ 'requests' package not installed")
        return False
    except Exception as e:
        print(f"   ❌ Telegram test failed: {e}")
        return False

def show_config_summary():
    """Show current configuration"""
    print("\n⚙️ CURRENT CONFIGURATION")
    print("=" * 28)
    
    config_items = [
        ('INITIAL_CAPITAL', '💰'),
        ('MAX_POSITION_SIZE', '📊'),
        ('STOP_LOSS_PERCENTAGE', '🛑'),
        ('TAKE_PROFIT_PERCENTAGE', '🎯'),
        ('LSTM_LOOKBACK_DAYS', '🧠'),
        ('PREDICTION_DAYS', '🔮'),
    ]
    
    for key, emoji in config_items:
        value = os.getenv(key, 'Not set')
        print(f"   {emoji} {key}: {value}")

def main():
    print_header()
    
    results = {}
    
    # Run all checks
    results['env_file'] = check_env_file()
    
    if results['env_file']:
        results['api_keys'] = verify_api_keys()
        results['data_connection'] = test_data_connection()
        results['telegram'] = test_telegram_connection()
        
        show_config_summary()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 50)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        test_display = test_name.replace('_', ' ').title()
        print(f"   {test_display}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 ALL CHECKS PASSED!")
        print("\n🚀 Your system is ready! Next steps:")
        print("   1. Run the dashboard: streamlit run dashboard.py")
        print("   2. Or start the bot: python main.py")
        print("   3. For full testing: python setup_and_test.py")
    else:
        print("\n⚠️ SOME ISSUES FOUND")
        print("Please fix the failed checks above before proceeding.")
        
        if not results.get('api_keys', True):
            print("\n🔧 To fix API key issues:")
            print("   1. Edit your .env file")
            print("   2. Make sure ALPHA_VANTAGE_API_KEY has your real key")
            print("   3. Get key at: https://www.alphavantage.co/support/#api-key")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main() 