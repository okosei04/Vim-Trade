# 🔑 Complete API Keys Setup Guide

## Step-by-Step Instructions

### 1. Alpha Vantage API Key (Required) 📊

**Why you need it:** Free real-time and historical stock market data

**Steps:**
1. Visit: https://www.alphavantage.co/support/#api-key
2. Click the blue "GET YOUR FREE API KEY TODAY" button
3. Fill out the registration form:
   ```
   First Name: [Your name]
   Last Name: [Your last name]
   Email: [Your email]
   Organization: Personal (or your company name)
   ```
4. Click "GET FREE API KEY"
5. Your key will appear instantly - copy it!

**Example key format:** `ABCD1234EFGH5678`

**Free tier includes:**
- ✅ 25 requests per day
- ✅ 5 requests per minute
- ✅ Real-time quotes
- ✅ Historical data
- ✅ Technical indicators

---

### 2. Telegram Bot Setup (Optional) 📱

**Why you need it:** Get instant notifications when your bot makes trades

#### Step 2A: Create Bot Token

1. Open Telegram and search for `@BotFather`
2. Start chat and send: `/start`
3. Send: `/newbot`
4. Bot will ask for a name: `My Stock Trading Bot`
5. Bot will ask for username: `mystockbot_[yourname]_bot` (must end with 'bot')
6. Copy the token that looks like: `123456789:ABCdef-GHIjklMNOpqrsTUVwxyz`

#### Step 2B: Get Your Chat ID

1. Send any message to your new bot
2. Open this URL in browser (replace YOUR_TOKEN):
   ```
   https://api.telegram.org/botYOUR_TOKEN/getUpdates
   ```
3. Look for `"chat":{"id":12345678` in the response
4. Copy the number after `"id":` (that's your Chat ID)

**Example:**
- Bot Token: `987654321:ABCdef-GHIjklMNOpqrsTUVwxyz`
- Chat ID: `123456789`

---

## 🛠️ Setting Up Your .env File

### Step 1: Create .env file
```bash
# Copy the example file
cp environment_example.txt .env
```

### Step 2: Edit with your keys
Open `.env` in any text editor and replace the placeholder values:

```env
# Replace with your actual Alpha Vantage key
ALPHA_VANTAGE_API_KEY=ABCD1234EFGH5678

# Optional: Replace with your Telegram bot details
TELEGRAM_BOT_TOKEN=987654321:ABCdef-GHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789

# Trading settings (you can modify these)
INITIAL_CAPITAL=10000
MAX_POSITION_SIZE=0.1
STOP_LOSS_PERCENTAGE=0.05
TAKE_PROFIT_PERCENTAGE=0.15

# AI Model settings
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
```

### Step 3: Test your setup
```bash
python setup_and_test.py
```

---

## 🔧 Troubleshooting

### Alpha Vantage Issues

**Problem:** "API key not working"
- ✅ Check you copied the entire key (no spaces)
- ✅ Make sure you're not exceeding 25 requests/day
- ✅ Wait 12 seconds between requests

**Problem:** "No data returned"
- ✅ Check your internet connection
- ✅ Try a different stock symbol (use 'AAPL' for testing)
- ✅ Check if markets are open

### Telegram Issues

**Problem:** "Bot not responding"
- ✅ Make sure you messaged the bot first
- ✅ Check the bot token is correct
- ✅ Verify the chat ID is a number (not text)

**Problem:** "Can't get Chat ID"
- ✅ Send ANY message to your bot first
- ✅ Check the URL has your correct bot token
- ✅ Look for the first "id" field in the JSON response

### Environment File Issues

**Problem:** "Config not loading"
- ✅ File must be named exactly `.env` (with the dot)
- ✅ No spaces around the = sign
- ✅ No quotes around values unless specified
- ✅ Save the file in the same folder as your Python scripts

---

## 💡 Pro Tips

### 1. Keep Your Keys Safe
- ✅ Never share your API keys publicly
- ✅ Don't commit .env file to version control
- ✅ The .env file should stay on your local machine only

### 2. Test with Paper Trading First
- ✅ Start with simulation mode
- ✅ Use small amounts for testing
- ✅ Monitor the bot's decisions manually

### 3. Monitor Usage
- ✅ Alpha Vantage free tier: 25 requests/day
- ✅ Bot makes ~1-3 requests per stock analysis
- ✅ Upgrade to premium if you need more data

### 4. Alternative Data Sources
If you want more requests, consider:
- **Alpha Vantage Premium**: $49.99/month for 1,200 requests/min
- **Yahoo Finance**: Free but less reliable
- **IEX Cloud**: Free tier with 500,000 requests/month

---

## 🎯 Quick Verification

Run this to test your setup:

```bash
# Test everything at once
python setup_and_test.py

# Or test individual components
python -c "from data_fetcher import DataFetcher; f=DataFetcher(); print('✅ API working!' if f.get_stock_data('AAPL') is not None else '❌ API failed')"
```

### Expected Output:
```
🔍 Checking dependencies...
  ✅ pandas
  ✅ numpy
  ✅ yfinance
  [... more packages ...]

🔍 Checking environment configuration...
  ✅ .env file: Found
  ✅ ALPHA_VANTAGE_API_KEY

🔍 Testing data fetching...
  📊 Testing Yahoo Finance...
    ✅ Retrieved 30 days of AAPL data
    📈 Latest close: $150.25

✅ All tests passed! Your system is ready to use.
```

---

Need help? Check the troubleshooting section above or run the test script to identify specific issues! 