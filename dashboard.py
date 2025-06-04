import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import asyncio
from datetime import datetime, timedelta
import time

from trading_engine import TradingEngine
from data_fetcher import DataFetcher
from config import Config

# Page configuration
st.set_page_config(
    page_title="AI Stock Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'trading_engine' not in st.session_state:
    st.session_state.trading_engine = TradingEngine()
if 'auto_trading' not in st.session_state:
    st.session_state.auto_trading = False

def main():
    st.title("🤖 AI Stock Trading Bot Dashboard")
    
    # Sidebar
    with st.sidebar:
        st.header("Control Panel")
        
        # Trading controls
        st.subheader("Trading Controls")
        auto_trading = st.toggle("Auto Trading", value=st.session_state.auto_trading)
        st.session_state.auto_trading = auto_trading
        
        if st.button("🔄 Refresh Data", type="primary"):
            st.rerun()
        
        # Stock selection
        st.subheader("Stock Selection")
        selected_symbols = st.multiselect(
            "Select stocks to monitor:",
            Config.DEFAULT_SYMBOLS,
            default=Config.DEFAULT_SYMBOLS[:4]
        )
        
        # Risk settings
        st.subheader("Risk Settings")
        max_position = st.slider("Max Position Size (%)", 1, 20, int(Config.MAX_POSITION_SIZE * 100))
        stop_loss = st.slider("Stop Loss (%)", 1, 10, int(Config.STOP_LOSS_PERCENTAGE * 100))
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Portfolio", "📈 Market Analysis", "🤖 AI Predictions", "📰 News & Sentiment", "⚙️ Trading History"])
    
    with tab1:
        display_portfolio_tab()
    
    with tab2:
        display_market_analysis_tab(selected_symbols)
    
    with tab3:
        display_ai_predictions_tab(selected_symbols)
    
    with tab4:
        display_news_sentiment_tab(selected_symbols)
    
    with tab5:
        display_trading_history_tab()

def display_portfolio_tab():
    st.header("Portfolio Overview")
    
    engine = st.session_state.trading_engine
    portfolio = engine.portfolio
    
    # Portfolio metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Get current prices for portfolio calculation
    current_prices = {}
    for symbol in portfolio.positions.keys():
        data = engine.data_fetcher.get_stock_data(symbol, period='1d')
        if data is not None and not data.empty:
            current_prices[symbol] = data['Close'].iloc[-1]
    
    total_value = portfolio.get_portfolio_value(current_prices)
    total_positions_value = sum(portfolio.get_position_value(symbol, current_prices.get(symbol, 0)) 
                               for symbol in portfolio.positions.keys())
    
    with col1:
        st.metric("Total Portfolio Value", f"${total_value:,.2f}")
    
    with col2:
        st.metric("Cash Available", f"${portfolio.cash:,.2f}")
    
    with col3:
        st.metric("Positions Value", f"${total_positions_value:,.2f}")
    
    with col4:
        pnl = total_value - portfolio.initial_capital
        pnl_pct = (pnl / portfolio.initial_capital) * 100
        st.metric("Total P&L", f"${pnl:,.2f}", f"{pnl_pct:.2f}%")
    
    # Current positions
    if portfolio.positions:
        st.subheader("Current Positions")
        
        positions_data = []
        for symbol, position in portfolio.positions.items():
            current_price = current_prices.get(symbol, 0)
            position_value = position['shares'] * current_price
            pnl = (current_price - position['avg_price']) * position['shares']
            pnl_pct = ((current_price - position['avg_price']) / position['avg_price']) * 100
            
            positions_data.append({
                'Symbol': symbol,
                'Shares': position['shares'],
                'Avg Price': f"${position['avg_price']:.2f}",
                'Current Price': f"${current_price:.2f}",
                'Position Value': f"${position_value:.2f}",
                'P&L': f"${pnl:.2f}",
                'P&L %': f"{pnl_pct:.2f}%"
            })
        
        st.dataframe(pd.DataFrame(positions_data), use_container_width=True)
    else:
        st.info("No current positions")

def display_market_analysis_tab(symbols):
    st.header("Market Analysis")
    
    if not symbols:
        st.warning("Please select symbols in the sidebar")
        return
    
    # Market overview
    engine = st.session_state.trading_engine
    market_indicators = engine.data_fetcher.get_market_indicators()
    
    if market_indicators:
        st.subheader("Market Overview")
        
        indicator_cols = st.columns(len(market_indicators))
        for i, (symbol, data) in enumerate(market_indicators.items()):
            with indicator_cols[i]:
                clean_symbol = symbol.replace('^', '')
                st.metric(
                    clean_symbol,
                    f"${data['price']:.2f}",
                    f"{data['change_pct']:.2f}%"
                )
    
    # Individual stock analysis
    st.subheader("Stock Analysis")
    
    for symbol in symbols:
        with st.expander(f"📊 {symbol} Analysis"):
            data = engine.data_fetcher.get_stock_data(symbol, period='3mo')
            if data is not None and not data.empty:
                
                # Add technical indicators
                data_with_indicators = engine.data_fetcher.calculate_technical_indicators(data)
                
                # Create price chart
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=(f'{symbol} Price & Volume', 'RSI', 'MACD'),
                    vertical_spacing=0.05,
                    row_heights=[0.6, 0.2, 0.2]
                )
                
                # Price and moving averages
                fig.add_trace(
                    go.Candlestick(
                        x=data_with_indicators.index,
                        open=data_with_indicators['Open'],
                        high=data_with_indicators['High'],
                        low=data_with_indicators['Low'],
                        close=data_with_indicators['Close'],
                        name="Price"
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data_with_indicators.index,
                        y=data_with_indicators['SMA_20'],
                        name="SMA 20",
                        line=dict(color='orange')
                    ),
                    row=1, col=1
                )
                
                # RSI
                fig.add_trace(
                    go.Scatter(
                        x=data_with_indicators.index,
                        y=data_with_indicators['RSI'],
                        name="RSI",
                        line=dict(color='purple')
                    ),
                    row=2, col=1
                )
                
                # MACD
                fig.add_trace(
                    go.Scatter(
                        x=data_with_indicators.index,
                        y=data_with_indicators['MACD'],
                        name="MACD",
                        line=dict(color='blue')
                    ),
                    row=3, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data_with_indicators.index,
                        y=data_with_indicators['MACD_Signal'],
                        name="MACD Signal",
                        line=dict(color='red')
                    ),
                    row=3, col=1
                )
                
                fig.update_layout(height=600, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Current metrics
                latest = data_with_indicators.iloc[-1]
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Price", f"${latest['Close']:.2f}")
                with col2:
                    st.metric("RSI", f"{latest['RSI']:.1f}")
                with col3:
                    st.metric("Volume Ratio", f"{latest['Volume_Ratio']:.2f}")
                with col4:
                    change = ((latest['Close'] - data_with_indicators['Close'].iloc[-2]) / 
                             data_with_indicators['Close'].iloc[-2]) * 100
                    st.metric("Daily Change", f"{change:.2f}%")

def display_ai_predictions_tab(symbols):
    st.header("AI Predictions & Signals")
    
    if not symbols:
        st.warning("Please select symbols in the sidebar")
        return
    
    engine = st.session_state.trading_engine
    
    for symbol in symbols:
        with st.expander(f"🤖 {symbol} AI Analysis"):
            # Run async analysis
            analysis = asyncio.run(engine.analyze_stock(symbol))
            
            if analysis:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Price Predictions")
                    if analysis['predictions']:
                        pred_df = pd.DataFrame({
                            'Day': range(1, len(analysis['predictions']) + 1),
                            'Predicted Price': analysis['predictions']
                        })
                        
                        fig = px.line(pred_df, x='Day', y='Predicted Price', 
                                     title=f"{symbol} Price Predictions (Next {len(analysis['predictions'])} days)")
                        fig.add_hline(y=analysis['current_price'], line_dash="dash", 
                                     annotation_text="Current Price")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No predictions available")
                
                with col2:
                    st.subheader("Trading Signal")
                    if analysis['signal']:
                        signal = analysis['signal']
                        
                        # Signal strength indicator
                        confidence = signal['confidence']
                        color = 'green' if signal['recommendation'] in ['BUY', 'STRONG_BUY'] else 'red'
                        
                        st.markdown(f"""
                        **Recommendation:** <span style='color: {color}; font-size: 24px; font-weight: bold'>{signal['recommendation']}</span>
                        
                        **Confidence:** {confidence:.2f}
                        
                        **Signal Strength:** {signal['signal']:.3f}
                        """, unsafe_allow_html=True)
                        
                        # Confidence gauge
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = confidence,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Confidence"},
                            gauge = {'axis': {'range': [None, 1]},
                                   'bar': {'color': color},
                                   'steps': [
                                       {'range': [0, 0.5], 'color': "lightgray"},
                                       {'range': [0.5, 0.8], 'color': "yellow"},
                                       {'range': [0.8, 1], 'color': "green"}],
                                   'threshold': {'line': {'color': "red", 'width': 4},
                                               'thickness': 0.75, 'value': 0.9}}))
                        
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No signal available")
            else:
                st.error(f"Could not analyze {symbol}")

def display_news_sentiment_tab(symbols):
    st.header("News & Sentiment Analysis")
    
    if not symbols:
        st.warning("Please select symbols in the sidebar")
        return
    
    engine = st.session_state.trading_engine
    
    for symbol in symbols:
        with st.expander(f"📰 {symbol} News & Sentiment"):
            # Get sentiment data
            sentiment_data = asyncio.run(engine.data_fetcher.fetch_news_sentiment(symbol))
            
            if sentiment_data and sentiment_data['articles']:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Sentiment Score")
                    score = sentiment_data['sentiment_score']
                    
                    # Sentiment gauge
                    color = 'green' if score > 0 else 'red' if score < 0 else 'gray'
                    
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Sentiment"},
                        delta = {'reference': 0},
                        gauge = {'axis': {'range': [-1, 1]},
                               'bar': {'color': color},
                               'steps': [
                                   {'range': [-1, -0.5], 'color': "red"},
                                   {'range': [-0.5, 0.5], 'color': "yellow"},
                                   {'range': [0.5, 1], 'color': "green"}],
                               'threshold': {'line': {'color': "black", 'width': 4},
                                           'thickness': 0.75, 'value': score}}))
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.metric("Total Articles", sentiment_data['total_articles'])
                
                with col2:
                    st.subheader("Recent News")
                    for article in sentiment_data['articles'][:5]:
                        sentiment_score = article['sentiment']['compound']
                        sentiment_emoji = '😊' if sentiment_score > 0.1 else '😟' if sentiment_score < -0.1 else '😐'
                        
                        st.markdown(f"""
                        **{article['title']}** {sentiment_emoji}
                        
                        *Sentiment: {sentiment_score:.3f}*
                        
                        [Read more]({article['url']})
                        
                        ---
                        """)
            else:
                st.info(f"No recent news found for {symbol}")

def display_trading_history_tab():
    st.header("Trading History")
    
    engine = st.session_state.trading_engine
    trades = engine.portfolio.trade_history
    
    if trades:
        df = pd.DataFrame(trades)
        
        # Summary metrics
        total_trades = len(df)
        buy_trades = len(df[df['action'] == 'BUY'])
        sell_trades = len(df[df['action'] == 'SELL'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Trades", total_trades)
        with col2:
            st.metric("Buy Orders", buy_trades)
        with col3:
            st.metric("Sell Orders", sell_trades)
        
        # Trades table
        st.subheader("Trade Details")
        st.dataframe(df[['timestamp', 'symbol', 'action', 'shares', 'price', 'value', 'reason']], 
                    use_container_width=True)
        
        # Trade volume chart
        if len(df) > 1:
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            daily_volume = df.groupby('date')['value'].sum().reset_index()
            
            fig = px.bar(daily_volume, x='date', y='value', title="Daily Trading Volume")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trades executed yet")

if __name__ == "__main__":
    main() 