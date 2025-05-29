import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import json
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Round-Number Order Clustering Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StockClusteringAnalyzer:
    """
    Core class implementing the Round-Number Order Clustering strategy
    based on "Can Daily Closing Prices Predict Future Movements? 
    The Role of Limit Order Clustering" by Xiao Zhang.
    """

    def __init__(self):
        self.round_threshold = 0.1  # Distance from round number to consider clustering

    def get_round_number_distance(self, price: float) -> float:
        """
        Calculate distance from nearest round number (integer).
        Returns: Distance as decimal (e.g., 100.1 returns 0.1, 99.9 returns -0.1)
        """
        nearest_round = round(price)
        return price - nearest_round

    def classify_closing_position(self, price: float) -> str:
        """
        Classify closing price relative to round numbers.
        X.1 category: Just above round number (bullish signal)
        X.9 category: Just below round number (bearish signal)
        """
        distance = self.get_round_number_distance(price)

        if 0.05 <= distance <= 0.15:  # X.1 range (e.g., 100.05 to 100.15)
            return "X.1_BULLISH"
        elif -0.15 <= distance <= -0.05:  # X.9 range (e.g., 99.85 to 99.95)
            return "X.9_BEARISH"
        else:
            return "NEUTRAL"

    def generate_signal(self, closing_price: float) -> Dict:
        """
        Generate trading signal based on closing price clustering.
        """
        classification = self.classify_closing_position(closing_price)
        distance = self.get_round_number_distance(closing_price)

        signal_data = {
            "price": closing_price,
            "distance_from_round": distance,
            "classification": classification,
            "signal": "NEUTRAL",
            "confidence": 0.0,
            "expected_direction": "HOLD"
        }

        if classification == "X.1_BULLISH":
            signal_data.update({
                "signal": "BUY",
                "confidence": min(0.8, 0.5 + abs(distance) * 2),  # Higher confidence closer to X.1
                "expected_direction": "UP",
                "reasoning": "Price closed just above round number - potential support level"
            })
        elif classification == "X.9_BEARISH":
            signal_data.update({
                "signal": "SELL",
                "confidence": min(0.8, 0.5 + abs(distance) * 2),
                "expected_direction": "DOWN",
                "reasoning": "Price closed just below round number - potential resistance level"
            })
        else:
            signal_data["reasoning"] = "Price not in clustering zone"

        return signal_data

class DataManager:
    """Handle stock data fetching and caching"""

    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes cache

    def fetch_stock_data(self, symbol: str, period: str = "1mo") -> pd.DataFrame:
        """Fetch stock data with caching"""
        cache_key = f"{symbol}_{period}"
        current_time = time.time()

        # Check cache
        if (cache_key in self.cache and 
            current_time - self.cache[cache_key]['timestamp'] < self.cache_duration):
            return self.cache[cache_key]['data']

        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)

            if not data.empty:
                # Cache the data
                self.cache[cache_key] = {
                    'data': data,
                    'timestamp': current_time
                }
                return data
            else:
                st.error(f"No data found for symbol {symbol}")
                return pd.DataFrame()

        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def get_real_time_price(self, symbol: str) -> Dict:
        """Get current price and basic info"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1d", interval="1m")

            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                return {
                    'symbol': symbol,
                    'current_price': current_price,
                    'previous_close': info.get('previousClose', current_price),
                    'change': current_price - info.get('previousClose', current_price),
                    'change_percent': ((current_price - info.get('previousClose', current_price)) / 
                                     info.get('previousClose', current_price)) * 100,
                    'volume': hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0
                }
        except Exception as e:
            st.error(f"Error getting real-time data for {symbol}: {str(e)}")

        return {'symbol': symbol, 'current_price': 0, 'change': 0, 'change_percent': 0}

def create_price_chart(data: pd.DataFrame, symbol: str, signals: List[Dict]) -> go.Figure:
    """Create interactive price chart with clustering signals"""

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=[f'{symbol} Price & Clustering Signals', 'Volume'],
        row_heights=[0.7, 0.3]
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=symbol
        ),
        row=1, col=1
    )

    # Add signal markers
    buy_signals = [s for s in signals if s['signal'] == 'BUY']
    sell_signals = [s for s in signals if s['signal'] == 'SELL']

    if buy_signals:
        buy_dates = [s['date'] for s in buy_signals]
        buy_prices = [s['price'] for s in buy_signals]
        fig.add_trace(
            go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode='markers',
                marker=dict(symbol='triangle-up', size=12, color='green'),
                name='Buy Signal (X.1)',
                hovertemplate='<b>BUY Signal</b><br>Price: $%{y:.2f}<br>Date: %{x}<extra></extra>'
            ),
            row=1, col=1
        )

    if sell_signals:
        sell_dates = [s['date'] for s in sell_signals]
        sell_prices = [s['price'] for s in sell_signals]
        fig.add_trace(
            go.Scatter(
                x=sell_dates,
                y=sell_prices,
                mode='markers',
                marker=dict(symbol='triangle-down', size=12, color='red'),
                name='Sell Signal (X.9)',
                hovertemplate='<b>SELL Signal</b><br>Price: $%{y:.2f}<br>Date: %{x}<extra></extra>'
            ),
            row=1, col=1
        )

    # Volume chart
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color='rgba(158,202,225,0.8)'
        ),
        row=2, col=1
    )

    fig.update_layout(
        title=f'{symbol} - Round Number Clustering Analysis',
        xaxis_rangeslider_visible=False,
        height=600
    )

    return fig

def create_performance_chart(performance_data: pd.DataFrame) -> go.Figure:
    """Create performance comparison chart"""

    fig = go.Figure()

    # Buy signals performance
    if 'buy_returns' in performance_data.columns:
        fig.add_trace(
            go.Scatter(
                x=performance_data.index,
                y=performance_data['buy_returns'].cumsum(),
                mode='lines',
                name='X.1 Signals (Buy)',
                line=dict(color='green', width=2)
            )
        )

    # Sell signals performance
    if 'sell_returns' in performance_data.columns:
        fig.add_trace(
            go.Scatter(
                x=performance_data.index,
                y=performance_data['sell_returns'].cumsum(),
                mode='lines',
                name='X.9 Signals (Sell)',
                line=dict(color='red', width=2)
            )
        )

    # Benchmark (buy and hold)
    if 'benchmark_returns' in performance_data.columns:
        fig.add_trace(
            go.Scatter(
                x=performance_data.index,
                y=performance_data['benchmark_returns'].cumsum(),
                mode='lines',
                name='Buy & Hold',
                line=dict(color='blue', width=2, dash='dash')
            )
        )

    fig.update_layout(
        title='Cumulative Returns: Clustering Strategy vs Benchmark',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns (%)',
        height=400
    )

    return fig

def analyze_historical_performance(data: pd.DataFrame, analyzer: StockClusteringAnalyzer, 
                                 holding_period: int = 5) -> Tuple[List[Dict], pd.DataFrame]:
    """Analyze historical performance of clustering signals"""

    signals = []
    performance_data = []

    for i in range(len(data) - holding_period):
        current_close = data['Close'].iloc[i]
        signal_data = analyzer.generate_signal(current_close)

        if signal_data['signal'] != 'NEUTRAL':
            # Calculate forward returns
            future_prices = data['Close'].iloc[i+1:i+1+holding_period]

            if len(future_prices) == holding_period:
                forward_return = (future_prices.iloc[-1] - current_close) / current_close * 100

                signal_data.update({
                    'date': data.index[i],
                    'forward_return': forward_return,
                    'holding_period': holding_period
                })

                signals.append(signal_data)

                # Track performance
                perf_row = {
                    'date': data.index[i],
                    'signal': signal_data['signal'],
                    'return': forward_return if signal_data['signal'] == 'BUY' else -forward_return
                }
                performance_data.append(perf_row)

    # Create performance DataFrame
    if performance_data:
        perf_df = pd.DataFrame(performance_data)
        perf_df.set_index('date', inplace=True)

        # Separate buy and sell signal returns
        buy_mask = perf_df['signal'] == 'BUY'
        sell_mask = perf_df['signal'] == 'SELL'

        perf_summary = pd.DataFrame(index=perf_df.index)
        perf_summary['buy_returns'] = perf_df.loc[buy_mask, 'return'] if buy_mask.any() else 0
        perf_summary['sell_returns'] = perf_df.loc[sell_mask, 'return'] if sell_mask.any() else 0
        perf_summary['benchmark_returns'] = data['Close'].pct_change().iloc[1:len(perf_summary)+1] * 100

        perf_summary.fillna(0, inplace=True)

        return signals, perf_summary

    return signals, pd.DataFrame()

def main():
    st.title("📈 Round-Number Order Clustering Monitor")
    st.markdown("""
    This application monitors stock prices and generates trading signals based on the **Round-Number Order Clustering** strategy 
    described in *"Can Daily Closing Prices Predict Future Movements? The Role of Limit Order Clustering"* by Xiao Zhang.

    **Strategy Overview:**
    - **X.1 Signals (Buy)**: Stocks closing just above round numbers (e.g., $100.10) tend to find support
    - **X.9 Signals (Sell)**: Stocks closing just below round numbers (e.g., $99.90) tend to face resistance
    """)

    # Initialize components
    analyzer = StockClusteringAnalyzer()
    data_manager = DataManager()

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Watchlist management
    st.sidebar.subheader("📋 Watchlist")

    # Default watchlist
    default_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]

    # Get user input for symbols
    symbols_input = st.sidebar.text_area(
        "Enter stock symbols (one per line):",
        value="\n".join(default_symbols),
        height=120
    )

    symbols = [s.strip().upper() for s in symbols_input.split('\n') if s.strip()]

    # Risk management settings
    st.sidebar.subheader("⚙️ Strategy Settings")
    holding_period = st.sidebar.slider("Holding Period (days)", 1, 10, 5)
    analysis_period = st.sidebar.selectbox("Analysis Period", ["1mo", "3mo", "6mo", "1y"], index=2)

    # Auto-refresh settings
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 30, 300, 60)

    if auto_refresh:
        # Create a placeholder for the refresh countdown
        countdown_placeholder = st.sidebar.empty()

        # Auto-refresh logic
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = time.time()

        time_since_refresh = time.time() - st.session_state.last_refresh

        if time_since_refresh >= refresh_interval:
            st.session_state.last_refresh = time.time()
            st.rerun()
        else:
            remaining_time = refresh_interval - int(time_since_refresh)
            countdown_placeholder.write(f"⏱️ Next refresh in: {remaining_time}s")

    # Main dashboard
    if symbols:
        # Real-time monitoring section
        st.header("🔴 Live Monitoring Dashboard")

        # Create columns for the watchlist
        cols = st.columns(min(len(symbols), 3))

        current_signals = []

        for idx, symbol in enumerate(symbols):
            with cols[idx % 3]:
                # Get real-time data
                real_time_data = data_manager.get_real_time_price(symbol)

                if real_time_data['current_price'] > 0:
                    # Generate signal
                    signal_data = analyzer.generate_signal(real_time_data['current_price'])
                    current_signals.append({**signal_data, 'symbol': symbol})

                    # Display stock card
                    change_color = "green" if real_time_data['change'] >= 0 else "red"
                    signal_color = {"BUY": "green", "SELL": "red", "NEUTRAL": "gray"}[signal_data['signal']]

                    st.markdown(f"""
                    <div style="border: 2px solid {signal_color}; border-radius: 10px; padding: 15px; margin: 10px 0;">
                        <h3 style="margin: 0; color: {signal_color};">{symbol}</h3>
                        <p style="font-size: 24px; margin: 5px 0;">${real_time_data['current_price']:.2f}</p>
                        <p style="color: {change_color}; margin: 5px 0;">
                            {real_time_data['change']:+.2f} ({real_time_data['change_percent']:+.1f}%)
                        </p>
                        <p style="margin: 5px 0;"><strong>Signal:</strong> 
                            <span style="color: {signal_color}; font-weight: bold;">{signal_data['signal']}</span>
                        </p>
                        <p style="margin: 5px 0; font-size: 12px;">
                            Distance from round: {signal_data['distance_from_round']:+.3f}
                        </p>
                        <p style="margin: 5px 0; font-size: 12px;">
                            Confidence: {signal_data['confidence']:.1%}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

        # Active signals summary
        active_signals = [s for s in current_signals if s['signal'] != 'NEUTRAL']

        if active_signals:
            st.header("🚨 Active Trading Signals")

            signal_df = pd.DataFrame(active_signals)
            st.dataframe(
                signal_df[['symbol', 'signal', 'price', 'distance_from_round', 'confidence', 'reasoning']],
                use_container_width=True
            )

        # Detailed analysis section
        st.header("📊 Detailed Analysis")

        selected_symbol = st.selectbox("Select symbol for detailed analysis:", symbols)

        if selected_symbol:
            # Fetch historical data
            historical_data = data_manager.fetch_stock_data(selected_symbol, analysis_period)

            if not historical_data.empty:
                # Analyze historical performance
                signals, performance_data = analyze_historical_performance(
                    historical_data, analyzer, holding_period
                )

                # Create tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["Price Chart", "Performance", "Signal History", "Statistics"])

                with tab1:
                    # Price chart with signals
                    chart = create_price_chart(historical_data, selected_symbol, signals)
                    st.plotly_chart(chart, use_container_width=True)

                with tab2:
                    if not performance_data.empty:
                        # Performance chart
                        perf_chart = create_performance_chart(performance_data)
                        st.plotly_chart(perf_chart, use_container_width=True)

                        # Performance metrics
                        col1, col2, col3 = st.columns(3)

                        buy_returns = performance_data['buy_returns'][performance_data['buy_returns'] != 0]
                        sell_returns = performance_data['sell_returns'][performance_data['sell_returns'] != 0]

                        with col1:
                            st.metric(
                                "Buy Signals (X.1)",
                                f"{len(buy_returns)} signals",
                                f"{buy_returns.mean():.2f}% avg return" if len(buy_returns) > 0 else "N/A"
                            )

                        with col2:
                            st.metric(
                                "Sell Signals (X.9)",
                                f"{len(sell_returns)} signals",
                                f"{sell_returns.mean():.2f}% avg return" if len(sell_returns) > 0 else "N/A"
                            )

                        with col3:
                            total_return = performance_data['buy_returns'].sum() + performance_data['sell_returns'].sum()
                            st.metric(
                                "Total Strategy Return",
                                f"{total_return:.2f}%",
                                f"vs {performance_data['benchmark_returns'].sum():.2f}% benchmark"
                            )

                with tab3:
                    # Signal history table
                    if signals:
                        signal_history = pd.DataFrame(signals)
                        signal_history = signal_history[[
                            'date', 'signal', 'price', 'distance_from_round', 
                            'confidence', 'forward_return'
                        ]].round(3)
                        st.dataframe(signal_history, use_container_width=True)
                    else:
                        st.info("No signals generated for the selected period.")

                with tab4:
                    # Statistical analysis
                    if signals:
                        buy_signals = [s for s in signals if s['signal'] == 'BUY']
                        sell_signals = [s for s in signals if s['signal'] == 'SELL']

                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("Buy Signals (X.1) Statistics")
                            if buy_signals:
                                buy_returns = [s['forward_return'] for s in buy_signals]
                                st.write(f"**Total Signals:** {len(buy_signals)}")
                                st.write(f"**Average Return:** {np.mean(buy_returns):.2f}%")
                                st.write(f"**Success Rate:** {sum(1 for r in buy_returns if r > 0) / len(buy_returns):.1%}")
                                st.write(f"**Best Trade:** {max(buy_returns):.2f}%")
                                st.write(f"**Worst Trade:** {min(buy_returns):.2f}%")
                            else:
                                st.write("No buy signals in this period")

                        with col2:
                            st.subheader("Sell Signals (X.9) Statistics")
                            if sell_signals:
                                sell_returns = [s['forward_return'] for s in sell_signals]
                                st.write(f"**Total Signals:** {len(sell_signals)}")
                                st.write(f"**Average Return:** {np.mean(sell_returns):.2f}%")
                                st.write(f"**Success Rate:** {sum(1 for r in sell_returns if r < 0) / len(sell_returns):.1%}")
                                st.write(f"**Best Trade:** {min(sell_returns):.2f}%")
                                st.write(f"**Worst Trade:** {max(sell_returns):.2f}%")
                            else:
                                st.write("No sell signals in this period")

    # Documentation section
    with st.expander("📚 Strategy Documentation"):
        st.markdown("""
        ### Round-Number Order Clustering Strategy

        This strategy is based on behavioral finance research that shows traders tend to place orders at round numbers,
        creating predictable support and resistance levels.

        **Key Concepts:**
        - **X.1 Pattern**: When stocks close just above round numbers (e.g., $100.10), it suggests strong buying pressure
          that pushed the price past the psychological resistance, indicating potential continued upward momentum.

        - **X.9 Pattern**: When stocks close just below round numbers (e.g., $99.90), it suggests selling pressure
          prevented the price from reaching the round number, indicating potential continued downward pressure.

        **Signal Generation:**
        - Buy signals are generated when closing prices are 0.05-0.15 above round numbers
        - Sell signals are generated when closing prices are 0.05-0.15 below round numbers
        - Confidence increases with proximity to the X.1 or X.9 levels

        **Risk Management:**
        - Use appropriate position sizing
        - Set stop-loss levels based on your risk tolerance
        - Consider the holding period (typically 1-5 days for this strategy)
        - Monitor volume and market conditions for confirmation

        **Limitations:**
        - Works best in trending markets
        - May produce false signals in highly volatile conditions
        - Should be combined with other technical analysis tools
        - Past performance does not guarantee future results
        """)

if __name__ == "__main__":
    main()