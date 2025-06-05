
"""
Data management module for stock data fetching and caching
"""
import pandas as pd
import yfinance as yf
import streamlit as st
import time
from typing import Dict


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

                # Get day's range from current day's data
                day_low = hist['Low'].min()
                day_high = hist['High'].max()

                return {
                    'symbol': symbol,
                    'current_price': current_price,
                    'previous_close': info.get('previousClose', current_price),
                    'change': current_price - info.get('previousClose', current_price),
                    'change_percent': ((current_price - info.get('previousClose', current_price)) / 
                                     info.get('previousClose', current_price)) * 100,
                    'volume': hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0,
                    'day_low': day_low,
                    'day_high': day_high,
                    'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
                    'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0)
                }
        except Exception as e:
            st.error(f"Error getting real-time data for {symbol}: {str(e)}")

        return {
            'symbol': symbol, 
            'current_price': 0, 
            'change': 0, 
            'change_percent': 0,
            'day_low': 0,
            'day_high': 0,
            'fifty_two_week_low': 0,
            'fifty_two_week_high': 0
        }
