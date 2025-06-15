
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
        """Get current price and basic info with comprehensive error handling"""
        # Default return structure
        default_result = {
            'symbol': symbol, 
            'current_price': 0, 
            'change': 0, 
            'change_percent': 0,
            'day_low': 0,
            'day_high': 0,
            'fifty_two_week_low': 0,
            'fifty_two_week_high': 0,
            'volume': 0
        }
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Try to get basic info first
            try:
                info = ticker.info
            except Exception as info_error:
                # If info fails, try without it
                info = {}
                
            # Try to get historical data with fallback periods
            hist = None
            for period_attempt in ["1d", "5d"]:
                try:
                    hist = ticker.history(period=period_attempt, interval="1m" if period_attempt == "1d" else "1d")
                    if not hist.empty:
                        break
                except Exception:
                    continue
            
            if hist is None or hist.empty:
                # Final fallback - try daily data
                try:
                    hist = ticker.history(period="1d")
                except Exception:
                    return default_result

            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
                
                # Validate price data
                if current_price <= 0 or pd.isna(current_price):
                    return default_result

                # Get day's range from current day's data
                day_low = float(hist['Low'].min()) if not hist['Low'].empty else current_price
                day_high = float(hist['High'].max()) if not hist['High'].empty else current_price
                
                # Get previous close with fallbacks
                previous_close = info.get('previousClose', current_price)
                if previous_close is None or pd.isna(previous_close) or previous_close <= 0:
                    if len(hist) > 1:
                        previous_close = float(hist['Close'].iloc[-2])
                    else:
                        previous_close = current_price

                # Calculate change safely
                change = current_price - previous_close
                change_percent = (change / previous_close * 100) if previous_close > 0 else 0

                # Get volume safely
                volume = 0
                if 'Volume' in hist.columns and not hist['Volume'].empty:
                    try:
                        volume = int(hist['Volume'].iloc[-1])
                        if pd.isna(volume):
                            volume = 0
                    except (ValueError, TypeError):
                        volume = 0

                return {
                    'symbol': symbol,
                    'current_price': current_price,
                    'previous_close': previous_close,
                    'change': change,
                    'change_percent': change_percent,
                    'volume': volume,
                    'day_low': day_low,
                    'day_high': day_high,
                    'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0) or 0,
                    'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0) or 0
                }
                
        except Exception as e:
            # Only show error for debugging, don't overwhelm UI
            error_msg = str(e)
            if "rate limit" in error_msg.lower() or "too many requests" in error_msg.lower():
                # Rate limit error - return default but don't show error
                pass
            elif len(symbol) <= 5:  # Only log for valid symbols
                st.error(f"Network error for {symbol}: Connection timeout")

        return default_result
