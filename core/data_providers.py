
"""
Enhanced data providers with multiple sources and fallback mechanisms
"""
import yfinance as yf
import requests
import pandas as pd
import streamlit as st
import time
import asyncio
import aiohttp
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import json
from abc import ABC, abstractmethod


class DataProvider(ABC):
    """Abstract base class for data providers"""
    
    @abstractmethod
    def get_quote(self, symbol: str) -> Dict:
        """Get real-time quote for a symbol"""
        pass
    
    @abstractmethod
    def get_historical(self, symbol: str, period: str) -> pd.DataFrame:
        """Get historical data for a symbol"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available"""
        pass


class YahooFinanceProvider(DataProvider):
    """Yahoo Finance data provider"""
    
    def __init__(self):
        self.name = "Yahoo Finance"
        self.rate_limit_delay = 0.5
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def get_quote(self, symbol: str) -> Dict:
        """Get real-time quote from Yahoo Finance"""
        try:
            self._rate_limit()
            ticker = yf.Ticker(symbol)
            
            # Get basic info and recent history
            info = ticker.info
            hist = ticker.history(period="2d")
            
            if hist.empty:
                return self._empty_quote(symbol)
            
            current_price = float(hist['Close'].iloc[-1])
            previous_close = info.get('previousClose', current_price)
            
            if len(hist) > 1 and (previous_close is None or previous_close == current_price):
                previous_close = float(hist['Close'].iloc[-2])
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'previous_close': previous_close,
                'change': current_price - previous_close,
                'change_percent': ((current_price - previous_close) / previous_close * 100) if previous_close > 0 else 0,
                'volume': int(hist['Volume'].iloc[-1]) if not pd.isna(hist['Volume'].iloc[-1]) else 0,
                'day_low': float(hist['Low'].iloc[-1]),
                'day_high': float(hist['High'].iloc[-1]),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0) or 0,
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0) or 0,
                'provider': self.name
            }
            
        except Exception as e:
            st.warning(f"Yahoo Finance error for {symbol}: {str(e)}")
            return self._empty_quote(symbol)
    
    def get_historical(self, symbol: str, period: str) -> pd.DataFrame:
        """Get historical data from Yahoo Finance"""
        try:
            self._rate_limit()
            ticker = yf.Ticker(symbol)
            return ticker.history(period=period)
        except Exception as e:
            st.warning(f"Yahoo Finance historical data error for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def is_available(self) -> bool:
        """Check if Yahoo Finance is available"""
        try:
            test_ticker = yf.Ticker("AAPL")
            test_data = test_ticker.history(period="1d")
            return not test_data.empty
        except:
            return False
    
    def _empty_quote(self, symbol: str) -> Dict:
        """Return empty quote structure"""
        return {
            'symbol': symbol,
            'current_price': 0,
            'previous_close': 0,
            'change': 0,
            'change_percent': 0,
            'volume': 0,
            'day_low': 0,
            'day_high': 0,
            'fifty_two_week_low': 0,
            'fifty_two_week_high': 0,
            'provider': self.name
        }


class AlphaVantageProvider(DataProvider):
    """Alpha Vantage data provider"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.name = "Alpha Vantage"
        # Safely access secrets with fallback
        try:
            self.api_key = api_key or st.secrets.get("ALPHA_VANTAGE_API_KEY", "demo")
        except:
            self.api_key = "demo"
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_delay = 1.0  # Alpha Vantage has stricter limits
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def get_quote(self, symbol: str) -> Dict:
        """Get real-time quote from Alpha Vantage"""
        try:
            self._rate_limit()
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            if 'Global Quote' in data:
                quote = data['Global Quote']
                current_price = float(quote['05. price'])
                previous_close = float(quote['08. previous close'])
                
                return {
                    'symbol': symbol,
                    'current_price': current_price,
                    'previous_close': previous_close,
                    'change': float(quote['09. change']),
                    'change_percent': float(quote['10. change percent'].rstrip('%')),
                    'volume': int(quote['06. volume']),
                    'day_low': float(quote['04. low']),
                    'day_high': float(quote['03. high']),
                    'fifty_two_week_low': 0,  # Not available in this endpoint
                    'fifty_two_week_high': 0,  # Not available in this endpoint
                    'provider': self.name
                }
            else:
                return self._empty_quote(symbol)
                
        except Exception as e:
            st.warning(f"Alpha Vantage error for {symbol}: {str(e)}")
            return self._empty_quote(symbol)
    
    def get_historical(self, symbol: str, period: str) -> pd.DataFrame:
        """Get historical data from Alpha Vantage"""
        try:
            self._rate_limit()
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': self.api_key,
                'outputsize': 'compact' if period in ['1mo', '3mo'] else 'full'
            }
            
            response = requests.get(self.base_url, params=params, timeout=15)
            data = response.json()
            
            if 'Time Series (Daily)' in data:
                df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                
                # Rename columns to match yfinance format
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                df = df.astype(float)
                
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            st.warning(f"Alpha Vantage historical data error for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def is_available(self) -> bool:
        """Check if Alpha Vantage is available"""
        if self.api_key == "demo":
            return False
        try:
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': 'AAPL',
                'apikey': self.api_key
            }
            response = requests.get(self.base_url, params=params, timeout=5)
            data = response.json()
            return 'Global Quote' in data
        except:
            return False
    
    def _empty_quote(self, symbol: str) -> Dict:
        """Return empty quote structure"""
        return {
            'symbol': symbol,
            'current_price': 0,
            'previous_close': 0,
            'change': 0,
            'change_percent': 0,
            'volume': 0,
            'day_low': 0,
            'day_high': 0,
            'fifty_two_week_low': 0,
            'fifty_two_week_high': 0,
            'provider': self.name
        }


class FinnhubProvider(DataProvider):
    """Finnhub data provider"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.name = "Finnhub"
        # Safely access secrets with fallback
        try:
            self.api_key = api_key or st.secrets.get("FINNHUB_API_KEY", "")
        except:
            self.api_key = ""
        self.base_url = "https://finnhub.io/api/v1"
        self.rate_limit_delay = 0.1
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def get_quote(self, symbol: str) -> Dict:
        """Get real-time quote from Finnhub"""
        try:
            self._rate_limit()
            headers = {'X-Finnhub-Token': self.api_key}
            
            # Get current quote
            quote_url = f"{self.base_url}/quote"
            params = {'symbol': symbol}
            response = requests.get(quote_url, headers=headers, params=params, timeout=10)
            quote_data = response.json()
            
            if 'c' in quote_data:  # 'c' is current price
                current_price = float(quote_data['c'])
                previous_close = float(quote_data['pc'])
                
                return {
                    'symbol': symbol,
                    'current_price': current_price,
                    'previous_close': previous_close,
                    'change': current_price - previous_close,
                    'change_percent': ((current_price - previous_close) / previous_close * 100) if previous_close > 0 else 0,
                    'volume': 0,  # Not available in quote endpoint
                    'day_low': float(quote_data['l']),
                    'day_high': float(quote_data['h']),
                    'fifty_two_week_low': 0,  # Requires separate API call
                    'fifty_two_week_high': 0,  # Requires separate API call
                    'provider': self.name
                }
            else:
                return self._empty_quote(symbol)
                
        except Exception as e:
            st.warning(f"Finnhub error for {symbol}: {str(e)}")
            return self._empty_quote(symbol)
    
    def get_historical(self, symbol: str, period: str) -> pd.DataFrame:
        """Get historical data from Finnhub"""
        try:
            self._rate_limit()
            headers = {'X-Finnhub-Token': self.api_key}
            
            # Calculate date range
            end_date = datetime.now()
            if period == "1mo":
                start_date = end_date - timedelta(days=30)
            elif period == "3mo":
                start_date = end_date - timedelta(days=90)
            elif period == "6mo":
                start_date = end_date - timedelta(days=180)
            elif period == "1y":
                start_date = end_date - timedelta(days=365)
            else:
                start_date = end_date - timedelta(days=30)
            
            # Convert to Unix timestamps
            start_timestamp = int(start_date.timestamp())
            end_timestamp = int(end_date.timestamp())
            
            candles_url = f"{self.base_url}/stock/candle"
            params = {
                'symbol': symbol,
                'resolution': 'D',
                'from': start_timestamp,
                'to': end_timestamp
            }
            
            response = requests.get(candles_url, headers=headers, params=params, timeout=15)
            data = response.json()
            
            if data['s'] == 'ok':  # 's' is status
                df = pd.DataFrame({
                    'Open': data['o'],
                    'High': data['h'],
                    'Low': data['l'],
                    'Close': data['c'],
                    'Volume': data['v']
                })
                
                # Convert timestamps to datetime index
                df.index = pd.to_datetime(data['t'], unit='s')
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            st.warning(f"Finnhub historical data error for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def is_available(self) -> bool:
        """Check if Finnhub is available"""
        if not self.api_key:
            return False
        try:
            headers = {'X-Finnhub-Token': self.api_key}
            params = {'symbol': 'AAPL'}
            response = requests.get(f"{self.base_url}/quote", headers=headers, params=params, timeout=5)
            data = response.json()
            return 'c' in data
        except:
            return False
    
    def _empty_quote(self, symbol: str) -> Dict:
        """Return empty quote structure"""
        return {
            'symbol': symbol,
            'current_price': 0,
            'previous_close': 0,
            'change': 0,
            'change_percent': 0,
            'volume': 0,
            'day_low': 0,
            'day_high': 0,
            'fifty_two_week_low': 0,
            'fifty_two_week_high': 0,
            'provider': self.name
        }


class MarketHoursDetector:
    """Detect market hours and trading sessions"""
    
    def __init__(self):
        self.market_timezone = 'US/Eastern'
    
    def is_market_open(self) -> bool:
        """Check if the market is currently open"""
        try:
            import pytz
            
            # Get current time in market timezone
            market_tz = pytz.timezone(self.market_timezone)
            current_time = datetime.now(market_tz)
            
            # Market is closed on weekends
            if current_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False
            
            # Regular market hours: 9:30 AM - 4:00 PM ET
            market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
            
            return market_open <= current_time <= market_close
            
        except ImportError:
            # Fallback if pytz is not available
            current_hour = datetime.now().hour
            current_minute = datetime.now().minute
            
            # Rough approximation (assumes ET timezone)
            market_start = 9 * 60 + 30  # 9:30 AM in minutes
            market_end = 16 * 60  # 4:00 PM in minutes
            current_minutes = current_hour * 60 + current_minute
            
            return market_start <= current_minutes <= market_end
    
    def get_market_status(self) -> str:
        """Get detailed market status"""
        if self.is_market_open():
            return "OPEN"
        else:
            return "CLOSED"
    
    def is_premarket(self) -> bool:
        """Check if we're in pre-market hours (4:00 AM - 9:30 AM ET)"""
        try:
            import pytz
            
            market_tz = pytz.timezone(self.market_timezone)
            current_time = datetime.now(market_tz)
            
            if current_time.weekday() >= 5:
                return False
            
            premarket_start = current_time.replace(hour=4, minute=0, second=0, microsecond=0)
            market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
            
            return premarket_start <= current_time < market_open
            
        except ImportError:
            return False
    
    def is_afterhours(self) -> bool:
        """Check if we're in after-hours trading (4:00 PM - 8:00 PM ET)"""
        try:
            import pytz
            
            market_tz = pytz.timezone(self.market_timezone)
            current_time = datetime.now(market_tz)
            
            if current_time.weekday() >= 5:
                return False
            
            market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
            afterhours_end = current_time.replace(hour=20, minute=0, second=0, microsecond=0)
            
            return market_close < current_time <= afterhours_end
            
        except ImportError:
            return False


class EnhancedDataManager:
    """Enhanced data manager with multiple providers and fallback mechanisms"""
    
    def __init__(self):
        # Initialize providers
        self.providers = [
            YahooFinanceProvider(),
            AlphaVantageProvider(),
            FinnhubProvider()
        ]
        
        # Initialize market hours detector
        self.market_detector = MarketHoursDetector()
        
        # Cache settings
        self.cache = {}
        self.cache_duration = 60  # 1 minute cache for quotes during market hours
        self.historical_cache = {}
        self.historical_cache_duration = 300  # 5 minutes cache for historical data
        
        # Provider health tracking
        self.provider_health = {provider.name: True for provider in self.providers}
        self.last_health_check = time.time()
        self.health_check_interval = 300  # 5 minutes
    
    def _check_provider_health(self):
        """Check the health of all providers"""
        current_time = time.time()
        if current_time - self.last_health_check > self.health_check_interval:
            for provider in self.providers:
                self.provider_health[provider.name] = provider.is_available()
            self.last_health_check = current_time
    
    def get_available_providers(self) -> List[DataProvider]:
        """Get list of available providers"""
        self._check_provider_health()
        return [p for p in self.providers if self.provider_health[p.name]]
    
    def get_real_time_price(self, symbol: str) -> Dict:
        """Get real-time price with fallback providers"""
        # Check cache first
        cache_key = f"quote_{symbol}"
        current_time = time.time()
        
        # Adjust cache duration based on market hours
        cache_duration = self.cache_duration if self.market_detector.is_market_open() else 300
        
        if (cache_key in self.cache and 
            current_time - self.cache[cache_key]['timestamp'] < cache_duration):
            return self.cache[cache_key]['data']
        
        # Try providers in order of preference
        available_providers = self.get_available_providers()
        
        for provider in available_providers:
            try:
                result = provider.get_quote(symbol)
                if result['current_price'] > 0:  # Valid data
                    # Add market status information
                    result['market_status'] = self.market_detector.get_market_status()
                    result['is_premarket'] = self.market_detector.is_premarket()
                    result['is_afterhours'] = self.market_detector.is_afterhours()
                    
                    # Cache the result
                    self.cache[cache_key] = {
                        'data': result,
                        'timestamp': current_time
                    }
                    
                    return result
                    
            except Exception as e:
                # Mark provider as unhealthy and try next
                self.provider_health[provider.name] = False
                continue
        
        # All providers failed, return empty result
        return {
            'symbol': symbol,
            'current_price': 0,
            'previous_close': 0,
            'change': 0,
            'change_percent': 0,
            'volume': 0,
            'day_low': 0,
            'day_high': 0,
            'fifty_two_week_low': 0,
            'fifty_two_week_high': 0,
            'provider': 'None (All providers failed)',
            'market_status': self.market_detector.get_market_status(),
            'is_premarket': self.market_detector.is_premarket(),
            'is_afterhours': self.market_detector.is_afterhours()
        }
    
    def fetch_stock_data(self, symbol: str, period: str = "1mo") -> pd.DataFrame:
        """Fetch historical stock data with fallback providers"""
        # Check cache first
        cache_key = f"historical_{symbol}_{period}"
        current_time = time.time()
        
        if (cache_key in self.historical_cache and 
            current_time - self.historical_cache[cache_key]['timestamp'] < self.historical_cache_duration):
            return self.historical_cache[cache_key]['data']
        
        # Try providers in order of preference
        available_providers = self.get_available_providers()
        
        for provider in available_providers:
            try:
                result = provider.get_historical(symbol, period)
                if not result.empty:
                    # Cache the result
                    self.historical_cache[cache_key] = {
                        'data': result,
                        'timestamp': current_time
                    }
                    return result
                    
            except Exception as e:
                # Mark provider as unhealthy and try next
                self.provider_health[provider.name] = False
                continue
        
        # All providers failed
        st.error(f"Unable to fetch historical data for {symbol} from any provider")
        return pd.DataFrame()
    
    def get_provider_status(self) -> Dict:
        """Get status of all data providers"""
        self._check_provider_health()
        return {
            'providers': self.provider_health,
            'market_status': self.market_detector.get_market_status(),
            'is_market_open': self.market_detector.is_market_open(),
            'is_premarket': self.market_detector.is_premarket(),
            'is_afterhours': self.market_detector.is_afterhours()
        }
