"""
Enhanced data management module with multiple providers and fallback mechanisms
"""
import pandas as pd
import streamlit as st
import time
from typing import Dict
from .data_providers import EnhancedDataManager


class DataManager:
    """Handle stock data fetching and caching with enhanced providers"""

    def __init__(self):
        # Initialize the enhanced data manager
        self.enhanced_manager = EnhancedDataManager()

        # Backward compatibility cache
        self.cache = {}
        self.cache_duration = 300  # 5 minutes cache

    def fetch_stock_data(self, symbol: str, period: str = "1mo") -> pd.DataFrame:
        """Fetch stock data using enhanced multi-provider system"""
        return self.enhanced_manager.fetch_stock_data(symbol, period)

    def get_real_time_price(self, symbol: str) -> Dict:
        """Get current price using enhanced multi-provider system"""
        return self.enhanced_manager.get_real_time_price(symbol)

    def get_provider_status(self) -> Dict:
        """Get status of all data providers"""
        return self.enhanced_manager.get_provider_status()

    def get_market_status(self) -> str:
        """Get current market status"""
        return self.enhanced_manager.market_detector.get_market_status()

    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        return self.enhanced_manager.market_detector.is_market_open()
```

```
"""
Enhanced data management module with multiple providers and fallback mechanisms
"""
import pandas as pd
import streamlit as st
import time
from typing import Dict
from .data_providers import EnhancedDataManager


class DataManager:
    """Handle stock data fetching and caching with enhanced providers"""

    def __init__(self):
        # Initialize the enhanced data manager
        self.enhanced_manager = EnhancedDataManager()

        # Backward compatibility cache
        self.cache = {}
        self.cache_duration = 300  # 5 minutes cache

    def fetch_stock_data(self, symbol: str, period: str = "1mo") -> pd.DataFrame:
        """Fetch stock data using enhanced multi-provider system"""
        return self.enhanced_manager.fetch_stock_data(symbol, period)

    def get_real_time_price(self, symbol: str) -> Dict:
        """Get current price using enhanced multi-provider system"""
        return self.enhanced_manager.get_real_time_price(symbol)

    def get_provider_status(self) -> Dict:
        """Get status of all data providers"""
        return self.enhanced_manager.get_provider_status()

    def get_market_status(self) -> str:
        """Get current market status"""
        return self.enhanced_manager.market_detector.get_market_status()

    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        return self.enhanced_manager.market_detector.is_market_open()