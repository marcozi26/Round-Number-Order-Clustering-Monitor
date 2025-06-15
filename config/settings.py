
"""
Configuration settings for the Round-Number Order Clustering Monitor
"""
import warnings
warnings.filterwarnings('ignore')

# Strategy Parameters
ROUND_THRESHOLD = 0.1  # Distance from round number to consider clustering
X1_MIN_DISTANCE = 0.05  # Minimum distance for X.1 classification
X1_MAX_DISTANCE = 0.15  # Maximum distance for X.1 classification
X9_MIN_DISTANCE = -0.15  # Minimum distance for X.9 classification (negative)
X9_MAX_DISTANCE = -0.05  # Maximum distance for X.9 classification (negative)

# Data Management
CACHE_DURATION = 300  # 5 minutes cache duration

# Data Provider Configuration
API_KEYS = {
    'ALPHA_VANTAGE': None,  # Set in Streamlit secrets as ALPHA_VANTAGE_API_KEY
    'FINNHUB': None,        # Set in Streamlit secrets as FINNHUB_API_KEY
    'POLYGON': None,        # Set in Streamlit secrets as POLYGON_API_KEY (future enhancement)
    'IEX_CLOUD': None,      # Set in Streamlit secrets as IEX_CLOUD_API_KEY (future enhancement)
}

# Data Provider Settings
PROVIDER_SETTINGS = {
    'YAHOO_FINANCE': {
        'rate_limit_delay': 0.5,
        'timeout': 10,
        'priority': 1
    },
    'ALPHA_VANTAGE': {
        'rate_limit_delay': 1.0,
        'timeout': 15,
        'priority': 2
    },
    'FINNHUB': {
        'rate_limit_delay': 0.1,
        'timeout': 10,
        'priority': 3
    }
}

# Default Symbols
DEFAULT_WATCHLIST = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "CRWD", "KO", "JPM", "JNJ"]

# Extended Stock Universe for Scanning
STOCK_UNIVERSE = [
    "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA", "NFLX", "AMD", "CRM",
    "ORCL", "ADBE", "INTC", "CSCO", "IBM", "QCOM", "TXN", "AVGO", "NOW", "INTU",
    "MU", "AMAT", "LRCX", "KLAC", "MRVL", "MCHP", "SNPS", "CDNS", "FTNT", "PANW",
    "ZS", "OKTA", "CRWD", "DDOG", "NET", "MDB", "SNOW", "PLTR", "U", "DOCN",
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "AXP", "V", "MA",
    "JNJ", "PFE", "UNH", "ABBV", "MRK", "LLY", "TMO", "ABT", "DHR", "BMY",
    "KO", "PEP", "WMT", "PG", "HD", "MCD", "COST", "LOW", "TGT", "SBUX",
    "DIS", "CMCSA", "VZ", "T", "CHTR", "TMUS", "DISH", "PARA", "WBD", "NWSA",
    "XOM", "CVX", "COP", "EOG", "SLB", "OXY", "PSX", "VLO", "MPC", "HES",
    "BA", "RTX", "LMT", "NOC", "GD", "HON", "MMM", "CAT", "DE", "EMR"
]

# Streamlit Configuration
PAGE_CONFIG = {
    "page_title": "Round-Number Order Clustering Monitor",
    "page_icon": "📈",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}
