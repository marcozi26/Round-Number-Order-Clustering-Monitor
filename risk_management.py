
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta, time
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class RiskManager:
    """
    Comprehensive risk management system for the Round-Number Order Clustering Monitor
    """
    
    def __init__(self):
        # Default risk parameters
        self.default_risk_percent = 2.0  # 2% risk per trade
        self.default_atr_multiplier = 1.5
        self.default_max_drawdown = 10.0  # 10% max drawdown
        self.default_max_sector_exposure = 30.0  # 30% max sector exposure
        self.default_correlation_threshold = 0.7
        
        # Sector mapping for common stocks
        self.sector_mapping = {
            'AAPL': 'Technology', 'GOOGL': 'Technology', 'MSFT': 'Technology', 'NVDA': 'Technology',
            'META': 'Technology', 'TSLA': 'Technology', 'NFLX': 'Technology', 'AMD': 'Technology',
            'CRM': 'Technology', 'ORCL': 'Technology', 'ADBE': 'Technology', 'INTC': 'Technology',
            'CSCO': 'Technology', 'IBM': 'Technology', 'QCOM': 'Technology', 'TXN': 'Technology',
            'AVGO': 'Technology', 'NOW': 'Technology', 'INTU': 'Technology', 'MU': 'Technology',
            'AMAT': 'Technology', 'LRCX': 'Technology', 'KLAC': 'Technology', 'MRVL': 'Technology',
            'MCHP': 'Technology', 'SNPS': 'Technology', 'CDNS': 'Technology', 'FTNT': 'Technology',
            'PANW': 'Technology', 'ZS': 'Technology', 'OKTA': 'Technology', 'CRWD': 'Technology',
            'DDOG': 'Technology', 'NET': 'Technology', 'MDB': 'Technology', 'SNOW': 'Technology',
            'PLTR': 'Technology', 'U': 'Technology', 'DOCN': 'Technology',
            
            'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial', 'C': 'Financial',
            'GS': 'Financial', 'MS': 'Financial', 'BLK': 'Financial', 'AXP': 'Financial',
            'V': 'Financial', 'MA': 'Financial',
            
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare', 'ABBV': 'Healthcare',
            'MRK': 'Healthcare', 'LLY': 'Healthcare', 'TMO': 'Healthcare', 'ABT': 'Healthcare',
            'DHR': 'Healthcare', 'BMY': 'Healthcare',
            
            'KO': 'Consumer', 'PEP': 'Consumer', 'WMT': 'Consumer', 'PG': 'Consumer',
            'HD': 'Consumer', 'MCD': 'Consumer', 'COST': 'Consumer', 'LOW': 'Consumer',
            'TGT': 'Consumer', 'SBUX': 'Consumer',
            
            'DIS': 'Media', 'CMCSA': 'Media', 'VZ': 'Telecom', 'T': 'Telecom',
            'CHTR': 'Telecom', 'TMUS': 'Telecom', 'DISH': 'Media', 'PARA': 'Media',
            'WBD': 'Media', 'NWSA': 'Media',
            
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'EOG': 'Energy',
            'SLB': 'Energy', 'OXY': 'Energy', 'PSX': 'Energy', 'VLO': 'Energy',
            'MPC': 'Energy', 'HES': 'Energy',
            
            'BA': 'Industrial', 'RTX': 'Industrial', 'LMT': 'Industrial', 'NOC': 'Industrial',
            'GD': 'Industrial', 'HON': 'Industrial', 'MMM': 'Industrial', 'CAT': 'Industrial',
            'DE': 'Industrial', 'EMR': 'Industrial'
        }
        
        # Market hours (Eastern Time)
        self.market_open = time(9, 30)
        self.market_close = time(16, 0)
        
        # Portfolio tracking
        if 'portfolio_balance' not in st.session_state:
            st.session_state.portfolio_balance = 100000.0  # Default $100k
        if 'portfolio_peak' not in st.session_state:
            st.session_state.portfolio_peak = st.session_state.portfolio_balance
        if 'current_positions' not in st.session_state:
            st.session_state.current_positions = {}
        if 'portfolio_allocation' not in st.session_state:
            st.session_state.portfolio_allocation = {}

    def get_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range (ATR) for volatility measurement
        """
        try:
            if len(data) < period + 1:
                return 0.0
            
            # Calculate True Range
            data = data.copy()
            data['H-L'] = data['High'] - data['Low']
            data['H-PC'] = abs(data['High'] - data['Close'].shift(1))
            data['L-PC'] = abs(data['Low'] - data['Close'].shift(1))
            data['TR'] = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
            
            # Calculate ATR
            atr = data['TR'].rolling(window=period).mean().iloc[-1]
            return atr if not pd.isna(atr) else 0.0
            
        except Exception as e:
            st.error(f"Error calculating ATR: {str(e)}")
            return 0.0

    def calculate_position_size(self, account_balance: float, risk_percent: float, 
                              entry_price: float, stop_loss_price: float) -> int:
        """
        Calculate position size based on risk management rules
        """
        try:
            if stop_loss_price <= 0 or entry_price <= 0:
                return 0
            
            # Calculate risk per share
            risk_per_share = abs(entry_price - stop_loss_price)
            
            if risk_per_share == 0:
                return 0
            
            # Calculate total risk amount
            total_risk = account_balance * (risk_percent / 100)
            
            # Calculate position size
            position_size = int(total_risk / risk_per_share)
            
            # Ensure position doesn't exceed reasonable limits
            max_position_value = account_balance * 0.1  # Max 10% of account per position
            max_shares = int(max_position_value / entry_price)
            
            return min(position_size, max_shares)
            
        except Exception as e:
            st.error(f"Error calculating position size: {str(e)}")
            return 0

    def calculate_stop_loss(self, entry_price: float, signal_type: str, atr: float, 
                           atr_multiplier: float = 1.5) -> float:
        """
        Calculate dynamic stop-loss based on ATR
        """
        try:
            if atr <= 0:
                # Fallback to percentage-based stop
                if signal_type == 'BUY':
                    return entry_price * 0.95  # 5% stop loss
                else:
                    return entry_price * 1.05  # 5% stop loss
            
            atr_distance = atr * atr_multiplier
            
            if signal_type == 'BUY':
                return entry_price - atr_distance
            else:  # SELL signal
                return entry_price + atr_distance
                
        except Exception as e:
            st.error(f"Error calculating stop loss: {str(e)}")
            return entry_price * 0.95 if signal_type == 'BUY' else entry_price * 1.05

    def calculate_risk_score(self, symbol: str, current_price: float, volume: float, 
                           data: pd.DataFrame) -> Dict:
        """
        Calculate risk-adjusted signal scoring
        """
        try:
            score = 0.0
            factors = {}
            
            # Volatility factor (30% weight)
            atr = self.get_atr(data)
            volatility_pct = (atr / current_price) * 100 if current_price > 0 else 0
            
            if volatility_pct < 2.0:  # Low volatility
                volatility_score = 1.0
            elif volatility_pct < 4.0:  # Medium volatility
                volatility_score = 0.6
            else:  # High volatility
                volatility_score = 0.3
            
            factors['volatility'] = volatility_score
            score += volatility_score * 0.3
            
            # Volume factor (25% weight)
            if len(data) >= 20:
                avg_volume = data['Volume'].tail(20).mean()
                volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
                
                if volume_ratio > 1.5:  # High volume
                    volume_score = 1.0
                elif volume_ratio > 0.8:  # Normal volume
                    volume_score = 0.7
                else:  # Low volume
                    volume_score = 0.4
            else:
                volume_score = 0.5
            
            factors['volume'] = volume_score
            score += volume_score * 0.25
            
            # Trend alignment (25% weight)
            if len(data) >= 20:
                sma_20 = data['Close'].tail(20).mean()
                trend_score = 1.0 if current_price > sma_20 else 0.3
            else:
                trend_score = 0.5
            
            factors['trend'] = trend_score
            score += trend_score * 0.25
            
            # Market timing (20% weight)
            current_time = datetime.now().time()
            if self.market_open <= current_time <= time(11, 0):  # First 1.5 hours
                timing_score = 1.0
            elif time(11, 0) < current_time <= time(14, 0):  # Mid-day
                timing_score = 0.8
            else:  # Last 2 hours
                timing_score = 0.5
            
            factors['timing'] = timing_score
            score += timing_score * 0.2
            
            # Determine risk level
            if score >= 0.8:
                risk_level = 'HIGH'
                color = 'green'
            elif score >= 0.6:
                risk_level = 'MEDIUM'
                color = 'yellow'
            else:
                risk_level = 'LOW'
                color = 'red'
            
            return {
                'score': score,
                'risk_level': risk_level,
                'color': color,
                'factors': factors,
                'volatility_pct': volatility_pct
            }
            
        except Exception as e:
            st.error(f"Error calculating risk score: {str(e)}")
            return {
                'score': 0.5,
                'risk_level': 'MEDIUM',
                'color': 'yellow',
                'factors': {},
                'volatility_pct': 0.0
            }

    def check_portfolio_correlation(self, symbols: List[str], new_symbol: str = None) -> Dict:
        """
        Check correlation between portfolio symbols
        """
        try:
            # Download 6 months of data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
            
            if new_symbol and new_symbol not in symbols:
                check_symbols = symbols + [new_symbol]
            else:
                check_symbols = symbols
            
            # Get price data
            price_data = {}
            for symbol in check_symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(start=start_date, end=end_date)
                    if not hist.empty:
                        price_data[symbol] = hist['Close']
                except:
                    continue
            
            if len(price_data) < 2:
                return {'correlation_matrix': pd.DataFrame(), 'warnings': []}
            
            # Create correlation matrix
            price_df = pd.DataFrame(price_data)
            price_df = price_df.dropna()
            
            if price_df.empty:
                return {'correlation_matrix': pd.DataFrame(), 'warnings': []}
            
            correlation_matrix = price_df.corr()
            
            # Check for high correlations
            warnings = []
            if new_symbol:
                for existing_symbol in symbols:
                    if existing_symbol in correlation_matrix.index and new_symbol in correlation_matrix.columns:
                        corr = correlation_matrix.loc[existing_symbol, new_symbol]
                        if abs(corr) > self.default_correlation_threshold:
                            warnings.append(f"High correlation ({corr:.2f}) between {new_symbol} and {existing_symbol}")
            
            return {
                'correlation_matrix': correlation_matrix,
                'warnings': warnings
            }
            
        except Exception as e:
            st.error(f"Error checking portfolio correlation: {str(e)}")
            return {'correlation_matrix': pd.DataFrame(), 'warnings': []}

    def sync_portfolio_with_watchlist(self, symbols: List[str], data_manager) -> Dict[str, float]:
        """
        Sync portfolio positions with watchlist symbols
        """
        try:
            position_values = {}
            total_balance = st.session_state.portfolio_balance
            
            if not symbols:
                return position_values
            
            # Get current allocations or create equal weight distribution
            if not st.session_state.portfolio_allocation or set(symbols) != set(st.session_state.portfolio_allocation.keys()):
                # Create equal weight allocation for all watchlist symbols
                equal_weight = 100.0 / len(symbols)
                st.session_state.portfolio_allocation = {symbol: equal_weight for symbol in symbols}
            
            # Calculate position values based on current prices and allocations
            for symbol in symbols:
                try:
                    allocation_pct = st.session_state.portfolio_allocation.get(symbol, 0)
                    allocation_value = total_balance * (allocation_pct / 100)
                    position_values[symbol] = allocation_value
                except Exception as e:
                    st.error(f"Error calculating position for {symbol}: {str(e)}")
                    position_values[symbol] = 0
            
            return position_values
            
        except Exception as e:
            st.error(f"Error syncing portfolio with watchlist: {str(e)}")
            return {}

    def check_sector_exposure(self, symbols: List[str], position_values: Dict[str, float]) -> Dict:
        """
        Check sector exposure limits
        """
        try:
            sector_exposure = {}
            total_value = sum(position_values.values())
            
            if total_value == 0:
                return {'sector_exposure': {}, 'warnings': [], 'total_exposure': 0}
            
            # Calculate sector exposures
            for symbol, value in position_values.items():
                sector = self.sector_mapping.get(symbol, 'Unknown')
                if sector not in sector_exposure:
                    sector_exposure[sector] = 0
                sector_exposure[sector] += (value / total_value) * 100
            
            # Check for violations
            warnings = []
            for sector, exposure in sector_exposure.items():
                if exposure > self.default_max_sector_exposure:
                    warnings.append(f"Sector exposure limit exceeded: {sector} ({exposure:.1f}%)")
            
            return {
                'sector_exposure': sector_exposure,
                'warnings': warnings,
                'total_exposure': sum(sector_exposure.values())
            }
            
        except Exception as e:
            st.error(f"Error checking sector exposure: {str(e)}")
            return {'sector_exposure': {}, 'warnings': [], 'total_exposure': 0}

class DrawdownProtector:
    """
    Manages portfolio drawdown protection
    """
    
    def __init__(self, max_drawdown_percent: float = 10.0):
        self.max_drawdown_percent = max_drawdown_percent
    
    def calculate_current_drawdown(self, current_balance: float, peak_balance: float) -> float:
        """Calculate current drawdown percentage"""
        if peak_balance <= 0:
            return 0.0
        return ((peak_balance - current_balance) / peak_balance) * 100
    
    def get_risk_multiplier(self, current_drawdown: float) -> float:
        """Get risk multiplier based on current drawdown"""
        if current_drawdown <= 0:
            return 1.0
        elif current_drawdown >= self.max_drawdown_percent:
            return 0.1  # Minimum risk
        else:
            # Linear scaling from 1.0 to 0.1
            return 1.0 - (current_drawdown / self.max_drawdown_percent) * 0.9
    
    def should_reduce_risk(self, current_balance: float, peak_balance: float) -> Tuple[bool, float, float]:
        """Check if risk should be reduced due to drawdown"""
        drawdown = self.calculate_current_drawdown(current_balance, peak_balance)
        multiplier = self.get_risk_multiplier(drawdown)
        
        return drawdown > 5.0, drawdown, multiplier  # Start reducing at 5% drawdown

def create_risk_management_interface(risk_manager: RiskManager, symbols: List[str], data_manager=None):
    """
    Create the risk management interface in Streamlit
    """
    st.header("⚠️ Risk Management Dashboard")
    
    # Portfolio Configuration Section
    st.subheader("💼 Portfolio Configuration")
    
    if symbols:
        st.write(f"**Watchlist Symbols:** {', '.join(symbols)}")
        
        # Portfolio allocation interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**Portfolio Allocation (%):**")
            allocation_changed = False
            
            # Initialize allocations if not set
            if not st.session_state.portfolio_allocation or set(symbols) != set(st.session_state.portfolio_allocation.keys()):
                equal_weight = 100.0 / len(symbols)
                st.session_state.portfolio_allocation = {symbol: equal_weight for symbol in symbols}
            
            # Create allocation sliders
            allocation_cols = st.columns(min(len(symbols), 3))
            for i, symbol in enumerate(symbols):
                with allocation_cols[i % 3]:
                    current_allocation = st.session_state.portfolio_allocation.get(symbol, 0)
                    new_allocation = st.slider(
                        f"{symbol}", 
                        min_value=0.0, 
                        max_value=50.0, 
                        value=current_allocation, 
                        step=1.0,
                        key=f"allocation_{symbol}"
                    )
                    if new_allocation != current_allocation:
                        st.session_state.portfolio_allocation[symbol] = new_allocation
                        allocation_changed = True
        
        with col2:
            total_allocation = sum(st.session_state.portfolio_allocation.values())
            if abs(total_allocation - 100.0) > 0.1:
                st.warning(f"⚠️ Total allocation: {total_allocation:.1f}%")
                if st.button("🔄 Normalize to 100%"):
                    # Normalize allocations to sum to 100%
                    if total_allocation > 0:
                        for symbol in symbols:
                            st.session_state.portfolio_allocation[symbol] = (
                                st.session_state.portfolio_allocation[symbol] / total_allocation * 100
                            )
                        st.rerun()
            else:
                st.success(f"✅ Total allocation: {total_allocation:.1f}%")
            
            if st.button("⚖️ Equal Weight"):
                equal_weight = 100.0 / len(symbols)
                for symbol in symbols:
                    st.session_state.portfolio_allocation[symbol] = equal_weight
                st.rerun()
    else:
        st.info("Add symbols to your watchlist to configure your portfolio.")
    
    # Risk Parameters Section
    st.subheader("🎛️ Risk Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_percent = st.slider("Risk per Trade (%)", 0.5, 5.0, risk_manager.default_risk_percent, 0.1)
        atr_multiplier = st.slider("ATR Multiplier", 1.0, 3.0, risk_manager.default_atr_multiplier, 0.1)
    
    with col2:
        max_drawdown = st.slider("Max Drawdown (%)", 5.0, 20.0, risk_manager.default_max_drawdown, 1.0)
        max_sector_exposure = st.slider("Max Sector Exposure (%)", 10.0, 50.0, risk_manager.default_max_sector_exposure, 5.0)
    
    with col3:
        correlation_threshold = st.slider("Correlation Threshold", 0.5, 0.9, risk_manager.default_correlation_threshold, 0.05)
        account_balance = st.number_input("Account Balance ($)", min_value=1000.0, value=st.session_state.portfolio_balance, step=1000.0)
    
    # Update session state
    st.session_state.portfolio_balance = account_balance
    if account_balance > st.session_state.portfolio_peak:
        st.session_state.portfolio_peak = account_balance
    
    # Portfolio Risk Gauge
    st.subheader("📊 Portfolio Risk Gauge")
    
    drawdown_protector = DrawdownProtector(max_drawdown)
    should_reduce, current_drawdown, risk_multiplier = drawdown_protector.should_reduce_risk(
        st.session_state.portfolio_balance, 
        st.session_state.portfolio_peak
    )
    
    # Risk gauge visualization
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = current_drawdown,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Current Drawdown (%)"},
        delta = {'reference': 0},
        gauge = {
            'axis': {'range': [None, 20]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 5], 'color': "lightgray"},
                {'range': [5, 10], 'color': "yellow"},
                {'range': [10, 20], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_drawdown
            }
        }
    ))
    
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Drawdown Protection Status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Peak Balance", f"${st.session_state.portfolio_peak:,.2f}")
    
    with col2:
        st.metric("Current Balance", f"${st.session_state.portfolio_balance:,.2f}")
    
    with col3:
        st.metric("Current Drawdown", f"{current_drawdown:.1f}%", 
                 delta=f"{current_drawdown:.1f}%" if current_drawdown > 0 else None)
    
    with col4:
        st.metric("Risk Multiplier", f"{risk_multiplier:.2f}", 
                 delta=f"{(risk_multiplier-1)*100:+.0f}%" if risk_multiplier != 1.0 else None)
    
    # Warning System
    if should_reduce:
        st.warning(f"⚠️ Drawdown protection active! Position sizes reduced by {(1-risk_multiplier)*100:.0f}%")
    
    # Sector Exposure Analysis
    if symbols:
        st.subheader("🏭 Sector Exposure Analysis")
        
        # Use actual portfolio positions from watchlist
        position_values = risk_manager.sync_portfolio_with_watchlist(symbols, data_manager)
        
        # Display current positions
        if position_values:
            st.write("**Current Position Values:**")
            pos_cols = st.columns(min(len(position_values), 4))
            for i, (symbol, value) in enumerate(position_values.items()):
                with pos_cols[i % 4]:
                    allocation_pct = st.session_state.portfolio_allocation.get(symbol, 0)
                    st.metric(
                        label=symbol,
                        value=f"${value:,.0f}",
                        delta=f"{allocation_pct:.1f}%"
                    )
        
        sector_analysis = risk_manager.check_sector_exposure(symbols, position_values)
        
        if sector_analysis['sector_exposure']:
            # Sector pie chart
            fig_sector = px.pie(
                values=list(sector_analysis['sector_exposure'].values()),
                names=list(sector_analysis['sector_exposure'].keys()),
                title="Current Sector Allocation"
            )
            fig_sector.update_layout(height=400)
            st.plotly_chart(fig_sector, use_container_width=True)
            
            # Sector warnings
            for warning in sector_analysis['warnings']:
                st.warning(f"🚨 {warning}")
    
    # Correlation Matrix
    if len(symbols) > 1:
        st.subheader("🔗 Portfolio Correlation Analysis")
        
        correlation_data = risk_manager.check_portfolio_correlation(symbols)
        
        if not correlation_data['correlation_matrix'].empty:
            # Correlation heatmap
            fig_corr = px.imshow(
                correlation_data['correlation_matrix'],
                text_auto=True,
                aspect="auto",
                title="Portfolio Correlation Matrix",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1
            )
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Correlation warnings
            for warning in correlation_data['warnings']:
                st.warning(f"🚨 {warning}")
    
    # Portfolio Performance Tracking
    if symbols and position_values:
        st.subheader("📈 Portfolio Performance")
        
        # Calculate total portfolio value
        total_portfolio_value = sum(position_values.values())
        portfolio_return = ((total_portfolio_value - st.session_state.portfolio_balance) / st.session_state.portfolio_balance) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Portfolio Value", f"${total_portfolio_value:,.0f}")
        
        with col2:
            st.metric("Initial Balance", f"${st.session_state.portfolio_balance:,.0f}")
        
        with col3:
            st.metric("Total Return", f"{portfolio_return:+.2f}%")
        
        with col4:
            num_positions = len([v for v in position_values.values() if v > 0])
            st.metric("Active Positions", num_positions)
        
        # Portfolio allocation pie chart
        if len(position_values) > 1:
            fig_portfolio = px.pie(
                values=list(position_values.values()),
                names=list(position_values.keys()),
                title="Current Portfolio Allocation"
            )
            fig_portfolio.update_layout(height=400)
            st.plotly_chart(fig_portfolio, use_container_width=True)
    
    # Time-based Risk Adjustment
    st.subheader("⏰ Time-based Risk Adjustment")
    
    current_time = datetime.now().time()
    market_hours_adjustment = get_time_risk_adjustment(current_time)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Current Time", current_time.strftime("%H:%M ET"))
    
    with col2:
        st.metric("Time Risk Multiplier", f"{market_hours_adjustment:.0%}")
    
    if market_hours_adjustment < 1.0:
        st.info(f"ℹ️ Position sizes reduced due to end-of-day risk management")
    
    return {
        'risk_percent': risk_percent,
        'atr_multiplier': atr_multiplier,
        'max_drawdown': max_drawdown,
        'max_sector_exposure': max_sector_exposure,
        'correlation_threshold': correlation_threshold,
        'account_balance': account_balance,
        'risk_multiplier': risk_multiplier,
        'time_multiplier': market_hours_adjustment
    }

def get_time_risk_adjustment(current_time: time) -> float:
    """
    Calculate time-based risk adjustment
    """
    market_open = time(9, 30)
    market_close = time(16, 0)
    late_day_start = time(14, 0)  # Last 2 hours
    very_late_start = time(15, 0)  # Last hour
    
    if current_time < market_open or current_time > market_close:
        return 0.0  # Market closed
    elif current_time >= very_late_start:
        return 0.5  # 50% in last hour
    elif current_time >= late_day_start:
        return 0.75  # 75% in last 2 hours
    else:
        return 1.0  # 100% during normal hours

def calculate_enhanced_signal_with_risk(signal_data: Dict, risk_manager: RiskManager, 
                                      data: pd.DataFrame, risk_params: Dict) -> Dict:
    """
    Enhance signal with comprehensive risk management
    """
    try:
        symbol = signal_data.get('symbol', '')
        current_price = signal_data.get('price', 0)
        signal_type = signal_data.get('signal', 'NEUTRAL')
        volume = signal_data.get('volume', 0)
        
        if signal_type == 'NEUTRAL' or current_price <= 0:
            return signal_data
        
        # Calculate ATR
        atr = risk_manager.get_atr(data)
        
        # Calculate stop loss
        stop_loss = risk_manager.calculate_stop_loss(
            current_price, signal_type, atr, risk_params['atr_multiplier']
        )
        
        # Calculate risk score
        risk_score_data = risk_manager.calculate_risk_score(symbol, current_price, volume, data)
        
        # Calculate position size
        base_position_size = risk_manager.calculate_position_size(
            risk_params['account_balance'],
            risk_params['risk_percent'],
            current_price,
            stop_loss
        )
        
        # Apply risk multipliers
        final_position_size = int(
            base_position_size * 
            risk_params['risk_multiplier'] * 
            risk_params['time_multiplier'] *
            risk_score_data['score']  # Risk score adjustment
        )
        
        # Calculate position value and risk
        position_value = final_position_size * current_price
        risk_amount = final_position_size * abs(current_price - stop_loss)
        
        # Enhanced signal data
        enhanced_signal = {
            **signal_data,
            'atr': atr,
            'atr_percent': (atr / current_price) * 100 if current_price > 0 else 0,
            'stop_loss': stop_loss,
            'stop_loss_percent': abs((stop_loss - current_price) / current_price) * 100,
            'position_size': final_position_size,
            'position_value': position_value,
            'risk_amount': risk_amount,
            'risk_score': risk_score_data['score'],
            'risk_level': risk_score_data['risk_level'],
            'risk_color': risk_score_data['color'],
            'risk_factors': risk_score_data['factors'],
            'volatility_pct': risk_score_data['volatility_pct']
        }
        
        return enhanced_signal
        
    except Exception as e:
        st.error(f"Error enhancing signal with risk data: {str(e)}")
        return signal_data

def display_enhanced_signal_card(signal_data: Dict, real_time_data: Dict):
    """
    Display enhanced signal card with risk management information using Streamlit components
    """
    symbol = signal_data.get('symbol', '')
    signal_type = signal_data.get('signal', 'NEUTRAL')
    risk_level = signal_data.get('risk_level', 'UNKNOWN')
    
    # Color coding based on risk level
    border_colors = {'HIGH': '#28a745', 'MEDIUM': '#ffc107', 'LOW': '#dc3545', 'UNKNOWN': '#6c757d'}
    border_color = border_colors.get(risk_level, '#6c757d')
    
    change = real_time_data.get('change', 0)
    change_percent = real_time_data.get('change_percent', 0)
    current_price = real_time_data.get('current_price', 0)
    
    # Create container with custom styling
    with st.container():
        # Custom CSS for the card
        st.markdown(f"""
        <style>
        .signal-card {{
            border: 3px solid {border_color};
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        }}
        .signal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        .risk-badge {{
            background: {border_color};
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 10px;
            font-weight: bold;
        }}
        </style>
        """, unsafe_allow_html=True)
        
        # Header with symbol and risk level
        col1, col2 = st.columns([3, 1])
        with col1:
            signal_color = {"BUY": "🟢", "SELL": "🔴", "NEUTRAL": "⚪"}[signal_type]
            st.markdown(f"### {signal_color} {symbol}")
        with col2:
            st.markdown(f'<span class="risk-badge">{risk_level}</span>', unsafe_allow_html=True)
        
        # Price information
        price_color = "green" if change >= 0 else "red"
        st.markdown(f"**${current_price:.2f}**")
        st.markdown(f":{price_color}[{change:+.2f} ({change_percent:+.1f}%)]")
        
        # Signal information
        signal_emoji = {"BUY": "📈", "SELL": "📉", "NEUTRAL": "➖"}[signal_type]
        st.markdown(f"**Signal:** {signal_emoji} {signal_type}")
        
        # Risk management metrics
        if signal_data.get('position_size', 0) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Position Size", f"{signal_data.get('position_size', 0):,} shares")
                st.metric("Stop Loss", f"${signal_data.get('stop_loss', 0):.2f}")
                st.metric("ATR", f"${signal_data.get('atr', 0):.2f}")
            
            with col2:
                st.metric("Position Value", f"${signal_data.get('position_value', 0):,.0f}")
                st.metric("Risk Amount", f"${signal_data.get('risk_amount', 0):.0f}")
                st.metric("Risk Score", f"{signal_data.get('risk_score', 0):.2f}/1.0")
        
        # Additional information
        with st.expander("Additional Details"):
            st.write(f"**Day's Range:** ${real_time_data.get('day_low', 0):.2f} - ${real_time_data.get('day_high', 0):.2f}")
            st.write(f"**52W Range:** ${real_time_data.get('fifty_two_week_low', 0):.2f} - ${real_time_data.get('fifty_two_week_high', 0):.2f}")
            st.write(f"**Distance from round:** {signal_data.get('distance_from_round', 0):+.3f}")
            if signal_data.get('atr_percent'):
                st.write(f"**ATR %:** {signal_data.get('atr_percent', 0):.1f}%")
            if signal_data.get('stop_loss_percent'):
                st.write(f"**Stop Loss %:** {signal_data.get('stop_loss_percent', 0):.1f}%")
