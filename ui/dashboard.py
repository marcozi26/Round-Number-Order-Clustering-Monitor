"""
Dashboard UI components and layouts
"""
import streamlit as st
import pandas as pd
import time
from datetime import datetime
from typing import List, Dict
from config.settings import STOCK_UNIVERSE
from core.data_manager import DataManager
from core.analyzer import StockClusteringAnalyzer
from risk_management import (
    calculate_enhanced_signal_with_risk,
    display_enhanced_signal_card
)


def create_data_sources_status(data_manager: DataManager) -> None:
    """Create data sources status monitoring section"""
    st.subheader("📡 Data Sources Status")
    
    try:
        provider_status = data_manager.get_provider_status()
        
        # Market status
        market_status = provider_status['market_status']
        is_open = provider_status['is_market_open']
        is_premarket = provider_status['is_premarket']
        is_afterhours = provider_status['is_afterhours']
        
        # Display market status
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if is_open:
                st.success(f"🟢 Market: {market_status}")
            else:
                st.info(f"🔴 Market: {market_status}")
        
        with col2:
            if is_premarket:
                st.warning("🌅 Pre-Market Active")
            else:
                st.empty()
        
        with col3:
            if is_afterhours:
                st.warning("🌙 After-Hours Active")
            else:
                st.empty()
        
        with col4:
            refresh_rate = "1 min" if is_open else "5 min"
            st.info(f"⏱️ Refresh: {refresh_rate}")
        
        # Provider status
        st.write("**Data Provider Status:**")
        provider_cols = st.columns(len(provider_status['providers']))
        
        for i, (provider_name, is_healthy) in enumerate(provider_status['providers'].items()):
            with provider_cols[i]:
                if is_healthy:
                    st.success(f"✅ {provider_name}")
                else:
                    st.error(f"❌ {provider_name}")
        
        # Add provider details in expander
        with st.expander("📊 Provider Details"):
            st.write("**Primary Providers:**")
            st.write("• **Yahoo Finance**: Free, reliable, good for most stocks")
            st.write("• **Alpha Vantage**: Professional data, requires API key")
            st.write("• **Finnhub**: Real-time data, requires API key")
            
            st.write("\n**Fallback Strategy:**")
            st.write("1. Try Yahoo Finance first (fastest, most reliable)")
            st.write("2. Fallback to Alpha Vantage if Yahoo fails")
            st.write("3. Fallback to Finnhub as last resort")
            st.write("4. Cache successful responses to reduce API calls")
            
            if not provider_status['providers'].get('Alpha Vantage', False):
                st.info("💡 **Tip**: Add Alpha Vantage API key in Secrets for backup data source")
            
            if not provider_status['providers'].get('Finnhub', False):
                st.info("💡 **Tip**: Add Finnhub API key in Secrets for real-time data")
    
    except Exception as e:
        st.error(f"Error retrieving data sources status: {str(e)}")
        st.info("Using fallback data source (Yahoo Finance only)")


def create_sidebar_config(default_symbols: List[str]) -> Dict:
    """Create sidebar configuration interface"""
    st.sidebar.header("Configuration")

    # Watchlist management
    st.sidebar.subheader("📋 Watchlist")

    # Get user input for symbols
    symbols_input = st.sidebar.text_area(
        "Enter stock symbols (one per line):",
        value="\n".join(default_symbols),
        height=120
    )

    symbols = [s.strip().upper() for s in symbols_input.split('\n') if s.strip()]

    # Strategy settings
    st.sidebar.subheader("⚙️ Strategy Settings")
    holding_period = st.sidebar.slider("Holding Period (days)", 1, 10, 5)
    analysis_period = st.sidebar.selectbox("Analysis Period", ["1mo", "3mo", "6mo", "1y"], index=2)

    # Auto-refresh settings
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 30, 300, 60)

    return {
        'symbols': symbols,
        'holding_period': holding_period,
        'analysis_period': analysis_period,
        'auto_refresh': auto_refresh,
        'refresh_interval': refresh_interval
    }


def handle_auto_refresh(auto_refresh: bool, refresh_interval: int):
    """Handle auto-refresh functionality"""
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


def create_buy_scanner(analyzer: StockClusteringAnalyzer, data_manager: DataManager) -> None:
    """Create the buy opportunities scanner section"""
    st.header("🎯 Potential Buy Stocks Scanner")

    if st.button("🔍 Scan for Buy Opportunities", type="primary"):
        with st.spinner("Scanning stocks for buy signals..."):
            potential_buys = []

            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, symbol in enumerate(STOCK_UNIVERSE):
                try:
                    # Update progress
                    progress = (i + 1) / len(STOCK_UNIVERSE)
                    progress_bar.progress(progress)
                    status_text.text(f"Scanning {symbol}... ({i+1}/{len(STOCK_UNIVERSE)})")

                    # Get real-time data with retry logic
                    max_retries = 2
                    retry_count = 0
                    real_time_data = None
                    
                    while retry_count < max_retries and real_time_data is None:
                        try:
                            real_time_data = data_manager.get_real_time_price(symbol)
                            if real_time_data['current_price'] <= 0:
                                real_time_data = None
                                raise ValueError("Invalid price data")
                        except Exception as retry_error:
                            retry_count += 1
                            if retry_count < max_retries:
                                time.sleep(1.0)  # Wait longer before retry
                            else:
                                status_text.text(f"Failed to get data for {symbol} after {max_retries} attempts")

                    if real_time_data and real_time_data['current_price'] > 0:
                        # Generate signal
                        signal_data = analyzer.generate_signal(real_time_data['current_price'])

                        # Check if it's a buy signal
                        if signal_data['signal'] == 'BUY':
                            potential_buys.append({
                                'Symbol': symbol,
                                'Current Price': real_time_data['current_price'],
                                'Distance from Round': signal_data['distance_from_round'],
                                'Confidence': signal_data['confidence'],
                                'Change %': real_time_data['change_percent'],
                                'Volume': real_time_data.get('volume', 0),
                                'Reasoning': signal_data['reasoning']
                            })

                    # Increased delay and better rate limiting
                    time.sleep(0.5)  # 500ms delay between requests

                except Exception as e:
                    # Log specific error types for debugging
                    error_msg = str(e)
                    if "rate limit" in error_msg.lower() or "too many requests" in error_msg.lower():
                        status_text.text(f"Rate limit hit for {symbol}, waiting...")
                        time.sleep(2.0)  # Longer wait for rate limits
                    continue  # Skip problematic symbols

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

            display_buy_scanner_results(potential_buys)


def display_buy_scanner_results(potential_buys: List[Dict]) -> None:
    """Display buy scanner results"""
    if potential_buys:
        # Sort by confidence (highest first)
        potential_buys.sort(key=lambda x: x['Confidence'], reverse=True)

        # Limit to top 20
        top_buys = potential_buys[:20]

        st.success(f"Found {len(potential_buys)} stocks with buy signals. Showing top 20:")

        # Create DataFrame for display
        buy_df = pd.DataFrame(top_buys)

        # Format columns for better display
        buy_df['Current Price'] = buy_df['Current Price'].apply(lambda x: f"${x:.2f}")
        buy_df['Distance from Round'] = buy_df['Distance from Round'].apply(lambda x: f"{x:+.3f}")
        buy_df['Confidence'] = buy_df['Confidence'].apply(lambda x: f"{x:.1%}")
        buy_df['Change %'] = buy_df['Change %'].apply(lambda x: f"{x:+.2f}%")
        buy_df['Volume'] = buy_df['Volume'].apply(lambda x: f"{x:,.0f}" if x > 0 else "N/A")

        # Display table
        st.dataframe(
            buy_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                "Current Price": st.column_config.TextColumn("Price", width="small"),
                "Distance from Round": st.column_config.TextColumn("Distance", width="small"),
                "Confidence": st.column_config.TextColumn("Confidence", width="small"),
                "Change %": st.column_config.TextColumn("Change %", width="small"),
                "Volume": st.column_config.TextColumn("Volume", width="medium"),
                "Reasoning": st.column_config.TextColumn("Reasoning", width="large")
            }
        )

        display_scanner_insights(potential_buys)

    else:
        st.warning("No stocks currently meet the buy criteria (X.1 pattern). Try again later as market conditions change.")
        st.info("The X.1 pattern requires stocks to close between 0.05 and 0.15 above round numbers (e.g., $100.05 to $100.15)")


def display_scanner_insights(potential_buys: List[Dict]) -> None:
    """Display additional insights from scanner results"""
    # Additional insights
    col1, col2, col3 = st.columns(3)

    with col1:
        avg_confidence = sum(x['Confidence'] for x in potential_buys) / len(potential_buys)
        st.metric("Average Confidence", f"{avg_confidence:.1%}")

    with col2:
        positive_change = sum(1 for x in potential_buys if x['Change %'] > 0)
        st.metric("Stocks Up Today", f"{positive_change}/{len(potential_buys)}")

    with col3:
        strong_signals = sum(1 for x in potential_buys if x['Confidence'] > 0.7)
        st.metric("High Confidence Signals", f"{strong_signals}")

    # Export functionality
    csv_data = pd.DataFrame(potential_buys).to_csv(index=False)
    st.download_button(
        label="📊 Download Buy List as CSV",
        data=csv_data,
        file_name=f"buy_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        type="secondary"
    )


def create_watchlist_monitoring(symbols: List[str], analyzer: StockClusteringAnalyzer, 
                               data_manager: DataManager, risk_manager, risk_params) -> List[Dict]:
    """Create the watchlist monitoring section"""
    st.header("🔴 Live Monitoring Dashboard")

    current_signals = []
    delisted_symbols = []

    if symbols:
        # Create columns for the watchlist
        cols = st.columns(min(len(symbols), 3))

        for idx, symbol in enumerate(symbols):
            with cols[idx % 3]:
                # Get real-time data
                real_time_data = data_manager.get_real_time_price(symbol)

                # Check if stock is delisted
                if real_time_data.get('delisted', False):
                    delisted_symbols.append({
                        'symbol': symbol,
                        'date': real_time_data.get('delisted_date', 'Unknown'),
                        'reason': real_time_data.get('delisted_reason', 'Unknown'),
                        'replacement': real_time_data.get('replacement', None)
                    })
                    
                    # Display delisted stock warning
                    st.error(f"⚠️ **{symbol} DELISTED**")
                    st.caption(f"Delisted: {real_time_data.get('delisted_date', 'Unknown')}")
                    st.caption(f"Reason: {real_time_data.get('delisted_reason', 'Unknown')}")
                    if real_time_data.get('replacement'):
                        st.info(f"Consider: {real_time_data.get('replacement')}")
                    continue

                if real_time_data['current_price'] > 0:
                    # Generate basic signal
                    signal_data = analyzer.generate_signal(real_time_data['current_price'])
                    signal_data['symbol'] = symbol
                    signal_data['volume'] = real_time_data.get('volume', 0)

                    # Get historical data for risk calculations
                    historical_data = data_manager.fetch_stock_data(symbol, "1mo")

                    # Enhance signal with risk management if risk params are available
                    if risk_params and not historical_data.empty:
                        enhanced_signal = calculate_enhanced_signal_with_risk(
                            signal_data, risk_manager, historical_data, risk_params
                        )
                        current_signals.append(enhanced_signal)

                        # Display enhanced signal card
                        display_enhanced_signal_card(enhanced_signal, real_time_data)
                    else:
                        current_signals.append(signal_data)
                        display_basic_signal_card(signal_data, real_time_data)
        # Display delisted stocks warning if any
        if delisted_symbols:
            st.divider()
            st.subheader("⚠️ Delisted Stocks Alert")
            
            delisted_df = pd.DataFrame(delisted_symbols)
            st.error("The following stocks in your watchlist have been delisted:")
            st.dataframe(delisted_df, use_container_width=True, hide_index=True)
            
            st.warning("**Action Required:** Remove these symbols from your watchlist and consider alternatives.")
            
            if st.button("🗑️ Remove All Delisted Stocks from Watchlist"):
                # This would require a callback to modify the watchlist
                st.info("Please manually remove delisted stocks from your watchlist configuration.")

        # Use actual portfolio positions from watchlist (exclude delisted)
        active_symbols = [s for s in symbols if s not in [d['symbol'] for d in delisted_symbols]]
        portfolio_positions = risk_manager.sync_portfolio_with_watchlist(active_symbols, data_manager)
        
        # Display current positions
        if portfolio_positions:
            st.write("**Current Positions:**")
            pos_cols = st.columns(min(len(portfolio_positions), 4))
            for i, (symbol, position) in enumerate(portfolio_positions.items()):
                with pos_cols[i % 4]:
                    current_value = position['current_value']
                    shares = position['shares']
                    unrealized_pnl = position['unrealized_pnl']
                    unrealized_pnl_pct = position['unrealized_pnl_pct']

                    st.metric(
                        label=symbol,
                        value=f"${current_value:,.0f}",
                        delta=f"{unrealized_pnl:+,.0f} ({unrealized_pnl_pct:+.1f}%)",
                    )
                    st.caption(f"{shares:,.0f} Shares")
    else:
        st.info("Please add some stock symbols to your watchlist in the sidebar to start monitoring.")

    return current_signals


def display_basic_signal_card(signal_data: Dict, real_time_data: Dict) -> None:
    """Display basic signal card for fallback"""
    change = real_time_data['change']
    change_percent = real_time_data['change_percent']
    current_price = real_time_data['current_price']
    symbol = signal_data['symbol']

    # Use Streamlit components instead of HTML
    signal_emoji = {"BUY": "📈", "SELL": "📉", "NEUTRAL": "➖"}[signal_data['signal']]

    with st.container():
        st.markdown(f"### {signal_emoji} {symbol}")
        st.markdown(f"**${current_price:.2f}**")

        # Change indicator
        if change >= 0:
            st.success(f"+{change:.2f} (+{change_percent:.1f}%)")
        else:
            st.error(f"{change:.2f} ({change_percent:.1f}%)")

        st.markdown(f"**Signal:** {signal_data['signal']}")

        # Additional details in expander
        with st.expander("Details"):
            st.write(f"**Day's Range:** ${real_time_data['day_low']:.2f} - ${real_time_data['day_high']:.2f}")
            st.write(f"**52W Range:** ${real_time_data['fifty_two_week_low']:.2f} - ${real_time_data['fifty_two_week_high']:.2f}")
            st.write(f"**Distance from round:** {signal_data['distance_from_round']:+.3f}")
            st.write(f"**Confidence:** {signal_data['confidence']:.1%}")

        st.divider()  # Add separator between cards


def display_active_signals(current_signals: List[Dict]) -> None:
    """Display active trading signals summary"""
    active_signals = [s for s in current_signals if s['signal'] != 'NEUTRAL']

    if active_signals:
        st.header("🚨 Active Trading Signals")

        # Create enhanced signals table
        if active_signals and 'risk_score' in active_signals[0]:
            # Enhanced signals with risk data
            enhanced_df = pd.DataFrame(active_signals)
            display_columns = ['symbol', 'signal', 'price', 'position_size', 'risk_amount', 'risk_level', 'confidence']
            available_columns = [col for col in display_columns if col in enhanced_df.columns]
            st.dataframe(enhanced_df[available_columns], use_container_width=True)
        else:
            # Basic signals
            signal_df = pd.DataFrame(active_signals)
            basic_columns = ['symbol', 'signal', 'price', 'distance_from_round', 'confidence', 'reasoning']
            available_columns = [col for col in basic_columns if col in signal_df.columns]
            st.dataframe(signal_df[available_columns], use_container_width=True)


def create_documentation_section():
    """Create strategy documentation section"""
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