import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Import configuration
from config.settings import DEFAULT_WATCHLIST, PAGE_CONFIG

# Import core modules
from core.analyzer import StockClusteringAnalyzer
from core.data_manager import DataManager

# Import visualization
from visualization.charts import create_price_chart, create_performance_chart

# Import analysis
from analysis.performance import analyze_historical_performance

# Import UI components
from ui.dashboard import (
    create_sidebar_config,
    handle_auto_refresh,
    create_buy_scanner,
    create_watchlist_monitoring,
    display_active_signals,
    create_documentation_section
)

# Import risk management components
from risk_management import (
    RiskManager, 
    create_risk_management_interface
)

# Configure Streamlit page
st.set_page_config(**PAGE_CONFIG)


def create_detailed_analysis_tabs(selected_symbol: str, data_manager: DataManager, 
                                analyzer: StockClusteringAnalyzer, analysis_period: str, 
                                holding_period: int):
    """Create detailed analysis tabs for selected symbol"""
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
    risk_manager = RiskManager()

    # Create sidebar configuration
    config = create_sidebar_config(DEFAULT_WATCHLIST)

    # Handle auto-refresh
    handle_auto_refresh(config['auto_refresh'], config['refresh_interval'])

    # Create main tabs
    main_tab1, main_tab2 = st.tabs(["📊 Trading Dashboard", "⚠️ Risk Management"])

    with main_tab2:
        # Risk Management Interface
        risk_params = create_risk_management_interface(risk_manager, config['symbols'])

    with main_tab1:
        # Main dashboard
        if config['symbols']:
            # Buy scanner section
            create_buy_scanner(analyzer, data_manager)

            # Watchlist monitoring
            current_signals = create_watchlist_monitoring(
                config['symbols'], analyzer, data_manager, risk_manager, 
                risk_params if 'risk_params' in locals() else None
            )

            # Active signals summary
            display_active_signals(current_signals)

        # Detailed analysis section
        st.header("📊 Detailed Analysis")

        if config['symbols']:
            selected_symbol = st.selectbox(
                "Select symbol for detailed analysis:", 
                config['symbols'], 
                key="detailed_analysis_symbol"
            )

            if selected_symbol:
                create_detailed_analysis_tabs(
                    selected_symbol, data_manager, analyzer, 
                    config['analysis_period'], config['holding_period']
                )

    # Documentation section
    create_documentation_section()


if __name__ == "__main__":
    main()
# Import risk management components  
from risk_management import (
    DrawdownProtector, 
    calculate_enhanced_signal_with_risk,
    display_enhanced_signal_card
)