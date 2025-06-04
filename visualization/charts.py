
"""
Charts and visualization module
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict


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
