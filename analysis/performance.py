
"""
Performance analysis module for the Round-Number Order Clustering strategy
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from core.analyzer import StockClusteringAnalyzer


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


def calculate_performance_metrics(signals: List[Dict]) -> Dict:
    """Calculate comprehensive performance metrics"""
    if not signals:
        return {}
    
    buy_signals = [s for s in signals if s['signal'] == 'BUY']
    sell_signals = [s for s in signals if s['signal'] == 'SELL']
    
    metrics = {}
    
    # Buy signal metrics
    if buy_signals:
        buy_returns = [s['forward_return'] for s in buy_signals]
        metrics['buy_metrics'] = {
            'total_signals': len(buy_signals),
            'avg_return': np.mean(buy_returns),
            'win_rate': sum(1 for r in buy_returns if r > 0) / len(buy_returns),
            'best_trade': max(buy_returns),
            'worst_trade': min(buy_returns),
            'std_return': np.std(buy_returns)
        }
    
    # Sell signal metrics
    if sell_signals:
        sell_returns = [s['forward_return'] for s in sell_signals]
        metrics['sell_metrics'] = {
            'total_signals': len(sell_signals),
            'avg_return': np.mean(sell_returns),
            'win_rate': sum(1 for r in sell_returns if r < 0) / len(sell_returns),  # Profitable sells are negative returns
            'best_trade': min(sell_returns),
            'worst_trade': max(sell_returns),
            'std_return': np.std(sell_returns)
        }
    
    return metrics
