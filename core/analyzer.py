
"""
Core stock clustering analyzer module
"""
import pandas as pd
import numpy as np
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


class StockClusteringAnalyzer:
    """
    Core class implementing the Round-Number Order Clustering strategy
    based on "Can Daily Closing Prices Predict Future Movements? 
    The Role of Limit Order Clustering" by Xiao Zhang.
    """

    def __init__(self):
        self.round_threshold = 0.1  # Distance from round number to consider clustering

    def get_round_number_distance(self, price: float) -> float:
        """
        Calculate distance from nearest round number (integer).
        Returns: Distance as decimal (e.g., 100.1 returns 0.1, 99.9 returns -0.1)
        """
        nearest_round = round(price)
        return price - nearest_round

    def classify_closing_position(self, price: float) -> str:
        """
        Classify closing price relative to round numbers.
        X.1 category: Just above round number (bullish signal)
        X.9 category: Just below round number (bearish signal)
        """
        distance = self.get_round_number_distance(price)

        if 0.05 <= distance <= 0.15:  # X.1 range (e.g., 100.05 to 100.15)
            return "X.1_BULLISH"
        elif -0.15 <= distance <= -0.05:  # X.9 range (e.g., 99.85 to 99.95)
            return "X.9_BEARISH"
        else:
            return "NEUTRAL"

    def generate_signal(self, closing_price: float) -> Dict:
        """
        Generate trading signal based on closing price clustering.
        """
        classification = self.classify_closing_position(closing_price)
        distance = self.get_round_number_distance(closing_price)

        signal_data = {
            "price": closing_price,
            "distance_from_round": distance,
            "classification": classification,
            "signal": "NEUTRAL",
            "confidence": 0.0,
            "expected_direction": "HOLD"
        }

        if classification == "X.1_BULLISH":
            signal_data.update({
                "signal": "BUY",
                "confidence": min(0.8, 0.5 + abs(distance) * 2),  # Higher confidence closer to X.1
                "expected_direction": "UP",
                "reasoning": "Price closed just above round number - potential support level"
            })
        elif classification == "X.9_BEARISH":
            signal_data.update({
                "signal": "SELL",
                "confidence": min(0.8, 0.5 + abs(distance) * 2),
                "expected_direction": "DOWN",
                "reasoning": "Price closed just below round number - potential resistance level"
            })
        else:
            signal_data["reasoning"] = "Price not in clustering zone"

        return signal_data
