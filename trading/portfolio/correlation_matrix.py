"""
Strategy Correlation Matrix & Risk Cluster Analysis.

Computes:
1. Strategy Return Matrix
2. Pairwise Strategy Correlation Matrix
3. Trade Overlap Matrix (co-occurrence of active trades)
4. Risk Clustering (prevents allocating across strategies that fail in the same events)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple


class StrategyCorrelationMatrix:
    """Analyzes correlations across strategy returns and active trade overlaps."""

    @staticmethod
    def compute_correlation_matrix(strategy_returns: Dict[str, List[float]]) -> pd.DataFrame:
        """
        Compute pairwise correlation matrix across strategies.
        """
        if not strategy_returns:
            return pd.DataFrame()

        df = pd.DataFrame(strategy_returns).fillna(0.0)
        return df.corr().fillna(0.0)

    @staticmethod
    def compute_trade_overlap(
        strategy_positions: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """
        Computes Jaccard trade overlap percentage between strategy portfolios.
        strategy_positions: dict of strat_name -> list of held symbols across bars
        """
        names = list(strategy_positions.keys())
        overlap_matrix = pd.DataFrame(index=names, columns=names, dtype=float)

        for s1 in names:
            set1 = set(strategy_positions.get(s1, []))
            for s2 in names:
                set2 = set(strategy_positions.get(s2, []))
                if not set1 and not set2:
                    overlap_matrix.loc[s1, s2] = 0.0
                else:
                    intersection = len(set1 & set2)
                    union = len(set1 | set2)
                    overlap_matrix.loc[s1, s2] = round(intersection / max(1, union), 3)

        return overlap_matrix

    @staticmethod
    def identify_risk_clusters(
        corr_matrix: pd.DataFrame, threshold: float = 0.70
    ) -> List[List[str]]:
        """
        Group strategies with return correlation >= threshold into distinct risk clusters.
        """
        if corr_matrix.empty:
            return []

        strategies = list(corr_matrix.columns)
        assigned = set()
        clusters = []

        for s1 in strategies:
            if s1 in assigned:
                continue
            cluster = [s1]
            assigned.add(s1)
            for s2 in strategies:
                if s2 not in assigned and corr_matrix.loc[s1, s2] >= threshold:
                    cluster.append(s2)
                    assigned.add(s2)
            clusters.append(cluster)

        return clusters
