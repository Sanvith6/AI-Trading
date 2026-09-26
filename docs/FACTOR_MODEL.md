# Multi-Factor Exposure & Style Model

## Factor Taxonomy
The system decomposes equity returns into 6 systematic risk factors plus Market Beta:
1. **Market Beta ($\beta$)**: Covariance with SPY (US) or Nifty 50 (India) over variance.
2. **Momentum**: 20-day and 252-day relative strength cross-sectional Z-score.
3. **Size**: Log market-capitalization proxy (Large vs Small cap tilt).
4. **Value**: Earnings Yield ($E/P$) and Book-to-Market Z-score.
5. **Quality**: ROE, Operating Margin, and Free Cash Flow margin composite.
6. **Volatility**: 60-day annualized realized volatility Z-score.
7. **Growth**: 1-year YoY revenue expansion rate.

## Alpha / Beta Decomposition Formula
For portfolio return $R_p$ over benchmark $R_m$:
$$R_p = \beta_{port} \cdot R_m + \alpha_{selection}$$
This separates systemic market tailwinds from genuine algorithmic stock-selection skill.
