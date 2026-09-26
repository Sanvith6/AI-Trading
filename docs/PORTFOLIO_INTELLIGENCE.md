# Portfolio Intelligence & Execution Capacity Models

## Risk Parity / Equal Risk Contribution (ERC)
Equal Risk Contribution allocates weights such that each asset contributes equally to the total portfolio volatility:
$$RC_i = w_i \cdot \frac{(\Sigma w)_i}{\sqrt{w^T \Sigma w}} = \frac{\sigma_{port}}{N}$$
Solved using convex optimization (SciPy SLSQP) subject to:
- Maximum position weight: $25\%$
- Maximum sector weight: $40\%$
- Minimum cash: $5\%$
- Correlation cluster cap: $40\%$

## Square-Root Law Market Impact Capacity
For scaling AUM from $\$10\text{K}$ to $\$10\text{M}$:
$$\text{Impact (bps)} = Y \cdot \sigma_{daily} \cdot \sqrt{\frac{\text{Order Size}}{\text{ADV}}} \cdot 10,000$$
Where $Y = 0.60$ for liquid US equities. Tracks participation rate (capped at $2\%$ ADV) and net Sharpe degradation after slippage drag.
