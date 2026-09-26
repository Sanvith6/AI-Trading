# Point-in-Time Data & Survivorship-Bias Mitigation

## Overview
Point-in-Time (PIT) correctness guarantees that at any historical timestamp $T$, the backtest engine only accesses information that was legally, publicly, and physically available to a market participant at or before $T$.

## Core Mitigations Implemented

### 1. Inception & IPO Date Gating
Historical backtests must never evaluate or hold an asset prior to its official IPO date.
- Examples enforced:
  - `ABNB`: Listed 2020-12-10
  - `COIN`: Listed 2021-04-14
  - `PLTR`: Listed 2020-09-30
  - `SNOW`: Listed 2020-09-16
  - `ZOMATO`: Listed 2021-07-23

### 2. Delisting & Bankruptcy Filtering
Instruments that entered insolvency or were acquired must not remain tradeable after their effective delisting date.
- Examples enforced:
  - `TWTR`: Delisted 2022-10-27 (Acquired by X Corp)
  - `SBNY`: Delisted 2023-03-12 (FDIC Receivership)
  - `FRC`: Delisted 2023-05-01 (FDIC Receivership)
  - `DHFL`: Delisted 2021-06-14 (Insolvency)

### 3. Survivorship-Bias Disclosure
Free and open-source market data provides partial point-in-time filtering for high-profile IPOs and bankruptcies. Full historical index constituent reconstitution (e.g. S&P 500 daily member tracking back to 2010) requires commercial historical constituent databases (e.g., S&P Historical Constituents). The system explicitly quantifies this limitation in `get_survivorship_bias_disclosure()`.
