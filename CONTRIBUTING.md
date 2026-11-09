# Contributing to AI-Trading

Thank you for your interest in contributing to **AI-Trading**, an institutional-grade autonomous quantitative trading and research platform.

We welcome contributions from quantitative researchers, machine learning engineers, Python developers, and traders.

---

## 🛠️ Development Setup

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/Sanvith6/AI-Trading.git
cd AI-Trading

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

By default, the platform runs in **100% Free Paper-Trading Mode** using Yahoo Finance and simulated paper ledgers without requiring any paid broker API keys.

---

## 🧪 Running Tests & Quality Verification

All contributions must pass the verification suite before merging:

```bash
# Run the entire test suite (218 unit, integration, and quantitative invariant tests)
pytest trading/tests -v

# Run the 15-point startup preflight audit
python -m trading.cli preflight
```

### Core Architecture Principles

1. **Zero Data Leakage Invariant**:
   - Quarterly fundamentals and corporate events must respect point-in-time causality (`available_at`). Never leak future data into feature generation or backtests.
2. **The LLM is a Research Orchestrator, Not an Unchecked Trader**:
   - Numerical models, portfolio theory (Kelly criterion, CVaR, Ledoit-Wolf shrinkage), and hard risk circuit breakers govern execution.
3. **Defense in Depth**:
   - Every order must pass through risk engine limits (order caps, position caps, daily loss limits, correlation guards, and kill switches).

---

## 📂 Project Organization

- `trading/`: Core trading runtime, portfolio optimization, risk management, and broker adapters.
  - `portfolio/`: Institutional portfolio constructor, covariance shrinkage, CVaR, Kelly sizing.
  - `market/`: Market data providers (Alpaca REST/WebSockets, Yahoo Finance fallback).
  - `brokers/`: Execution adapters (Alpaca Paper, local simulated ledger, Angel One, Zerodha).
  - `laya_specialist/` & `laya_specialist_v2/`: Neural financial intelligence & RL meta-decision policies.
  - `runtime/`: Orchestrator, safety guards, session lifecycle, and Prometheus metrics.
  - `tests/`: 218 unit, integration, and quantitative invariant tests.
- `reports/`: Audit reports, walk-forward results, and empirical research benchmarks.
- `docs/`: In-depth runbooks, broker setup guides, and architecture specifications.

---

## 📝 Pull Request Guidelines

1. **Feature Branches**: Create a descriptive branch (e.g., `feat/factor-momentum`, `fix/alpaca-feed`).
2. **Code Standards**: Follow PEP 8, include strict type hints, and write modular, testable code.
3. **Test Coverage**: Accompany new features with corresponding unit tests in `trading/tests/`.
4. **No Secrets**: Never commit `.env`, live API keys, tokens, or credential files.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
