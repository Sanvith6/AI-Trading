# Phase 1: Quantitative Research Foundation Upgrade — Architectural Specification

## Executive Overview
Phase 1 solidifies the quantitative research foundation of the AI Trading Platform across both US Equities (NYSE/NASDAQ) and Indian Equities (NSE/BSE). It establishes institutional-grade data correctness, cryptographic experiment lineage, point-in-time causality, and portfolio intelligence without introducing unnecessary distributed infrastructure or live broker execution risk.

---

## High-Level Architecture Diagram

```mermaid
graph TD
    subgraph Market Data Layer
        YF[Yahoo Finance Provider]
        EDGAR[SEC EDGAR / XBRL Provider]
        CORP[Corporate Actions Engine]
    end

    subgraph Research Data Lake
        RAW[Raw Market & Fundamental Ingestion] --> PQ[PyArrow Parquet Store]
        PQ --> DUCK[DuckDB Zero-Copy SQL Engine]
        PQ --> POLARS[Polars High-Throughput Preprocessing]
        PQ --> REG[Dataset Registry & Cryptographic Manifests]
    end

    subgraph Quantitative Feature & Factor Engine
        FR[Feature Registry - Semantic Versioning]
        MC[US Market Context - Macro & Sectors]
        EV[Causal Event Foundation - Filings & Dates]
        FM[Multi-Factor Model - Style & Beta Decomposition]
    end

    subgraph Portfolio & Execution Intelligence
        PC[PortfolioConstructor - Equal Risk Contribution]
        MRC[Marginal Risk Contribution & Clustering]
        CAP[Capacity Engine - Square-Root Market Impact]
        SKF[Skfolio / SLSQP Optimization Adapter]
    end

    subgraph MLOps & Model Governance
        MLF[MLflow Experiment Lineage Tracker]
        OPT[Optuna Bounded Tuner - Holdout Isolated]
        DRIFT[DriftDetector - Multi-Window KS & PSI]
    end

    subgraph Verification & Guardrails
        REPRO[Reproducibility Engine - Environment Fingerprinting]
        GOLDEN[Golden Backtest Suite - Regression Guard]
        TRACE[Decision Trace & Point-in-Time Replay]
        OBS[Prometheus Observability Telemetry]
        RISK[Immutable RiskEngine - 13 Non-Bypassable Gates]
    end

    DUCK --> FR
    POLARS --> FR
    FR --> FM
    MC --> PC
    FM --> PC
    PC --> RISK
    RISK --> TRACE
```

---

## Key Invariants Enforced in Phase 1

1. **Zero Future Fundamental Leakage**: Financial statements are strictly gated by `available_at` (SEC public filing timestamp), never by period-end.
2. **Zero Pre-IPO / Delisted Contamination**: Point-in-time universe filters exclude instruments before their IPO date or after their delisting date.
3. **Zero Holdout Contamination**: Hyperparameter search (Optuna) is strictly bounded within training/validation folds with zero access to protected holdouts.
4. **Authoritative RiskEngine**: The portfolio construction layer cannot bypass the 13 immutable risk limits enforced by `RiskEngine`.
5. **No Speculative Complexity**: Phase 1 is built entirely using open-source in-process technologies (DuckDB, Parquet, Polars, Optuna, SciPy) without Kafka, Flink, or Kubernetes.
