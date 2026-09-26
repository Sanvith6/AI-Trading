# Fundamental Data Layer & SEC EDGAR Point-in-Time Schema

## Causality Invariant
Corporate financial quarters end months before their official 10-Q / 10-K statements are filed with the SEC.
```
Quarter End (period_end)        SEC Public Acceptance (filing_date)        Available to Model (available_at)
      2023-12-31                                2024-02-02                                2024-02-02T21:30:00Z
          |-----------------------------------------|---------------------------------------------|
                 (Financial Audit Window)               (Public SEC Dissemination)
```
**CRITICAL RULE**: Models running on `2024-01-15` CANNOT use Q4 2023 fundamentals. Doing so introduces lookahead leakage. The system strictly queries by `available_at <= as_of_date`.

## Tracked Metrics
- Revenue, Revenue YoY Growth
- Gross Margin, Operating Margin
- EBITDA, Net Income, Diluted EPS
- Free Cash Flow, Cash & Equivalents
- Total Debt, Total Assets, Total Liabilities
- Shares Outstanding, ROE, ROIC
- Composite Fundamental Quality Score
