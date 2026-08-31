# Portfolio Risk Engine — Value-at-Risk & Expected Shortfall

A regulatory-grade tail-risk system that computes 1-day **Value-at-Risk (VaR)** and
**Expected Shortfall (ES)** for a multi-asset portfolio using three independent
methodologies, then validates them with the statistical backtests used in bank
risk management under **Basel III / FRTB**.

## What it does

| Component | Description |
|-----------|-------------|
| **Historical Simulation** | Empirical quantile of the loss distribution — no distributional assumption |
| **Parametric (Gaussian & Student-t)** | Closed-form VaR/ES; the Student-t variant captures fat tails |
| **Monte Carlo** | Simulates return paths and reads off the tail |
| **Kupiec POF test** | Checks whether the *number* of breaches is statistically correct |
| **Christoffersen test** | Checks whether breaches are *independent* or cluster during stress |

## Why it matters

Banks report VaR/ES daily for regulatory capital. FRTB shifted the mandated measure
from 99% VaR to **97.5% Expected Shortfall** because ES captures tail severity that VaR
ignores. Too many backtest exceptions push a bank into a penalty multiplier that raises
its capital requirement — so calibration is not academic, it is money.

## How to run

**Google Colab (easiest):** `File → Upload notebook → Portfolio_Risk_Engine.ipynb`, then `Runtime → Run all`.

**Locally:**
```bash
pip install -r requirements.txt
jupyter notebook Portfolio_Risk_Engine.ipynb
```

No API keys required — data comes from Yahoo Finance via `yfinance`.

## Configuration

Edit the `CONFIG` block in Cell 2 to change tickers, weights, dates, notional, or
confidence levels. Weights must sum to 1.0 (validated at runtime).

## Outputs

- Method-comparison bar charts (VaR & ES)
- Return distribution with tail-risk cutoffs and fitted-normal overlay
- Rolling backtest with exception markers
- Printable end-of-day risk report

## Key findings to discuss

- Gaussian VaR systematically **understates** tail risk vs. Historical/Student-t
- If the Christoffersen test fails, breaches cluster in crises — motivating the
  EWMA/GARCH volatility upgrade

## Potential upgrades

EWMA/GARCH conditional volatility · filtered historical simulation ·
component/marginal VaR for risk attribution · Streamlit dashboard · FRTB liquidity-horizon ES scaling

## Tech stack

`numpy` · `pandas` · `scipy` · `matplotlib` · `yfinance`

---
*Built as a demonstration of quantitative risk methodology. Not investment advice.*
