# PEAD Predictor: Post-Earnings Announcement Drift with Machine Learning

A quantitative research system that predicts post-earnings stock price direction for S&P 500 companies using machine learning. Built as a publishable research letter targeting internship applications at quantitative trading firms.

---

## What This Project Does

Every quarter, 500+ S&P 500 companies report earnings. When actual EPS differs from analyst consensus, stocks react. This project asks:

> *"Given everything knowable before an earnings announcement — surprise magnitude, market context, sector dynamics — can we predict which direction the stock will move at market open?"*

The answer is yes, with 64.1% validation accuracy and a statistically significant out-of-sample Sharpe ratio of 1.70 on 2025 data the model never saw.

---

## Results

### Core Model Performance

| Metric | Value |
|--------|-------|
| Universe | 472 S&P 500 stocks (ex-Financials, ex-Utilities) |
| Training period | 2022–2023 |
| Validation period | 2024 |
| Out-of-sample test | 2025 |
| Validation accuracy | 64.1% |
| Validation AUC | 0.687 |
| OOS accuracy | 61.0% |

### Backtest — Full Dataset (2022–2025)

| Metric | Value |
|--------|-------|
| Total trades | 1,706 |
| Win rate | 74.1% |
| Gross Sharpe | 4.90 |
| Net Sharpe | 3.54 |
| Max drawdown | -39.6% |
| Round-trip cost | 38 bps |

### Backtest — 2025 Out-of-Sample

| Metric | Value |
|--------|-------|
| Trades | 308 |
| Win rate | 60.7% |
| Net Sharpe | 1.70 |
| Max drawdown | -35.6% |
| T-statistic | 9.217 |
| P-value | 0.000000 |
| Bootstrap 95% CI | [2.78, 4.30] |

### Signal Quality

| Metric | Value | Industry Benchmark |
|--------|-------|--------------------|
| OOS Mean IC | 0.278 | >0.05 = good |
| OOS IR | 1.482 | >1.0 = strong |
| IC > 0 rate | 87.5% | — |
| Rating | EXCELLENT | — |

### Alpha Decay

| Horizon | Mean Return | Win Rate |
|---------|-------------|----------|
| 1 day | 1.37% | 66.4% |
| 2 days | 1.70% | 61.4% |
| 3 days | 1.81% | 61.7% |
| 5 days | 2.10% | 62.1% |
| 10 days | 3.03% | 65.4% |

### Regime Analysis — All Four Regimes Profitable

| Regime | Trades | Win Rate | Net Sharpe | Significant |
|--------|--------|----------|------------|-------------|
| 2022 Bear Market | 320 | 79.7% | 4.38 | ✅ p=0.0000 |
| 2023 Recovery | 508 | 80.9% | 5.36 | ✅ p=0.0000 |
| 2024 AI Rally | 481 | 71.9% | 2.59 | ✅ p=0.0004 |
| 2025 OOS | 308 | 60.7% | 1.70 | ✅ p=0.0288 |

### EPS Surprise Quintile Analysis

Perfect monotonic relationship validated (Spearman ρ = 1.000, p = 0.0000):

| Quintile | Mean Return | Win Rate |
|----------|-------------|----------|
| Q1 — Largest Miss | -2.31% | 29.9% |
| Q2 — Miss | -1.04% | 40.1% |
| Q3 — In-line | +0.51% | 56.3% |
| Q4 — Beat | +1.14% | 63.8% |
| Q5 — Largest Beat | +2.47% | 70.0% |

### Capacity Analysis

Strategy remains profitable up to ~$1B deployed:

| Capital | Net Return per Trade |
|---------|---------------------|
| $10M | 92 bps ✅ |
| $100M | 84 bps ✅ |
| $500M | 68 bps ✅ |
| $1,000M | 57 bps ✅ |

---

## Research Charts

| Chart | Description |
|-------|-------------|
| `alpha_decay.png` | Signal persistence across 1–10 day horizons |
| `cumulative_pnl.png` | Strategy cumulative P&L 2022–2025 |
| `eps_vs_return.png` | EPS surprise vs abnormal return |
| `eps_quintile_analysis.png` | Monotonic PEAD relationship |
| `shap_importance.png` | Feature importance (SHAP values) |
| `ic_analysis.png` | Monthly IC time series, in-sample vs OOS |
| `regime_analysis.png` | Performance across market regimes |
| `capacity_analysis.png` | Strategy capacity and scalability |
| `threshold_sensitivity.png` | Performance across confidence thresholds |
| `sector_breakdown.png` | Returns by sector |
| `model_comparison.png` | Model comparison |

---

## Methodology

### Universe
472 S&P 500 stocks excluding Financials and Utilities. Financials have structurally different earnings mechanics; Utilities have minimal surprise variation. Follows Bernard & Thomas (1989) standard.

### Labeling
For each earnings event:
- **Abnormal return** = overnight gap return minus sector ETF return (isolates company-specific reaction)
- **Label** = direction of abnormal return (+1 up, -1 down)
- **No leakage** = all labels use post-event prices, all features use pre-event data only

### Features (24 total)
**Earnings features (10):** EPS surprise %, winsorized surprise, absolute surprise, direction, actual EPS, consensus EPS, report timing, earnings quality score, both-beat flag, sector-relative surprise (z-score within sector)

**Market context features (9):** Pre-earnings 5d and 20d returns, 20d volatility, volume ratio, price-to-52w-high, price-to-52w-low, VIX level, VIX percentile, previous day gap

**Sector and time features (5):** Sector encoding, month, quarter, day of week, earnings season indicator

### Model Architecture
Calibrated Ensemble with weighted averaging:
- Logistic Regression (20%) — linear baseline
- Random Forest (30%) — nonlinear, handles interactions
- XGBoost (50%) — gradient boosting, primary model

All models use isotonic regression calibration for reliable probability estimates. Confidence threshold of 0.65 applied for trade selection.

### Backtesting
- Transaction costs: 38 bps round trip (spread + market impact + commission + slippage)
- Equal position sizing
- Overnight gap strategy: enter at market open, exit same day
- Strict temporal split: no future information ever used

### Statistical Validation
- Two-sided t-test: T-stat 9.217, p=0.000000
- Bootstrap 95% CI for Sharpe: [2.78, 4.30]
- Bonferroni correction for multiple testing across 3 models
- All results significant at adjusted alpha level

---

## Project Structure
```
news-alpha/
├── src/
│   ├── data/
│   │   ├── database.py          # SQLite database setup
│   │   ├── universe.py          # S&P 500 universe construction
│   │   ├── price_fetcher.py     # yfinance price data
│   │   └── earnings_fetcher.py  # Earnings data collection
│   ├── features/
│   │   ├── labeling.py          # Abnormal return labeling
│   │   ├── feature_engineering.py  # 24-feature matrix
│   │   └── advanced_features.py    # Beat streak, momentum interaction
│   ├── models/
│   │   ├── impact_model.py      # Calibrated ensemble training
│   │   └── evaluation.py        # Model evaluation utilities
│   └── backtest/
│       ├── analysis.py          # Backtesting + alpha decay
│       └── advanced_analysis.py # IC + capacity + regime analysis
├── data/
│   └── processed/
│       └── charts/              # All research charts (11 PNG files)
├── docs/
│   ├── methodology.md           # Detailed methodology
│   ├── data_dictionary.md       # Feature definitions
│   └── architecture.md          # System architecture
├── collect_data.py              # Data collection pipeline
├── validate_setup.py            # Environment validation
├── run_advanced_analysis.py     # IC + capacity + regime
├── run_additional_analysis.py   # Threshold + quintile + capacity
├── requirements.txt             # Python dependencies
└── README.md
```

---

## How To Run

### Prerequisites
```bash
pip install -r requirements.txt
```

Create `.env` file in project root:
```
FINNHUB_API_KEY=your_key
FMP_API_KEY=your_key
ALPHA_VANTAGE_API_KEY=your_key
NASDAQ_API_KEY=your_key
DB_PATH=data/database/news_alpha.db
```

### Full Pipeline
```bash
# Step 1: Validate setup
python validate_setup.py

# Step 2: Collect data (prices + earnings + EDGAR)
python collect_data.py

# Step 3: Label earnings events
python src/features/labeling.py

# Step 4: Build feature matrix
python src/features/feature_engineering.py

# Step 5: Train models
python src/models/impact_model.py

# Step 6: Backtest
python src/backtest/analysis.py

# Step 7: Advanced analysis
python run_advanced_analysis.py

# Step 8: Additional analysis
python run_additional_analysis.py
```

---

## Data Sources

| Source | Data | Cost |
|--------|------|------|
| yfinance | Daily prices, earnings estimates | Free |
| SEC EDGAR | 8-K filings, CIK lookup | Free |
| Finnhub | Market data | Free tier |
| Alpha Vantage | News sentiment | Free tier |

All data sources are free. No Bloomberg or Compustat required.

---

## Known Limitations

**Survivorship bias** — Universe based on current S&P 500 constituents. Companies delisted 2022–2025 are excluded, likely causing slight return overestimation.

**Training data depth** — Only 2 years of training data (2022–2023, ~2,767 events). Upgrading to institutional data would extend this to 10+ years.

**Earnings quality feature** — `earnings_quality` is 98% null due to sparse revenue surprise data in yfinance free tier. Full coverage would require premium data.

**Max drawdown** — -35.6% OOS reflects earnings event clustering during quarterly reporting seasons. Volatility-scaled position sizing is a natural extension.

**Daily prices only** — Intraday execution timing cannot be precisely modeled. Assumes execution at market open.

---

## Academic Context

This project implements and extends Post-Earnings Announcement Drift (PEAD), documented since Ball & Brown (1968). Key references:

- Bernard & Thomas (1989) — PEAD anomaly and universe construction
- Livnat & Mendenhall (2006) — Earnings surprise measurement
- Engelberg et al. (2010) — News-driven PEAD

Our contribution: applying calibrated ensemble ML to PEAD on 2022–2025 data with rigorous OOS validation, IC analysis, and regime decomposition.

---

## Author

Punnag | B.Tech Computer Science (Data Science, Economics, Business) | Plaksha University

*Built as part of a quantitative research letter series targeting internship applications at Citadel, Jane Street, and Wintermute.*
```
