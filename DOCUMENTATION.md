# news-alpha — Full Technical Documentation

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Data Layer](#2-data-layer)
3. [NLP Layer](#3-nlp-layer)
4. [Feature Engineering](#4-feature-engineering)
5. [Predictive Model](#5-predictive-model)
6. [Backtest Engine](#6-backtest-engine)
7. [Database Schema](#7-database-schema)
8. [Configuration Reference](#8-configuration-reference)
9. [Error Handling & Logging](#9-error-handling--logging)
10. [Extending the Pipeline](#10-extending-the-pipeline)

---

## 1. System Architecture

```
News Sources                   Market Data
(NewsAPI, RSS, AV)             (yfinance, AV)
        │                            │
        ▼                            ▼
  news_fetcher.py           price_fetcher.py
  earnings_fetcher.py       universe.py
        │                            │
        └────────────┬───────────────┘
                     ▼
              database.py  ──►  SQLite (news_alpha.db)
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
  entity_extraction.py      sentiment.py
  (spaCy NER)               (FinBERT + VADER)
        │                         │
        └────────────┬────────────┘
                     ▼
          news_features.py   market_features.py
                     │              │
                     └──────┬───────┘
                            ▼
                    impact_model.py
                    (LightGBM / XGB)
                            │
                            ▼
                      strategy.py
                   (signal → portfolio)
                            │
                            ▼
                      analysis.py
                     (tearsheet / IC)
```

---

## 2. Data Layer

### 2.1 `src/data/universe.py`

Builds the investable universe for US (S&P 500 constituents) and/or
India (Nifty 200 constituents).

**Key functions:**
- `get_us_universe() -> list[str]` — returns S&P 500 tickers scraped from Wikipedia
- `get_india_universe() -> list[str]` — returns Nifty 200 tickers
- `build_universe(market) -> dict` — combined ticker metadata with sector/industry

**Output columns:** `ticker`, `name`, `sector`, `industry`, `market`, `market_cap`

---

### 2.2 `src/data/news_fetcher.py`

Fetches news articles from multiple sources and normalises them into a
standard schema.

**Sources:**
- **NewsAPI** — `GET /v2/everything?q={ticker}&language=en`
- **Alpha Vantage** — `ALPHA_VANTAGE_NEWS_SENTIMENT` endpoint
- **RSS feeds** — Reuters, Bloomberg, Seeking Alpha (via `feedparser`)

**Standard article schema:**

| Field | Type | Description |
|-------|------|-------------|
| `article_id` | str | SHA-256 hash of URL |
| `ticker` | str | Primary ticker (from query) |
| `source` | str | Publisher name |
| `published_at` | datetime | UTC publish timestamp |
| `title` | str | Headline text |
| `description` | str | Lede paragraph |
| `url` | str | Canonical article URL |
| `fetched_at` | datetime | UTC fetch timestamp |

**Rate limiting:** Uses `tenacity` exponential back-off. NewsAPI free tier: 100
requests/day. Alpha Vantage free tier: 25 requests/day.

---

### 2.3 `src/data/price_fetcher.py`

Downloads OHLCV data via `yfinance` and stores in parquet.

**Key functions:**
- `fetch_prices(tickers, start, end, interval) -> pd.DataFrame`
- `fetch_intraday(ticker, interval) -> pd.DataFrame`
- `get_benchmark(market) -> pd.Series` — SPY for US, NIFTYBEES.NS for India

**Storage:** `data/raw/prices_{market}_{date}.parquet`

---

### 2.4 `src/data/earnings_fetcher.py`

Fetches earnings calendar and EPS surprise data.

**Key functions:**
- `get_earnings_calendar(tickers, days_ahead) -> pd.DataFrame`
- `get_eps_surprise(ticker) -> pd.DataFrame`

**Output columns:** `ticker`, `report_date`, `eps_estimate`, `eps_actual`,
`surprise_pct`, `revenue_estimate`, `revenue_actual`

---

### 2.5 `src/data/database.py`

SQLite interface via `SQLAlchemy`. All raw articles and processed sentiment
scores are persisted here to avoid re-fetching.

**Key functions:**
- `init_db()` — create tables if not exists
- `insert_articles(df)` — upsert by `article_id`
- `insert_sentiment(df)` — upsert by `article_id`
- `query_articles(ticker, start, end) -> pd.DataFrame`
- `query_sentiment(ticker, start, end) -> pd.DataFrame`

---

## 3. NLP Layer

### 3.1 `src/nlp/entity_extraction.py`

Maps free-text articles to equity tickers using spaCy NER + a fuzzy
company-name → ticker lookup table.

**Pipeline:**
1. spaCy `en_core_web_sm` extracts `ORG` entities from title + description
2. Each entity is matched against a company-name dictionary (from universe)
3. Fuzzy match (`difflib.SequenceMatcher`) with threshold ≥ 0.85
4. Returns list of `(ticker, confidence)` pairs per article

**Key functions:**
- `build_company_lookup(universe_df) -> dict`
- `extract_tickers(text, lookup) -> list[tuple[str, float]]`
- `batch_extract(articles_df, lookup) -> pd.DataFrame`

---

### 3.2 `src/nlp/sentiment.py`

Scores each article on a [-1, +1] sentiment scale using an ensemble of:

**FinBERT** (primary):
- Model: `ProsusAI/finbert` from HuggingFace
- Output: `{positive, negative, neutral}` probabilities
- Sentiment score: `positive_prob - negative_prob`

**VADER** (baseline/fallback):
- `nltk.sentiment.vader.SentimentIntensityAnalyzer`
- Used when FinBERT is unavailable or as a sanity check
- Score: `compound` value from VADER

**Ensemble score:** `0.7 × finbert_score + 0.3 × vader_score`

**Key functions:**
- `load_finbert() -> pipeline`
- `score_article(text) -> dict`  — `{finbert, vader, ensemble, label}`
- `batch_score(articles_df, batch_size=32) -> pd.DataFrame`

**Output columns:** `article_id`, `finbert_score`, `vader_score`,
`ensemble_score`, `sentiment_label`, `scored_at`

---

## 4. Feature Engineering

### 4.1 `src/features/news_features.py`

Aggregates article-level sentiment scores into ticker-day features.

**Key features computed:**

| Feature | Description |
|---------|-------------|
| `sentiment_mean_1d` | Mean ensemble score on day t |
| `sentiment_mean_3d` | Rolling 3-day mean |
| `sentiment_mean_7d` | Rolling 7-day mean |
| `article_count_1d` | Number of articles on day t |
| `article_count_7d` | 7-day article volume |
| `sentiment_surprise` | Today's sentiment vs 30-day mean |
| `positive_ratio` | Fraction of positive articles |
| `negative_ratio` | Fraction of negative articles |
| `source_diversity` | Number of unique publishers |
| `earnings_proximity` | Days to next earnings (from calendar) |

**Key functions:**
- `aggregate_sentiment(sentiment_df, freq="D") -> pd.DataFrame`
- `compute_news_features(sentiment_df, earnings_df) -> pd.DataFrame`

---

### 4.2 `src/features/market_features.py`

Adds price-derived context features for use as model inputs.

**Key features:**

| Feature | Description |
|---------|-------------|
| `ret_1d` | Prior day return |
| `ret_5d` | 5-day trailing return |
| `ret_21d` | 21-day trailing return |
| `vol_21d` | 21-day annualised realised vol |
| `vol_surprise` | Today's vol vs 63-day mean |
| `rsi_14` | 14-day RSI |
| `above_ma50` | Binary: price > 50-day MA |
| `eps_surprise_pct` | Most recent earnings surprise % |

**Key functions:**
- `compute_price_features(price_df) -> pd.DataFrame`
- `merge_features(news_df, price_df, earnings_df) -> pd.DataFrame`

---

## 5. Predictive Model

### 5.1 `src/models/impact_model.py`

Trains a gradient-boosted tree to predict next-day return given
today's news and market features.

**Target variable:** `fwd_ret_1d` — forward 1-day return (label at t+1)

**Train / validation split:** Expanding window (walk-forward) — never
uses future data to train. Min 252 days of history before first prediction.

**Model:** `LightGBM` (primary), `XGBoost` (secondary / ensemble)

**Hyperparameters (defaults):**
```python
lgb_params = {
    "objective":        "regression",
    "metric":           "rmse",
    "learning_rate":    0.05,
    "num_leaves":       63,
    "min_child_samples": 30,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "n_estimators":     500,
    "early_stopping_rounds": 50,
}
```

**Key functions:**
- `train(feature_df, target_col, params) -> model`
- `predict(model, feature_df) -> pd.Series`
- `walk_forward_predict(feature_df) -> pd.DataFrame`

---

### 5.2 `src/models/evaluation.py`

Model and signal evaluation metrics.

**Metrics computed:**

| Metric | Description |
|--------|-------------|
| `IC` | Spearman rank correlation of signal vs fwd return |
| `ICIR` | IC mean / IC std (information ratio of the signal) |
| `hit_rate` | Fraction of correct directional predictions |
| `long_ret` | Mean return of top-quintile stocks |
| `short_ret` | Mean return of bottom-quintile stocks |
| `spread` | Long return − short return |
| `t_stat` | t-statistic of the IC series |

**Key functions:**
- `compute_ic(signal, returns) -> float`
- `compute_icir(signal_df, return_df) -> pd.DataFrame`
- `feature_importance(model) -> pd.DataFrame`
- `evaluation_report(predictions_df, returns_df) -> dict`

---

## 6. Backtest Engine

### 6.1 `src/backtest/strategy.py`

Converts daily model predictions into a rebalanced long-short portfolio.

**Portfolio construction:**
- **Signal:** predicted `fwd_ret_1d` rank across universe on each date
- **Long bucket:** top N stocks by predicted return
- **Short bucket:** bottom N stocks by predicted return
- **Weighting:** equal weight within each bucket
- **Rebalance:** configurable — daily `D`, weekly `W`, monthly `M`
- **Transaction cost:** one-way cost deducted on new positions

**Key functions:**
- `build_positions(predictions_df, top_n, rebal_freq) -> pd.DataFrame`
- `compute_portfolio_returns(positions_df, price_df, tc_bps) -> pd.DataFrame`

---

### 6.2 `src/backtest/analysis.py`

Performance analytics and tearsheet generation.

**Statistics computed:**

| Stat | Description |
|------|-------------|
| `cagr` | Compound annual growth rate |
| `sharpe` | Annualised Sharpe ratio (rf = 0) |
| `sortino` | Sortino ratio |
| `max_dd` | Maximum drawdown |
| `calmar` | CAGR / max drawdown |
| `hit_rate` | % of profitable days |
| `avg_holding` | Mean holding period in days |
| `turnover` | Daily average portfolio turnover |

**Key functions:**
- `compute_stats(returns) -> dict`
- `compute_drawdown(returns) -> pd.Series`
- `monthly_returns_heatmap(returns) -> pd.DataFrame`
- `generate_tearsheet(results) -> dict`

---

## 7. Database Schema

**Table: `articles`**

| Column | Type | Notes |
|--------|------|-------|
| `article_id` | TEXT PK | SHA-256 of URL |
| `ticker` | TEXT | Primary ticker |
| `source` | TEXT | Publisher |
| `published_at` | DATETIME | UTC |
| `title` | TEXT | |
| `description` | TEXT | |
| `url` | TEXT | |
| `fetched_at` | DATETIME | UTC |

**Table: `sentiment`**

| Column | Type | Notes |
|--------|------|-------|
| `article_id` | TEXT PK FK | |
| `finbert_score` | REAL | [-1, 1] |
| `vader_score` | REAL | [-1, 1] |
| `ensemble_score` | REAL | [-1, 1] |
| `sentiment_label` | TEXT | positive / neutral / negative |
| `scored_at` | DATETIME | |

**Table: `prices`**

| Column | Type | Notes |
|--------|------|-------|
| `ticker` | TEXT | |
| `date` | DATE | |
| `open` | REAL | |
| `high` | REAL | |
| `low` | REAL | |
| `close` | REAL | |
| `volume` | INTEGER | |
| PRIMARY KEY | `(ticker, date)` | |

---

## 8. Configuration Reference

All configuration via `.env`. See `.env` for descriptions.

| Variable | Default | Description |
|----------|---------|-------------|
| `NEWSAPI_KEY` | — | NewsAPI.org key |
| `ALPHANEWS_KEY` | — | Alpha Vantage key |
| `OPENAI_API_KEY` | — | Optional GPT-4o key |
| `HF_TOKEN` | — | HuggingFace token |
| `DB_PATH` | `data/database/news_alpha.db` | SQLite path |
| `UNIVERSE` | `us` | `us` / `india` / `both` |
| `REBALANCE_FREQ` | `D` | `D` / `W` / `M` |
| `TOP_N` | `20` | Long/short bucket size |
| `TC_BPS` | `5` | Transaction cost (bps) |

---

## 9. Error Handling & Logging

All modules use `loguru` for structured logging:

```python
from loguru import logger
logger.info("Fetching {n} articles for {ticker}", n=count, ticker=ticker)
logger.warning("Rate limit hit — sleeping {s}s", s=sleep)
logger.error("FinBERT failed: {exc}", exc=e)
```

API calls are wrapped with `tenacity` retry:

```python
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
def fetch_with_retry(url, params):
    ...
```

---

## 10. Extending the Pipeline

**Add a new news source:**
1. Add a fetcher function in `src/data/news_fetcher.py`
2. Return a DataFrame matching the standard article schema
3. Call it from the main fetch loop

**Add a new feature:**
1. Add computation to `src/features/news_features.py` or `market_features.py`
2. The new column will automatically be picked up by `impact_model.py`
3. Check feature importance post-training in `evaluation.py`

**Swap the ML model:**
1. Implement `train()` and `predict()` in a new file under `src/models/`
2. Update `strategy.py` to call the new model's `predict()` function


# News-Driven Alpha: Project Documentation

## Overview
A quantitative research system that detects market-moving news events
and predicts their impact on S&P 500 stock prices. Built as both an
internship portfolio project and a publishable research letter
targeting firms including Citadel, Jane Street, and Wintermute.

**Research Question:**
Can publicly available news be used to predict abnormal short-term
returns in US equities?

---

## Three-Chapter Research Design

| Chapter | Event Type | Mechanism | Status |
|---------|-----------|-----------|--------|
| 1 | Earnings Surprises | Actual vs consensus EPS/revenue | In Progress |
| 3 | Idiosyncratic Corporate News | Recalls, lawsuits, breaches | Planned |
| 2 | FDA Binary Events | Drug approval/rejection | Planned |

### Chapter Ordering Rationale
- **Chapter 1 (Earnings):** Most structured, largest dataset,
  directly comparable to academic benchmarks (PEAD literature)
- **Chapter 3 (Corporate):** Unstructured, pure NLP challenge,
  original contribution — recalls, lawsuits, data breaches
- **Chapter 2 (FDA):** Binary extreme case, cleanest possible labels

---

## Universe

- **Base:** S&P 500 constituents
- **Exclusions:** Financial Services, Utilities
- **Rationale:** Follows Bernard & Thomas (1989),
  Livnat & Mendenhall (2006) — financials have structurally
  different earnings mechanisms; utilities have regulated
  returns with minimal surprise variation
- **Final size:** 472 stocks across 10 sectors
- **Historical window:** January 2020 — December 2024

### Sector Breakdown
| Sector | Stocks |
|--------|--------|
| Industrials | 79 |
| Financials* | 76 |
| Information Technology | 71 |
| Health Care | 60 |
| Consumer Discretionary | 48 |
| Consumer Staples | 36 |
| Real Estate | 31 |
| Materials | 26 |
| Communication Services | 23 |
| Energy | 22 |

*Financials shown for reference — excluded from model universe

---

## Data Sources

### Confirmed Working (Phase 1 Validated)

| Source | Data | Endpoint | Cost | Notes |
|--------|------|----------|------|-------|
| Financial Modeling Prep | Earnings EPS/Revenue | `/stable/earnings` | Free | 250 calls/day |
| SEC EDGAR | 8-K filings | `data.sec.gov/submissions` | Free | No key needed |
| Alpha Vantage | News + sentiment | `NEWS_SENTIMENT` | Free | 25 calls/day |
| yfinance | Daily OHLCV prices | Yahoo Finance | Free | Use v0.2.58+ |
| Wikipedia | S&P 500 universe | HTML scrape | Free | Needs browser headers |

### Deprecated/Rejected Sources
| Source | Reason Rejected |
|--------|----------------|
| FMP `/api/v3` | Deprecated August 31 2025 — returns legacy error |
| Finnhub company news | Free tier returns empty for historical data |
| yfinance 0.2.38 | Incompatible with Python 3.12 — JSONDecodeError |

---

## Architecture
```
Data Layer → NLP Layer → Feature Layer → Model Layer → Backtest Layer
```

### Data Layer
- News: SEC EDGAR (Ch1) + Alpha Vantage (Ch3)
- Prices: yfinance daily OHLCV
- Earnings: Financial Modeling Prep stable API
- Storage: SQLite (news_alpha.db)

### NLP Layer (Phase 3)
- Sentiment: FinBERT (ProsusAI/finbert via HuggingFace)
- Entity extraction: spaCy (en_core_web_sm)
- Event classification: Rule-based + XGBoost

### Feature Layer (Phase 4)
- News features: sentiment score, headline length, novelty
- Market features: pre-event volatility, volume, sector momentum
- Earnings features: EPS surprise %, revenue surprise %, guidance

### Model Layer (Phase 5)
- Primary: XGBoost classifier (direction) + regressor (magnitude)
- Interpretability: SHAP values
- Validation: Walk-forward out-of-sample testing

### Backtest Layer (Phase 6)
- Framework: vectorbt
- Costs: Bid-ask spread + market impact + slippage
- Metrics: Sharpe ratio, alpha decay curve, hit rate

---

## Database Schema

### companies
| Column | Type | Description |
|--------|------|-------------|
| ticker | TEXT PK | Stock ticker symbol |
| name | TEXT | Company name |
| sector | TEXT | GICS sector |
| industry | TEXT | GICS sub-industry |
| market_cap | REAL | Market capitalization |
| added_date | TEXT | Date added to universe |

### earnings_events
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto increment |
| ticker | TEXT | Stock ticker |
| report_date | TEXT | Earnings report date |
| actual_eps | REAL | Reported EPS |
| consensus_eps | REAL | Analyst consensus EPS |
| eps_surprise | REAL | Actual minus consensus |
| eps_surprise_pct | REAL | Surprise as % of consensus |
| actual_revenue | REAL | Reported revenue |
| consensus_revenue | REAL | Analyst consensus revenue |
| revenue_surprise_pct | REAL | Revenue surprise % |

### news_articles
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto increment |
| ticker | TEXT | Associated ticker |
| headline | TEXT | Article headline |
| summary | TEXT | Article summary |
| source | TEXT | News source |
| url | TEXT | Article URL |
| published_at | TEXT | Publication timestamp |
| event_type | TEXT | EARNINGS_8K or CORPORATE_NEWS |
| chapter | INTEGER | 1 = Earnings, 3 = Corporate |

### price_data
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto increment |
| ticker | TEXT | Stock ticker |
| date | TEXT | Trading date |
| open/high/low/close | REAL | OHLC prices |
| volume | INTEGER | Daily volume |
| adj_close | REAL | Adjusted close |

### labeled_events
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto increment |
| event_type | TEXT | Type of news event |
| ticker | TEXT | Stock ticker |
| event_date | TEXT | Date of event |
| return_1h | REAL | 1-hour forward return |
| return_4h | REAL | 4-hour forward return |
| return_1d | REAL | 1-day forward return |
| return_3d | REAL | 3-day forward return |
| abnormal_return_1d | REAL | Return minus sector ETF |
| label_direction | INTEGER | +1 up, -1 down |
| label_magnitude | REAL | Absolute return magnitude |
| is_market_moving | INTEGER | 1 if beyond 2x volatility |

---

## Tech Stack

| Component | Tool | Version |
|-----------|------|---------|
| Language | Python | 3.12.3 |
| Data wrangling | pandas | 2.1.4 |
| Numerical | numpy | 1.26.2 |
| ML | xgboost | 2.0.3 |
| Interpretability | shap | 0.46.0 |
| NLP | transformers | 4.36.2 |
| Deep learning | torch | 2.2.0 |
| NER | spacy | 3.7.2 |
| Backtesting | vectorbt | TBD |
| Visualization | matplotlib, seaborn | 3.8.2, 0.13.1 |
| Statistics | scipy | 1.11.4 |
| Database | SQLite via sqlalchemy | 2.0.23 |
| Earnings data | FMP stable API | REST |
| News (Ch1) | SEC EDGAR | REST |
| News (Ch3) | Alpha Vantage | REST |
| Prices | yfinance | 0.2.58 |

---

## Statistical Methodology

### Labeling (Chapter 1)
```
Abnormal Return = Stock Return - Sector ETF Return (same window)
Market Moving = |Abnormal Return| > 2 * Pre-Event Rolling Volatility
```

### Train/Test Split
- Training: 2020-01-01 to 2023-06-30
- Validation: 2023-07-01 to 2023-12-31
- Out-of-sample: 2024-01-01 to 2024-12-31

### Significance Tests
- Two-sided t-test on mean returns
- Bootstrap 95% CI for Sharpe ratio
- Bonferroni correction for multiple testing

---

## Known Limitations

1. **Survivorship bias:** Universe based on current S&P 500
2. **Latency:** Cannot compete with co-located HFT
3. **News coverage:** Alpha Vantage free tier: 25 calls/day
4. **Language:** English-only sources
5. **Regime sensitivity:** 2020-2024 includes COVID distortions

---

## Phase Progress

- [x] Phase 1: Environment + Data Pipeline
- [ ] Phase 2: Data Collection (bulk fetch all 472 stocks)
- [ ] Phase 3: Labeling Pipeline
- [ ] Phase 4: NLP Feature Engineering
- [ ] Phase 5: Model Training + Evaluation
- [ ] Phase 6: Backtesting + Research Note

---

