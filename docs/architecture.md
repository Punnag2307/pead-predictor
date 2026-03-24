# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                       news-alpha                            │
│                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │ Data Layer   │   │  NLP Layer   │   │Feature Engineer│  │
│  │              │   │              │   │                │  │
│  │ news_fetcher │──▶│ entity_extr. │──▶│ news_features  │  │
│  │ price_fetch  │   │ sentiment    │   │ market_feats   │  │
│  │ earnings_f   │   └──────────────┘   └───────┬────────┘  │
│  │ universe     │                              │           │
│  │ database     │◀─────────────────────────────┘           │
│  └──────────────┘                              │           │
│                                                ▼           │
│                                     ┌──────────────────┐   │
│                                     │  Model Layer     │   │
│                                     │  impact_model    │   │
│                                     │  evaluation      │   │
│                                     └────────┬─────────┘   │
│                                              │             │
│                                              ▼             │
│                                     ┌──────────────────┐   │
│                                     │ Backtest Layer   │   │
│                                     │  strategy        │   │
│                                     │  analysis        │   │
│                                     └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Design Principles

1. **No look-ahead bias** — Every model training and prediction uses only data
   available at the time of prediction. Walk-forward expanding window.

2. **Modular layers** — Each layer (data / NLP / features / models / backtest)
   is independently testable and replaceable.

3. **Persistent storage** — Raw data is stored in SQLite to avoid re-fetching.
   Processed features are saved as parquet for fast loading.

4. **Graceful degradation** — If FinBERT is unavailable, falls back to VADER.
   If NewsAPI is unavailable, uses RSS feeds only.

## Data Flow

```
NewsAPI / RSS / AlphaVantage
          │
          ▼
    news_fetcher.py ──────────────────────────────┐
          │                                       │
          ▼                                       ▼
   entity_extraction.py                     database.py
   (ORG NER → ticker)                      (SQLite store)
          │
          ▼
    sentiment.py
   (FinBERT + VADER)
          │
          ▼
   news_features.py ◀── earnings_fetcher.py
   (ticker × date)
          │
          ├──── market_features.py ◀── price_fetcher.py
          │
          ▼
   impact_model.py
   (LightGBM walk-fwd)
          │
          ▼
    strategy.py
   (top_n long/short)
          │
          ▼
    analysis.py
   (tearsheet + stats)
```

## Technology Stack

| Component | Library |
|-----------|---------|
| Data fetching | `yfinance`, `newsapi-python`, `requests`, `feedparser` |
| Storage | `SQLite` via `SQLAlchemy` + `parquet` via `pyarrow` |
| NLP | `transformers` (FinBERT), `spacy`, `nltk` (VADER) |
| ML | `lightgbm`, `xgboost`, `scikit-learn` |
| Analysis | `pandas`, `numpy`, `scipy`, `statsmodels` |
| Visualisation | `plotly`, `matplotlib`, `seaborn` |
