# Methodology

## 1. News Sentiment Signal

### Hypothesis
News sentiment contains forward-looking information about stock returns.
A company mentioned positively in news today is more likely to outperform
in the short term (1–5 days) versus a company mentioned negatively.

### Sentiment Model

**FinBERT** (`ProsusAI/finbert`) is a BERT model fine-tuned on financial
text. It produces three probabilities: positive, neutral, negative.
Score = P(positive) − P(negative) ∈ [−1, 1].

**VADER** is a rule-based lexicon approach requiring no fine-tuning.
It provides a robust baseline for short headlines.

**Ensemble score** = 0.7 × FinBERT + 0.3 × VADER. The weights were
chosen based on empirical IC on a held-out validation set. FinBERT is
prioritised because it was trained on financial text.

### Entity Linking
Articles are linked to tickers via spaCy NER (ORG entities) plus fuzzy
matching against a company-name dictionary. Only articles with
confidence ≥ 0.82 are linked. This threshold balances precision vs recall.

---

## 2. Feature Engineering

### Sentiment Features
Raw article-level scores are aggregated to ticker × day:
- **Mean score** captures overall sentiment.
- **Rolling means** (3d, 7d) smooth noise and capture trend.
- **Sentiment surprise** = today − 30d trailing mean. A sudden shift
  in sentiment is a stronger signal than a persistent mild positive.
- **Article volume** acts as an attention signal — stocks in the news
  attract greater information flow.
- **Source diversity** distinguishes broad coverage (multiple outlets)
  from single-source stories.
- **Earnings proximity** controls for the earnings announcement effect,
  which is a known regime change in news-return dynamics.

### Market Features
Price-based features capture the market context into which the news lands:
- **Returns** (1d, 5d, 21d) capture momentum and mean-reversion context.
- **Volatility** (21d) and **vol surprise** capture regime.
- **RSI** and **MA signals** capture technical state.
- **EPS surprise** captures fundamental surprise history.

---

## 3. Predictive Model

### Architecture
LightGBM regression. Tree-based models handle tabular data well and are
robust to missing values and feature scale. The gradient boosting
objective minimises RMSE over predicted next-day returns.

### Walk-Forward Training
To avoid look-ahead bias:
1. The model is first trained on days 1 through T−1.
2. Predictions are made for day T.
3. At each rebalance interval (default: every 21 days), the model is
   retrained on all available data.
4. The first T = 252 trading days are warm-up and produce no predictions.

### Feature Importance
After each training run, SHAP values / split counts are logged.
Typically sentiment_surprise and article_count features rank highly,
alongside ret_1d and vol_21d.

---

## 4. Portfolio Construction

### Signal Ranking
At each rebalance date, stocks are ranked by predicted next-day return
in descending order. Top N → long bucket, Bottom N → short bucket.

### Weighting
Equal weight within each bucket. This is robust to prediction errors in
individual stocks and avoids concentration risk.

### Transaction Costs
One-way cost = TC_BPS basis points applied to positions that are new
versus the prior rebalance. Stocks that remain in the same bucket incur
no cost (they are not traded). Realistic for liquid large-cap equities.

### Rebalance Frequency
- **Daily (D)**: Maximum signal timeliness; highest turnover and TC.
- **Weekly (W)**: Moderate balance of signal decay vs transaction cost.
- **Monthly (M)**: Lowest cost; signal decay is a risk.

For short-horizon news alpha, daily or weekly rebalancing is recommended.

---

## 5. Performance Measurement

### Primary Metrics
- **IC / ICIR**: Signal quality metrics independent of position sizing.
  ICIR > 0.5 is considered a strong signal.
- **Sharpe Ratio**: Risk-adjusted return. > 1.0 after transaction costs
  is the target.
- **Max Drawdown**: Must be tolerable in live trading. Target < −20%.

### Benchmark
Long-short portfolios are benchmarked against:
- SPY (S&P 500 ETF) for US universe
- NIFTYBEES.NS for India universe

### Overfitting Controls
- Walk-forward out-of-sample evaluation only.
- No hyperparameter tuning on test data.
- Feature count limited to avoid noise factors.
- Minimum 252 trading days of training history before predictions begin.
- Early stopping on validation set.

---

## 6. Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| NewsAPI free tier: 100 req/day | Limits historical breadth | Use Alpha Vantage for more history |
| FinBERT on CPU: ~1s per article | Slow batch scoring | Use GPU; cache in DB |
| yfinance data quality | Occasional gaps / adjusted price issues | Validate prices; fill or drop |
| Short history of NLP features | Model warm-up period is long | Augment with historical news archives |
| Survivorship bias | Universe contains only current members | Use point-in-time universe if available |
| Market impact ignored | Costs underestimated for large positions | Add slippage model for live trading |
