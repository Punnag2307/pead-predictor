# Data Dictionary

## Raw Data

### articles (SQLite table)

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| article_id | TEXT PK | No | SHA-256 hash of article URL |
| ticker | TEXT | Yes | Primary ticker from the fetch query |
| source | TEXT | Yes | Publisher name (e.g. "Reuters") |
| published_at | DATETIME | Yes | UTC publication timestamp |
| title | TEXT | Yes | Headline text |
| description | TEXT | Yes | Lede / summary paragraph |
| url | TEXT | Yes | Canonical article URL |
| fetched_at | DATETIME | No | UTC timestamp when the row was inserted |

### sentiment (SQLite table)

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| article_id | TEXT PK/FK | No | Foreign key to articles.article_id |
| finbert_score | REAL | Yes | FinBERT score: positive_prob - negative_prob ∈ [-1, 1] |
| vader_score | REAL | Yes | VADER compound score ∈ [-1, 1] |
| ensemble_score | REAL | Yes | 0.7 × finbert + 0.3 × vader ∈ [-1, 1] |
| sentiment_label | TEXT | Yes | "positive" / "neutral" / "negative" |
| scored_at | DATETIME | No | UTC timestamp when scoring ran |

### prices (SQLite table)

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | INTEGER PK | No | Auto-increment |
| ticker | TEXT | No | Stock ticker |
| date | TEXT | No | YYYY-MM-DD format |
| open | REAL | Yes | Opening price (adjusted) |
| high | REAL | Yes | Daily high (adjusted) |
| low | REAL | Yes | Daily low (adjusted) |
| close | REAL | Yes | Closing price (adjusted) |
| volume | INTEGER | Yes | Trading volume |

---

## Processed Features

### News Features (ticker × date)

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| sentiment_mean_1d | float | [-1, 1] | Mean ensemble score, day t |
| sentiment_mean_3d | float | [-1, 1] | Rolling 3-day mean |
| sentiment_mean_7d | float | [-1, 1] | Rolling 7-day mean |
| sentiment_std_7d | float | [0, ∞) | Rolling 7-day std |
| article_count_1d | int | [0, ∞) | Articles on day t |
| article_count_7d | int | [0, ∞) | 7-day article volume |
| sentiment_surprise | float | (-∞, ∞) | Today's score minus 30-day mean |
| positive_ratio | float | [0, 1] | Fraction of positive articles |
| negative_ratio | float | [0, 1] | Fraction of negative articles |
| source_diversity | int | [0, ∞) | Unique publishers on day t |
| earnings_proximity | int | [0, 90] | Days to next earnings (NaN if unknown) |

### Market Features (ticker × date)

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| ret_1d | float | (-1, ∞) | Prior 1-day return |
| ret_5d | float | (-1, ∞) | Prior 5-day return |
| ret_21d | float | (-1, ∞) | Prior 21-day return |
| vol_21d | float | [0, ∞) | 21-day annualised realised vol |
| vol_surprise | float | (-∞, ∞) | vol_21d minus 63-day mean vol |
| rsi_14 | float | [0, 100] | 14-day RSI |
| above_ma50 | float | {0, 1} | 1 if price > 50-day MA |
| above_ma200 | float | {0, 1} | 1 if price > 200-day MA |
| eps_surprise_pct | float | (-∞, ∞) | Most recent EPS surprise % |

### Model Target

| Feature | Type | Description |
|---------|------|-------------|
| fwd_ret_1d | float | Next-day forward return (the label) |
| predicted_return | float | Model's predicted fwd_ret_1d |

---

## Performance Metrics

| Metric | Formula | Good value |
|--------|---------|------------|
| CAGR | (1 + total_ret)^(252/n) - 1 | > 0 |
| Ann. Vol | std(daily_ret) × √252 | Lower is better |
| Sharpe | CAGR / Ann. Vol | > 1 good, > 2 excellent |
| Sortino | CAGR / Downside Std | > 1 |
| Max Drawdown | min((cum - roll_max) / roll_max) | > -20% good |
| Calmar | CAGR / |Max DD| | > 0.5 good |
| IC | Spearman(signal, fwd_ret) | > 0.05 useful |
| ICIR | IC mean / IC std | > 0.5 good |
| Hit Rate | P(sign(signal) == sign(fwd_ret)) | > 52% good |
