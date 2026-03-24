import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.database import get_connection
from src.features.advanced_features import (
    calculate_beat_streaks,
    calculate_momentum_interaction,
    calculate_previous_earnings_gap,
    build_guidance_features
)

# ─────────────────────────────────────────────
# SECTOR ENCODING
# ─────────────────────────────────────────────

SECTOR_ENCODING = {
    'Information Technology': 0,
    'Health Care': 1,
    'Consumer Discretionary': 2,
    'Consumer Staples': 3,
    'Industrials': 4,
    'Energy': 5,
    'Materials': 6,
    'Real Estate': 7,
    'Communication Services': 8
}

# ─────────────────────────────────────────────
# PRICE DATA LOADING
# ─────────────────────────────────────────────

def load_price_cache() -> dict:
    """Load all price data into memory as dictionary."""
    conn = get_connection()
    tickers_df = pd.read_sql(
        "SELECT DISTINCT ticker FROM price_data", conn
    )
    conn.close()

    prices = {}
    for ticker in tickers_df['ticker']:
        conn = get_connection()
        df = pd.read_sql("""
            SELECT date, open, high, low, close, volume
            FROM price_data
            WHERE ticker = ?
            ORDER BY date ASC
        """, conn, params=(ticker,))
        conn.close()

        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df['daily_return'] = df['close'].pct_change()
            prices[ticker] = df

    print(f"Loaded price cache: {len(prices)} tickers")
    return prices

# ─────────────────────────────────────────────
# CATEGORY 1: EARNINGS FEATURES
# ─────────────────────────────────────────────

def build_earnings_features(
    row: pd.Series,
    sector_stats: pd.DataFrame = None
) -> dict:
    """
    Build features derived directly from earnings data.
    Includes earnings quality score and sector-relative surprise.
    All available at announcement time — no leakage.
    """
    eps_surprise_pct = row.get('eps_surprise_pct', np.nan)
    actual_eps = row.get('actual_eps', np.nan)
    consensus_eps = row.get('consensus_eps', np.nan)
    report_time = row.get('report_time', None)
    revenue_surprise = row.get('revenue_surprise_pct', np.nan)

    # EPS surprise features
    eps_surprise_abs = abs(eps_surprise_pct) if not pd.isna(eps_surprise_pct) else np.nan
    eps_surprise_direction = (
        1 if eps_surprise_pct > 0
        else -1 if eps_surprise_pct < 0
        else 0
    ) if not pd.isna(eps_surprise_pct) else 0

    # Winsorize at ±30% (tighter than before — removes data errors)
    eps_surprise_winsorized = np.clip(
        eps_surprise_pct, -30, 30
    ) if not pd.isna(eps_surprise_pct) else np.nan

    # Report timing flag
    report_time_flag = (
        1 if report_time == 'AMC'
        else 0 if report_time == 'BMO'
        else -1
    )

    # ── NEW: Earnings Quality Score ──
    # Positive = revenue beat exceeded EPS beat (organic growth)
    # Negative = EPS beat without revenue (cost cuts or buybacks)
    # Market rewards quality beats more than accounting beats
    if not pd.isna(eps_surprise_pct) and not pd.isna(revenue_surprise):
        earnings_quality = float(revenue_surprise) - float(eps_surprise_pct)
        earnings_quality = np.clip(earnings_quality, -50, 50)
    else:
        earnings_quality = np.nan

    # Both beat flag — highest quality signal
    both_beat = int(
        (not pd.isna(eps_surprise_pct) and eps_surprise_pct > 0) and
        (not pd.isna(revenue_surprise) and revenue_surprise > 0)
    ) if not pd.isna(eps_surprise_pct) and not pd.isna(revenue_surprise) else 0

    # Both miss flag — strongest negative signal
    both_miss = int(
        (not pd.isna(eps_surprise_pct) and eps_surprise_pct < 0) and
        (not pd.isna(revenue_surprise) and revenue_surprise < 0)
    ) if not pd.isna(eps_surprise_pct) and not pd.isna(revenue_surprise) else 0

    # ── NEW: Sector-Relative Surprise ──
    # Z-score of EPS surprise within sector
    # A 5% beat means different things in tech vs utilities
    sector = row.get('sector', 'Unknown')
    sector_relative_surprise = np.nan

    if sector_stats is not None and sector in sector_stats.index:
        s_mean = sector_stats.loc[sector, 'sector_surprise_mean']
        s_std = sector_stats.loc[sector, 'sector_surprise_std']
        if not pd.isna(eps_surprise_pct) and s_std > 0:
            sector_relative_surprise = (
                float(eps_surprise_pct) - s_mean
            ) / s_std
            # Clip to ±3 standard deviations
            sector_relative_surprise = np.clip(
                sector_relative_surprise, -3, 3
            )

    return {
        'eps_surprise_pct': eps_surprise_pct,
        'eps_surprise_pct_winsorized': eps_surprise_winsorized,
        'eps_surprise_abs': eps_surprise_abs,
        'eps_surprise_direction': eps_surprise_direction,
        'actual_eps': actual_eps,
        'consensus_eps': consensus_eps,
        'report_time_flag': report_time_flag,
        'earnings_quality': earnings_quality,
        'both_beat': both_beat,
        'both_miss': both_miss,
        'sector_relative_surprise': sector_relative_surprise
    }

# ─────────────────────────────────────────────
# CATEGORY 2: MARKET CONTEXT FEATURES
# ─────────────────────────────────────────────

def build_market_features(
    ticker: str,
    event_date: pd.Timestamp,
    prices_cache: dict
) -> dict:
    """
    Build pre-event market context features.
    All use data strictly BEFORE event_date — no leakage.
    """
    features = {
        'pre_return_5d': np.nan,
        'pre_return_20d': np.nan,
        'pre_vol_20d': np.nan,
        'pre_volume_ratio': np.nan,
        'price_to_52w_high': np.nan,
        'price_to_52w_low': np.nan,
        'vix_level': np.nan,
        'vix_percentile': np.nan,
        'opening_gap_prev': np.nan
    }

    if ticker not in prices_cache:
        return features

    prices = prices_cache[ticker]
    pre_prices = prices[prices.index < event_date].copy()

    if len(pre_prices) < 20:
        return features

    try:
        # 5-day and 20-day pre-event returns
        features['pre_return_5d'] = (
            pre_prices['close'].iloc[-1] /
            pre_prices['close'].iloc[-6] - 1
        ) if len(pre_prices) >= 6 else np.nan

        features['pre_return_20d'] = (
            pre_prices['close'].iloc[-1] /
            pre_prices['close'].iloc[-21] - 1
        ) if len(pre_prices) >= 21 else np.nan

        # 20-day pre-event volatility (annualized)
        features['pre_vol_20d'] = (
            pre_prices['daily_return'].tail(20).std() * np.sqrt(252)
        )

        # Volume ratio
        recent_vol = pre_prices['volume'].tail(5).mean()
        avg_vol = pre_prices['volume'].tail(20).mean()
        features['pre_volume_ratio'] = (
            recent_vol / avg_vol if avg_vol > 0 else np.nan
        )

        # Price relative to 52-week high and low
        prices_52w = pre_prices['close'].tail(252)
        if len(prices_52w) > 0:
            high_52w = prices_52w.max()
            low_52w = prices_52w.min()
            current_price = pre_prices['close'].iloc[-1]
            features['price_to_52w_high'] = (
                current_price / high_52w if high_52w > 0 else np.nan
            )
            features['price_to_52w_low'] = (
                current_price / low_52w if low_52w > 0 else np.nan
            )

        # Previous day opening gap
        # Captures pre-earnings drift signal
        if len(pre_prices) >= 2:
            prev_open = pre_prices['open'].iloc[-1]
            prev_prev_close = pre_prices['close'].iloc[-2]
            features['opening_gap_prev'] = (
                prev_open / prev_prev_close - 1
            )

    except Exception:
        pass

    # VIX features
    if 'VIX' in prices_cache:
        vix_prices = prices_cache['VIX']
        pre_vix = vix_prices[vix_prices.index < event_date]

        if len(pre_vix) > 0:
            features['vix_level'] = pre_vix['close'].iloc[-1]
            vix_1yr = pre_vix['close'].tail(252)
            if len(vix_1yr) > 10:
                features['vix_percentile'] = (
                    (vix_1yr < features['vix_level']).mean()
                )

    return features

# ─────────────────────────────────────────────
# CATEGORY 3: SECTOR & TIME FEATURES
# ─────────────────────────────────────────────

def build_time_sector_features(
    row: pd.Series,
    event_date: pd.Timestamp
) -> dict:
    """Build sector and calendar features."""
    sector = row.get('sector', 'Unknown')
    month = event_date.month
    quarter = (month - 1) // 3 + 1
    day_of_week = event_date.dayofweek
    is_earnings_season = int(
        month in [1, 2, 4, 5, 7, 8, 10, 11]
    )

    return {
        'sector_encoded': SECTOR_ENCODING.get(sector, -1),
        'sector': sector,
        'month': month,
        'quarter': quarter,
        'day_of_week': day_of_week,
        'is_earnings_season': is_earnings_season
    }

# ─────────────────────────────────────────────
# RETURN DECOMPOSITION
# ─────────────────────────────────────────────

def calculate_return_decomposition(
    ticker: str,
    event_date: pd.Timestamp,
    report_time: str,
    prices_cache: dict
) -> dict:
    """
    Decompose total return into:
    - Opening gap: previous close to reaction day open
    - Intraday drift: reaction day open to close

    This tells us whether alpha comes from the
    immediate gap or from continued intraday drift.
    """
    result = {
        'opening_gap': np.nan,
        'intraday_drift': np.nan,
        'total_decomposed': np.nan
    }

    if ticker not in prices_cache:
        return result

    prices = prices_cache[ticker]
    all_dates = prices.index.sort_values()
    future_dates = all_dates[all_dates >= event_date]

    if len(future_dates) < 2:
        return result

    if report_time == 'BMO':
        reaction_day = future_dates[0]
    else:
        reaction_day = future_dates[1] if len(future_dates) > 1 else future_dates[0]

    prev_dates = all_dates[all_dates < reaction_day]
    if len(prev_dates) == 0:
        return result

    try:
        prev_day = prev_dates[-1]
        prev_close = prices.loc[prev_day, 'close']
        reaction_open = prices.loc[reaction_day, 'open']
        reaction_close = prices.loc[reaction_day, 'close']

        result['opening_gap'] = (reaction_open / prev_close) - 1
        result['intraday_drift'] = (reaction_close / reaction_open) - 1
        result['total_decomposed'] = (reaction_close / prev_close) - 1

    except Exception:
        pass

    return result

# ─────────────────────────────────────────────
# MAIN FEATURE ENGINEERING FUNCTION
# ─────────────────────────────────────────────

def build_feature_matrix() -> pd.DataFrame:
    """
    Main feature engineering pipeline.
    Builds 24 features across 4 categories:
    1. Earnings features (11) — including quality + sector surprise
    2. Market context (9) — pre-event price/volume/VIX
    3. Sector and time (6) — calendar and sector encoding
    4. Return decomposition (3) — gap vs drift analysis
    """
# Load labeled events
    labeled_df = pd.read_csv('data/processed/labeled_earnings.csv')
    labeled_df['event_date'] = pd.to_datetime(
        labeled_df['event_date']
    )
    print(f"Building features for {len(labeled_df)} events...")

    # ── Load earnings history for beat streak ──
    print("Calculating beat streak features...")
    conn = get_connection()
    earnings_history = pd.read_sql("""
        SELECT ticker, report_date, eps_surprise_pct,
               actual_eps, consensus_eps
        FROM earnings_events
        WHERE actual_eps IS NOT NULL
        ORDER BY ticker, report_date ASC
    """, conn)
    conn.close()

    earnings_with_streaks = calculate_beat_streaks(
        earnings_history
    )
    streak_map = {}
    for _, row in earnings_with_streaks.iterrows():
        key = (row['ticker'], str(row['report_date'])[:10])
        streak_map[key] = {
            'beat_streak': row['beat_streak'],
            'prev_surprise_1q': row['prev_surprise_1q'],
            'prev_surprise_2q': row['prev_surprise_2q'],
            'surprise_trend': row['surprise_trend']
        }
    print(f"Beat streak map built: {len(streak_map)} entries")

    # ── Momentum interaction features ──

    # ── Previous earnings gap ──
    print("Loading prices for previous earnings gap...")
    prices_cache_temp = load_price_cache()
    labeled_df = calculate_previous_earnings_gap(
        labeled_df, prices_cache_temp
    )
    print("Previous earnings gap calculated.")
    # ── Calculate sector statistics on training data only ──
    # Using training cutoff to prevent leakage into val/test
    train_cutoff = '2023-12-31'
    train_data = labeled_df[
        labeled_df['event_date'] <= train_cutoff
    ]
    sector_stats = train_data.groupby('sector').agg(
        sector_surprise_mean=('eps_surprise_pct', 'mean'),
        sector_surprise_std=('eps_surprise_pct', 'std')
    ).fillna(0)
    print(f"Sector stats from {len(train_data)} training events")
    print(sector_stats.to_string())

    # Load price cache once
    print("\nLoading price cache...")
    prices_cache = load_price_cache()

    all_features = []
    skipped = 0

    for i, row in labeled_df.iterrows():
        ticker = row['ticker']
        event_date = pd.Timestamp(row['event_date'])
        report_time = row.get('report_time', None)

        try:
             # Category 1: Earnings features
            earnings_feats = build_earnings_features(
                row, sector_stats
            )

            # Beat streak features
            streak_key = (
                ticker,
                str(row['event_date'])[:10]
            )
            streak_feats = streak_map.get(streak_key, {
                'beat_streak': 0,
                'prev_surprise_1q': np.nan,
                'prev_surprise_2q': np.nan,
                'surprise_trend': np.nan
            })

            # Previous earnings gap
            prev_gap = row.get('prev_earnings_gap', np.nan)

            # Category 2: Market context features
            market_feats = build_market_features(
                ticker, event_date, prices_cache
            )

            # Momentum interaction (calculated after market_feats)
            pre_ret_20d = market_feats.get('pre_return_20d', np.nan)
            pre_ret_5d = market_feats.get('pre_return_5d', np.nan)
            eps_surp_dir = earnings_feats.get('eps_surprise_direction', 0)
            eps_surp_pct = earnings_feats.get('eps_surprise_pct', np.nan)

            momentum_interaction = (
                pre_ret_20d * eps_surp_dir
                if not np.isnan(pre_ret_20d) else np.nan
            )
            signal_alignment = (
                1 if (not np.isnan(pre_ret_20d) and pre_ret_20d > 0
                      and not np.isnan(eps_surp_pct) and eps_surp_pct > 0)
                else -1 if (not np.isnan(pre_ret_20d) and pre_ret_20d < 0
                            and not np.isnan(eps_surp_pct) and eps_surp_pct < 0)
                else 0
            )
            momentum_5d_surprise = (
                pre_ret_5d * eps_surp_dir
                if not np.isnan(pre_ret_5d) else np.nan
            )

            # Category 3: Sector and time features
            time_feats = build_time_sector_features(
                row, event_date
            )

            # Category 4: Return decomposition
            decomp = calculate_return_decomposition(
                ticker, event_date, report_time, prices_cache
            )

            # Combine all features with labels
            combined = {
                # Identifiers
                'event_id': row['event_id'],
                'ticker': ticker,
                'event_date': row['event_date'],

                # Labels
                'label_direction': row['label_direction'],
                'label_magnitude': row['label_magnitude'],
                'is_market_moving': row['is_market_moving'],
                'abnormal_return_1d': row['abnormal_return_1d'],
                'return_1d': row['return_1d'],
                'return_3d': row['return_3d'],

                 # All features
                **earnings_feats,
                **market_feats,
                **time_feats,
                

                # Advanced features
                'beat_streak': streak_feats['beat_streak'],
                'prev_surprise_1q': streak_feats['prev_surprise_1q'],
                'prev_surprise_2q': streak_feats['prev_surprise_2q'],
                'surprise_trend': streak_feats['surprise_trend'],
                'momentum_surprise_interaction': momentum_interaction,
                'signal_alignment': signal_alignment,
                'momentum_5d_surprise': momentum_5d_surprise,
                'prev_earnings_gap': prev_gap,
            }

            all_features.append(combined)

        except Exception as e:
            skipped += 1
            continue

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(labeled_df)}...")

    feature_df = pd.DataFrame(all_features)

    print(f"\nFeature engineering complete:")
    print(f"  Events: {len(feature_df)}")
    print(f"  Skipped: {skipped}")
    print(f"  Total columns: {len(feature_df.columns)}")

    return feature_df

def validate_features(feature_df: pd.DataFrame):
    """Validate feature matrix quality."""
    print("\n=== FEATURE VALIDATION ===")

    label_cols = [
        'event_id', 'ticker', 'event_date', 'sector',
        'label_direction', 'label_magnitude', 'is_market_moving',
        'abnormal_return_1d', 'return_1d', 'return_3d',
        'opening_gap', 'intraday_drift', 'total_decomposed'
    ]
    feature_cols = [
        c for c in feature_df.columns if c not in label_cols
    ]

    print(f"\nTotal features: {len(feature_cols)}")
    print(f"Total events: {len(feature_df)}")

    # Null check
    print(f"\nNull value check:")
    for col in feature_cols:
        null_pct = feature_df[col].isna().mean() * 100
        status = "WARNING" if null_pct > 20 else "OK"
        print(f"  {status} {col}: {null_pct:.1f}% null")

    # New feature distributions
    print(f"\nNew feature distributions:")
    new_features = [
        'earnings_quality', 'both_beat', 'both_miss',
        'sector_relative_surprise', 'opening_gap_prev'
    ]
    for feat in new_features:
        if feat in feature_df.columns:
            non_null = feature_df[feat].dropna()
            print(f"  {feat}:")
            print(f"    Mean: {non_null.mean():.4f}")
            print(f"    Std:  {non_null.std():.4f}")
            print(f"    Null: {feature_df[feat].isna().mean()*100:.1f}%")

    # Correlations with target
    print(f"\nNew feature correlations with return_1d:")
    numeric_feats = feature_df[feature_cols].select_dtypes(
        include=[np.number]
    ).columns
    correlations = feature_df[numeric_feats].corrwith(
        feature_df['abnormal_return_1d']
    ).abs().sort_values(ascending=False)
    print(correlations.head(15).to_string())

    # Advanced feature correlations
    print(f"\nAdvanced feature correlations with return_1d:")
    adv_features = [
        'beat_streak', 'prev_surprise_1q', 'surprise_trend',
        'momentum_surprise_interaction', 'signal_alignment',
        'momentum_5d_surprise', 'prev_earnings_gap'
    ]
    for feat in adv_features:
        if feat in feature_df.columns:
            corr = feature_df[feat].corr(
                feature_df['abnormal_return_1d']
            )
            print(f"  {feat}: {corr:.4f}")

if __name__ == "__main__":
    feature_df = build_feature_matrix()

    if not feature_df.empty:
        validate_features(feature_df)
        os.makedirs('data/processed', exist_ok=True)
        feature_df.to_csv(
            'data/processed/feature_matrix.csv', index=False
        )
        print(f"\nSaved to data/processed/feature_matrix.csv")
        print(f"Shape: {feature_df.shape}")
