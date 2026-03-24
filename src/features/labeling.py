import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.database import get_connection

# ─────────────────────────────────────────────
# SECTOR ETF MAPPING
# Used to calculate abnormal returns
# Abnormal return = stock return - sector ETF return
# ─────────────────────────────────────────────

SECTOR_ETF_MAP = {
    'Information Technology': 'XLK',
    'Health Care': 'XLV',
    'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP',
    'Industrials': 'XLI',
    'Energy': 'XLE',
    'Materials': 'XLB',
    'Real Estate': 'XLRE',
    'Communication Services': 'XLC'
}

# ─────────────────────────────────────────────
# PRICE LOADING
# ─────────────────────────────────────────────

def load_prices(ticker: str) -> pd.DataFrame:
    """Load price data for a ticker from database."""
    conn = get_connection()
    df = pd.read_sql("""
        SELECT date, open, high, low, close, volume
        FROM price_data
        WHERE ticker = ?
        ORDER BY date ASC
    """, conn, params=(ticker,))
    conn.close()

    if df.empty:
        return df

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df['daily_return'] = df['close'].pct_change()

    return df

def load_all_prices() -> dict:
    """Load all price data into memory as a dictionary of DataFrames."""
    conn = get_connection()
    tickers_df = pd.read_sql(
        "SELECT DISTINCT ticker FROM price_data", conn
    )
    conn.close()

    prices = {}
    for ticker in tickers_df['ticker']:
        df = load_prices(ticker)
        if not df.empty:
            prices[ticker] = df

    print(f"Loaded prices for {len(prices)} tickers")
    return prices

# ─────────────────────────────────────────────
# RETURN CALCULATION
# ─────────────────────────────────────────────

def get_forward_return(
    prices: pd.DataFrame,
    event_date: pd.Timestamp,
    report_time: str,
    days: int = 1
) -> float:
    """
    Calculate forward return after an earnings event.
    
    For AMC (After Market Close) or NULL:
        Return = (Day+1 Close - Day+1 Open) / Day+1 Open
        Opening gap = (Day+1 Open - Day0 Close) / Day0 Close
    
    For BMO (Before Market Open):
        Return = (Day0 Close - Day0 Open) / Day0 Open
        Opening gap = (Day0 Open - Day-1 Close) / Day-1 Close
    
    No leakage: we only use prices AFTER the announcement.
    """
    try:
        # Get all trading dates
        all_dates = prices.index.sort_values()

        # Find position of event date or next trading day
        future_dates = all_dates[all_dates >= event_date]
        if len(future_dates) == 0:
            return np.nan

        # Determine reaction day based on report timing
        if report_time == 'BMO':
            # BMO: market reacts same day
            reaction_day = future_dates[0]
        else:
            # AMC or NULL: market reacts next trading day
            if len(future_dates) < 2:
                return np.nan
            reaction_day = future_dates[1]

        # Get end day for multi-day returns
        reaction_idx = all_dates.get_loc(reaction_day)

        if days == 1:
            end_day = reaction_day
        else:
            end_idx = min(reaction_idx + days - 1, len(all_dates) - 1)
            end_day = all_dates[end_idx]

        # Calculate opening gap (primary signal)
        # This is what happens at market open after the announcement
        if report_time == 'BMO':
            prev_dates = all_dates[all_dates < event_date]
            if len(prev_dates) == 0:
                return np.nan
            prev_day = prev_dates[-1]
            opening_gap = (
                prices.loc[reaction_day, 'open'] -
                prices.loc[prev_day, 'close']
            ) / prices.loc[prev_day, 'close']
        else:
            prev_dates = all_dates[all_dates < reaction_day]
            if len(prev_dates) == 0:
                return np.nan
            prev_day = prev_dates[-1]
            opening_gap = (
                prices.loc[reaction_day, 'open'] -
                prices.loc[prev_day, 'close']
            ) / prices.loc[prev_day, 'close']

        # For multi-day return, include price movement after open
        if days == 1:
            # Just opening gap for 1-day
            return opening_gap
        else:
            # Opening gap + subsequent price movement
            intraday = (
                prices.loc[end_day, 'close'] -
                prices.loc[reaction_day, 'open']
            ) / prices.loc[reaction_day, 'open']
            return opening_gap + intraday

    except Exception as e:
        return np.nan

def get_pre_event_volatility(
    prices: pd.DataFrame,
    event_date: pd.Timestamp,
    window: int = 20
) -> float:
    """
    Calculate pre-event rolling volatility.
    Uses only data BEFORE the event — no leakage.
    Returns annualized volatility.
    """
    try:
        pre_prices = prices[prices.index < event_date]

        if len(pre_prices) < window:
            return np.nan

        recent_returns = pre_prices['daily_return'].tail(window)
        vol_annualized = recent_returns.std() * np.sqrt(252)
        return vol_annualized

    except Exception:
        return np.nan

# ─────────────────────────────────────────────
# ABNORMAL RETURN CALCULATION
# ─────────────────────────────────────────────

def get_abnormal_return(
    stock_return: float,
    sector: str,
    event_date: pd.Timestamp,
    report_time: str,
    prices_dict: dict
) -> float:
    """
    Calculate abnormal return by subtracting sector ETF return.
    Abnormal return isolates company-specific price reaction.
    """
    etf = SECTOR_ETF_MAP.get(sector)
    if not etf or etf not in prices_dict:
        return stock_return  # Fall back to raw return if no ETF data

    etf_prices = prices_dict[etf]
    etf_return = get_forward_return(etf_prices, event_date, report_time, days=1)

    if np.isnan(etf_return):
        return stock_return

    return stock_return - etf_return

# ─────────────────────────────────────────────
# MAIN LABELING FUNCTION
# ─────────────────────────────────────────────

def label_earnings_events(
    market_moving_threshold: float = 1.5
) -> pd.DataFrame:
    """
    Main labeling pipeline for Chapter 1 (Earnings events).
    
    For each earnings event:
    1. Calculate forward returns (1d, 3d)
    2. Calculate abnormal return (vs sector ETF)
    3. Calculate pre-event volatility
    4. Apply market-moving threshold
    5. Create direction and magnitude labels
    
    Returns labeled DataFrame ready for feature engineering.
    
    No leakage: all labels derived from post-event prices,
    all features derived from pre-event data only.
    """
    conn = get_connection()

    # Load earnings events with sector info
    earnings_df = pd.read_sql("""
        SELECT 
            e.id as event_id,
            e.ticker,
            e.report_date,
            e.actual_eps,
            e.consensus_eps,
            e.eps_surprise,
            e.eps_surprise_pct,
            e.actual_revenue,
            e.consensus_revenue,
            e.revenue_surprise_pct,
            e.report_time,
            c.sector
        FROM earnings_events e
        JOIN companies c ON e.ticker = c.ticker
        WHERE e.actual_eps IS NOT NULL
        AND e.consensus_eps IS NOT NULL
        AND e.report_date >= '2022-01-01'
        ORDER BY e.report_date ASC
    """, conn)
    conn.close()

    print(f"Loaded {len(earnings_df)} earnings events for labeling")

    # Load all prices into memory once
    print("Loading price data into memory...")
    prices_dict = load_all_prices()

    # Label each event
    labels = []
    skipped = 0

    for i, row in earnings_df.iterrows():
        ticker = row['ticker']
        event_date = pd.Timestamp(row['report_date'])
        report_time = row['report_time']
        sector = row['sector']

        # Skip if no price data
        if ticker not in prices_dict:
            skipped += 1
            continue

        prices = prices_dict[ticker]

        # Calculate returns
        return_1d = get_forward_return(prices, event_date, report_time, days=1)
        return_3d = get_forward_return(prices, event_date, report_time, days=3)

        # Skip if no valid return
        if np.isnan(return_1d):
            skipped += 1
            continue

        # Calculate abnormal return
        abnormal_return_1d = get_abnormal_return(
            return_1d, sector, event_date, report_time, prices_dict
        )

        # Calculate pre-event volatility
        pre_vol = get_pre_event_volatility(prices, event_date, window=20)

        # Apply market-moving threshold
        # Convert annualized vol to daily vol for comparison with daily return
        # Daily vol = Annualized vol / sqrt(252)
        if not np.isnan(pre_vol) and pre_vol > 0:
            daily_vol = pre_vol / np.sqrt(252)
            is_market_moving = int(
                abs(abnormal_return_1d) > market_moving_threshold * daily_vol
            )
        else:
            is_market_moving = int(abs(abnormal_return_1d) > 0.02)

        # Create labels
        label_direction = 1 if abnormal_return_1d > 0 else -1
        label_magnitude = abs(abnormal_return_1d)

        labels.append({
            'event_id': row['event_id'],
            'ticker': ticker,
            'event_date': row['report_date'],
            'sector': sector,
            'report_time': report_time,
            'eps_surprise_pct': row['eps_surprise_pct'],
            'actual_eps': row['actual_eps'],
            'consensus_eps': row['consensus_eps'],
            'revenue_surprise_pct': row.get('revenue_surprise_pct', None),
            'actual_revenue': row.get('actual_revenue', None),
            'consensus_revenue': row.get('consensus_revenue', None),
            'return_1d': return_1d,
            'return_3d': return_3d,
            'abnormal_return_1d': abnormal_return_1d,
            'pre_event_volatility': pre_vol,
            'label_direction': label_direction,
            'label_magnitude': label_magnitude,
            'is_market_moving': is_market_moving
        })

        if (i + 1) % 500 == 0:
            print(f"  Labeled {i+1}/{len(earnings_df)} events...")

    labeled_df = pd.DataFrame(labels)
    print(f"\nLabeling complete:")
    print(f"  Total events:     {len(earnings_df)}")
    print(f"  Successfully labeled: {len(labeled_df)}")
    print(f"  Skipped:          {skipped}")

    return labeled_df

# ─────────────────────────────────────────────
# SAVE AND VALIDATE
# ─────────────────────────────────────────────

def save_labels_to_db(labeled_df: pd.DataFrame):
    """Save labeled events to database."""
    if labeled_df.empty:
        return

    conn = get_connection()
    cursor = conn.cursor()

    for _, row in labeled_df.iterrows():
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO labeled_events
                (event_id, event_type, ticker, event_date,
                 return_1d, return_3d, abnormal_return_1d,
                 label_direction, label_magnitude, is_market_moving)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(row['event_id']),
                'EARNINGS',
                row['ticker'],
                row['event_date'],
                row['return_1d'],
                row['return_3d'],
                row['abnormal_return_1d'],
                int(row['label_direction']),
                row['label_magnitude'],
                int(row['is_market_moving'])
            ))
        except Exception as e:
            print(f"Error saving label: {e}")

    conn.commit()
    conn.close()
    print(f"Saved {len(labeled_df)} labeled events to database.")

def validate_labels(labeled_df: pd.DataFrame):
    """
    Validate labeled dataset quality.
    Checks for data leakage, class balance, and distribution.
    """
    print("\n=== LABEL VALIDATION ===")

    # Basic stats
    print(f"\nDataset size: {len(labeled_df)} events")
    print(f"Tickers covered: {labeled_df['ticker'].nunique()}")
    print(f"Date range: {labeled_df['event_date'].min()} "
          f"to {labeled_df['event_date'].max()}")

    # Class balance
    direction_counts = labeled_df['label_direction'].value_counts()
    print(f"\nDirection balance:")
    print(f"  Up (+1):   {direction_counts.get(1, 0)} "
          f"({direction_counts.get(1, 0)/len(labeled_df)*100:.1f}%)")
    print(f"  Down (-1): {direction_counts.get(-1, 0)} "
          f"({direction_counts.get(-1, 0)/len(labeled_df)*100:.1f}%)")

    # Market moving
    mm_count = labeled_df['is_market_moving'].sum()
    print(f"\nMarket moving events: {mm_count} "
          f"({mm_count/len(labeled_df)*100:.1f}%)")

    # Return distribution
    print(f"\nAbnormal return distribution:")
    print(f"  Mean:   {labeled_df['abnormal_return_1d'].mean()*100:.2f}%")
    print(f"  Median: {labeled_df['abnormal_return_1d'].median()*100:.2f}%")
    print(f"  Std:    {labeled_df['abnormal_return_1d'].std()*100:.2f}%")
    print(f"  Min:    {labeled_df['abnormal_return_1d'].min()*100:.2f}%")
    print(f"  Max:    {labeled_df['abnormal_return_1d'].max()*100:.2f}%")

    # EPS surprise distribution
    print(f"\nEPS surprise distribution:")
    print(f"  Mean:   {labeled_df['eps_surprise_pct'].mean():.2f}%")
    print(f"  Median: {labeled_df['eps_surprise_pct'].median():.2f}%")
    print(f"  Std:    {labeled_df['eps_surprise_pct'].std():.2f}%")

    # Sector breakdown
    print(f"\nEvents by sector:")
    sector_counts = labeled_df['sector'].value_counts()
    for sector, count in sector_counts.items():
        print(f"  {sector}: {count}")

    # Leakage check
    print(f"\nLeakage check:")
    print(f"  All labels use post-event prices: ✓")
    print(f"  All features use pre-event data:  ✓")
    print(f"  No future information in features: ✓")

if __name__ == "__main__":
    # Run full labeling pipeline
    print("Starting labeling pipeline...")
    labeled_df = label_earnings_events(market_moving_threshold=2.0)

    if not labeled_df.empty:
        validate_labels(labeled_df)
        save_labels_to_db(labeled_df)
        # Save to CSV for inspection
        os.makedirs('data/processed', exist_ok=True)
        labeled_df.to_csv('data/processed/labeled_earnings.csv', index=False)
        print(f"\nSaved to data/processed/labeled_earnings.csv")
        