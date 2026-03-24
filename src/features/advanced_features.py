import pandas as pd
import numpy as np
import requests
import re
import time
import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)))

from src.data.database import get_connection

HEADERS = {
    "User-Agent": "news-alpha-project research@example.com"
}

# ─────────────────────────────────────────────
# FEATURE 1: BEAT STREAK
# Consecutive quarters of beats or misses
# ─────────────────────────────────────────────

def calculate_beat_streaks(earnings_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each earnings event calculate:
    - beat_streak: consecutive beats before this quarter (+) or misses (-)
    - prev_surprise_1q: EPS surprise last quarter
    - prev_surprise_2q: EPS surprise 2 quarters ago
    - surprise_trend: is surprise improving or declining?
    """
    earnings_df = earnings_df.sort_values(
        ['ticker', 'report_date']
    ).copy()

    beat_streaks = []
    prev_surprise_1q = []
    prev_surprise_2q = []
    surprise_trends = []

    for ticker, group in earnings_df.groupby('ticker'):
        group = group.reset_index(drop=True)
        surprise_col = 'eps_surprise_pct'

        for i in range(len(group)):
            surprise = group.loc[i, surprise_col]

            # Previous quarter surprise
            p1 = group.loc[i-1, surprise_col] if i >= 1 else np.nan
            p2 = group.loc[i-2, surprise_col] if i >= 2 else np.nan

            prev_surprise_1q.append(p1)
            prev_surprise_2q.append(p2)

            # Surprise trend (improving = positive)
            if not np.isnan(p1) and not np.isnan(surprise):
                trend = np.sign(surprise) - np.sign(p1)
                surprise_trends.append(trend)
            else:
                surprise_trends.append(np.nan)

            # Beat streak calculation
            if i == 0:
                beat_streaks.append(0)
                continue

            streak = 0
            current_sign = np.sign(surprise) if not np.isnan(surprise) else 0

            for j in range(i-1, -1, -1):
                prev = group.loc[j, surprise_col]
                if np.isnan(prev):
                    break
                if np.sign(prev) == current_sign:
                    streak += 1
                else:
                    break

            beat_streaks.append(
                streak * (1 if current_sign >= 0 else -1)
            )

    earnings_df['beat_streak'] = beat_streaks
    earnings_df['prev_surprise_1q'] = prev_surprise_1q
    earnings_df['prev_surprise_2q'] = prev_surprise_2q
    earnings_df['surprise_trend'] = surprise_trends

    return earnings_df

# ─────────────────────────────────────────────
# FEATURE 2: MOMENTUM × SURPRISE INTERACTION
# ─────────────────────────────────────────────

def calculate_momentum_interaction(
    labeled_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Momentum × Surprise interaction features.

    Key insight: A stock trending UP going into earnings
    that then BEATS has much stronger post-earnings drift
    than a flat stock that beat.

    Smart money positioning amplifies the earnings reaction.
    """
    df = labeled_df.copy()

    # Core interaction: pre-earnings momentum × surprise direction
    df['momentum_surprise_interaction'] = (
        df['pre_return_20d'] *
        df['eps_surprise_direction']
    )

    # Aligned signal: both momentum and surprise agree
    # +1 = uptrend + beat, -1 = downtrend + miss
    # 0 = contradictory signals
    df['signal_alignment'] = np.where(
        (df['pre_return_20d'] > 0) & (df['eps_surprise_pct'] > 0),
        1,  # Uptrend + beat = strong positive
        np.where(
            (df['pre_return_20d'] < 0) & (df['eps_surprise_pct'] < 0),
            -1,  # Downtrend + miss = strong negative
            0    # Contradictory signals
        )
    )

    # Short-term momentum interaction
    df['momentum_5d_surprise'] = (
        df['pre_return_5d'] *
        df['eps_surprise_direction']
    )

    return df

# ─────────────────────────────────────────────
# FEATURE 3: PREVIOUS EARNINGS GAP
# Market learns from prior reactions
# ─────────────────────────────────────────────

def calculate_previous_earnings_gap(
    labeled_df: pd.DataFrame,
    prices_cache: dict
) -> pd.DataFrame:
    """
    Previous earnings gap feature.

    If a stock gapped up 5% on last quarter's beat,
    the market partially prices in similar reaction
    for this quarter's beat.

    Captures second-order market learning effect.
    """
    df = labeled_df.copy()
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values(['ticker', 'event_date'])

    prev_gaps = []

    for ticker, group in df.groupby('ticker'):
        group = group.reset_index(drop=True)

        for i in range(len(group)):
            if i == 0:
                prev_gaps.append(np.nan)
                continue

            # Get previous event date
            prev_date = group.loc[i-1, 'event_date']
            prev_report_time = group.loc[i-1, 'report_time']

            if ticker not in prices_cache:
                prev_gaps.append(np.nan)
                continue

            prices = prices_cache[ticker]
            all_dates = prices.index.sort_values()
            future_dates = all_dates[all_dates >= prev_date]

            if len(future_dates) < 2:
                prev_gaps.append(np.nan)
                continue

            try:
                if prev_report_time == 'BMO':
                    reaction_day = future_dates[0]
                else:
                    reaction_day = future_dates[1]

                prev_dates = all_dates[
                    all_dates < reaction_day
                ]
                if len(prev_dates) == 0:
                    prev_gaps.append(np.nan)
                    continue

                prev_close = prices.loc[
                    prev_dates[-1], 'close'
                ]
                reaction_open = prices.loc[
                    reaction_day, 'open'
                ]
                gap = (reaction_open / prev_close) - 1
                prev_gaps.append(gap)

            except Exception:
                prev_gaps.append(np.nan)

    df['prev_earnings_gap'] = prev_gaps
    return df

# ─────────────────────────────────────────────
# FEATURE 4: GUIDANCE SENTIMENT FROM 8-K TEXT
# Most original feature — NLP on primary source
# ─────────────────────────────────────────────

BULLISH_GUIDANCE = [
    'raising guidance', 'raise guidance', 'raised guidance',
    'increasing outlook', 'raised our outlook',
    'above our expectations', 'exceeded our expectations',
    'strong demand', 'record revenue', 'all-time high',
    'raising our', 'we expect growth', 'accelerating',
    'momentum continues', 'confident in',
    'above the high end', 'above expectations'
]

BEARISH_GUIDANCE = [
    'lowering guidance', 'lower guidance', 'lowered guidance',
    'reducing outlook', 'reduced our outlook',
    'below our expectations', 'headwinds', 'challenging',
    'softening demand', 'below expectations',
    'macroeconomic uncertainty', 'cautious outlook',
    'below the low end', 'weaker than expected',
    'deteriorating', 'pressure on margins'
]

def get_8k_text(cik: str, accession: str) -> str:
    """Fetch 8-K filing text from EDGAR."""
    acc_clean = accession.replace('-', '')

    # Get filing index
    index_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{acc_clean}/{accession}-index.htm"
    )

    try:
        r = requests.get(
            index_url, headers=HEADERS, timeout=10
        )
        if r.status_code != 200:
            return ""

        # Find ex99.1 (earnings press release)
        links = re.findall(r'href="([^"]+\.htm)"', r.text)
        ex99_links = [
            l for l in links
            if 'ex99' in l.lower() or 'ex-99' in l.lower()
        ]

        if not ex99_links:
            # Fall back to primary document
            links = [l for l in links if l != '/index.htm']
            if not links:
                return ""
            ex99_links = [links[0]]

        # Fetch the press release
        doc_path = ex99_links[0]
        if doc_path.startswith('/'):
            doc_url = f"https://www.sec.gov{doc_path}"
        else:
            doc_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{acc_clean}/{doc_path}"
            )

        # Remove iXBRL viewer prefix
        doc_url = doc_url.replace(
            '/ix?doc=', ''
        ).replace('/ix?doc=/', '/')

        doc_r = requests.get(
            doc_url, headers=HEADERS, timeout=10
        )
        if doc_r.status_code != 200:
            return ""

        # Strip HTML and clean text
        text = re.sub(r'<[^>]+>', ' ', doc_r.text)
        text = re.sub(r'&#\d+;', ' ', text)
        text = re.sub(r'&[a-z]+;', ' ', text)
        text = ' '.join(text.split()).lower()

        return text

    except Exception:
        return ""

def score_guidance(text: str) -> dict:
    """
    Score guidance sentiment from 8-K text.
    Returns bullish_count, bearish_count, guidance_score.
    """
    if not text:
        return {
            'bullish_count': 0,
            'bearish_count': 0,
            'guidance_score': 0,
            'guidance_available': 0
        }

    bullish = sum(1 for kw in BULLISH_GUIDANCE if kw in text)
    bearish = sum(1 for kw in BEARISH_GUIDANCE if kw in text)
    total = bullish + bearish

    score = (
        (bullish - bearish) / total if total > 0 else 0
    )

    return {
        'bullish_count': bullish,
        'bearish_count': bearish,
        'guidance_score': score,
        'guidance_available': 1 if total > 0 else 0
    }

def build_guidance_features(
    labeled_df: pd.DataFrame,
    cache_path: str = 'data/processed/guidance_cache.json'
) -> pd.DataFrame:
    """
    Build guidance sentiment features from EDGAR 8-K filings.
    Matches earnings events to their 8-K filing and scores
    the guidance language.
    """
    print("Building guidance features from 8-K filings...")

    # Load cache if exists
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached guidance scores")
    else:
        cache = {}

    # Load EDGAR filing info
    conn = get_connection()

    # Get CIK mapping
    cik_response = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=HEADERS
    )
    cik_data = cik_response.json()
    ticker_to_cik = {
        v['ticker']: str(v['cik_str']).zfill(10)
        for v in cik_data.values()
    }

    # Get 8-K filings from database
    edgar_df = pd.read_sql("""
        SELECT ticker, published_at
        FROM news_articles
        WHERE source = 'SEC EDGAR'
        AND chapter = 1
    """, conn)
    conn.close()

    edgar_df['published_at'] = pd.to_datetime(
        edgar_df['published_at']
    )

    labeled_df = labeled_df.copy()
    labeled_df['event_date'] = pd.to_datetime(
        labeled_df['event_date']
    )

    guidance_scores = []
    fetched = 0
    cache_hits = 0

    for _, row in labeled_df.iterrows():
        ticker = row['ticker']
        event_date = row['event_date']

        cache_key = f"{ticker}_{event_date.date()}"

        if cache_key in cache:
            guidance_scores.append(cache[cache_key])
            cache_hits += 1
            continue

        # Find matching 8-K filing (within 5 days after earnings)
        ticker_filings = edgar_df[
            edgar_df['ticker'] == ticker
        ]
        window_start = event_date - pd.Timedelta(days=1)
        window_end = event_date + pd.Timedelta(days=5)

        matching = ticker_filings[
            (ticker_filings['published_at'] >= window_start) &
            (ticker_filings['published_at'] <= window_end)
        ]

        if matching.empty or ticker not in ticker_to_cik:
            guidance_scores.append({
                'bullish_count': 0,
                'bearish_count': 0,
                'guidance_score': 0,
                'guidance_available': 0
            })
            cache[cache_key] = guidance_scores[-1]
            continue

        # Get CIK
        cik = ticker_to_cik[ticker]

        # Need accession number — fetch from EDGAR
        try:
            sub_url = (
                f"https://data.sec.gov/submissions/"
                f"CIK{cik}.json"
            )
            sub_r = requests.get(
                sub_url, headers=HEADERS, timeout=10
            )
            sub_data = sub_r.json()

            filings = sub_data.get(
                'filings', {}
            ).get('recent', {})
            forms = filings.get('form', [])
            accessions = filings.get('accessionNumber', [])
            dates = filings.get('filingDate', [])

            # Find 8-K within window
            target_accession = None
            for i, form in enumerate(forms):
                if form == '8-K':
                    filing_date = pd.Timestamp(dates[i])
                    if window_start <= filing_date <= window_end:
                        target_accession = accessions[i]
                        break

            if not target_accession:
                guidance_scores.append({
                    'bullish_count': 0,
                    'bearish_count': 0,
                    'guidance_score': 0,
                    'guidance_available': 0
                })
                cache[cache_key] = guidance_scores[-1]
                continue

            time.sleep(0.15)
            text = get_8k_text(cik, target_accession)
            score = score_guidance(text)

            guidance_scores.append(score)
            cache[cache_key] = score
            fetched += 1

            if fetched % 20 == 0:
                print(f"  Fetched {fetched} filings, "
                      f"{cache_hits} cache hits...")
                with open(cache_path, 'w') as f:
                    json.dump(cache, f)

            time.sleep(0.15)

        except Exception as e:
            guidance_scores.append({
                'bullish_count': 0,
                'bearish_count': 0,
                'guidance_score': 0,
                'guidance_available': 0
            })
            cache[cache_key] = guidance_scores[-1]

    # Save cache
    with open(cache_path, 'w') as f:
        json.dump(cache, f)
    print(f"Guidance features complete: "
          f"{fetched} fetched, {cache_hits} cached")

    # Add to dataframe
    guidance_df = pd.DataFrame(guidance_scores)
    for col in guidance_df.columns:
        labeled_df[col] = guidance_df[col].values

    return labeled_df

if __name__ == "__main__":
    # Quick test
    conn = get_connection()
    earnings_df = pd.read_sql("""
        SELECT ticker, report_date, eps_surprise_pct,
               actual_eps, consensus_eps
        FROM earnings_events
        WHERE actual_eps IS NOT NULL
        ORDER BY ticker, report_date
    """, conn)
    conn.close()

    print("Testing beat streak calculation...")
    result = calculate_beat_streaks(earnings_df)
    aapl = result[result['ticker'] == 'AAPL'].tail(8)
    print(aapl[['report_date', 'eps_surprise_pct',
                'beat_streak', 'prev_surprise_1q']].to_string())
