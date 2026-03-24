import os
import sys
import time
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.data.database import initialize_database, get_connection
from src.data.universe import get_universe_from_db
from src.data.price_fetcher import fetch_price_data, save_prices_to_db
from src.data.earnings_fetcher import get_historical_earnings, save_earnings_to_db
from src.data.news_fetcher import get_company_cik, get_8k_filings, save_edgar_filings_to_db

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

PRICE_START = "2020-01-01"
PRICE_END   = "2025-12-31"


FMP_DAILY_LIMIT = 240  # Leave 10 calls as buffer  #changed it into yfinance now (only 3years)

# EDGAR: 10 requests/second max
EDGAR_DELAY = 0.15

# yfinance: no hard limit but be respectful
PRICE_DELAY = 0.1

# ─────────────────────────────────────────────
# PROGRESS TRACKING
# ─────────────────────────────────────────────

PROGRESS_FILE = "data/collection_progress.json"

def load_progress() -> dict:
    """
    Load collection progress to resume if interrupted.
    If no progress file exists, start fresh.
    """
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            progress = json.load(f)
            print(f"Resuming from saved progress file.")
            return progress

    # Fresh start
    return {
        "prices_done": [],
        "earnings_done": [],
        "edgar_done": [],
        "prices_failed": [],
        "earnings_failed": [],
        "edgar_failed": []
    }

def save_progress(progress: dict):
    """
    Save progress after every single ticker.
    This ensures nothing is lost if script is interrupted.
    """
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

# ─────────────────────────────────────────────
# COLLECTION FUNCTIONS
# ─────────────────────────────────────────────

def collect_prices(tickers: list, progress: dict):
    """Fetch and save price data for all tickers."""
    print("\n" + "="*60)
    print("COLLECTING PRICE DATA")
    print("="*60)

    remaining = [t for t in tickers if t not in progress['prices_done']
                 and t not in progress['prices_failed']]
    print(f"Already done:    {len(progress['prices_done'])}")
    print(f"Remaining:       {len(remaining)}")
    print(f"Previously failed: {len(progress['prices_failed'])}")

    if not remaining:
        print("All prices already collected. Skipping.")
        return

    for i, ticker in enumerate(remaining):
        try:
            print(f"[{i+1}/{len(remaining)}] Prices: {ticker}...", end=" ", flush=True)
            df = fetch_price_data(ticker, PRICE_START, PRICE_END)

            if not df.empty:
                save_prices_to_db(ticker, df)
                progress['prices_done'].append(ticker)
                print(f"OK ({len(df)} rows)")
            else:
                progress['prices_failed'].append(ticker)
                print("EMPTY")

            # Save after every ticker — no data loss on interruption
            save_progress(progress)
            time.sleep(PRICE_DELAY)

        except KeyboardInterrupt:
            print("\nInterrupted by user. Progress saved.")
            save_progress(progress)
            return

        except Exception as e:
            print(f"ERROR: {e}")
            progress['prices_failed'].append(ticker)
            save_progress(progress)

    print(f"\nPrices complete: {len(progress['prices_done'])} done, "
          f"{len(progress['prices_failed'])} failed")

def collect_earnings(tickers: list, progress: dict):
    """Fetch and save earnings data for all tickers."""
    print("\n" + "="*60)
    print("COLLECTING EARNINGS DATA")
    print("="*60)

    remaining = [t for t in tickers if t not in progress['earnings_done']
                 and t not in progress['earnings_failed']]
    print(f"Already done:    {len(progress['earnings_done'])}")
    print(f"Remaining:       {len(remaining)}")
    print(f"Previously failed: {len(progress['earnings_failed'])}")

    if not remaining:
        print("All earnings already collected. Skipping.")
        return

    # Respect FMP daily limit
    to_fetch = remaining[:FMP_DAILY_LIMIT]
    if len(remaining) > FMP_DAILY_LIMIT:
        print(f"\nFMP daily limit: {FMP_DAILY_LIMIT} calls")
        print(f"Fetching {len(to_fetch)} today, "
              f"{len(remaining) - len(to_fetch)} remaining for tomorrow")

    for i, ticker in enumerate(to_fetch):
        try:
            print(f"[{i+1}/{len(to_fetch)}] Earnings: {ticker}...", end=" ", flush=True)
            df = get_historical_earnings(ticker)

            if not df.empty:
                save_earnings_to_db(ticker, df)
                progress['earnings_done'].append(ticker)
                print(f"OK ({len(df)} quarters)")
            else:
                progress['earnings_failed'].append(ticker)
                print("EMPTY")

            # Save after every ticker
            save_progress(progress)
            time.sleep(0.3)  # FMP rate limit

        except KeyboardInterrupt:
            print("\nInterrupted by user. Progress saved.")
            save_progress(progress)
            return

        except Exception as e:
            print(f"ERROR: {e}")
            progress['earnings_failed'].append(ticker)
            save_progress(progress)

    print(f"\nEarnings complete today: {len(progress['earnings_done'])} done, "
          f"{len(progress['earnings_failed'])} failed")

    if len(remaining) > FMP_DAILY_LIMIT:
        print(f"Run script again tomorrow to fetch remaining "
              f"{len(remaining) - FMP_DAILY_LIMIT} tickers.")

def collect_edgar_filings(tickers: list, progress: dict):
    """Fetch and save EDGAR 8-K filings for all tickers."""
    print("\n" + "="*60)
    print("COLLECTING EDGAR 8-K FILINGS")
    print("="*60)

    remaining = [t for t in tickers if t not in progress['edgar_done']
                 and t not in progress['edgar_failed']]
    print(f"Already done:    {len(progress['edgar_done'])}")
    print(f"Remaining:       {len(remaining)}")
    print(f"Previously failed: {len(progress['edgar_failed'])}")

    if not remaining:
        print("All EDGAR filings already collected. Skipping.")
        return

    for i, ticker in enumerate(remaining):
        try:
            print(f"[{i+1}/{len(remaining)}] EDGAR: {ticker}...", end=" ", flush=True)

            # Step 1: Get CIK number
            cik = get_company_cik(ticker)
            if not cik:
                print("NO CIK FOUND")
                progress['edgar_failed'].append(ticker)
                save_progress(progress)
                time.sleep(EDGAR_DELAY)
                continue

            time.sleep(EDGAR_DELAY)

            # Step 2: Get 8-K filings
            filings_df = get_8k_filings(cik, ticker)

            if not filings_df.empty:
                # Filter to our date range
                filings_df['filing_date'] = filings_df['filing_date'].astype(str)
                filings_df = filings_df[
                    (filings_df['filing_date'] >= PRICE_START) &
                    (filings_df['filing_date'] <= PRICE_END)
                ]
                save_edgar_filings_to_db(ticker, filings_df)
                progress['edgar_done'].append(ticker)
                print(f"OK ({len(filings_df)} 8-Ks in range)")
            else:
                progress['edgar_failed'].append(ticker)
                print("EMPTY")

            # Save after every ticker
            save_progress(progress)
            time.sleep(EDGAR_DELAY)

        except KeyboardInterrupt:
            print("\nInterrupted by user. Progress saved.")
            save_progress(progress)
            return

        except Exception as e:
            print(f"ERROR: {e}")
            progress['edgar_failed'].append(ticker)
            save_progress(progress)

    print(f"\nEDGAR complete: {len(progress['edgar_done'])} done, "
          f"{len(progress['edgar_failed'])} failed")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("="*60)
    print("NEWS-ALPHA: BULK DATA COLLECTION")
    print("="*60)
    print("Progress is saved after every ticker.")
    print("Safe to interrupt and resume at any time.")
    print("="*60)

    # Initialize database
    initialize_database()

    # Load universe
    universe_df = get_universe_from_db()
    if universe_df.empty:
        print("ERROR: Universe not found in database.")
        print("Run validate_setup.py first.")
        return

    tickers = universe_df['ticker'].tolist()
    print(f"\nUniverse loaded: {len(tickers)} stocks")

    # Load progress
    progress = load_progress()
    print(f"Progress loaded:")
    print(f"  Prices done:   {len(progress['prices_done'])}")
    print(f"  Earnings done: {len(progress['earnings_done'])}")
    print(f"  EDGAR done:    {len(progress['edgar_done'])}")

    # Run collection
    collect_prices(tickers, progress)
    collect_earnings(tickers, progress)
    collect_edgar_filings(tickers, progress)

    # Final summary
    print("\n" + "="*60)
    print("COLLECTION SUMMARY")
    print("="*60)
    print(f"Prices:   {len(progress['prices_done'])} done, "
          f"{len(progress['prices_failed'])} failed")
    print(f"Earnings: {len(progress['earnings_done'])} done, "
          f"{len(progress['earnings_failed'])} failed")
    print(f"EDGAR:    {len(progress['edgar_done'])} done, "
          f"{len(progress['edgar_failed'])} failed")

    # Database row counts
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM price_data")
    price_rows = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM earnings_events")
    earnings_rows = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM news_articles WHERE chapter=1")
    news_rows = cursor.fetchone()[0]
    conn.close()

    print(f"\nDatabase totals:")
    print(f"  Price rows:    {price_rows:,}")
    print(f"  Earnings rows: {earnings_rows:,}")
    print(f"  8-K filings:   {news_rows:,}")
    print("\nTo resume collection, just run this script again.")
    print("Already completed tickers will be skipped automatically.")

if __name__ == "__main__":
    main()