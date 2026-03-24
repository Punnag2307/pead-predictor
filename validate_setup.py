import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()
print("=" * 60)
print("NEWS-ALPHA PROJECT VALIDATION")
print("=" * 60)
# ─────────────────────────────────────────────
# CHECK 1: API Keys
# ─────────────────────────────────────────────
print("\n[1] Checking API keys...")
finnhub_key = os.getenv("FINNHUB_API_KEY")
fmp_key = os.getenv("FMP_API_KEY")
av_key = os.getenv("ALPHA_VANTAGE_API_KEY")
db_path = os.getenv("DB_PATH")
print(f"  Finnhub API key:        {'OK' if finnhub_key else 'MISSING'}")
print(f"  FMP API key:            {'OK' if fmp_key else 'MISSING'}")
print(f"  Alpha Vantage API key:  {'OK' if av_key else 'MISSING'}")
print(f"  DB path:                {db_path}")
# ─────────────────────────────────────────────
# CHECK 2: Database
# ─────────────────────────────────────────────
print("\n[2] Initializing database...")
from src.data.database import initialize_database, get_connection
initialize_database()
conn = get_connection()
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
conn.close()
print(f"  Tables created: {tables}")
# ─────────────────────────────────────────────
# CHECK 3: Universe
# ─────────────────────────────────────────────
print("\n[3] Fetching S&P 500 universe...")
from src.data.universe import get_sp500_tickers, save_universe_to_db
import pandas as pd
try:
    universe_df = get_sp500_tickers()
    if not universe_df.empty:
        save_universe_to_db(universe_df)
        print(f"  Universe size: {len(universe_df)} stocks")
    else:
        print("  WARNING: Empty universe returned - will retry with headers fix")
except Exception as e:
    print(f"  WARNING: Universe fetch failed ({e}) - continuing validation")
    universe_df = pd.DataFrame()
# ─────────────────────────────────────────────
# CHECK 4: Price Fetcher
# ─────────────────────────────────────────────
print("\n[4] Testing price fetcher (AAPL, Jan 2024)...")
from src.data.price_fetcher import fetch_price_data
import yfinance as yf
try:
    # Test yfinance directly first
    ticker_obj = yf.Ticker("AAPL")
    info = ticker_obj.fast_info
    print(f"  yfinance connection OK - AAPL last price: ${info.last_price:.2f}")

    price_df = fetch_price_data("AAPL", "2024-01-01", "2024-01-31")
    if not price_df.empty:
        print(f"  Rows fetched: {len(price_df)}")
        print(f"  Date range: {price_df['date'].min()} to {price_df['date'].max()}")
        print(f"  Sample close price: ${price_df['close'].iloc[0]:.2f}")
    else:
        print("  WARNING: No historical data returned - checking yfinance version")
        import yfinance
        print(f"  yfinance version: {yfinance.__version__}")
except Exception as e:
    print(f"  WARNING: Price fetcher issue: {e}")
# ─────────────────────────────────────────────
# CHECK 5: Earnings Fetcher
# ─────────────────────────────────────────────
print("\n[5] Testing earnings fetcher (AAPL)...")
from src.data.earnings_fetcher import get_historical_earnings
earnings_df = get_historical_earnings("AAPL")
if not earnings_df.empty:
    print(f"  Earnings records fetched: {len(earnings_df)}")
    latest = earnings_df.iloc[0]
    print(f"  Latest report date: {latest.get('date', 'N/A')}")
    print(f"  EPS actual: {latest.get('epsActual', 'N/A')}")
    print(f"  EPS estimated: {latest.get('epsEstimated', 'N/A')}")
    print(f"  EPS surprise %: {latest.get('eps_surprise_pct', 'N/A'):.2f}%")
else:
    print("  ERROR: No earnings data returned")
# ─────────────────────────────────────────────
# CHECK 6: EDGAR News Fetcher
# ─────────────────────────────────────────────
print("\n[6] Testing EDGAR news fetcher (AAPL CIK lookup)...")
from src.data.news_fetcher import get_company_cik, get_8k_filings
cik = get_company_cik("AAPL")
if cik:
    print(f"  AAPL CIK: {cik}")
    filings_df = get_8k_filings(cik, "AAPL")
    print(f"  Total 8-K filings found: {len(filings_df)}")
    if not filings_df.empty:
        most_recent = str(filings_df.iloc[0]['filing_date'])[:10]
        print(f"  Most recent 8-K: {most_recent}")
else:
    print("  ERROR: Could not find CIK for AAPL")
# ─────────────────────────────────────────────
# CHECK 7: Alpha Vantage News Fetcher
# ─────────────────────────────────────────────
print("\n[7] Testing Alpha Vantage news fetcher (AAPL)...")
from src.data.news_fetcher import fetch_alpha_vantage_news
av_df = fetch_alpha_vantage_news("AAPL", limit=5)
if not av_df.empty:
    print(f"  Articles fetched: {len(av_df)}")
    print(f"  Sample headline: {av_df.iloc[0]['headline'][:80]}...")
    print(f"  Sample sentiment: {av_df.iloc[0]['overall_sentiment']:.4f}")
else:
    print("  ERROR: No Alpha Vantage news returned")
# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)
print("\nIf all 7 checks passed, Phase 1 is complete.")
print("Next: Phase 2 — Labeling Pipeline")
