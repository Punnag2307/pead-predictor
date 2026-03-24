import yfinance as yf
import pandas as pd
import os
import time
from dotenv import load_dotenv
from src.data.database import get_connection

load_dotenv()

def get_historical_earnings(ticker: str) -> pd.DataFrame:
    """
    Fetch historical earnings data using yfinance earnings_dates.
    Returns ~12 quarters of EPS actual vs estimate with surprise %.
    
    Key fields returned:
    - report_date: exact earnings announcement date + time
    - actual_eps: reported EPS
    - consensus_eps: analyst consensus estimate
    - eps_surprise_pct: surprise percentage
    - report_time: AMC or BMO (derived from timestamp hour)
    """
    try:
        tk = yf.Ticker(ticker)
        ed = tk.earnings_dates
        
        if ed is None or ed.empty:
            print(f"No earnings data found for {ticker}")
            return pd.DataFrame()
        
        # Reset index to make date a column
        df = ed.reset_index()
        df['ticker'] = ticker
        
        # Rename columns to our standard names
        df = df.rename(columns={
            'Earnings Date': 'report_date',
            'EPS Estimate': 'consensus_eps',
            'Reported EPS': 'actual_eps',
            'Surprise(%)': 'eps_surprise_pct'
        })
        
        # Calculate EPS surprise amount
        if 'actual_eps' in df.columns and 'consensus_eps' in df.columns:
            df['eps_surprise'] = df['actual_eps'] - df['consensus_eps']
        
        # Derive report_time from timestamp hour
        # AMC = After Market Close (typically 16:00-21:00)
        # BMO = Before Market Open (typically 04:00-09:00)
        if 'report_date' in df.columns:
            df['report_date'] = pd.to_datetime(df['report_date'])
            df['report_hour'] = df['report_date'].dt.hour
            df['report_time'] = df['report_hour'].apply(
                lambda h: 'BMO' if h < 12 else 'AMC'
            )
            df['report_date_str'] = df['report_date'].dt.strftime('%Y-%m-%d')
        
        # Filter to only rows with actual reported earnings
        # Future dates have estimates but no actuals yet
        df = df[df['actual_eps'].notna()].copy()
        
        # Sort by date descending
        df = df.sort_values('report_date', ascending=False)
        
        return df
        
    except Exception as e:
        print(f"Error fetching earnings for {ticker}: {e}")
        return pd.DataFrame()

def save_earnings_to_db(ticker: str, df: pd.DataFrame):
    """Save earnings data to database."""
    if df.empty:
        return
    
    conn = get_connection()
    cursor = conn.cursor()
    
    saved = 0
    for _, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO earnings_events
                (ticker, report_date, actual_eps, consensus_eps,
                 eps_surprise, eps_surprise_pct, report_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                row.get('report_date_str', ''),
                row.get('actual_eps', None),
                row.get('consensus_eps', None),
                row.get('eps_surprise', None),
                row.get('eps_surprise_pct', None),
                row.get('report_time', None)
            ))
            saved += 1
        except Exception as e:
            print(f"Error saving earnings for {ticker}: {e}")
    
    conn.commit()
    conn.close()

def fetch_and_save_bulk_earnings(tickers: list, delay: float = 0.3):
    """
    Fetch earnings for a list of tickers with rate limiting.
    yfinance has no hard limit but we add delay to be respectful.
    """
    results = {}
    failed = []
    
    for i, ticker in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] Earnings: {ticker}...", end=" ", flush=True)
        df = get_historical_earnings(ticker)
        
        if not df.empty:
            save_earnings_to_db(ticker, df)
            results[ticker] = len(df)
            print(f"OK ({len(df)} quarters)")
        else:
            failed.append(ticker)
            print("EMPTY")
        
        time.sleep(delay)
    
    print(f"\nCompleted: {len(results)} succeeded, {len(failed)} failed")
    if failed:
        print(f"Failed tickers: {failed[:10]}"
              f"{'...' if len(failed) > 10 else ''}")
    
    return results

if __name__ == "__main__":
    # Quick test
    test_tickers = ["AAPL", "MSFT", "GOOGL", "MMM"]
    fetch_and_save_bulk_earnings(test_tickers)