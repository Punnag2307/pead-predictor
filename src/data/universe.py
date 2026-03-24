import io
import pandas as pd
import requests
import yfinance as yf
import time
from src.data.database import get_connection
# Sectors to exclude following standard quant research practice
# Bernard & Thomas (1989), Livnat & Mendenhall (2006)
EXCLUDED_SECTORS = ['Financial Services', 'Utilities']
def get_sp500_tickers() -> pd.DataFrame:
    """
    Fetch current S&P 500 constituents from Wikipedia.
    Returns DataFrame excluding Financials and Utilities.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        # Add browser-like headers to avoid 403 blocking
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        tables = pd.read_html(io.StringIO(response.text))
        df = tables[0]
        # Rename columns
        df = df.rename(columns={
            'Symbol': 'ticker',
            'Security': 'name',
            'GICS Sector': 'sector',
            'GICS Sub-Industry': 'industry'
        })
        # Keep only needed columns
        df = df[['ticker', 'name', 'sector', 'industry']].copy()
        # Fix ticker format (BRK.B -> BRK-B for yfinance compatibility)
        df['ticker'] = df['ticker'].str.replace('.', '-', regex=False)
        # Exclude Financial Services and Utilities
        df_filtered = df[~df['sector'].isin(EXCLUDED_SECTORS)].copy()
        df_filtered = df_filtered.reset_index(drop=True)
        print(f"Total S&P 500 stocks: {len(df)}")
        print(f"After excluding {EXCLUDED_SECTORS}: {len(df_filtered)} stocks")
        print(f"\nSectors included:")
        for sector, count in df_filtered['sector'].value_counts().items():
            print(f"  {sector}: {count} stocks")
        return df_filtered
    except Exception as e:
        print(f"Error fetching S&P 500 universe: {e}")
        return pd.DataFrame()
def save_universe_to_db(df: pd.DataFrame):
    """Save company universe to database."""
    if df.empty:
        return
    conn = get_connection()
    cursor = conn.cursor()
    for _, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO companies
                (ticker, name, sector, industry, added_date)
                VALUES (?, ?, ?, ?, date('now'))
            """, (
                row['ticker'],
                row['name'],
                row['sector'],
                row['industry']
            ))
        except Exception as e:
            print(f"Error saving company {row['ticker']}: {e}")
    conn.commit()
    conn.close()
    print(f"\nSaved {len(df)} companies to database.")
def get_universe_from_db() -> pd.DataFrame:
    """Load universe from database."""
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM companies", conn)
    conn.close()
    return df
def get_tickers_by_sector(sector: str) -> list:
    """Get all tickers for a specific sector."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT ticker FROM companies WHERE sector = ?", (sector,)
    )
    tickers = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tickers
if __name__ == "__main__":
    df = get_sp500_tickers()
    save_universe_to_db(df)
    print(f"\nSample tickers:")
    print(df[['ticker', 'sector']].head(10))
