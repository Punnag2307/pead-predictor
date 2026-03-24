import yfinance as yf
import pandas as pd
import os
from datetime import datetime
from src.data.database import get_connection
def fetch_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV price data using yfinance.
    start and end format: 'YYYY-MM-DD'
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end, interval="1d")
        if df.empty:
            print(f"No price data found for {ticker}")
            return pd.DataFrame()
        df = df.reset_index()
        df['ticker'] = ticker
        df['date'] = df['Date'].dt.strftime('%Y-%m-%d')
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        df['adj_close'] = df['close']
        # Keep only columns we need
        df = df[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']]
        print(f"Fetched {len(df)} days of price data for {ticker}")
        return df
    except Exception as e:
        print(f"Error fetching price data for {ticker}: {e}")
        return pd.DataFrame()
def save_prices_to_db(ticker: str, df: pd.DataFrame):
    """Save price data to database."""
    if df.empty:
        return
    conn = get_connection()
    cursor = conn.cursor()
    for _, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO price_data
                (ticker, date, open, high, low, close, volume, adj_close)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                row['date'],
                row.get('open'),
                row.get('high'),
                row.get('low'),
                row.get('close'),
                row.get('volume'),
                row.get('adj_close')
            ))
        except Exception as e:
            print(f"Error saving price for {ticker} on {row.get('date')}: {e}")
    conn.commit()
    conn.close()
def fetch_and_save_bulk_prices(tickers: list, start: str, end: str):
    """Fetch and save prices for a list of tickers."""
    for i, ticker in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] Fetching prices for {ticker}...")
        df = fetch_price_data(ticker, start, end)
        if not df.empty:
            save_prices_to_db(ticker, df)
if __name__ == "__main__":
    # Quick test with 3 tickers
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    fetch_and_save_bulk_prices(
        test_tickers,
        start="2020-01-01",
        end="2024-12-31"
    )
