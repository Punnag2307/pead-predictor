import sqlite3
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
DB_PATH = os.getenv("DB_PATH", "data/database/news_alpha.db")
def get_connection():
    """Get SQLite database connection."""
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)
def initialize_database():
    """Create all tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    # Table 1: Companies (universe)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            ticker TEXT PRIMARY KEY,
            name TEXT,
            sector TEXT,
            industry TEXT,
            market_cap REAL,
            added_date TEXT
        )
    """)
    # Table 2: Earnings events (Chapter 1)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS earnings_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            report_date TEXT,
            fiscal_quarter TEXT,
            actual_eps REAL,
            consensus_eps REAL,
            eps_surprise REAL,
            eps_surprise_pct REAL,
            actual_revenue REAL,
            consensus_revenue REAL,
            revenue_surprise_pct REAL,
            report_time TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (ticker) REFERENCES companies(ticker)
        )
    """)
    # Table 3: News articles (Chapter 1 + Chapter 3)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS news_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            headline TEXT,
            summary TEXT,
            source TEXT,
            url TEXT,
            published_at TEXT,
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
            event_type TEXT,
            chapter INTEGER,
            FOREIGN KEY (ticker) REFERENCES companies(ticker)
        )
    """)
    # Table 4: Price data
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            adj_close REAL,
            FOREIGN KEY (ticker) REFERENCES companies(ticker),
            UNIQUE(ticker, date)
        )
    """)
    # Table 5: Labeled events (training data)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS labeled_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER,
            event_type TEXT,
            ticker TEXT,
            event_date TEXT,
            return_1h REAL,
            return_4h REAL,
            return_1d REAL,
            return_3d REAL,
            abnormal_return_1d REAL,
            label_direction INTEGER,
            label_magnitude REAL,
            is_market_moving INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    print("Database initialized successfully.")
if __name__ == "__main__":
    initialize_database()
