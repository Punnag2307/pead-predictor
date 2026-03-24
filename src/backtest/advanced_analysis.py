import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)))

from scipy import stats
from src.data.database import get_connection

OUTPUT_DIR = 'data/processed/charts'

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load feature matrix with predictions."""
    df = pd.read_csv(
        'data/processed/feature_matrix_with_preds.csv'
    )
    df['event_date'] = pd.to_datetime(df['event_date'])
    return df

def load_price_cache() -> dict:
    """Load prices for backtest."""
    conn = get_connection()
    tickers_df = pd.read_sql(
        "SELECT DISTINCT ticker FROM price_data", conn
    )
    conn.close()
    prices = {}
    for ticker in tickers_df['ticker']:
        conn = get_connection()
        df = pd.read_sql("""
            SELECT date, open, close
            FROM price_data
            WHERE ticker = ?
            ORDER BY date ASC
        """, conn, params=(ticker,))
        conn.close()
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            prices[ticker] = df
    return prices

def get_actual_trade_returns(
    df: pd.DataFrame,
    prices_cache: dict,
    confidence_threshold: float = 0.65
) -> pd.DataFrame:
    """
    Calculate actual overnight gap returns for each trade.
    Uses PREDICTED direction (not label) — no leakage.
    """
    ROUND_TRIP_COST = 0.0038
    trades = []

    hc = df[df['predicted_proba'] > confidence_threshold].copy()

    for _, row in hc.iterrows():
        ticker = row['ticker']
        event_date = pd.Timestamp(row['event_date'])
        predicted_dir = row['predicted_direction']
        report_time = row.get('report_time', None)

        if ticker not in prices_cache:
            continue

        prices = prices_cache[ticker]
        all_dates = prices.index.sort_values()
        future_dates = all_dates[all_dates >= event_date]

        if len(future_dates) < 2:
            continue

        if report_time == 'BMO':
            reaction_day = future_dates[0]
        else:
            reaction_day = future_dates[1]

        prev_dates = all_dates[all_dates < reaction_day]
        if len(prev_dates) == 0:
            continue

        try:
            prev_day = prev_dates[-1]
            prev_close = prices.loc[prev_day, 'close']
            reaction_open = prices.loc[reaction_day, 'open']

            gross_return = (
                reaction_open / prev_close - 1
            ) * predicted_dir
            net_return = gross_return - ROUND_TRIP_COST

            trades.append({
                'date': reaction_day,
                'ticker': ticker,
                'event_date': event_date,
                'predicted_dir': predicted_dir,
                'predicted_proba': row['predicted_proba'],
                'actual_direction': row['label_direction'],
                'gross_return': gross_return,
                'net_return': net_return,
                'correct': int(
                    predicted_dir == row['label_direction']
                ),
                'sector': row.get('sector', 'Unknown'),
                'year': event_date.year
            })
        except Exception:
            continue

    return pd.DataFrame(trades)

# ─────────────────────────────────────────────
# 1. INFORMATION COEFFICIENT ANALYSIS
# ─────────────────────────────────────────────

def run_ic_analysis(df: pd.DataFrame) -> dict:
    """
    Information Coefficient Analysis.
    IMPORTANT: Calculate IC on OUT-OF-SAMPLE data only (2025)
    to avoid in-sample inflation.

    IC = Spearman rank correlation between predicted
    probability and actual return, calculated monthly.
    """
    print("\n=== INFORMATION COEFFICIENT ANALYSIS ===")

    # Split into in-sample and out-of-sample
    train_val = df[df['event_date'] <= '2024-12-31'].copy()
    oos = df[df['event_date'] >= '2025-01-01'].copy()

    print(f"\nIn-sample events:  {len(train_val)}")
    print(f"OOS events:        {len(oos)}")

    # Calculate IC on full dataset by month
    df['year_month'] = df['event_date'].dt.to_period('M')

    monthly_ic = []
    for period, group in df.groupby('year_month'):
        if len(group) < 5:
            continue
        ic, p_val = stats.spearmanr(
            group['predicted_proba'],
            group['abnormal_return_1d']
        )
        is_oos = str(period) >= '2025-01'
        monthly_ic.append({
            'period': period,
            'ic': ic,
            'p_value': p_val,
            'n_events': len(group),
            'is_oos': is_oos
        })

    ic_df = pd.DataFrame(monthly_ic)

    # Separate in-sample vs OOS IC
    ic_insample = ic_df[~ic_df['is_oos']]['ic']
    ic_oos = ic_df[ic_df['is_oos']]['ic']

    print(f"\nIn-sample IC (train+val, 2022-2024):")
    print(f"  Mean IC:    {ic_insample.mean():.4f}")
    print(f"  IC Std:     {ic_insample.std():.4f}")
    print(f"  IC > 0:     {(ic_insample > 0).mean()*100:.1f}%")

    print(f"\nOut-of-sample IC (2025 — model never saw this):")
    if len(ic_oos) > 0:
        mean_oos_ic = ic_oos.mean()
        std_oos_ic = ic_oos.std()
        ir_oos = mean_oos_ic / std_oos_ic if std_oos_ic > 0 else 0
        print(f"  Mean IC:    {mean_oos_ic:.4f}")
        print(f"  IC Std:     {std_oos_ic:.4f}")
        print(f"  IR:         {ir_oos:.4f}")
        print(f"  IC > 0:     {(ic_oos > 0).mean()*100:.1f}%")
    else:
        mean_oos_ic = np.nan
        ir_oos = np.nan
        print("  Not enough OOS months")

    # Industry context based on OOS IC
    print(f"\nIndustry context (based on OOS IC):")
    if not np.isnan(mean_oos_ic):
        if mean_oos_ic > 0.10:
            rating = "EXCELLENT (top tier quant fund level)"
        elif mean_oos_ic > 0.05:
            rating = "GOOD (competitive with professional signals)"
        elif mean_oos_ic > 0.02:
            rating = "MODERATE (typical single-factor signal)"
        else:
            rating = "WEAK (below typical professional threshold)"
        print(f"  OOS IC Rating: {rating}")

    # Significance test on full IC series
    all_ic = ic_df['ic']
    t_stat, p_val = stats.ttest_1samp(all_ic, 0)
    print(f"\nIC significance (full series):")
    print(f"  T-statistic: {t_stat:.3f}")
    print(f"  P-value:     {p_val:.4f}")
    print(f"  Significant: {p_val < 0.05}")

    return {
        'ic_df': ic_df,
        'mean_ic_insample': ic_insample.mean(),
        'mean_ic_oos': mean_oos_ic,
        'ir_oos': ir_oos,
        't_stat': t_stat,
        'p_value': p_val
    }

def plot_ic_analysis(ic_results: dict):
    """Generate IC time series chart."""
    ic_df = ic_results['ic_df']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Color by in-sample vs OOS
    colors = [
        '#FF9800' if row['is_oos'] else
        ('#4CAF50' if row['ic'] > 0 else '#FF5722')
        for _, row in ic_df.iterrows()
    ]

    periods = [str(p) for p in ic_df['period']]
    ax1.bar(range(len(ic_df)), ic_df['ic'],
            color=colors, alpha=0.8)
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.axhline(
        ic_results['mean_ic_insample'],
        color='blue', linewidth=2, linestyle='--',
        label=f"In-sample mean: "
              f"{ic_results['mean_ic_insample']:.4f}"
    )
    if not np.isnan(ic_results['mean_ic_oos']):
        ax1.axhline(
            ic_results['mean_ic_oos'],
            color='orange', linewidth=2, linestyle='-.',
            label=f"OOS mean: "
                  f"{ic_results['mean_ic_oos']:.4f}"
        )
    ax1.axhline(0.05, color='green', linewidth=1,
                linestyle=':', label='Benchmark (0.05)')

    # Add OOS region shading
    oos_start = next(
        (i for i, r in enumerate(
            ic_df['is_oos']
        ) if r), None
    )
    if oos_start:
        ax1.axvspan(
            oos_start - 0.5, len(ic_df) - 0.5,
            alpha=0.1, color='orange',
            label='OOS period (2025)'
        )

    ax1.set_xticks(range(0, len(ic_df), 3))
    ax1.set_xticklabels(
        [periods[i] for i in range(0, len(ic_df), 3)],
        rotation=45, fontsize=8
    )
    ax1.set_ylabel('Information Coefficient', fontsize=12)
    ax1.set_title(
        'Monthly IC Time Series\n'
        'Blue=In-sample, Orange=Out-of-sample (2025)',
        fontsize=13, fontweight='bold'
    )
    ax1.legend(fontsize=9)

    # IC distribution split
    ic_is = ic_df[~ic_df['is_oos']]['ic']
    ic_oos = ic_df[ic_df['is_oos']]['ic']

    ax2.hist(ic_is, bins=15, color='#2196F3',
             alpha=0.6, label='In-sample', edgecolor='white')
    if len(ic_oos) > 0:
        ax2.hist(ic_oos, bins=10, color='#FF9800',
                 alpha=0.8, label='OOS (2025)',
                 edgecolor='white')
    ax2.axvline(0, color='black', linewidth=1)
    ax2.axvline(0.05, color='green', linewidth=1,
                linestyle=':', label='Benchmark (0.05)')
    ax2.set_xlabel('Information Coefficient', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('IC Distribution: In-sample vs OOS',
                  fontsize=13, fontweight='bold')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/ic_analysis.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: ic_analysis.png")

# ─────────────────────────────────────────────
# 2. CAPACITY AND SCALABILITY ANALYSIS
# ─────────────────────────────────────────────

def run_capacity_analysis(df: pd.DataFrame) -> dict:
    """
    Capacity and Scalability Analysis.
    Uses square-root market impact model.
    """
    print("\n=== CAPACITY AND SCALABILITY ANALYSIS ===")

    conn = get_connection()
    high_conviction = df[df['predicted_proba'] > 0.65]
    tickers = high_conviction['ticker'].unique().tolist()

    placeholders = ','.join(['?' for _ in tickers])
    adv_data = pd.read_sql(f"""
        SELECT ticker,
               AVG(volume) as avg_volume,
               AVG(close) as avg_price
        FROM price_data
        WHERE ticker IN ({placeholders})
        AND date >= '2024-01-01'
        GROUP BY ticker
    """, conn, params=tickers)
    conn.close()

    adv_data['adv_usd'] = (
        adv_data['avg_volume'] * adv_data['avg_price']
    )

    print(f"\nUniverse for capacity analysis:")
    print(f"  High-conviction tickers: {len(tickers)}")
    print(f"  Mean ADV: ${adv_data['adv_usd'].mean()/1e6:.1f}M")
    print(f"  Median ADV: "
          f"${adv_data['adv_usd'].median()/1e6:.1f}M")
    print(f"  Total ADV: "
          f"${adv_data['adv_usd'].sum()/1e6:.0f}M")

    # Mean return from actual backtest results
    mean_return_bps = 133.6
    fixed_costs_bps = 38
    n_trades_per_year = 862
    impact_coef = 0.1

    capital_levels = np.logspace(5, 9, 100)

    results = []
    for capital in capital_levels:
        capital_per_trade = capital / n_trades_per_year
        participation = capital_per_trade / (
            adv_data['adv_usd'].mean()
        )
        impact_bps = impact_coef * np.sqrt(
            participation
        ) * 10000
        net_return_bps = (
            mean_return_bps - impact_bps - fixed_costs_bps
        )

        results.append({
            'capital': capital,
            'capital_per_trade': capital_per_trade,
            'impact_bps': impact_bps,
            'net_return_bps': net_return_bps,
            'participation_rate': participation * 100
        })

    capacity_df = pd.DataFrame(results)
    positive = capacity_df[capacity_df['net_return_bps'] > 0]
    capacity_ceiling = (
        positive['capital'].max() if len(positive) > 0 else 0
    )

    print(f"\nCapacity estimates:")
    print(f"  Mean return/trade:  {mean_return_bps:.0f} bps")
    print(f"  Fixed costs:        {fixed_costs_bps} bps")
    print(f"  Available for impact: "
          f"{mean_return_bps-fixed_costs_bps:.0f} bps")

    checkpoints = [1e6, 10e6, 50e6, 100e6,
                   250e6, 500e6, 1e9]
    print(f"\n  Capital → Net return:")
    for cap in checkpoints:
        row = capacity_df.iloc[
            (capacity_df['capital'] - cap).abs().argsort()[:1]
        ]
        net = row['net_return_bps'].values[0]
        imp = row['impact_bps'].values[0]
        print(f"  ${cap/1e6:>6.0f}M: "
              f"impact={imp:.0f}bps, "
              f"net={net:.0f}bps "
              f"{'✅' if net > 0 else '❌'}")

    print(f"\n  Capacity ceiling: ~${capacity_ceiling/1e6:.0f}M")

    return {
        'capacity_df': capacity_df,
        'capacity_ceiling': capacity_ceiling,
        'mean_return_bps': mean_return_bps
    }

def plot_capacity_analysis(capacity_results: dict):
    """Generate capacity analysis chart."""
    capacity_df = capacity_results['capacity_df']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.semilogx(
        capacity_df['capital'] / 1e6,
        capacity_df['net_return_bps'],
        color='#2196F3', linewidth=2.5
    )
    ax1.axhline(0, color='red', linestyle='--',
                linewidth=1.5, label='Break-even')
    ax1.axhline(38, color='orange', linestyle=':',
                linewidth=1.5, label='Cost floor (38 bps)')
    ax1.fill_between(
        capacity_df['capital'] / 1e6,
        capacity_df['net_return_bps'],
        0,
        where=capacity_df['net_return_bps'] > 0,
        alpha=0.2, color='#4CAF50',
        label='Profitable region'
    )
    ax1.set_xlabel('Capital Deployed ($M)', fontsize=12)
    ax1.set_ylabel('Net Return per Trade (bps)', fontsize=12)
    ax1.set_title(
        'Strategy Capacity Analysis\nNet Return vs Capital',
        fontsize=13, fontweight='bold'
    )
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.semilogx(
        capacity_df['capital'] / 1e6,
        capacity_df['impact_bps'],
        color='#FF5722', linewidth=2.5,
        label='Market impact (sqrt model)'
    )
    ax2.axhline(
        capacity_results['mean_return_bps'] - 38,
        color='green', linestyle='--', linewidth=1.5,
        label=f"Alpha available "
              f"({capacity_results['mean_return_bps']-38:.0f} bps)"
    )
    ax2.set_xlabel('Capital Deployed ($M)', fontsize=12)
    ax2.set_ylabel('Market Impact (bps)', fontsize=12)
    ax2.set_title(
        'Market Impact vs Capital\nSquare-Root Model',
        fontsize=13, fontweight='bold'
    )
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f'{OUTPUT_DIR}/capacity_analysis.png',
        dpi=150, bbox_inches='tight'
    )
    plt.close()
    print("Saved: capacity_analysis.png")

# ─────────────────────────────────────────────
# 3. REGIME ANALYSIS (FIXED — uses predicted direction)
# ─────────────────────────────────────────────

def run_regime_analysis(
    trades_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Regime Analysis using ACTUAL trade returns.
    Uses predicted_direction not label_direction.
    No leakage.
    """
    print("\n=== REGIME ANALYSIS ===")

    if trades_df.empty:
        print("No trades data available.")
        return pd.DataFrame()

    trades_df['year'] = pd.to_datetime(
        trades_df['date']
    ).dt.year

    regimes = {
        '2022 Bear': 2022,
        '2023 Recovery': 2023,
        '2024 AI Rally': 2024,
        '2025 OOS': 2025
    }

    regime_results = []

    for regime_name, year in regimes.items():
        regime_trades = trades_df[
            trades_df['year'] == year
        ].copy()

        if len(regime_trades) < 5:
            print(f"\n{regime_name}: insufficient trades "
                  f"({len(regime_trades)})")
            continue

        net_returns = regime_trades['net_return'].values
        mean_ret = net_returns.mean()
        std_ret = net_returns.std()
        sharpe = (
            mean_ret / std_ret * np.sqrt(252)
            if std_ret > 0 else 0
        )
        win_rate = regime_trades['correct'].mean()
        n_trades = len(regime_trades)

        # T-test
        t_stat, p_val = stats.ttest_1samp(net_returns, 0)

        regime_results.append({
            'regime': regime_name,
            'year': year,
            'n_trades': n_trades,
            'win_rate': win_rate,
            'mean_return_bps': mean_ret * 10000,
            'sharpe': sharpe,
            'p_value': p_val,
            'significant': p_val < 0.05
        })

        print(f"\n{regime_name} ({year}):")
        print(f"  Trades:    {n_trades}")
        print(f"  Win rate:  {win_rate*100:.1f}%")
        print(f"  Mean ret:  {mean_ret*10000:.0f} bps")
        print(f"  Sharpe:    {sharpe:.2f}")
        print(f"  P-value:   {p_val:.4f}")
        print(f"  Significant: {p_val < 0.05}")

    return pd.DataFrame(regime_results)

def plot_regime_analysis(regime_df: pd.DataFrame):
    """Generate regime analysis chart."""
    if regime_df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    colors = ['#FF5722', '#4CAF50', '#2196F3', '#FF9800']

    # Sharpe by regime
    axes[0].bar(
        range(len(regime_df)),
        regime_df['sharpe'],
        color=colors[:len(regime_df)], alpha=0.8
    )
    axes[0].axhline(1.0, color='red', linestyle='--',
                    label='Sharpe = 1.0', linewidth=1.5)
    axes[0].set_xticks(range(len(regime_df)))
    axes[0].set_xticklabels(
        regime_df['regime'],
        rotation=15, ha='right', fontsize=9
    )
    axes[0].set_ylabel('Net Sharpe Ratio', fontsize=11)
    axes[0].set_title('Net Sharpe by Regime',
                      fontsize=12, fontweight='bold')
    axes[0].legend()

    # Win rate by regime
    axes[1].bar(
        range(len(regime_df)),
        regime_df['win_rate'] * 100,
        color=colors[:len(regime_df)], alpha=0.8
    )
    axes[1].axhline(50, color='red', linestyle='--',
                    label='Random (50%)', linewidth=1.5)
    axes[1].set_xticks(range(len(regime_df)))
    axes[1].set_xticklabels(
        regime_df['regime'],
        rotation=15, ha='right', fontsize=9
    )
    axes[1].set_ylabel('Win Rate (%)', fontsize=11)
    axes[1].set_title('Win Rate by Regime',
                      fontsize=12, fontweight='bold')
    axes[1].legend()

    # Mean return by regime
    axes[2].bar(
        range(len(regime_df)),
        regime_df['mean_return_bps'],
        color=colors[:len(regime_df)], alpha=0.8
    )
    axes[2].axhline(38, color='orange', linestyle=':',
                    label='Cost floor (38 bps)',
                    linewidth=1.5)
    axes[2].axhline(0, color='black', linewidth=0.8)
    axes[2].set_xticks(range(len(regime_df)))
    axes[2].set_xticklabels(
        regime_df['regime'],
        rotation=15, ha='right', fontsize=9
    )
    axes[2].set_ylabel('Mean Return (bps)', fontsize=11)
    axes[2].set_title('Mean Return by Regime',
                      fontsize=12, fontweight='bold')
    axes[2].legend()

    plt.suptitle(
        'Strategy Performance Across Market Regimes\n'
        '(Using model predicted direction — no leakage)',
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig(
        f'{OUTPUT_DIR}/regime_analysis.png',
        dpi=150, bbox_inches='tight'
    )
    plt.close()
    print("Saved: regime_analysis.png")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("="*60)
    print("ADVANCED ANALYSIS")
    print("IC + Capacity + Regime")
    print("="*60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    df = load_data()
    print(f"Loaded {len(df)} events")

    # Load prices for actual trade returns
    print("Loading price cache for regime analysis...")
    prices_cache = load_price_cache()

    # Get actual trade returns (no leakage)
    print("Calculating actual trade returns...")
    trades_df = get_actual_trade_returns(df, prices_cache)
    print(f"Total trades: {len(trades_df)}")

    # 1. IC Analysis
    ic_results = run_ic_analysis(df)
    plot_ic_analysis(ic_results)

    # 2. Capacity Analysis
    capacity_results = run_capacity_analysis(df)
    plot_capacity_analysis(capacity_results)

    # 3. Regime Analysis (fixed)
    regime_df = run_regime_analysis(trades_df)
    plot_regime_analysis(regime_df)

    # Save summary
    summary = {
        'ic_mean_insample': float(
            ic_results['mean_ic_insample']
        ),
        'ic_mean_oos': float(ic_results['mean_ic_oos'])
        if not np.isnan(ic_results['mean_ic_oos']) else None,
        'ic_ir_oos': float(ic_results['ir_oos'])
        if not np.isnan(ic_results['ir_oos']) else None,
        'capacity_ceiling_usd': float(
            capacity_results['capacity_ceiling']
        ),
        'regime_results': regime_df.to_dict('records')
        if not regime_df.empty else []
    }

    with open(
        'data/processed/advanced_analysis.json', 'w'
    ) as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "="*60)
    print("ADVANCED ANALYSIS COMPLETE")
    print("="*60)
    print("\nCharts generated:")
    print("  data/processed/charts/ic_analysis.png")
    print("  data/processed/charts/capacity_analysis.png")
    print("  data/processed/charts/regime_analysis.png")

if __name__ == "__main__":
    main()
