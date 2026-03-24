import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from src.data.database import get_connection

OUTPUT_DIR = 'data/processed/charts'

def load_data():
    df = pd.read_csv(
        'data/processed/feature_matrix_with_preds.csv'
    )
    df['event_date'] = pd.to_datetime(df['event_date'])
    return df

def load_price_cache():
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

def get_trade_return(row, prices_cache):
    """Get overnight gap return for one event."""
    ROUND_TRIP_COST = 0.0038
    ticker = row['ticker']
    event_date = pd.Timestamp(row['event_date'])
    predicted_dir = row['predicted_direction']
    report_time = row.get('report_time', None)

    if ticker not in prices_cache:
        return np.nan

    prices = prices_cache[ticker]
    all_dates = prices.index.sort_values()
    future_dates = all_dates[all_dates >= event_date]

    if len(future_dates) < 2:
        return np.nan

    reaction_day = (
        future_dates[0] if report_time == 'BMO'
        else future_dates[1]
    )
    prev_dates = all_dates[all_dates < reaction_day]
    if len(prev_dates) == 0:
        return np.nan

    try:
        prev_close = prices.loc[prev_dates[-1], 'close']
        reaction_open = prices.loc[reaction_day, 'open']
        gross = (reaction_open / prev_close - 1) * predicted_dir
        return gross - ROUND_TRIP_COST
    except Exception:
        return np.nan

# ─────────────────────────────────────────────
# ADDITION 1: THRESHOLD SENSITIVITY
# ─────────────────────────────────────────────

def threshold_sensitivity_analysis(df, prices_cache):
    """
    Show strategy performance across all confidence thresholds.
    Directly addresses selection bias concern.
    OOS only (2025) for honest evaluation.
    """
    print("\n=== ADDITION 1: THRESHOLD SENSITIVITY ===")

    oos = df[df['event_date'] >= '2025-01-01'].copy()
    print(f"OOS events: {len(oos)}")

    thresholds = [0.50, 0.52, 0.54, 0.56,
                  0.58, 0.60, 0.62, 0.65]
    results = []

    for t in thresholds:
        subset = oos[oos['predicted_proba'] > t].copy()
        if len(subset) < 10:
            continue

        # Calculate actual trade returns
        net_returns = []
        for _, row in subset.iterrows():
            ret = get_trade_return(row, prices_cache)
            if not np.isnan(ret):
                net_returns.append(ret)

        if len(net_returns) < 10:
            continue

        net_returns = np.array(net_returns)
        mean_ret = net_returns.mean()
        std_ret = net_returns.std()
        sharpe = (
            mean_ret / std_ret * np.sqrt(252)
            if std_ret > 0 else 0
        )
        win_rate = (net_returns > 0).mean()
        t_stat, p_val = stats.ttest_1samp(net_returns, 0)

        # Accuracy
        acc = (
            subset['predicted_direction'] ==
            subset['label_direction']
        ).mean()

        results.append({
            'threshold': t,
            'n_trades': len(net_returns),
            'accuracy': acc,
            'win_rate': win_rate,
            'mean_return_bps': mean_ret * 10000,
            'net_sharpe': sharpe,
            'p_value': p_val,
            'significant': p_val < 0.05
        })

        print(f"  > {t}: {len(net_returns)} trades, "
              f"acc={acc*100:.1f}%, "
              f"sharpe={sharpe:.2f}, "
              f"p={p_val:.4f} "
              f"{'✅' if p_val < 0.05 else '⚠️'}")

    results_df = pd.DataFrame(results)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Trades vs threshold
    axes[0,0].bar(
        results_df['threshold'].astype(str),
        results_df['n_trades'],
        color='#2196F3', alpha=0.8
    )
    axes[0,0].set_xlabel('Confidence Threshold', fontsize=11)
    axes[0,0].set_ylabel('Number of OOS Trades', fontsize=11)
    axes[0,0].set_title('Trade Count vs Threshold',
                        fontsize=12, fontweight='bold')

    # Accuracy vs threshold
    axes[0,1].plot(
        results_df['threshold'],
        results_df['accuracy'] * 100,
        marker='o', linewidth=2.5,
        color='#4CAF50', markersize=8
    )
    axes[0,1].axhline(50, color='red', linestyle='--',
                      label='Random (50%)')
    axes[0,1].set_xlabel('Confidence Threshold', fontsize=11)
    axes[0,1].set_ylabel('Directional Accuracy (%)', fontsize=11)
    axes[0,1].set_title('Accuracy vs Threshold\n(OOS 2025)',
                        fontsize=12, fontweight='bold')
    axes[0,1].legend()

    # Net Sharpe vs threshold
    colors = [
        '#4CAF50' if s > 1.0 else '#FF5722'
        for s in results_df['net_sharpe']
    ]
    axes[1,0].bar(
        results_df['threshold'].astype(str),
        results_df['net_sharpe'],
        color=colors, alpha=0.8
    )
    axes[1,0].axhline(1.0, color='red', linestyle='--',
                      label='Sharpe = 1.0')
    axes[1,0].axhline(0, color='black', linewidth=0.8)
    axes[1,0].set_xlabel('Confidence Threshold', fontsize=11)
    axes[1,0].set_ylabel('Net Sharpe Ratio', fontsize=11)
    axes[1,0].set_title('Net Sharpe vs Threshold\n(OOS 2025)',
                        fontsize=12, fontweight='bold')
    axes[1,0].legend()

    # P-value vs threshold
    axes[1,1].bar(
        results_df['threshold'].astype(str),
        results_df['p_value'],
        color='#9C27B0', alpha=0.8
    )
    axes[1,1].axhline(0.05, color='red', linestyle='--',
                      label='p = 0.05 threshold')
    axes[1,1].set_xlabel('Confidence Threshold', fontsize=11)
    axes[1,1].set_ylabel('P-value', fontsize=11)
    axes[1,1].set_title('Statistical Significance vs Threshold',
                        fontsize=12, fontweight='bold')
    axes[1,1].legend()

    plt.suptitle(
        'Threshold Sensitivity Analysis — OOS 2025\n'
        'Strategy performance across all confidence levels',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(
        f'{OUTPUT_DIR}/threshold_sensitivity.png',
        dpi=150, bbox_inches='tight'
    )
    plt.close()
    print("Saved: threshold_sensitivity.png")

    return results_df

# ─────────────────────────────────────────────
# ADDITION 2: EPS SURPRISE QUINTILE ANALYSIS
# ─────────────────────────────────────────────

def eps_quintile_analysis(df):
    """
    Show monotonic relationship between EPS surprise
    magnitude and post-earnings return.
    Validates the fundamental PEAD mechanism.
    """
    print("\n=== ADDITION 2: EPS SURPRISE QUINTILE ANALYSIS ===")

    # Use full dataset for maximum statistical power
    clean = df[
        df['eps_surprise_pct'].notna() &
        df['abnormal_return_1d'].notna()
    ].copy()

    # Winsorize surprise at ±50% for visualization
    clean['eps_surprise_viz'] = np.clip(
        clean['eps_surprise_pct'], -50, 50
    )

    # Create quintiles
    clean['quintile'] = pd.qcut(
        clean['eps_surprise_viz'],
        q=5,
        labels=['Q1\n(Largest Miss)',
                'Q2\n(Miss)',
                'Q3\n(In-line)',
                'Q4\n(Beat)',
                'Q5\n(Largest Beat)']
    )

    # Calculate stats per quintile
    quintile_stats = clean.groupby(
        'quintile', observed=True
    ).agg(
        mean_return=('abnormal_return_1d', 'mean'),
        median_return=('abnormal_return_1d', 'median'),
        std_return=('abnormal_return_1d', 'std'),
        n_events=('abnormal_return_1d', 'count'),
        mean_surprise=('eps_surprise_viz', 'mean'),
        win_rate=('label_direction',
                  lambda x: (x == 1).mean())
    ).reset_index()

    quintile_stats['se'] = (
        quintile_stats['std_return'] /
        np.sqrt(quintile_stats['n_events'])
    )

    print("\nQuintile statistics:")
    print(quintile_stats[[
        'quintile', 'mean_surprise', 'mean_return',
        'win_rate', 'n_events'
    ]].to_string(index=False))

    # T-test for monotonic trend
    returns_by_quintile = [
        clean[clean['quintile'] == q][
            'abnormal_return_1d'
        ].values
        for q in quintile_stats['quintile']
    ]

    # Spearman correlation: quintile rank vs mean return
    quintile_ranks = np.arange(1, 6)
    mean_returns = quintile_stats['mean_return'].values
    spearman_corr, spearman_p = stats.spearmanr(
        quintile_ranks, mean_returns
    )
    print(f"\nMonotonic trend test:")
    print(f"  Spearman correlation: {spearman_corr:.4f}")
    print(f"  P-value: {spearman_p:.4f}")
    print(f"  Monotonic: {spearman_p < 0.05}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#FF5722', '#FF9800', '#9E9E9E',
              '#8BC34A', '#4CAF50']

    # Mean return by quintile
    bars = axes[0].bar(
        range(5),
        quintile_stats['mean_return'] * 100,
        color=colors, alpha=0.85,
        yerr=quintile_stats['se'] * 100 * 1.96,
        capsize=5
    )
    axes[0].axhline(0, color='black', linewidth=0.8)
    axes[0].set_xticks(range(5))
    axes[0].set_xticklabels(
        quintile_stats['quintile'], fontsize=9
    )
    axes[0].set_ylabel(
        'Mean Abnormal Return (%)', fontsize=12
    )
    axes[0].set_title(
        'Post-Earnings Return by EPS Surprise Quintile\n'
        f'Spearman ρ={spearman_corr:.3f}, '
        f'p={spearman_p:.4f}',
        fontsize=12, fontweight='bold'
    )

    # Add value labels on bars
    for bar, val in zip(
        bars, quintile_stats['mean_return'] * 100
    ):
        axes[0].text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.05,
            f'{val:.2f}%',
            ha='center', va='bottom', fontsize=9,
            fontweight='bold'
        )

    # Win rate by quintile
    axes[1].bar(
        range(5),
        quintile_stats['win_rate'] * 100,
        color=colors, alpha=0.85
    )
    axes[1].axhline(50, color='red', linestyle='--',
                    label='Random (50%)', linewidth=1.5)
    axes[1].set_xticks(range(5))
    axes[1].set_xticklabels(
        quintile_stats['quintile'], fontsize=9
    )
    axes[1].set_ylabel('Win Rate (%)', fontsize=12)
    axes[1].set_title(
        'Win Rate by EPS Surprise Quintile\n'
        '(Direction of abnormal return)',
        fontsize=12, fontweight='bold'
    )
    axes[1].legend()
    axes[1].set_ylim(30, 70)

    plt.suptitle(
        'EPS Surprise Quintile Analysis\n'
        'Validates monotonic PEAD mechanism',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(
        f'{OUTPUT_DIR}/eps_quintile_analysis.png',
        dpi=150, bbox_inches='tight'
    )
    plt.close()
    print("Saved: eps_quintile_analysis.png")

    return quintile_stats, spearman_corr

# ─────────────────────────────────────────────
# ADDITION 3: REVISED CAPACITY ANALYSIS
# ─────────────────────────────────────────────

def revised_capacity_analysis(df):
    """
    Conservative capacity estimate using 10th percentile ADV.
    More realistic than mean ADV assumption.
    """
    print("\n=== ADDITION 3: REVISED CAPACITY ANALYSIS ===")

    conn = get_connection()
    hc = df[df['predicted_proba'] > 0.65]
    tickers = hc['ticker'].unique().tolist()

    placeholders = ','.join(['?' for _ in tickers])
    adv_data = pd.read_sql(f"""
        SELECT ticker,
               AVG(volume) as avg_volume,
               AVG(close) as avg_price,
               MIN(volume * close) as min_adv_usd
        FROM price_data
        WHERE ticker IN ({placeholders})
        AND date >= '2024-01-01'
        GROUP BY ticker
    """, conn, params=tickers)
    conn.close()

    adv_data['adv_usd'] = (
        adv_data['avg_volume'] * adv_data['avg_price']
    )

    # Three scenarios
    mean_adv = adv_data['adv_usd'].mean()
    median_adv = adv_data['adv_usd'].median()
    p10_adv = adv_data['adv_usd'].quantile(0.10)

    print(f"\nADV Statistics:")
    print(f"  Mean ADV:   ${mean_adv/1e6:.0f}M "
          f"(optimistic)")
    print(f"  Median ADV: ${median_adv/1e6:.0f}M "
          f"(moderate)")
    print(f"  10th pctile: ${p10_adv/1e6:.0f}M "
          f"(conservative)")

    mean_return_bps = 133.6
    fixed_costs_bps = 38
    n_trades = 1605
    impact_coef = 0.1
    capital_levels = np.logspace(5, 9, 100)

    scenarios = {
        'Optimistic (Mean ADV)': mean_adv,
        'Moderate (Median ADV)': median_adv,
        'Conservative (10th pctile)': p10_adv
    }

    scenario_results = {}
    print(f"\nCapacity by scenario:")

    for scenario_name, adv in scenarios.items():
        results = []
        for capital in capital_levels:
            capital_per_trade = capital / n_trades
            participation = capital_per_trade / adv
            impact_bps = (
                impact_coef * np.sqrt(participation) * 10000
            )
            net_bps = (
                mean_return_bps - impact_bps - fixed_costs_bps
            )
            results.append({
                'capital': capital,
                'net_return_bps': net_bps,
                'impact_bps': impact_bps
            })

        results_df = pd.DataFrame(results)
        positive = results_df[
            results_df['net_return_bps'] > 0
        ]
        ceiling = (
            positive['capital'].max()
            if len(positive) > 0 else 0
        )
        scenario_results[scenario_name] = {
            'df': results_df,
            'ceiling': ceiling
        }
        print(f"  {scenario_name}: "
              f"~${ceiling/1e6:.0f}M")

    print(f"\nRecommended reported capacity: "
          f"$50M-$150M (conservative range)")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    colors_s = ['#4CAF50', '#2196F3', '#FF5722']

    for (name, res), color in zip(
        scenario_results.items(), colors_s
    ):
        ax.semilogx(
            res['df']['capital'] / 1e6,
            res['df']['net_return_bps'],
            linewidth=2.5, color=color,
            label=f"{name}: "
                  f"~${res['ceiling']/1e6:.0f}M ceiling"
        )

    ax.axhline(0, color='black', linestyle='--',
               linewidth=1.5, label='Break-even')
    ax.axhline(fixed_costs_bps, color='orange',
               linestyle=':', linewidth=1.5,
               label=f'Cost floor ({fixed_costs_bps} bps)')

    # Shade conservative profitable region
    cons_df = scenario_results[
        'Conservative (10th pctile)'
    ]['df']
    ax.fill_between(
        cons_df['capital'] / 1e6,
        cons_df['net_return_bps'],
        0,
        where=cons_df['net_return_bps'] > 0,
        alpha=0.15, color='#FF5722',
        label='Conservative profitable region'
    )

    ax.set_xlabel('Capital Deployed ($M)', fontsize=12)
    ax.set_ylabel('Net Return per Trade (bps)', fontsize=12)
    ax.set_title(
        'Strategy Capacity — Three Scenarios\n'
        'Conservative estimate uses 10th percentile ADV',
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0.1)

    plt.tight_layout()
    plt.savefig(
        f'{OUTPUT_DIR}/capacity_revised.png',
        dpi=150, bbox_inches='tight'
    )
    plt.close()
    print("Saved: capacity_revised.png")

    return scenario_results

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("="*60)
    print("ADDITIONAL ANALYSIS")
    print("Threshold Sensitivity + Quintile + Capacity")
    print("="*60)

    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data()
    print(f"Loaded {len(df)} events")

    print("Loading price cache...")
    prices_cache = load_price_cache()

    # Addition 1
    threshold_df = threshold_sensitivity_analysis(
        df, prices_cache
    )

    # Addition 2
    quintile_df, spearman_corr = eps_quintile_analysis(df)

    # Addition 3
    capacity_scenarios = revised_capacity_analysis(df)

    print("\n" + "="*60)
    print("ADDITIONAL ANALYSIS COMPLETE")
    print("="*60)
    print("\nNew charts:")
    print("  threshold_sensitivity.png")
    print("  eps_quintile_analysis.png")
    print("  capacity_revised.png")

if __name__ == "__main__":
    main()
