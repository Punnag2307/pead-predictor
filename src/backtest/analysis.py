import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.database import get_connection

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

BID_ASK_SPREAD_BPS = 5
MARKET_IMPACT_BPS = 10
COMMISSION_BPS = 1
SLIPPAGE_BPS = 3
TOTAL_ONE_WAY_BPS = (
    BID_ASK_SPREAD_BPS + MARKET_IMPACT_BPS +
    COMMISSION_BPS + SLIPPAGE_BPS
)
ROUND_TRIP_COST = (TOTAL_ONE_WAY_BPS * 2) / 10000

OUTPUT_DIR = 'data/processed/charts'

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_backtest_data() -> pd.DataFrame:
    """Load feature matrix with model predictions."""
    feature_df = pd.read_csv(
        'data/processed/feature_matrix_with_preds.csv'
    )
    feature_df['event_date'] = pd.to_datetime(
        feature_df['event_date']
    )
    return feature_df

def load_price_cache() -> dict:
    """Load all prices into memory."""
    conn = get_connection()
    tickers_df = pd.read_sql(
        "SELECT DISTINCT ticker FROM price_data", conn
    )
    conn.close()

    prices = {}
    for ticker in tickers_df['ticker']:
        conn = get_connection()
        df = pd.read_sql("""
            SELECT date, open, high, low, close, volume
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

# ─────────────────────────────────────────────
# ALPHA DECAY ANALYSIS
# ─────────────────────────────────────────────

def calculate_alpha_decay(
    feature_df: pd.DataFrame,
    prices_cache: dict
) -> pd.DataFrame:
    """
    Calculate average returns at different holding horizons.
    Uses model predicted direction for realistic decay.
    Only includes high conviction trades (proba > 0.60).
    Horizons: 1d, 2d, 3d, 5d, 10d
    """
    horizons = [1, 2, 3, 5, 10]
    results = {h: [] for h in horizons}

    for _, row in feature_df.iterrows():
        ticker = row['ticker']
        event_date = pd.Timestamp(row['event_date'])
        direction = row['predicted_direction']
        report_time = row.get('report_time', None)
        predicted_proba = row.get('predicted_proba', 0.5)

        # Only high conviction trades
        if predicted_proba < 0.65:
            continue

        if ticker not in prices_cache:
            continue

        prices = prices_cache[ticker]
        all_dates = prices.index.sort_values()
        future_dates = all_dates[all_dates >= event_date]

        if len(future_dates) < 2:
            continue

        if report_time == 'BMO':
            reaction_day_idx = 0
        else:
            reaction_day_idx = 1

        if reaction_day_idx >= len(future_dates):
            continue

        reaction_day = future_dates[reaction_day_idx]
        reaction_pos = all_dates.get_loc(reaction_day)

        # Get prev close for overnight gap
        prev_dates = all_dates[all_dates < reaction_day]
        if len(prev_dates) == 0:
            continue
        prev_day = prev_dates[-1]

        for horizon in horizons:
            end_pos = min(
                reaction_pos + horizon - 1,
                len(all_dates) - 1
            )
            end_day = all_dates[end_pos]

            try:
                prev_close = prices.loc[prev_day, 'close']
                reaction_open = prices.loc[reaction_day, 'open']
                opening_gap = (reaction_open / prev_close - 1)

                if horizon > 1:
                    subsequent = (
                        prices.loc[end_day, 'close'] /
                        prices.loc[reaction_day, 'open'] - 1
                    )
                    total_return = opening_gap + subsequent
                else:
                    total_return = opening_gap

                directional_return = total_return * direction
                results[horizon].append(directional_return)

            except Exception:
                continue

    # Calculate statistics per horizon
    decay_data = []
    for horizon in horizons:
        rets = results[horizon]
        if rets:
            decay_data.append({
                'horizon_days': horizon,
                'mean_return': np.mean(rets) * 100,
                'std_return': np.std(rets) * 100,
                'se': np.std(rets) / np.sqrt(len(rets)) * 100,
                'n_trades': len(rets),
                'win_rate': np.mean([r > 0 for r in rets]) * 100
            })

    return pd.DataFrame(decay_data)

# ─────────────────────────────────────────────
# BACKTESTING
# ─────────────────────────────────────────────

def run_backtest(
    feature_df: pd.DataFrame,
    predictions: np.ndarray,
    prices_cache: dict,
    label: str = "Strategy",
    confidence_threshold: float = 0.65,
    use_conviction_sizing: bool = False
) -> dict:
    """
    Simulate trading on model predictions.

    Entry: overnight gap (prev close to reaction day open)
    Exit: reaction day open (gap capture only)

    Parameters:
    - confidence_threshold: minimum predicted probability to trade
    - use_conviction_sizing: if True, scale position by conviction
      if False, equal weight all trades (default — better net Sharpe)

    Transaction costs: 38 bps round trip
    (5 bps spread + 10 bps impact + 1 bps commission + 3 bps slippage)
    """
    trades = []

    for i, (_, row) in enumerate(feature_df.iterrows()):
        ticker = row['ticker']
        event_date = pd.Timestamp(row['event_date'])
        predicted_dir = predictions[i]
        actual_return = row['abnormal_return_1d']
        report_time = row.get('report_time', None)

        # Conviction filter
        predicted_proba = row.get('predicted_proba', 0.5)
        if predicted_proba < confidence_threshold:
            continue

        # Position sizing
        if use_conviction_sizing:
            position_size = (predicted_proba - 0.5) * 2
        else:
            position_size = 1.0

        if ticker not in prices_cache:
            continue

        prices = prices_cache[ticker]
        all_dates = prices.index.sort_values()
        future_dates = all_dates[all_dates >= event_date]

        if len(future_dates) < 2:
            continue

        # Determine reaction day
        if report_time == 'BMO':
            reaction_day = future_dates[0]
        else:
            reaction_day = future_dates[1]

        try:
            # Overnight gap return
            prev_dates = all_dates[all_dates < reaction_day]
            if len(prev_dates) == 0:
                continue
            prev_day = prev_dates[-1]

            prev_close = prices.loc[prev_day, 'close']
            reaction_open = prices.loc[reaction_day, 'open']

            # Gross return scaled by position size
            gross_return = (
                reaction_open / prev_close - 1
            ) * predicted_dir * position_size

            # Net return after round-trip costs
            net_return = gross_return - ROUND_TRIP_COST

            trades.append({
                'date': reaction_day,
                'ticker': ticker,
                'predicted_dir': predicted_dir,
                'predicted_proba': predicted_proba,
                'position_size': position_size,
                'actual_return': actual_return,
                'gross_return': gross_return,
                'net_return': net_return,
                'correct': int(
                    predicted_dir == row['label_direction']
                )
            })

        except Exception:
            continue

    if not trades:
        print(f"  No trades found for {label}")
        return {}

    trades_df = pd.DataFrame(trades)
    trades_df = trades_df.sort_values('date')

    # Cumulative returns
    trades_df['cum_gross'] = (
        1 + trades_df['gross_return']
    ).cumprod()
    trades_df['cum_net'] = (
        1 + trades_df['net_return']
    ).cumprod()

    # Sharpe ratios
    gross_sharpe = (
        trades_df['gross_return'].mean() /
        trades_df['gross_return'].std() * np.sqrt(252)
    ) if trades_df['gross_return'].std() > 0 else 0

    net_sharpe = (
        trades_df['net_return'].mean() /
        trades_df['net_return'].std() * np.sqrt(252)
    ) if trades_df['net_return'].std() > 0 else 0

    # Max drawdown
    cum_returns = trades_df['cum_net'].values
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min()

    # Win rate
    win_rate = trades_df['correct'].mean()

    # Cost drag
    cost_drag = abs(
        gross_sharpe - net_sharpe
    ) / max(abs(gross_sharpe), 0.01) * 100

    print(f"\n{label} Backtest Results:")
    print(f"  Total trades:    {len(trades_df)}")
    print(f"  Win rate:        {win_rate*100:.1f}%")
    print(f"  Gross Sharpe:    {gross_sharpe:.2f}")
    print(f"  Net Sharpe:      {net_sharpe:.2f}")
    print(f"  Max Drawdown:    {max_drawdown*100:.1f}%")
    print(f"  Round-trip cost: {ROUND_TRIP_COST*100:.2f}%")
    print(f"  Cost drag:       {cost_drag:.1f}% of gross Sharpe")

    return {
        'trades_df': trades_df,
        'gross_sharpe': gross_sharpe,
        'net_sharpe': net_sharpe,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'n_trades': len(trades_df)
    }

# ─────────────────────────────────────────────
# CHART GENERATION
# ─────────────────────────────────────────────

def generate_research_charts(
    feature_df: pd.DataFrame,
    decay_df: pd.DataFrame,
    backtest_results: dict,
    shap_df: pd.DataFrame
):
    """Generate all 6 charts for the research note."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0', '#FF9800']

    # ── Chart 1: Alpha Decay Curve ──
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        decay_df['horizon_days'],
        decay_df['mean_return'],
        yerr=decay_df['se'] * 1.96,
        marker='o', linewidth=2.5,
        color=colors[0], capsize=5,
        label='Mean directional return (model, proba>0.60)'
    )
    ax.axhline(0, color='red', linestyle='--', alpha=0.7)
    ax.fill_between(
        decay_df['horizon_days'],
        decay_df['mean_return'] - decay_df['se'] * 1.96,
        decay_df['mean_return'] + decay_df['se'] * 1.96,
        alpha=0.2, color=colors[0]
    )
    ax.set_xlabel('Holding Period (Days)', fontsize=12)
    ax.set_ylabel('Mean Directional Return (%)', fontsize=12)
    ax.set_title(
        'Alpha Decay Curve — Post-Earnings Signal',
        fontsize=14, fontweight='bold'
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        f'{OUTPUT_DIR}/alpha_decay.png',
        dpi=150, bbox_inches='tight'
    )
    plt.close()
    print("Saved: alpha_decay.png")

    # ── Chart 2: EPS Surprise vs Abnormal Return ──
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = feature_df[
        (feature_df['eps_surprise_pct_winsorized'].abs() < 50) &
        (feature_df['abnormal_return_1d'].abs() < 0.3)
    ].copy()

    ax.scatter(
        plot_df['eps_surprise_pct_winsorized'],
        plot_df['abnormal_return_1d'] * 100,
        alpha=0.3, s=10,
        c=plot_df['label_direction'],
        cmap='RdYlGn'
    )

    # Trend line
    valid = plot_df.dropna(
        subset=['eps_surprise_pct_winsorized', 'abnormal_return_1d']
    )
    z = np.polyfit(
        valid['eps_surprise_pct_winsorized'],
        valid['abnormal_return_1d'] * 100,
        1
    )
    p = np.poly1d(z)
    x_line = np.linspace(
        valid['eps_surprise_pct_winsorized'].min(),
        valid['eps_surprise_pct_winsorized'].max(),
        100
    )
    ax.plot(x_line, p(x_line), 'r-', linewidth=2, label='Trend')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('EPS Surprise (%)', fontsize=12)
    ax.set_ylabel('Abnormal Return (%)', fontsize=12)
    ax.set_title(
        'EPS Surprise vs Post-Earnings Abnormal Return',
        fontsize=14, fontweight='bold'
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        f'{OUTPUT_DIR}/eps_vs_return.png',
        dpi=150, bbox_inches='tight'
    )
    plt.close()
    print("Saved: eps_vs_return.png")

    # ── Chart 3: SHAP Feature Importance ──
    fig, ax = plt.subplots(figsize=(10, 7))
    top_features = shap_df.head(15)
    ax.barh(
        top_features['feature'][::-1],
        top_features['shap_importance'][::-1],
        color=colors[0], alpha=0.8
    )
    ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
    ax.set_title(
        'Feature Importance — XGBoost SHAP Values',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(
        f'{OUTPUT_DIR}/shap_importance.png',
        dpi=150, bbox_inches='tight'
    )
    plt.close()
    print("Saved: shap_importance.png")

    # ── Chart 4: Cumulative PnL ──
    if 'trades_df' in backtest_results:
        fig, ax = plt.subplots(figsize=(12, 6))
        trades_df = backtest_results['trades_df']

        ax.plot(
            trades_df['date'],
            (trades_df['cum_gross'] - 1) * 100,
            label=f"Gross (Sharpe: "
                  f"{backtest_results['gross_sharpe']:.2f})",
            color=colors[0], linewidth=2
        )
        ax.plot(
            trades_df['date'],
            (trades_df['cum_net'] - 1) * 100,
            label=f"Net of costs (Sharpe: "
                  f"{backtest_results['net_sharpe']:.2f})",
            color=colors[1], linewidth=2
        )
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.fill_between(
            trades_df['date'],
            (trades_df['cum_net'] - 1) * 100,
            0,
            where=(trades_df['cum_net'] - 1) > 0,
            alpha=0.1, color=colors[1]
        )
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax.set_title(
            'Strategy Cumulative PnL — Post-Earnings Signal',
            fontsize=14, fontweight='bold'
        )
        ax.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(
            f'{OUTPUT_DIR}/cumulative_pnl.png',
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        print("Saved: cumulative_pnl.png")

    # ── Chart 5: Sector Breakdown ──
    fig, ax = plt.subplots(figsize=(12, 6))
    sector_stats = feature_df.groupby('sector').agg(
        mean_return=('abnormal_return_1d', 'mean'),
        count=('abnormal_return_1d', 'count'),
        win_rate=(
            'label_direction',
            lambda x: (x == 1).mean()
        )
    ).reset_index()
    sector_stats = sector_stats.sort_values(
        'mean_return', ascending=True
    )
    bar_colors = [
        colors[1] if r > 0 else colors[2]
        for r in sector_stats['mean_return']
    ]
    ax.barh(
        sector_stats['sector'],
        sector_stats['mean_return'] * 100,
        color=bar_colors, alpha=0.8
    )
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Mean Abnormal Return (%)', fontsize=12)
    ax.set_title(
        'Post-Earnings Abnormal Return by Sector',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(
        f'{OUTPUT_DIR}/sector_breakdown.png',
        dpi=150, bbox_inches='tight'
    )
    plt.close()
    print("Saved: sector_breakdown.png")

    # ── Chart 6: Model Comparison ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    models = ['Logistic\nRegression', 'Random\nForest', 'XGBoost']
    accuracies = [62.7, 64.3, 64.3]
    aucs = [0.6955, 0.6871, 0.6895]

    ax1.bar(models, accuracies, color=colors[:3], alpha=0.8)
    ax1.axhline(
        50, color='red', linestyle='--',
        label='Random baseline (50%)'
    )
    ax1.set_ylabel('Directional Accuracy (%)', fontsize=12)
    ax1.set_title(
        'Model Accuracy Comparison',
        fontsize=13, fontweight='bold'
    )
    ax1.legend()
    ax1.set_ylim(45, 70)

    ax2.bar(models, aucs, color=colors[:3], alpha=0.8)
    ax2.axhline(
        0.5, color='red', linestyle='--',
        label='Random baseline (0.5)'
    )
    ax2.set_ylabel('AUC-ROC', fontsize=12)
    ax2.set_title(
        'Model AUC Comparison',
        fontsize=13, fontweight='bold'
    )
    ax2.legend()
    ax2.set_ylim(0.45, 0.75)

    plt.tight_layout()
    plt.savefig(
        f'{OUTPUT_DIR}/model_comparison.png',
        dpi=150, bbox_inches='tight'
    )
    plt.close()
    print("Saved: model_comparison.png")

    print(f"\nAll charts saved to {OUTPUT_DIR}/")

# ─────────────────────────────────────────────
# STATISTICAL VALIDATION
# ─────────────────────────────────────────────

def bootstrap_sharpe(
    returns: np.ndarray,
    n_bootstrap: int = 10000
) -> tuple:
    """Bootstrap 95% confidence interval for Sharpe ratio."""
    def sharpe(r):
        return (
            r.mean() / r.std() * np.sqrt(252)
            if r.std() > 0 else 0
        )

    bootstrap_sharpes = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(
            returns, size=len(returns), replace=True
        )
        bootstrap_sharpes.append(sharpe(sample))

    ci_low = np.percentile(bootstrap_sharpes, 2.5)
    ci_high = np.percentile(bootstrap_sharpes, 97.5)
    return ci_low, ci_high

def run_statistical_validation(backtest_results: dict):
    """Full statistical validation of strategy results."""
    if 'trades_df' not in backtest_results:
        return

    trades_df = backtest_results['trades_df']
    net_returns = trades_df['net_return'].values

    print("\n=== STATISTICAL VALIDATION ===")

    from scipy import stats
    t_stat, p_value = stats.ttest_1samp(net_returns, 0)
    print(f"\nT-test (H0: mean return = 0):")
    print(f"  T-statistic: {t_stat:.3f}")
    print(f"  P-value:     {p_value:.6f}")
    print(f"  Significant: {p_value < 0.05}")

    ci_low, ci_high = bootstrap_sharpe(net_returns)
    print(f"\nBootstrap Sharpe (95% CI):")
    print(f"  Point estimate: {backtest_results['net_sharpe']:.2f}")
    print(f"  95% CI:         [{ci_low:.2f}, {ci_high:.2f}]")

    print(f"\nMultiple testing note:")
    print(f"  Testing 3 models — Bonferroni adjusted alpha: 0.0167")
    print(
        f"  Significant at adjusted level: {p_value < 0.0167}"
    )

    return {
        't_stat': t_stat,
        'p_value': p_value,
        'sharpe_ci': (ci_low, ci_high)
    }

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("="*60)
    print("PHASE 6: BACKTESTING + RESEARCH NOTE")
    print("="*60)

    # Load data
    feature_df = load_backtest_data()
    print(f"Loaded {len(feature_df)} events")

    # Load prices
    print("Loading price cache...")
    prices_cache = load_price_cache()

    # Load SHAP importance
    shap_df = pd.read_csv('data/processed/shap_importance.csv')

    # ── Alpha Decay Analysis ──
    print("\n=== ALPHA DECAY ANALYSIS ===")
    print("Calculating returns at multiple horizons...")
    decay_df = calculate_alpha_decay(feature_df, prices_cache)
    print(f"\nAlpha decay results:")
    print(decay_df.to_string(index=False))
    decay_df.to_csv(
        'data/processed/alpha_decay.csv', index=False
    )

    # ── Backtesting ──
    print("\n=== BACKTESTING ===")
    predictions = feature_df['predicted_direction'].values

    # Primary strategy: >0.65 equal weight
    print("\n--- Primary Strategy: >0.65 equal weight ---")
    backtest_primary = run_backtest(
        feature_df, predictions, prices_cache,
        label="Full Dataset (>0.65 equal weight)",
        confidence_threshold=0.65,
        use_conviction_sizing=False
    )

    # 2025 OOS test
    test_df = feature_df[
        feature_df['event_date'] >= '2025-01-01'
    ].copy()
    test_predictions = test_df['predicted_direction'].values

    if len(test_df) > 0:
        print(f"\n--- 2025 Out-of-Sample Test ---")
        test_results = run_backtest(
            test_df, test_predictions, prices_cache,
            label="2025 OOS (>0.65 equal weight)",
            confidence_threshold=0.65,
            use_conviction_sizing=False
        )

    # Statistical validation on primary strategy
    run_statistical_validation(backtest_primary)

    # ── Generate Charts ──
    print("\n=== GENERATING RESEARCH CHARTS ===")
    generate_research_charts(
        feature_df, decay_df, backtest_primary, shap_df
    )

    print("\n" + "="*60)
    print("PHASE 6 COMPLETE")
    print("="*60)
    print("\nDeliverables:")
    print("  data/processed/alpha_decay.csv")
    print("  data/processed/charts/alpha_decay.png")
    print("  data/processed/charts/eps_vs_return.png")
    print("  data/processed/charts/shap_importance.png")
    print("  data/processed/charts/cumulative_pnl.png")
    print("  data/processed/charts/sector_breakdown.png")
    print("  data/processed/charts/model_comparison.png")

if __name__ == "__main__":
    main()
