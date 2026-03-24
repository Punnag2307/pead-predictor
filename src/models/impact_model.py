import pandas as pd
import numpy as np
import os
import sys
import random
import json
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)))
random.seed(42)
np.random.seed(42)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    mean_squared_error, mean_absolute_error
)
import xgboost as xgb
import shap

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

FEATURE_COLS = [
    'eps_surprise_pct_winsorized',
    'eps_surprise_abs',
    'eps_surprise_direction',
    'actual_eps',
    'consensus_eps',
    'report_time_flag',
    'earnings_quality',
    'both_beat',
    'both_miss',
    'sector_relative_surprise',
    'pre_return_5d',
    'pre_return_20d',
    'pre_vol_20d',
    'pre_volume_ratio',
    'price_to_52w_high',
    'price_to_52w_low',
    'vix_level',
    'vix_percentile',
    'opening_gap_prev',
    'sector_encoded',
    'month',
    'quarter',
    'day_of_week',
    'is_earnings_season'
]

DIRECTION_LABEL = 'label_direction'
MAGNITUDE_LABEL = 'abnormal_return_1d'

TRAIN_END = '2023-12-31'
VAL_END   = '2024-12-31'

# ─────────────────────────────────────────────
# DATA LOADING AND SPLITTING
# ─────────────────────────────────────────────

def load_feature_matrix() -> pd.DataFrame:
    """Load feature matrix from CSV."""
    path = 'data/processed/feature_matrix.csv'
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Feature matrix not found. "
            "Run feature_engineering.py first."
        )
    df = pd.read_csv(path)
    df['event_date'] = pd.to_datetime(df['event_date'])
    print(f"Loaded feature matrix: {df.shape}")
    return df

def time_based_split(df: pd.DataFrame) -> tuple:
    """Split data by time — no leakage."""
    train = df[df['event_date'] <= TRAIN_END].copy()
    val = df[
        (df['event_date'] > TRAIN_END) &
        (df['event_date'] <= VAL_END)
    ].copy()
    test = df[df['event_date'] > VAL_END].copy()

    print(f"\nTime-based split:")
    print(f"  Train:      {len(train)} events "
          f"({train['event_date'].min().date()} to "
          f"{train['event_date'].max().date()})")
    print(f"  Validation: {len(val)} events "
          f"({val['event_date'].min().date()} to "
          f"{val['event_date'].max().date()})")
    print(f"  Test (OOS): {len(test)} events "
          f"({test['event_date'].min().date()} to "
          f"{test['event_date'].max().date()})")

    return train, val, test

def prepare_arrays(df: pd.DataFrame) -> tuple:
    """Prepare feature matrix with median imputation."""
    X = df[FEATURE_COLS].copy()
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    y_direction = df[DIRECTION_LABEL].values
    y_magnitude = df[MAGNITUDE_LABEL].values
    return X, y_direction, y_magnitude

# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────

def train_logistic_regression(
    X_train, y_train, X_val, y_val
) -> dict:
    """Baseline: Logistic Regression with calibration."""
    print("\n--- Training Logistic Regression ---")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    lr = LogisticRegression(
        C=1.0, max_iter=1000,
        random_state=42, class_weight='balanced'
    )
    lr.fit(X_train_sc, y_train)

    val_preds = lr.predict(X_val_sc)
    val_proba = lr.predict_proba(X_val_sc)[:, 1]
    y_val_bin = (y_val == 1).astype(int)

    acc = accuracy_score(y_val, val_preds)
    auc = roc_auc_score(y_val_bin, val_proba)

    print(f"  Validation Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Validation AUC:      {auc:.4f}")

    return {
        'model': lr,
        'scaler': scaler,
        'val_accuracy': acc,
        'val_auc': auc,
        'val_preds_dir': val_preds,
        'val_proba': val_proba
    }

def train_random_forest(
    X_train, y_train_dir, y_train_mag,
    X_val, y_val_dir, y_val_mag
) -> dict:
    """Middle model: Random Forest with calibration."""
    print("\n--- Training Random Forest ---")

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    clf.fit(X_train, y_train_dir)

    # Calibrate probabilities
    calibrated = CalibratedClassifierCV(
        clf, method='isotonic', cv='prefit'
    )
    y_train_bin = (y_train_dir == 1).astype(int)
    calibrated.fit(X_train, y_train_bin)

    val_proba = calibrated.predict_proba(X_val)[:, 1]
    val_preds = np.where(val_proba > 0.5, 1, -1)
    y_val_bin = (y_val_dir == 1).astype(int)

    acc = accuracy_score(y_val_dir, val_preds)
    auc = roc_auc_score(y_val_bin, val_proba)
    print(f"  Classification - Accuracy: {acc:.4f}, "
          f"AUC: {auc:.4f}")

    # Regression
    reg = RandomForestRegressor(
        n_estimators=300, max_depth=8,
        min_samples_leaf=10, random_state=42,
        n_jobs=-1
    )
    reg.fit(X_train, y_train_mag)
    val_preds_mag = reg.predict(X_val)
    rmse = np.sqrt(
        mean_squared_error(y_val_mag, val_preds_mag)
    )
    corr = np.corrcoef(y_val_mag, val_preds_mag)[0, 1]
    print(f"  Regression    - RMSE: {rmse:.4f}, "
          f"Corr: {corr:.4f}")

    return {
        'classifier': calibrated,
        'regressor': reg,
        'val_accuracy': acc,
        'val_auc': auc,
        'val_preds_dir': val_preds,
        'val_proba': val_proba
    }

def train_xgboost(
    X_train, y_train_dir, y_train_mag,
    X_val, y_val_dir, y_val_mag
) -> dict:
    """Primary model: XGBoost with calibration + SHAP."""
    print("\n--- Training XGBoost ---")

    y_train_bin = (y_train_dir == 1).astype(int)
    y_val_bin = (y_val_dir == 1).astype(int)

    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='auc',
        early_stopping_rounds=10,
        verbosity=0
    )
    clf.fit(
        X_train, y_train_bin,
        eval_set=[(X_val, y_val_bin)],
        verbose=False
    )

    # Calibrate probabilities using isotonic regression
    calibrated = CalibratedClassifierCV(
        clf, method='isotonic', cv='prefit'
    )
    calibrated.fit(X_train, y_train_bin)

    val_proba = calibrated.predict_proba(X_val)[:, 1]
    val_preds = np.where(val_proba > 0.5, 1, -1)

    acc = accuracy_score(y_val_dir, val_preds)
    auc = roc_auc_score(y_val_bin, val_proba)
    print(f"  Classification - Accuracy: {acc:.4f}, "
          f"AUC: {auc:.4f}")

    # Regression
    reg = xgb.XGBRegressor(
        n_estimators=300, max_depth=4,
        learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=10,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42,
        early_stopping_rounds=10,
        verbosity=0
    )
    reg.fit(
        X_train, y_train_mag,
        eval_set=[(X_val, y_val_mag)],
        verbose=False
    )
    val_preds_mag = reg.predict(X_val)
    rmse = np.sqrt(
        mean_squared_error(y_val_mag, val_preds_mag)
    )
    corr = np.corrcoef(y_val_mag, val_preds_mag)[0, 1]
    print(f"  Regression    - RMSE: {rmse:.4f}, "
          f"Corr: {corr:.4f}")

    # SHAP on base classifier
    print("\n  Computing SHAP values...")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_val)

    shap_importance = pd.DataFrame({
        'feature': FEATURE_COLS,
        'shap_importance': np.abs(
            shap_values
        ).mean(axis=0)
    }).sort_values('shap_importance', ascending=False)

    print(f"\n  Top 10 features by SHAP importance:")
    print(shap_importance.head(10).to_string(index=False))

    return {
        'classifier': calibrated,
        'base_classifier': clf,
        'regressor': reg,
        'val_accuracy': acc,
        'val_auc': auc,
        'val_preds_dir': val_preds,
        'val_proba': val_proba,
        'shap_values': shap_values,
        'shap_importance': shap_importance
    }

def build_ensemble(
    lr_results: dict,
    rf_results: dict,
    xgb_results: dict,
    X_val, y_val_dir,
    weights: tuple = (0.2, 0.3, 0.5)
) -> dict:
    """
    Ensemble three models with weighted averaging.
    Weights: LR=0.2, RF=0.3, XGBoost=0.5
    Ensembles are more stable than individual models.
    """
    print("\n--- Building Ensemble ---")

    lr_proba = lr_results['val_proba']
    rf_proba = rf_results['val_proba']
    xgb_proba = xgb_results['val_proba']

    # Weighted ensemble
    w_lr, w_rf, w_xgb = weights
    ensemble_proba = (
        w_lr * lr_proba +
        w_rf * rf_proba +
        w_xgb * xgb_proba
    )

    ensemble_preds = np.where(ensemble_proba > 0.5, 1, -1)
    y_val_bin = (y_val_dir == 1).astype(int)

    acc = accuracy_score(y_val_dir, ensemble_preds)
    auc = roc_auc_score(y_val_bin, ensemble_proba)

    print(f"  Ensemble Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Ensemble AUC:      {auc:.4f}")
    print(f"  Proba range: "
          f"[{ensemble_proba.min():.3f}, "
          f"{ensemble_proba.max():.3f}]")

    # Threshold analysis
    print(f"\n  Threshold analysis:")
    for t in [0.50, 0.55, 0.60, 0.65]:
        mask = ensemble_proba > t
        if mask.sum() > 0:
            t_acc = accuracy_score(
                y_val_dir[mask],
                ensemble_preds[mask]
            )
            print(f"    > {t}: {mask.sum()} trades, "
                  f"{t_acc*100:.1f}% accuracy")

    return {
        'val_accuracy': acc,
        'val_auc': auc,
        'val_preds_dir': ensemble_preds,
        'val_proba': ensemble_proba
    }

# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def calculate_strategy_metrics(
    y_true_mag: np.ndarray,
    y_pred_dir: np.ndarray,
    label: str = ""
) -> dict:
    """Calculate strategy-level metrics."""
    strategy_returns = y_true_mag * y_pred_dir
    mean_return = strategy_returns.mean()
    std_return = strategy_returns.std()
    sharpe = (
        mean_return / std_return * np.sqrt(252)
        if std_return > 0 else 0
    )
    win_rate = (strategy_returns > 0).mean()

    from scipy import stats
    t_stat, p_value = stats.ttest_1samp(strategy_returns, 0)

    print(f"\n  Strategy metrics ({label}):")
    print(f"    Mean return:  {mean_return*100:.3f}%")
    print(f"    Win rate:     {win_rate*100:.1f}%")
    print(f"    Sharpe:       {sharpe:.2f}")
    print(f"    T-stat:       {t_stat:.2f}")
    print(f"    P-value:      {p_value:.4f}")
    print(f"    Significant:  {p_value < 0.05}")

    return {
        'mean_return': mean_return,
        'win_rate': win_rate,
        'sharpe': sharpe,
        't_stat': t_stat,
        'p_value': p_value
    }

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("="*60)
    print("PHASE 5: MODEL TRAINING & EVALUATION")
    print("="*60)

    # Load data
    df = load_feature_matrix()
    train, val, test = time_based_split(df)

    X_train, y_train_dir, y_train_mag = prepare_arrays(train)
    X_val, y_val_dir, y_val_mag = prepare_arrays(val)
    X_test, y_test_dir, y_test_mag = prepare_arrays(test)

    print(f"\nFeature matrix shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")

    # Train all three models
    lr_results = train_logistic_regression(
        X_train, y_train_dir, X_val, y_val_dir
    )
    rf_results = train_random_forest(
        X_train, y_train_dir, y_train_mag,
        X_val, y_val_dir, y_val_mag
    )
    xgb_results = train_xgboost(
        X_train, y_train_dir, y_train_mag,
        X_val, y_val_dir, y_val_mag
    )

    # Build ensemble
    ensemble_results = build_ensemble(
        lr_results, rf_results, xgb_results,
        X_val, y_val_dir
    )

    # Model comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)

    all_results = {
        'Logistic Regression': lr_results,
        'Random Forest': rf_results,
        'XGBoost': xgb_results,
        'Ensemble': ensemble_results
    }

    print(f"\n{'Model':<25} {'Accuracy':>10} {'AUC':>8}")
    print("-"*45)
    for name, res in all_results.items():
        print(f"{name:<25} "
              f"{res['val_accuracy']*100:>9.1f}% "
              f"{res['val_auc']:>8.4f}")

    # OOS test with ensemble
    print("\n" + "="*60)
    print("OUT-OF-SAMPLE TEST (2025)")
    print("="*60)

    if len(X_test) > 0:
        # Get OOS probabilities from all models
        scaler = lr_results['scaler']
        X_test_sc = scaler.transform(X_test)

        lr_test_proba = lr_results['model'].predict_proba(
            X_test_sc
        )[:, 1]
        rf_test_proba = rf_results['classifier'].predict_proba(
            X_test
        )[:, 1]
        xgb_test_proba = xgb_results['classifier'].predict_proba(
            X_test
        )[:, 1]

        # Ensemble OOS
        ensemble_test_proba = (
            0.2 * lr_test_proba +
            0.3 * rf_test_proba +
            0.5 * xgb_test_proba
        )
        ensemble_test_preds = np.where(
            ensemble_test_proba > 0.5, 1, -1
        )

        y_test_bin = (y_test_dir == 1).astype(int)
        test_acc = accuracy_score(
            y_test_dir, ensemble_test_preds
        )
        test_auc = roc_auc_score(y_test_bin, ensemble_test_proba)

        print(f"\nEnsemble OOS results:")
        print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
        print(f"  AUC:      {test_auc:.4f}")
        print(f"  Proba range: [{ensemble_test_proba.min():.3f}, "
              f"{ensemble_test_proba.max():.3f}]")

        # Threshold analysis on OOS
        print(f"\n  OOS threshold analysis:")
        for t in [0.50, 0.55, 0.60, 0.65]:
            mask = ensemble_test_proba > t
            if mask.sum() > 10:
                t_acc = accuracy_score(
                    y_test_dir[mask],
                    ensemble_test_preds[mask]
                )
                print(f"    > {t}: {mask.sum()} trades, "
                      f"{t_acc*100:.1f}% accuracy")

        # Save predictions for backtest
        df_copy = df.copy()
        X_all, y_all_dir, _ = prepare_arrays(df_copy)

        X_all_sc = scaler.transform(X_all)
        lr_all_proba = lr_results['model'].predict_proba(
            X_all_sc
        )[:, 1]
        rf_all_proba = rf_results['classifier'].predict_proba(
            X_all
        )[:, 1]
        xgb_all_proba = xgb_results['classifier'].predict_proba(
            X_all
        )[:, 1]

        ensemble_all_proba = (
            0.2 * lr_all_proba +
            0.3 * rf_all_proba +
            0.5 * xgb_all_proba
        )
        ensemble_all_preds = np.where(
            ensemble_all_proba > 0.5, 1, -1
        )

        df_copy['predicted_direction'] = ensemble_all_preds
        df_copy['predicted_proba'] = ensemble_all_proba
        df_copy.to_csv(
            'data/processed/feature_matrix_with_preds.csv',
            index=False
        )
        print(f"\nSaved ensemble predictions.")
        print(f"Proba range (all): "
              f"[{ensemble_all_proba.min():.3f}, "
              f"{ensemble_all_proba.max():.3f}]")

    # Save SHAP importance
    shap_df = xgb_results['shap_importance']
    shap_df.to_csv(
        'data/processed/shap_importance.csv', index=False
    )

    # Save results summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'train_size': len(train),
        'val_size': len(val),
        'test_size': len(test),
        'models': {
            name: {
                'val_accuracy': float(res['val_accuracy']),
                'val_auc': float(res['val_auc'])
            }
            for name, res in all_results.items()
        }
    }
    with open('data/processed/model_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\nPhase 5 complete.")

if __name__ == "__main__":
    main()
